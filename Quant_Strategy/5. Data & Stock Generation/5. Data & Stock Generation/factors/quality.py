# factors/quality.py

# ============================================================
# STAGE: 2 — QUALITY
#
# Responsibilities:
# - Apply quality filter AFTER momentum
# - Uses 5-year AVERAGE quality (stable)
# - Chronological month processing
# - Stage-safe via Generator flags
# - Idempotent and restart-safe
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from config.settings import (
    MASTER_DB_FILE,
    DATA_DIR,
    QUALITY_THRESHOLD,
    FLAG_YES,
    FLAG_NO,
    FLAG_MOMENTUM_DONE,
    FLAG_QUALITY_DONE,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"

ROE_FILE = RAW_DIR / "ROE.xlsx"
ROCE_FILE = RAW_DIR / "ROCE.xlsx"
ROA_FILE = RAW_DIR / "ROA.xlsx"

# =================================================
# HELPERS
# =================================================


def _fiscal_year(month_label: str) -> int:
    """
    Jan–Jun → previous FY
    Jul–Dec → same FY
    """
    mon, year = month_label.split()
    year = int(year)
    return year - 1 if mon in {"Jan", "Feb", "Mar", "Apr", "May", "Jun"} else year


def _last_n_fys(latest_fy: int, n: int = 5) -> List[int]:
    """
    Example:
    latest_fy = 2024 → [2024, 2023, 2022, 2021, 2020]
    """
    return [latest_fy - i for i in range(n)]


def _load_metric_long(file_path: Path, value_name: str) -> pd.DataFrame:
    """
    Converts:
        FiscalYear | TICKER1 | TICKER2 | ...
    →
        Ticker | FiscalYear | <value_name>
    """
    df = pd.read_excel(file_path)

    long = df.melt(
        id_vars=["FiscalYear"],
        var_name="Ticker",
        value_name=value_name,
    )

    long.dropna(subset=[value_name], inplace=True)
    long["FiscalYear"] = long["FiscalYear"].astype(int)

    return long


# =================================================
# CORE MONTH LOGIC
# =================================================


def _apply_quality_for_month(db: pd.DataFrame, month_label: str) -> None:
    """
    Apply quality logic for ONE month.
    Mutates db in-place.
    """

    latest_fy = _fiscal_year(month_label)
    required_fys = _last_n_fys(latest_fy, 5)

    month_mask = db["MonthYear"] == month_label

    # -------------------------------------------------
    # Universe gating (FINAL)
    # -------------------------------------------------

    mask = (
        (db["MonthYear"] == month_label)
        & (db["Generator"] == FLAG_MOMENTUM_DONE)
        & (db["TMI"] == FLAG_YES)
        & (db["Momentum"] == FLAG_YES)
    )

    universe = db.loc[mask, "Ticker"].unique().tolist()

    # -------------------------------------------------
    # Empty universe handling (MANDATORY)
    # -------------------------------------------------

    if not universe:
        db.loc[month_mask, "Quality"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_QUALITY_DONE
        return

    # -------------------------------------------------
    # Load quality metrics
    # -------------------------------------------------

    roe = _load_metric_long(ROE_FILE, "ROE")
    roce = _load_metric_long(ROCE_FILE, "ROCE")
    roa = _load_metric_long(ROA_FILE, "ROA")

    metrics = roe.merge(roce, on=["Ticker", "FiscalYear"], how="outer").merge(
        roa, on=["Ticker", "FiscalYear"], how="outer"
    )

    metrics = metrics[
        metrics["Ticker"].isin(universe) & metrics["FiscalYear"].isin(required_fys)
    ]

    # -------------------------------------------------
    # Quality rule (5Y AVERAGE)
    # -------------------------------------------------

    def is_profitable_5y_avg(ticker: str) -> bool:
        df_t = metrics[metrics["Ticker"] == ticker]

        # Must have all 5 fiscal years
        if df_t["FiscalYear"].nunique() < len(required_fys):
            return False

        roe_avg = df_t["ROE"].mean(skipna=True)
        roce_avg = df_t["ROCE"].mean(skipna=True)
        roa_avg = df_t["ROA"].mean(skipna=True)

        return (
            (roe_avg is not None and roe_avg > QUALITY_THRESHOLD)
            or (roce_avg is not None and roce_avg > QUALITY_THRESHOLD)
            or (roa_avg is not None and roa_avg > QUALITY_THRESHOLD)
        )

    # -------------------------------------------------
    # Ticker-level multithreading
    # -------------------------------------------------

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(is_profitable_5y_avg, universe))

    profitable_tickers = [t for t, ok in zip(universe, results) if ok]
    non_profitable_tickers = [t for t, ok in zip(universe, results) if not ok]

    # -------------------------------------------------
    # Persist flags
    # -------------------------------------------------

    db.loc[
        month_mask & db["Ticker"].isin(profitable_tickers),
        "Quality",
    ] = FLAG_YES

    db.loc[
        month_mask & db["Ticker"].isin(non_profitable_tickers),
        "Quality",
    ] = FLAG_NO

    # -------------------------------------------------
    # Stage transition (ALWAYS)
    # -------------------------------------------------

    db.loc[month_mask, "Generator"] = FLAG_QUALITY_DONE


# =================================================
# PUBLIC API
# =================================================


def run_quality_factor() -> None:
    """
    Stage-2 Factor: Quality

    Rules:
    - Runs ONLY when Generator == FLAG_MOMENTUM_DONE
    - Requires Momentum == YES
    - Uses 5-year average quality
    - Marks Generator → FLAG_QUALITY_DONE
    """

    db = pd.read_excel(MASTER_DB_FILE)

    pending = (
        db.loc[db["Generator"] == FLAG_MOMENTUM_DONE, "MonthYear"]
        .drop_duplicates()
        .to_frame(name="MonthYear")
    )

    if pending.empty:
        print("[PROF] No pending months")
        return

    # -------------------------------------------------
    # Chronological execution (CRITICAL)
    # -------------------------------------------------

    pending["MonthDate"] = pd.to_datetime(pending["MonthYear"], format="%b %Y")
    pending = pending.sort_values("MonthDate")

    months = pending["MonthYear"].tolist()

    for month_label in tqdm(
        months,
        desc="[PROF] Processing months",
        unit="month",
        ncols=100,
    ):
        _apply_quality_for_month(db, month_label)

    db.to_excel(MASTER_DB_FILE, index=False)

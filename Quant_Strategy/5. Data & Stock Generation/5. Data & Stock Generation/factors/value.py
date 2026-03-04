# factors/value.py

# ============================================================
# STAGE: 3 — Value
#
# Responsibilities:
# - Apply value AFTER quality
# - Chronological month processing
# - Stage-safe (Generator flag based)
# - Idempotent and restart-safe
# - Ticker-level multithreading
#
# CRITICAL FIXES:
# - Uses PREVIOUS MONTH prices (live-safe)
# - Correct fiscal year alignment
# - Pass-through when ticker count <= 10
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from config.settings import (
    MASTER_DB_FILE,
    DATA_DIR,
    FLAG_YES,
    FLAG_NO,
    FLAG_QUALITY_DONE,
    FLAG_VALUE_DONE,
    MIN_VALUE_TICKERS,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"

PRICE_FILE = RAW_DIR / "Prices.xlsx"
EPS_FILE = RAW_DIR / "EPS.xlsx"
BV_FILE = RAW_DIR / "BookValue.xlsx"
REV_FILE = RAW_DIR / "RevenuePerShare.xlsx"

# =================================================
# HELPERS
# =================================================


def _signal_asof_ts(month_label: str) -> pd.Timestamp:
    """
    For a signal month like 'Mar 2019',
    return the last calendar day of the PREVIOUS month.
    """
    first_of_month = pd.to_datetime(month_label, format="%b %Y")
    return first_of_month - pd.Timedelta(days=1)


def _fiscal_year(month_label: str) -> int:
    """
    Fiscal year mapping (India-style):
    Jan–Jun  → previous FY
    Jul–Dec  → same FY
    """
    mon, year = month_label.split()
    year = int(year)
    return year - 1 if mon in {"Jan", "Feb", "Mar", "Apr", "May", "Jun"} else year


def _last_price(
    prices: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
) -> Optional[float]:
    """
    Point-in-time safe price lookup
    """
    if ticker not in prices.columns:
        return None

    s = prices.loc[prices.index <= as_of, ticker]
    if s.empty:
        return None

    return float(s.iloc[-1])


def _load_fundamental_long(
    file_path: Path,
    value_name: str,
) -> pd.DataFrame:
    """
    Wide → long fundamentals
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


def _apply_value_for_month(
    db: pd.DataFrame,
    month_label: str,
) -> None:
    """
    Apply value logic for ONE month.
    """

    fy = _fiscal_year(month_label)
    as_of = _signal_asof_ts(month_label)

    # -------------------------------------------------
    # Universe gating (FINAL)
    # -------------------------------------------------

    mask = (
        (db["MonthYear"] == month_label)
        & (db["Generator"] == FLAG_QUALITY_DONE)
        & (db["TMI"] == FLAG_YES)
        & (db["Momentum"] == FLAG_YES)
        & (db["Quality"] == FLAG_YES)
    )

    universe = db.loc[mask].copy()
    month_mask = db["MonthYear"] == month_label

    if universe.empty:
        db.loc[month_mask, "Value"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_VALUE_DONE
        return

    # -------------------------------------------------
    # Load fundamentals
    # -------------------------------------------------

    eps = _load_fundamental_long(EPS_FILE, "EPS")
    bv = _load_fundamental_long(BV_FILE, "BV")
    rev = _load_fundamental_long(REV_FILE, "Revenue")

    fundamentals = eps.merge(bv, on=["Ticker", "FiscalYear"], how="outer").merge(
        rev, on=["Ticker", "FiscalYear"], how="outer"
    )

    fundamentals = fundamentals[fundamentals["FiscalYear"] == fy]

    universe = universe.merge(fundamentals, on="Ticker", how="left")

    # -------------------------------------------------
    # Prices (point-in-time safe)
    # -------------------------------------------------

    prices = pd.read_excel(PRICE_FILE, parse_dates=["Date"]).set_index("Date")

    with ThreadPoolExecutor(max_workers=4) as executor:
        universe["Price"] = list(
            executor.map(
                lambda t: _last_price(prices, t, as_of),
                universe["Ticker"],
            )
        )

    # -------------------------------------------------
    # Hard filters
    # -------------------------------------------------

    valid = universe[
        (universe["Price"] > 0)
        & (universe["EPS"] > 0)
        & (universe["BV"] > 0)
        & (universe["Revenue"] > 0)
    ].copy()

    if valid.empty:
        db.loc[month_mask, "Value"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_VALUE_DONE
        return

    # -------------------------------------------------
    # Value metrics (Old Logic Style)
    # -------------------------------------------------

    valid["EY"] = valid["EPS"] / valid["Price"]
    valid["BY"] = valid["BV"] / valid["Price"]
    valid["SY"] = valid["Revenue"] / valid["Price"]

    # Rank (higher yield = better value)
    valid["Rank_EY"] = valid["EY"].rank(ascending=False, method="average")
    valid["Rank_BY"] = valid["BY"].rank(ascending=False, method="average")
    valid["Rank_SY"] = valid["SY"].rank(ascending=False, method="average")

    n = len(valid)

    # Convert to percentile ranks
    valid["Pct_EY"] = valid["Rank_EY"] / n
    valid["Pct_BY"] = valid["Rank_BY"] / n
    valid["Pct_SY"] = valid["Rank_SY"] / n

    # Equal-weight average (same spirit as old code)
    valid["ValueScore"] = (valid["Pct_EY"] + valid["Pct_BY"] + valid["Pct_SY"]) / 3

    # -------------------------------------------------
    # Rank & select (FINAL RULE)
    # -------------------------------------------------

    ranked = valid.sort_values("ValueScore", ascending=False)
    n = len(ranked)

    if n <= MIN_VALUE_TICKERS:
        selected = set(ranked["Ticker"])
    else:
        cutoff = n // 2
        selected = set(ranked.iloc[:cutoff]["Ticker"])

    # -------------------------------------------------
    # Persist flags
    # -------------------------------------------------

    db.loc[
        month_mask & db["Ticker"].isin(selected),
        "Value",
    ] = FLAG_YES

    db.loc[
        month_mask & db["Ticker"].isin(valid["Ticker"]) & ~db["Ticker"].isin(selected),
        "Value",
    ] = FLAG_NO

    # -------------------------------------------------
    # Stage transition (ALWAYS)
    # -------------------------------------------------

    db.loc[month_mask, "Generator"] = FLAG_VALUE_DONE


# =================================================
# PUBLIC API
# =================================================


def run_value_factor() -> None:
    """
    Stage-4 Factor: Value
    """

    db = pd.read_excel(MASTER_DB_FILE)

    pending = (
        db.loc[db["Generator"] == FLAG_QUALITY_DONE, "MonthYear"]
        .drop_duplicates()
        .to_frame(name="MonthYear")
    )

    if pending.empty:
        print("[VAL] No pending months")
        return

    pending["MonthDate"] = pd.to_datetime(pending["MonthYear"], format="%b %Y")
    pending = pending.sort_values("MonthDate")

    for month_label in tqdm(
        pending["MonthYear"],
        desc="[VAL] Processing months",
        unit="month",
        ncols=100,
    ):
        _apply_value_for_month(db, month_label)

    db.to_excel(MASTER_DB_FILE, index=False)

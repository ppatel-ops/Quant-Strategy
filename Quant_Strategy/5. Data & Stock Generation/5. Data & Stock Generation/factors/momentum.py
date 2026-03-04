# factors/momentum.py

# ============================================================
# STAGE: 1 — MOMENTUM (PRIMARY FILTER)
#
# This factor identifies stocks exhibiting persistent,
# index-relative momentum using regression analysis.
#
# Methodology
# ------------
# 1. Universe gating:
#    - Generator == FLAG_TMI_DONE
#    - TMI == YES
#
# 2. Time window:
#    - Last N completed years (default: 5)
#    - Ending at last trading day of previous month
#
# 3. Returns construction:
#    - Daily log returns
#    - Converted to non-overlapping K-day returns (default: 10)
#
# 4. Regression model (per stock-index pair):
#    Stock_Returns = alpha + beta * Index_Returns + residuals
#
# 5. Metrics computed:
#    - alpha: excess return over index
#    - beta: sensitivity to index movements
#    - residual_vol: volatility of residuals
#    - information_ratio: alpha / residual_vol
#    - modified_shm: alpha * residual_vol
#
# 6. Selection rule:
#    - beta >= threshold
#    - momentum score > 0
#    - condition must hold for ALL years
#    - satisfying ANY ONE index is sufficient
# ============================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from config.settings import (
    MASTER_DB_FILE,
    PRICE_FILE,
    INDICES_PRICE_FILE,
    FLAG_YES,
    FLAG_NO,
    FLAG_TMI_DONE,
    FLAG_MOMENTUM_DONE,
    MOMENTUM_SCORE_METHOD,
    MOMENTUM_BETA_MIN,
    MOMENTUM_YEARS_REQUIRED,
    MOMENTUM_WINDOW_DAYS,
)

# =================================================
# Helpers: dates & returns
# =================================================


def _month_end_ts(month_label: str) -> pd.Timestamp:
    """Month end timestamp for 'Jan 2026'"""
    return pd.to_datetime(month_label, format="%b %Y") + pd.offsets.MonthEnd(0)


def _compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df) - np.log(df.shift(1))


def _non_overlapping_returns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Convert daily log returns into non-overlapping K-day returns.
    """
    df = df.dropna(how="all")
    grp = np.arange(len(df)) // window
    out = df.groupby(grp).sum()
    out.index = df.index[::window][: len(out)]
    return out


# =================================================
# Regression metrics
# =================================================


def _compute_momentum_metrics(
    stock_ret: np.ndarray,
    index_ret: np.ndarray,
) -> Dict[str, float] | None:
    """
    Compute regression-based momentum metrics.
    """
    if len(stock_ret) < 20:
        return None

    X = index_ret.reshape(-1, 1)
    y = stock_ret.reshape(-1, 1)

    model = LinearRegression().fit(X, y)

    alpha = float(model.intercept_[0])
    beta = float(model.coef_[0][0])

    residuals = y - model.predict(X)
    residual_vol = float(np.std(residuals))

    if residual_vol == 0:
        ir = 0.0
        mod_shm = 0.0
    else:
        ir = alpha / residual_vol
        mod_shm = alpha * residual_vol

    return {
        "alpha": alpha,
        "beta": beta,
        "residual_vol": residual_vol,
        "information_ratio": ir,
        "modified_shm": mod_shm,
    }


def _passes_metric(metrics: Dict[str, float]) -> bool:
    """
    Apply configured momentum score rule.
    """
    if metrics["beta"] < MOMENTUM_BETA_MIN:
        return False

    if MOMENTUM_SCORE_METHOD == "alpha":
        return metrics["alpha"] > 0

    if MOMENTUM_SCORE_METHOD == "information_ratio":
        return metrics["information_ratio"] > 0

    if MOMENTUM_SCORE_METHOD == "modified_shm":
        return metrics["modified_shm"] > 0

    raise ValueError(f"Unknown MOMENTUM_SCORE_METHOD: {MOMENTUM_SCORE_METHOD}")


# =================================================
# Core checks
# =================================================


def _passes_index_for_all_years(
    stock: str,
    index: str,
    yearly_returns: Dict[int, pd.DataFrame],
) -> bool:
    """
    Check momentum consistency for one stock vs one index
    across all required years.
    """
    for _, df in yearly_returns.items():
        if stock not in df.columns or index not in df.columns:
            return False

        res = _compute_momentum_metrics(
            df[stock].values,
            df[index].values,
        )

        if res is None or not _passes_metric(res):
            return False

    return True


def _passes_momentum(
    stock: str,
    indices: List[str],
    yearly_returns: Dict[int, pd.DataFrame],
) -> bool:
    """
    Stock passes momentum if ANY ONE index satisfies
    the momentum condition across ALL years.
    """
    for idx in indices:
        if _passes_index_for_all_years(stock, idx, yearly_returns):
            return True
    return False


# =================================================
# CORE MONTH LOGIC
# =================================================


def _apply_momentum_for_month(db: pd.DataFrame, month_label: str) -> None:
    """
    Apply momentum factor for ONE month.
    Mutates db in-place.
    """

    end_date = _month_end_ts(month_label) - pd.offsets.MonthEnd(1)

    # -------------------------------------------------
    # Universe gating (FINAL, CORRECT)
    # -------------------------------------------------

    mask = (
        (db["MonthYear"] == month_label)
        & (db["Generator"] == FLAG_TMI_DONE)
        & (db["TMI_Flag"] == FLAG_YES)
    )

    universe = db.loc[mask, "Ticker"].unique().tolist()
    month_mask = db["MonthYear"] == month_label

    if not universe:
        db.loc[month_mask, "Momentum"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_MOMENTUM_DONE
        return

    # -------------------------------------------------
    # Load prices
    # -------------------------------------------------

    prices = pd.read_excel(PRICE_FILE, parse_dates=["Date"]).set_index("Date")
    indices = pd.read_excel(INDICES_PRICE_FILE, parse_dates=["Date"]).set_index("Date")

    prices = prices.loc[:end_date, universe]
    indices = indices.loc[:end_date]

    # -------------------------------------------------
    # Compute log returns
    # -------------------------------------------------

    stock_lr = _compute_log_returns(prices)
    index_lr = _compute_log_returns(indices)

    # -------------------------------------------------
    # Build yearly windows
    # -------------------------------------------------

    yearly_returns: Dict[int, pd.DataFrame] = {}

    for i in range(MOMENTUM_YEARS_REQUIRED):
        y_end = end_date - pd.DateOffset(years=i)
        y_start = y_end - pd.DateOffset(years=1)

        sr = stock_lr.loc[y_start:y_end]
        ir = index_lr.loc[y_start:y_end]

        if sr.empty or ir.empty:
            continue

        merged = pd.concat([sr, ir], axis=1, join="inner")
        merged = _non_overlapping_returns(merged, MOMENTUM_WINDOW_DAYS)
        yearly_returns[i] = merged

    if len(yearly_returns) < MOMENTUM_YEARS_REQUIRED:
        db.loc[month_mask, "Momentum"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_MOMENTUM_DONE
        return

    index_list = indices.columns.tolist()

    # -------------------------------------------------
    # Multithreaded evaluation
    # -------------------------------------------------

    passed: set[str] = set()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_passes_momentum, stock, index_list, yearly_returns): stock
            for stock in universe
        }

        for f in as_completed(futures):
            stock = futures[f]
            try:
                if f.result():
                    passed.add(stock)
            except Exception:
                continue

    # -------------------------------------------------
    # Persist flags
    # -------------------------------------------------

    db.loc[
        month_mask & db["Ticker"].isin(passed),
        "Momentum",
    ] = FLAG_YES

    db.loc[
        month_mask & db["Ticker"].isin(universe) & ~db["Ticker"].isin(passed),
        "Momentum",
    ] = FLAG_NO

    # -------------------------------------------------
    # Stage transition
    # -------------------------------------------------

    db.loc[month_mask, "Generator"] = FLAG_MOMENTUM_DONE


# =================================================
# PUBLIC API
# =================================================


def run_momentum_factor() -> None:
    """
    Stage-1 Factor: Momentum

    Rules:
    - Runs ONLY when Generator == FLAG_TMI_DONE
    - Requires TMI == YES
    - Marks Generator → FLAG_MOMENTUM_DONE
    """

    db = pd.read_excel(MASTER_DB_FILE)

    # ---------------------------------------------
    # Pending months
    # ---------------------------------------------

    pending = (
        db.loc[db["Generator"] == FLAG_TMI_DONE, "MonthYear"]
        .drop_duplicates()
        .to_frame(name="MonthYear")
    )

    if pending.empty:
        print("[MOM] No pending months")
        return

    # ---------------------------------------------
    # Chronological execution
    # ---------------------------------------------

    pending["MonthDate"] = pd.to_datetime(pending["MonthYear"], format="%b %Y")
    pending = pending.sort_values("MonthDate")

    months = pending["MonthYear"].tolist()

    for month_label in tqdm(
        months,
        desc="[MOM] Processing months",
        unit="month",
        ncols=100,
    ):
        _apply_momentum_for_month(db, month_label)

    # ---------------------------------------------
    # Persist
    # ---------------------------------------------

    db.to_excel(MASTER_DB_FILE, index=False)

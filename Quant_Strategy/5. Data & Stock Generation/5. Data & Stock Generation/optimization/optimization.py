# optimization/optimization.py

# ============================================================
# PORTFOLIO OPTIMIZATION (BINARY MEAN–VARIANCE, SHARPE)
#
# Objective:
# - Select optimal subset of stocks using Sharpe maximization
# - Binary selection (0/1), NOT continuous weights
# - Modified covariance (correlation × avg variance)
#
# Rules:
# - Runs ONLY when Generator == FLAG_VALUE_DONE
# - Universe requires:
#     TMI == YES
#     Momentum == YES
#     Quality == YES
#     Value == YES
# - Select exactly 5 stocks (or fewer if universe < 5)
# - Deterministic per month
# - Ensemble optimization for stability
# - Marks Generator → FLAG_OPTIMIZATION_DONE
# ============================================================

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from config.settings import (
    MASTER_DB_FILE,
    PRICE_FILE,
    FLAG_YES,
    FLAG_NO,
    FLAG_VALUE_DONE,
    FLAG_OPTIMIZATION_DONE,
)

# =================================================
# DATE & RETURN HELPERS
# =================================================


def _month_end_ts(month_label: str) -> pd.Timestamp:
    """Month-end timestamp for 'Jan 2026'."""
    return pd.to_datetime(month_label, format="%b %Y") + pd.offsets.MonthEnd(0)


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices) - np.log(prices.shift(1))


def _non_overlapping_returns(
    log_returns: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Convert daily log returns into non-overlapping K-day returns.
    """
    log_returns = log_returns.dropna(how="all")
    groups = np.arange(len(log_returns)) // window
    out = log_returns.groupby(groups).sum()
    out.index = log_returns.index[::window][: len(out)]
    return out


# =================================================
# SHARPE OPTIMIZATION CORE
# =================================================


def _negative_sharpe_ratio(
    weights: np.ndarray,
    n_assets: int,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Negative Sharpe ratio for minimization.
    """
    mod_w = weights / n_assets
    port_ret = np.dot(expected_returns, mod_w)
    port_var = np.dot(mod_w.T, np.dot(covariance_matrix, mod_w))

    if port_var <= 0:
        return 1e6

    return -port_ret / np.sqrt(port_var)


def _optimize_max_sharpe_binary(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    n_select: int,
    n_tries: int = 5,
) -> np.ndarray:
    """
    Binary Sharpe maximization using continuous relaxation + rounding.
    """

    total_assets = len(expected_returns)

    bounds = [(0.0, 1.0)] * total_assets
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - n_select},
        {"type": "ineq", "fun": lambda w: w},
    )

    best_weights = None
    best_obj = np.inf

    for _ in range(n_tries):
        initial_guess = np.random.choice([0, 1], size=total_assets)

        result = minimize(
            _negative_sharpe_ratio,
            initial_guess,
            args=(n_select, expected_returns, covariance_matrix),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        if not result.success:
            continue

        w = result.x
        if np.allclose(w, 0):
            continue

        obj = _negative_sharpe_ratio(w, n_select, expected_returns, covariance_matrix)

        if obj < best_obj:
            best_obj = obj
            best_weights = w

    # -------------------------------------------------
    # Fallback: risk-adjusted ranking
    # -------------------------------------------------

    if best_weights is None:
        vols = np.sqrt(np.diag(covariance_matrix))
        scores = expected_returns / np.where(vols == 0, np.nan, vols)
        order = np.argsort(scores)[::-1]

        binary = np.zeros(total_assets, dtype=int)
        binary[order[:n_select]] = 1
        return binary

    # -------------------------------------------------
    # Hard rounding to binary
    # -------------------------------------------------

    threshold = np.sort(best_weights)[-n_select]
    binary = (best_weights >= threshold).astype(int)

    return binary


# =================================================
# MONTHLY OPTIMIZATION LOGIC
# =================================================


def _apply_optimization_for_month(db: pd.DataFrame, month_label: str) -> None:
    """
    Apply optimization for ONE month (mutates db in-place).
    """

    # Deterministic randomness per month
    seed = abs(hash(month_label)) % (2**32)
    np.random.seed(seed)

    end_date = _month_end_ts(month_label) - pd.offsets.MonthEnd(1)
    month_mask = db["MonthYear"] == month_label

    # -------------------------------------------------
    # FINAL UNIVERSE GATING
    # -------------------------------------------------

    mask = (
        month_mask
        & (db["Generator"] == FLAG_VALUE_DONE)
        & (db["TMI"] == FLAG_YES)
        & (db["Momentum"] == FLAG_YES)
        & (db["Quality"] == FLAG_YES)
        & (db["Value"] == FLAG_YES)
    )

    universe = db.loc[mask, "Ticker"].unique().tolist()

    if not universe:
        db.loc[month_mask, "Optimization"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_OPTIMIZATION_DONE
        return

    n_select = min(5, len(universe))

    # -------------------------------------------------
    # PRICE & RETURN CONSTRUCTION
    # -------------------------------------------------

    prices = (
        pd.read_excel(PRICE_FILE, parse_dates=["Date"])
        .set_index("Date")
        .loc[:end_date, universe]
        .ffill()
    )

    log_ret = _compute_log_returns(prices).dropna()
    window_ret_5y = _non_overlapping_returns(log_ret, window=10)

    if window_ret_5y.shape[0] < 20:
        db.loc[month_mask, "Optimization"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_OPTIMIZATION_DONE
        return

    # -------------------------------------------------
    # EXPECTED RETURNS (1Y)
    # -------------------------------------------------

    expected_returns = log_ret.tail(252).mean().to_numpy()

    # -------------------------------------------------
    # MODIFIED COVARIANCE (5Y)
    # -------------------------------------------------

    cov = np.cov(window_ret_5y, rowvar=False)

    if not np.isfinite(cov).all():
        db.loc[month_mask, "Optimization"] = FLAG_NO
        db.loc[month_mask, "Generator"] = FLAG_OPTIMIZATION_DONE
        return

    stds = np.sqrt(np.diag(cov))
    avg_std = np.sqrt(np.mean(np.diag(cov)))
    corr = cov / np.outer(stds, stds)
    mod_cov = corr * avg_std * avg_std

    # -------------------------------------------------
    # ENSEMBLE OPTIMIZATION (VOTING)
    # -------------------------------------------------

    VOTES = 15
    vote_counter = Counter()

    for _ in range(VOTES):
        allocation = _optimize_max_sharpe_binary(
            expected_returns,
            mod_cov,
            n_select,
        )
        selected = np.array(universe)[allocation == 1]
        vote_counter.update(selected)

    final_selected = [s for s, _ in vote_counter.most_common(n_select)]

    # -------------------------------------------------
    # PERSIST RESULTS
    # -------------------------------------------------

    db.loc[
        month_mask & db["Ticker"].isin(final_selected),
        "Optimization",
    ] = FLAG_YES

    db.loc[
        month_mask & db["Ticker"].isin(universe) & ~db["Ticker"].isin(final_selected),
        "Optimization",
    ] = FLAG_NO

    db.loc[month_mask, "Generator"] = FLAG_OPTIMIZATION_DONE


# =================================================
# PUBLIC API
# =================================================


def portfolio_optimization() -> None:
    """
    Stage-5: Portfolio Optimization (Binary Sharpe)

    - Runs ONLY when Generator == FLAG_VALUE_DONE
    - Selects up to 5 stocks per month
    - Deterministic, restart-safe
    """

    db = pd.read_excel(MASTER_DB_FILE)

    pending = (
        db.loc[db["Generator"] == FLAG_VALUE_DONE, "MonthYear"]
        .drop_duplicates()
        .to_frame(name="MonthYear")
    )

    if pending.empty:
        print("[OPT] No pending months")
        return

    pending["MonthDate"] = pd.to_datetime(pending["MonthYear"], format="%b %Y")
    pending = pending.sort_values("MonthDate")

    for month_label in tqdm(
        pending["MonthYear"].tolist(),
        desc="[OPT] Optimizing portfolios",
        unit="month",
        ncols=100,
    ):
        _apply_optimization_for_month(db, month_label)

    db.to_excel(MASTER_DB_FILE, index=False)

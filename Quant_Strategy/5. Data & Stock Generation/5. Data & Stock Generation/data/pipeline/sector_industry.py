# data/pipeline/sector_industry.py

from __future__ import annotations

import time
import logging
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config.settings import (
    DATA_DIR,
    MASTER_DB_FILE,
    UNIVERSE_COLUMNS,
)

# =================================================
# SILENCE NOISY LOGGERS (IMPORTANT)
# =================================================

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

UNIQUE_TICKERS_FILE = RAW_DIR / "unique_tickers.xlsx"

# =================================================
# CONFIG
# =================================================

METADATA_COLS = [
    "Sector",
    "Industry",
]

EXTRA_INFO_COLS = [
    "Business",
    "CurrentPrice",
    "FiftyTwoWeekLow",
    "FiftyTwoWeekHigh",
    "TargetHighPrice",
    "TargetLowPrice",
    "TargetMeanPrice",
    "TargetMedianPrice",
]

ALL_INFO_COLS = METADATA_COLS + EXTRA_INFO_COLS

MAX_WORKERS = 4
SLEEP_SECONDS = 0.25  # light Yahoo throttling

# =================================================
# HELPERS
# =================================================


def _load_universe() -> List[str]:
    if not UNIQUE_TICKERS_FILE.exists():
        raise RuntimeError("unique_tickers.xlsx not found")

    df = pd.read_excel(UNIQUE_TICKERS_FILE)

    if "Ticker" not in df.columns:
        raise ValueError("unique_tickers.xlsx must contain 'Ticker' column")

    return sorted(df["Ticker"].dropna().unique().tolist())


def _fetch_metadata(ticker: str) -> Dict | None:
    """
    Full Yahoo fetch.
    Returns metadata dict or None.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()

        if not info:
            return None

        return {
            "Ticker": ticker,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Business": info.get("longBusinessSummary"),
            "CurrentPrice": info.get("currentPrice"),
            "FiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "FiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "TargetHighPrice": info.get("targetHighPrice"),
            "TargetLowPrice": info.get("targetLowPrice"),
            "TargetMeanPrice": info.get("targetMeanPrice"),
            "TargetMedianPrice": info.get("targetMedianPrice"),
        }

    except Exception:
        return None
    finally:
        time.sleep(SLEEP_SECONDS)


# =================================================
# PIPELINE
# =================================================


def sector_industry() -> None:
    print("=" * 70)
    print("[METADATA] Yahoo full refresh")
    print("=" * 70)

    tickers = _load_universe()

    print(f"[METADATA] Universe tickers : {len(tickers):,}")

    records: List[Dict] = []
    tickers_with_data: Set[str] = set()
    tickers_no_data: Set[str] = set()
    failed_tickers: Set[str] = set()

    # -------------------------------------------------
    # MULTITHREADED FETCH
    # -------------------------------------------------

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_metadata, ticker): ticker for ticker in tickers
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="[METADATA] Fetching",
            ncols=100,
        ):
            ticker = futures[future]

            try:
                data = future.result()

                if data:
                    records.append(data)
                    tickers_with_data.add(ticker)
                else:
                    tickers_no_data.add(ticker)

            except Exception:
                failed_tickers.add(ticker)

    # -------------------------------------------------
    # BUILD FULL METADATA TABLE (DO NOT SHRINK UNIVERSE)
    # -------------------------------------------------

    # Start from full universe
    ut_full = pd.DataFrame({"Ticker": tickers})

    # Convert fetched metadata to DataFrame
    meta_df = pd.DataFrame(records)

    # Merge (left join keeps ALL tickers)
    ut = ut_full.merge(meta_df, on="Ticker", how="left")

    # Ensure schema stability
    for col in ALL_INFO_COLS:
        if col not in ut.columns:
            ut[col] = None

    ut = ut[["Ticker"] + ALL_INFO_COLS]
    ut.sort_values("Ticker", inplace=True)

    # Overwrite file BUT keep all tickers
    ut.to_excel(UNIQUE_TICKERS_FILE, index=False)

    # =================================================
    # MASTER DB — REFRESH SECTOR / INDUSTRY ONLY
    # =================================================

    print("\n" + "=" * 70)
    print("[MASTER_DB] Refreshing Sector / Industry")
    print("=" * 70)

    master = pd.read_excel(MASTER_DB_FILE)

    meta = ut[["Ticker"] + METADATA_COLS]

    master.drop(columns=METADATA_COLS, errors="ignore", inplace=True)

    master = master.merge(
        meta,
        on="Ticker",
        how="left",
        validate="many_to_one",
    )

    missing = [c for c in UNIVERSE_COLUMNS if c not in master.columns]
    if missing:
        raise RuntimeError(f"MASTER_DB missing columns: {missing}")

    master = master[UNIVERSE_COLUMNS]
    master.to_excel(MASTER_DB_FILE, index=False)

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    print("\n[METADATA] SUMMARY")
    print("-" * 70)
    print(f"Total tickers           : {len(tickers):,}")
    print(f"Tickers with data       : {len(tickers_with_data):,}")
    print(f"Tickers with NO data    : {len(tickers_no_data):,}")
    print(f"Tickers failed (errors) : {len(failed_tickers):,}")
    print("-" * 70)
    print(f"unique_tickers rows     : {len(ut):,}")
    print("=" * 70)
    print("[PIPELINE] COMPLETED SUCCESSFULLY")
    print("=" * 70)

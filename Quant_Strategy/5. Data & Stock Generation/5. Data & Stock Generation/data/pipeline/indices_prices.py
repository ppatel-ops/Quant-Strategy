# data/pipeline/indices_prices.py

"""
Indices Prices Data Pipeline

Downloads daily CLOSE prices for all NSE indices.
This pipeline is INCREMENTAL (no rebuild).

Output:
    data/raw/Indices.xlsx

Rows   : Trading days
Cols   : Index names
Values : Closing Index Value (NSE official)
"""

from datetime import date
import time
import io

import pandas as pd
import requests
from tqdm import tqdm

from config.settings import (
    DATA_DIR,
    DATA_START_YEAR,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRICE_FILE = RAW_DIR / "Indices.xlsx"
CALENDAR_FILE = PROCESSED_DIR / "calendar" / "trading_days_price.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)

SHEET_NAME = "indices"

NSE_INDICES_URL = "https://archives.nseindia.com/content/indices/"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv",
}

# =================================================
# HELPERS
# =================================================


def _download_snapshot(trading_date: date) -> pd.DataFrame | None:
    """
    Download NSE index snapshot for a given trading day.
    Returns None if file is missing or request fails.
    """
    url = f"{NSE_INDICES_URL}ind_close_all_{trading_date.strftime('%d%m%Y')}.csv"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None


def _load_trading_days() -> list[date]:
    if not CALENDAR_FILE.exists():
        raise RuntimeError("Trading calendar not found")

    df = pd.read_parquet(CALENDAR_FILE)
    return pd.to_datetime(df["date"]).dt.date.tolist()


def _load_existing_prices() -> tuple[pd.DataFrame, date | None]:
    """
    Load existing indices price file if present.
    Returns:
        prices_df (indexed by Date)
        last_available_date
    """
    if not PRICE_FILE.exists():
        return pd.DataFrame(), None

    df = pd.read_excel(PRICE_FILE, sheet_name=SHEET_NAME)

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str),
        errors="coerce",
    ).dt.date

    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)

    if df.empty:
        return pd.DataFrame(), None

    return df, df.index.max()


# =================================================
# MAIN ENTRY
# =================================================


def indices_prices() -> None:
    """
    Incremental download of NSE indices prices.
    """

    print("=" * 70)
    print("[INDICES] Indices Prices — INCREMENTAL UPDATE")
    print("=" * 70)

    try:
        # -------------------------------------------------
        # Load trading calendar
        # -------------------------------------------------

        trading_days = _load_trading_days()
        trading_days_set = set(trading_days)
        last_calendar_day = trading_days[-1]

        # -------------------------------------------------
        # Load existing data
        # -------------------------------------------------

        prices, last_available_day = _load_existing_prices()

        # -------------------------------------------------
        # Determine start date
        # -------------------------------------------------

        if last_available_day:
            start_date = last_available_day
        else:
            start_date = date(DATA_START_YEAR, 1, 1)

        pending_days = [
            d for d in trading_days if (start_date is None or d > start_date)
        ]

        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        print("[INDICES] DATE STATUS")
        print("-" * 70)
        print(f"Last available date  : {last_available_day}")
        print(f"Last calendar date   : {last_calendar_day}")
        print(f"Pending trading days : {len(pending_days)}")
        print("-" * 70)

        if not pending_days:
            print("[INDICES] Already up to date — no downloads required")
            print("=" * 70)
            return

        # -------------------------------------------------
        # Download snapshots
        # -------------------------------------------------

        for d in tqdm(
            pending_days,
            desc="[INDICES] Downloading",
            unit="day",
            ncols=90,
        ):
            try:
                snapshot = _download_snapshot(d)
                if snapshot is None:
                    continue

                snapshot["Index Name"] = (
                    snapshot["Index Name"].astype(str).str.strip().str.upper()
                )

                snapshot_map = dict(
                    zip(
                        snapshot["Index Name"],
                        snapshot["Closing Index Value"],
                    )
                )

                # Initialize dataframe on first successful fetch
                if prices.empty:
                    prices = pd.DataFrame(columns=sorted(snapshot_map.keys()))
                    prices.index.name = "Date"

                for idx in prices.columns:
                    prices.at[d, idx] = snapshot_map.get(idx)

                time.sleep(0.15)

            except Exception as exc:
                tqdm.write(f"[INDICES][WARN] {d}: {exc}")

        # -------------------------------------------------
        # DROP FULLY EMPTY ROWS (CRITICAL FIX)
        # -------------------------------------------------

        empty_rows = prices.isna().all(axis=1)
        dropped_count = empty_rows.sum()

        if dropped_count > 0:
            dropped_dates = prices.index[empty_rows]

            print("\n[INDICES][CLEANUP]")
            print("-" * 70)
            print(f"Rows before cleanup : {len(prices):,}")
            print(f"Dropped empty rows  : {dropped_count}")

            print("Sample dropped dates:")
            for d in dropped_dates[:10]:
                print(f"  - {d}")

            prices = prices.loc[~empty_rows].copy()
            print(f"Rows after cleanup  : {len(prices):,}")
        else:
            print("\n[INDICES][CLEANUP] No empty rows found")

        # -------------------------------------------------
        # Persist
        # -------------------------------------------------

        if prices.empty:
            print("[INDICES][WARN] No valid index data to save — aborting write")
            return

        prices = prices.sort_index()
        prices.reset_index(inplace=True)
        prices.rename(columns={"index": "Date"}, inplace=True)

        with pd.ExcelWriter(PRICE_FILE, engine="xlsxwriter") as writer:
            prices.to_excel(writer, sheet_name=SHEET_NAME, index=False)

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        print("\n[INDICES] SUMMARY")
        print("-" * 70)
        print(f"Rows in file      : {prices.shape[0]:,}")
        print(f"Indices count     : {prices.shape[1] - 1:,}")
        print(f"Days added        : {len(pending_days):,}")
        print("-" * 70)

        print("=" * 70)
        print("[INDICES] Step completed successfully")
        print("=" * 70)

    except Exception:
        print("=" * 70)
        print("[INDICES] FAILED")
        print("=" * 70)
        raise

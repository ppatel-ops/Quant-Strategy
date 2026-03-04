# data/pipeline/stocks_prices.py

"""
Stock Prices Data Pipeline

Builds a wide, point-in-time-safe CLOSE price matrix
for all tickers in unique_tickers.xlsx.

Also derives a PRICE-ALIGNED trading calendar
used as the ONLY source for backtesting dates.

Outputs:
    data/raw/Prices.xlsx
    data/processed/calendar/trading_days_price.parquet

Rows   : Trading days with sufficient price coverage
Cols   : Stock tickers (e.g. RELIANCE.NS)
Values : Yahoo Finance CLOSE prices
"""

from datetime import date, datetime
import time
import math
from typing import Dict

import pandas as pd
import requests
import pytz
from tqdm import tqdm

from config.settings import (
    DATA_DIR,
    YFINANCE_BASE_URL,
    TIMEZONE,
    DATA_START_YEAR,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CALENDAR_DIR = PROCESSED_DIR / "calendar"

PRICE_FILE = RAW_DIR / "Prices.xlsx"
UNIQUE_TICKERS_FILE = RAW_DIR / "unique_tickers.xlsx"
NSE_CALENDAR_FILE = CALENDAR_DIR / "trading_days_past.parquet"
PRICE_CALENDAR_FILE = CALENDAR_DIR / "trading_days_price.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CALENDAR_DIR.mkdir(parents=True, exist_ok=True)

TZ = pytz.timezone(TIMEZONE)

# =================================================
# HELPERS
# =================================================


def _load_unique_tickers() -> list[str]:
    if not UNIQUE_TICKERS_FILE.exists():
        raise RuntimeError("unique_tickers.xlsx not found")

    df = pd.read_excel(UNIQUE_TICKERS_FILE)
    if "Ticker" not in df.columns:
        raise ValueError("unique_tickers.xlsx must contain 'Ticker' column")

    return sorted(df["Ticker"].dropna().unique().tolist())


def _load_nse_trading_days() -> list[date]:
    if not NSE_CALENDAR_FILE.exists():
        raise RuntimeError("NSE trading calendar not found")

    df = pd.read_parquet(NSE_CALENDAR_FILE)
    return pd.to_datetime(df["date"]).dt.date.tolist()


def _download_close(
    ticker: str,
    start: date,
    end: date,
) -> Dict[date, float]:
    """
    Download daily CLOSE prices from Yahoo Finance.
    Returns {date: close_price}
    """

    start_dt = TZ.localize(datetime.combine(start, datetime.min.time()))
    end_dt = TZ.localize(datetime.combine(end, datetime.max.time()))

    url = (
        f"{YFINANCE_BASE_URL}{ticker}"
        f"?period1={int(start_dt.timestamp())}"
        f"&period2={int(end_dt.timestamp())}"
        f"&interval=1d"
    )

    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )

    if r.status_code == 404:
        return {}

    r.raise_for_status()

    payload = r.json().get("chart", {}).get("result")
    if not payload:
        return {}

    result = payload[0]
    timestamps = result.get("timestamp", [])
    quotes = result.get("indicators", {}).get("quote", [])

    if not timestamps or not quotes:
        return {}

    closes = quotes[0].get("close", [])
    data: Dict[date, float] = {}

    for i, ts in enumerate(timestamps):
        price = closes[i]
        if price is None:
            continue

        d = datetime.fromtimestamp(ts, tz=TZ).date()
        data[d] = price

    return data


# =================================================
# MAIN ENTRY
# =================================================


def stocks_prices() -> None:
    """
    FULL rebuild of stock CLOSE price matrix.

    Guarantees:
    - Point-in-time safety
    - Deterministic output
    - Backtest-safe calendar alignment
    """

    print("=" * 70)
    print("[PRICES] Stock Prices — FULL REBUILD")
    print("=" * 70)

    try:
        # -------------------------------------------------
        # Load universe + calendar
        # -------------------------------------------------

        tickers = _load_unique_tickers()
        nse_trading_days = _load_nse_trading_days()

        if not tickers:
            raise RuntimeError("No tickers found")

        if not nse_trading_days:
            raise RuntimeError("Trading calendar empty")

        trading_days_set = set(nse_trading_days)

        start_date = max(date(DATA_START_YEAR, 1, 1), nse_trading_days[0])
        end_date = nse_trading_days[-1]

        print(f"[PRICES] Date range   : {start_date} → {end_date}")
        print(f"[PRICES] Tickers      : {len(tickers):,}")
        print(f"[PRICES] NSE days     : {len(nse_trading_days):,}")
        print("-" * 70)

        # -------------------------------------------------
        # Initialize price matrix (NSE calendar as index)
        # -------------------------------------------------

        prices = pd.DataFrame(index=nse_trading_days, columns=tickers)

        tickers_with_data = set()
        tickers_no_data = set()
        failed_tickers = set()

        # -------------------------------------------------
        # Download prices (SERIAL, stable)
        # -------------------------------------------------

        for ticker in tqdm(
            tickers,
            desc="[PRICES] Downloading",
            unit="ticker",
            ncols=90,
        ):
            try:
                data = _download_close(ticker, start_date, end_date)

                if not data:
                    tickers_no_data.add(ticker)
                    continue

                tickers_with_data.add(ticker)

                for d, v in data.items():
                    if d in trading_days_set:
                        prices.at[d, ticker] = v

                time.sleep(0.1)

            except Exception as exc:
                failed_tickers.add(ticker)
                tqdm.write(f"[PRICES][WARN] {ticker}: {exc}")

        # -------------------------------------------------
        # DROP LOW-COVERAGE DAYS (≥5% rule)
        # -------------------------------------------------

        total_tickers = len(tickers)
        min_required = max(1, math.ceil(0.05 * total_tickers))

        non_null_counts = prices.notna().sum(axis=1)
        bad_rows = non_null_counts < min_required
        bad_dates = prices.index[bad_rows]

        if bad_rows.any():
            print("\n[PRICES][CLEANUP]")
            print("-" * 70)
            print(f"Trading days before cleanup : {len(prices):,}")
            print(f"Minimum prices required/day : {min_required}")
            print(f"Dropped low-coverage days   : {bad_rows.sum()}")

            print("Sample dropped dates:")
            for d in bad_dates[:10]:
                print(f"  - {d}")

            prices = prices.loc[~bad_rows].copy()
            print(f"Trading days after cleanup  : {len(prices):,}")
        else:
            print("\n[PRICES][CLEANUP] No low-coverage days found")

        # -------------------------------------------------
        # BUILD PRICE-ALIGNED TRADING CALENDAR (CRITICAL)
        # -------------------------------------------------

        price_days = pd.to_datetime(prices.index)

        price_calendar = pd.DataFrame(
            {
                "date": price_days,
                "year": price_days.year,
                "month": price_days.month,
            }
        )

        price_calendar.to_parquet(PRICE_CALENDAR_FILE, index=False)

        print(
            f"\n[PRICES] Price-aligned trading days : {len(price_calendar):,} "
            f"(dropped {len(nse_trading_days) - len(price_calendar)})"
        )

        # -------------------------------------------------
        # Persist Prices.xlsx
        # -------------------------------------------------

        prices = prices.sort_index()
        prices.reset_index(inplace=True)
        prices.rename(columns={"index": "Date"}, inplace=True)

        with pd.ExcelWriter(PRICE_FILE, engine="xlsxwriter") as writer:
            prices.to_excel(writer, sheet_name="prices", index=False)

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        print("\n[PRICES] SUMMARY")
        print("-" * 70)
        print(f"Total tickers             : {len(tickers):,}")
        print(f"Tickers with data         : {len(tickers_with_data):,}")
        print(f"Tickers with NO data      : {len(tickers_no_data):,}")
        print(f"Tickers failed (errors)   : {len(failed_tickers):,}")
        print(f"Final trading days saved  : {len(prices):,}")

        print("=" * 70)
        print("[PRICES] Completed successfully")
        print("=" * 70)

    except Exception:
        print("=" * 70)
        print("[PRICES] FAILED")
        print("=" * 70)
        raise

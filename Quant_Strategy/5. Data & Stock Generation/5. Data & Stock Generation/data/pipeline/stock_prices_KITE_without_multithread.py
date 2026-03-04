"""
Stock Prices Data Pipeline (KITE)

Builds a wide, point-in-time-safe CLOSE price matrix
using Zerodha Kite historical API.

Outputs:
    data/raw/Prices_KITE.xlsx
    data/processed/calendar/trading_days_price.parquet
"""

from datetime import date
import time
import math
from typing import Dict

import pandas as pd
import pytz
from tqdm import tqdm
from dotenv import load_dotenv
import os

from kiteconnect import KiteConnect

from config.settings import (
    DATA_DIR,
    TIMEZONE,
    DATA_START_YEAR,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CALENDAR_DIR = PROCESSED_DIR / "calendar"

PRICE_FILE = RAW_DIR / "Prices_KITE.xlsx"
UNIQUE_TICKERS_FILE = RAW_DIR / "unique_tickers.xlsx"
NSE_CALENDAR_FILE = CALENDAR_DIR / "trading_days_past.parquet"
PRICE_CALENDAR_FILE = CALENDAR_DIR / "trading_days_price.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CALENDAR_DIR.mkdir(parents=True, exist_ok=True)

TZ = pytz.timezone(TIMEZONE)

# =================================================
# LOAD KITE
# =================================================

load_dotenv()

kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

# =================================================
# HELPERS
# =================================================


def _load_unique_tickers() -> list[str]:

    if not UNIQUE_TICKERS_FILE.exists():
        raise RuntimeError("unique_tickers.xlsx not found")

    df = pd.read_excel(UNIQUE_TICKERS_FILE)

    if "Ticker" not in df.columns:
        raise ValueError("unique_tickers.xlsx must contain 'Ticker' column")

    return (
        df["Ticker"]
        .dropna()
        .astype(str)
        .str.replace(".NS", "", regex=False)
        .unique()
        .tolist()
    )


def _load_nse_trading_days() -> list[date]:

    if not NSE_CALENDAR_FILE.exists():
        raise RuntimeError("NSE trading calendar not found")

    df = pd.read_parquet(NSE_CALENDAR_FILE)

    return pd.to_datetime(df["date"]).dt.date.tolist()


def _get_instrument_map():

    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    df = df[df["segment"] == "NSE"]

    return dict(zip(df["tradingsymbol"], df["instrument_token"]))


def _download_close(token: int, start: date, end: date) -> Dict[date, float]:

    data = kite.historical_data(
        instrument_token=token,
        from_date=start,
        to_date=end,
        interval="day",
    )

    result = {}

    for row in data:

        d = pd.to_datetime(row["date"]).date()
        result[d] = row["close"]

    return result


# =================================================
# MAIN
# =================================================


def stocks_prices_kite() -> None:

    print("=" * 70)
    print("[KITE PRICES] Stock Prices — FULL REBUILD")
    print("=" * 70)

    tickers = _load_unique_tickers()
    nse_trading_days = _load_nse_trading_days()

    if not tickers:
        raise RuntimeError("No tickers found")

    if not nse_trading_days:
        raise RuntimeError("Trading calendar empty")

    trading_days_set = set(nse_trading_days)

    start_date = max(date(DATA_START_YEAR, 1, 1), nse_trading_days[0])
    end_date = nse_trading_days[-1]

    print(f"[KITE] Date range : {start_date} → {end_date}")
    print(f"[KITE] Tickers    : {len(tickers):,}")

    prices = pd.DataFrame(index=nse_trading_days, columns=tickers)

    instrument_map = _get_instrument_map()

    tickers_with_data = set()
    tickers_no_data = set()

    # -------------------------------------------------
    # Download prices
    # -------------------------------------------------

    for ticker in tqdm(tickers, desc="[KITE] Downloading", unit="ticker", ncols=90):

        token = instrument_map.get(ticker)

        if not token:
            tickers_no_data.add(ticker)
            continue

        try:

            data = _download_close(token, start_date, end_date)

            if not data:
                tickers_no_data.add(ticker)
                continue

            tickers_with_data.add(ticker)

            for d, v in data.items():

                if d in trading_days_set:
                    prices.at[d, ticker] = v

            time.sleep(0.35)

        except Exception as exc:
            tqdm.write(f"[KITE][WARN] {ticker}: {exc}")

    # -------------------------------------------------
    # DROP LOW COVERAGE DAYS
    # -------------------------------------------------

    total_tickers = len(tickers)
    min_required = max(1, math.ceil(0.05 * total_tickers))

    non_null_counts = prices.notna().sum(axis=1)
    bad_rows = non_null_counts < min_required

    prices = prices.loc[~bad_rows].copy()

    # -------------------------------------------------
    # BUILD PRICE CALENDAR
    # -------------------------------------------------

    price_days = pd.to_datetime(prices.index)

    calendar = pd.DataFrame(
        {
            "date": price_days,
            "year": price_days.year,
            "month": price_days.month,
        }
    )

    calendar.to_parquet(PRICE_CALENDAR_FILE, index=False)

    # -------------------------------------------------
    # SAVE PRICES
    # -------------------------------------------------

    prices = prices.sort_index()

    prices.reset_index(inplace=True)
    prices.rename(columns={"index": "Date"}, inplace=True)

    with pd.ExcelWriter(PRICE_FILE, engine="xlsxwriter") as writer:
        prices.to_excel(writer, sheet_name="prices", index=False)

    print("=" * 70)
    print("[KITE PRICES] Completed successfully")
    print("=" * 70)
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# =================================================
# PATH SETUP (AUTO PROJECT ROOT)
# =================================================

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKER_FILE = RAW_DIR / "unique_tickers.xlsx"
OUTPUT_FILE = RAW_DIR / "Prices_KITE.xlsx"

# =================================================
# CONFIG
# =================================================

START_DATE = "2009-01-01"
INTERVAL = "day"
EXCHANGE = "NSE"

CHUNK_YEARS = 5
MAX_WORKERS = 4
SLEEP_SECONDS = 0.35
MAX_RETRIES = 3

# =================================================
# LOAD ENV
# =================================================

load_dotenv()

api_key = os.getenv("KITE_API_KEY")
access_token = os.getenv("KITE_ACCESS_TOKEN")

if not api_key or not access_token:
    raise RuntimeError("Missing Kite credentials in environment variables")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# =================================================
# GENERATE DATE CHUNKS
# =================================================

def generate_date_chunks(start_date, chunk_years=5):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(datetime.today().date())
    chunks = []

    while start < end:
        chunk_end = start + pd.DateOffset(years=chunk_years) - timedelta(days=1)
        if chunk_end > end:
            chunk_end = end

        chunks.append((start.strftime("%Y-%m-%d"),
                       chunk_end.strftime("%Y-%m-%d")))
        start = chunk_end + timedelta(days=1)

    return chunks


date_chunks = generate_date_chunks(START_DATE, CHUNK_YEARS)

# =================================================
# LOAD TICKERS
# =================================================

df_tickers = pd.read_excel(TICKER_FILE)

tickers = (
    df_tickers["Ticker"]
    .dropna()
    .astype(str)
    .str.replace(".NS", "", regex=False)
    .unique()
    .tolist()
)

print(f"Total tickers: {len(tickers)}")

# =================================================
# INSTRUMENT MASTER
# =================================================

instruments = kite.instruments(EXCHANGE)
instrument_df = pd.DataFrame(instruments)
instrument_df = instrument_df[instrument_df["segment"] == "NSE"]

symbol_to_token = dict(
    zip(instrument_df["tradingsymbol"], instrument_df["instrument_token"])
)

# =================================================
# THREAD WORKER
# =================================================

def fetch_ticker_data(ticker):

    token = symbol_to_token.get(ticker)
    if not token:
        return ticker, None

    all_series = []

    for from_date, to_date in date_chunks:

        for attempt in range(MAX_RETRIES):
            try:
                data = kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=INTERVAL
                )

                if not data:
                    break

                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                s = df.set_index("date")["close"]

                all_series.append(s)
                time.sleep(SLEEP_SECONDS)
                break

            except Exception:
                time.sleep(1)

    if all_series:
        combined = pd.concat(all_series)
        combined = combined[~combined.index.duplicated(keep="last")]
        return ticker, combined

    return ticker, None

# =================================================
# MULTI THREAD EXECUTION
# =================================================

print("🚀 Starting multi-threaded download...")

price_data = {}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_ticker_data, t): t for t in tickers}

    for future in as_completed(futures):
        ticker, data = future.result()
        if data is not None:
            price_data[ticker] = data
            print(f"✅ {ticker}")
        else:
            print(f"⚠ Skipped {ticker}")

# =================================================
# BUILD MATRIX
# =================================================

price_matrix = pd.concat(price_data, axis=1)
price_matrix.sort_index(inplace=True)
price_matrix.index.name = "Date"
price_matrix = price_matrix[~price_matrix.index.duplicated(keep="last")]

price_matrix.to_excel(OUTPUT_FILE)

print("\n🎉 SUCCESS")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Shape: {price_matrix.shape}")
print(f"Date range: {price_matrix.index.min()} → {price_matrix.index.max()}")
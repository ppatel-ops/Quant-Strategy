# data/pipeline/master_universe.py

from datetime import date
from io import StringIO
import json

import pandas as pd
import requests

from config.settings import (
    DATA_DIR,
    MASTER_DB_FILE,
    NSE_TMI_URL,
    FLAG_YES,
    FLAG_NO,
    FLAG_TMI_DONE,
    UNIVERSE_COLUMNS,
)

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TMI_FILE = RAW_DIR / "tmi.json"
UNIQUE_TICKERS_FILE = RAW_DIR / "unique_tickers.xlsx"

MASTER_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# =================================================
# HELPERS
# =================================================


def _month_key(d: date) -> str:
    """Return month label like 'Jan 2026'"""
    return d.replace(day=1).strftime("%b %Y")


def _month_to_date(month_label: str) -> pd.Timestamp:
    """Convert 'Jan 2026' → Timestamp(2026-01-01)"""
    return pd.to_datetime(f"01 {month_label}", format="%d %b %Y")


def _load_tmi() -> dict:
    if not TMI_FILE.exists():
        raise RuntimeError("tmi.json not found. Historical TMI must exist.")
    return json.loads(TMI_FILE.read_text())


def _save_tmi(tmi: dict) -> None:
    TMI_FILE.write_text(json.dumps(tmi, indent=4))


def _download_current_tmi() -> pd.DataFrame:
    r = requests.get(NSE_TMI_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# =================================================
# MAIN ENTRY
# =================================================


def master_universe() -> None:
    """
    APPEND-ONLY MASTER UNIVERSE BUILDER

    Rules:
    - tmi.json is the universe source of truth
    - master_db.xlsx is append-only by (MonthYear, Ticker)
    - unique_tickers.xlsx is append-only by Ticker
    - MonthYear format is always 'Mon YYYY'
    - No Year column exists
    """

    today = date.today()
    current_month = _month_key(today)

    # -------------------------------------------------
    # STEP 1: LOAD / UPDATE TMI.JSON
    # -------------------------------------------------

    tmi = _load_tmi()

    if current_month not in tmi:
        print(f"[TMI] Adding universe for {current_month}")

        df = _download_current_tmi()
        tickers = sorted({f"{s}.NS" for s in df["Symbol"]})

        tmi[current_month] = tickers
        _save_tmi(tmi)
    else:
        print(f"[TMI] {current_month} already present")

    # -------------------------------------------------
    # STEP 2: BUILD / APPEND MASTER_DB.XLSX
    # -------------------------------------------------

    if not MASTER_DB_FILE.exists():
        # FIRST RUN: full backfill from tmi.json
        print("[TMI] master_db not found — building full history from tmi.json")

        records = []

        for month_label, tickers in tmi.items():
            month_dt = _month_to_date(month_label)

            for ticker in tickers:
                records.append(
                    {
                        # Identity
                        "MonthYear": month_label,
                        "MonthDate": month_dt,
                        "Ticker": ticker,
                        "Sector": None,
                        "Industry": None,
                        "Generator": FLAG_TMI_DONE,
                        "TMI_Flag": FLAG_YES,
                        "Momentum_Flag": FLAG_NO,
                        "Quality_Flag": FLAG_NO,
                        "Value_Flag": FLAG_NO,
                        "Optimization_Flag": FLAG_NO,
                    }
                )

        master_db = pd.DataFrame(records)

    else:
        master_db = pd.read_excel(MASTER_DB_FILE)

        # recreate MonthDate safely (legacy compatibility)
        if "MonthDate" not in master_db.columns:
            master_db["MonthDate"] = master_db["MonthYear"].apply(_month_to_date)

        month_exists = (master_db["MonthYear"] == current_month).any()

        if not month_exists:
            print(f"[TMI] Appending {current_month} to master_db.xlsx")

            month_dt = _month_to_date(current_month)

            new_rows = pd.DataFrame(
                {
                    # Identity
                    "MonthYear": current_month,
                    "MonthDate": month_dt,
                    "Ticker": tmi[current_month],
                    "Sector": None,
                    "Industry": None,
                    "Generator": FLAG_TMI_DONE,
                    "TMI_Flag": FLAG_YES,
                    "Momentum_Flag": FLAG_NO,
                    "Quality_Flag": FLAG_NO,
                    "Value_Flag": FLAG_NO,
                    "Optimization_Flag": FLAG_NO,
                }
            )

            master_db = pd.concat([master_db, new_rows], ignore_index=True)

    # -------------------------------------------------
    # FINAL SORT & SAVE
    # -------------------------------------------------

    master_db.drop_duplicates(subset=["MonthYear", "Ticker"], inplace=True)

    master_db.sort_values(["MonthDate", "Ticker"], inplace=True)

    # remove internal column before save
    master_db.drop(columns=["MonthDate"], inplace=True)

    master_db = master_db[UNIVERSE_COLUMNS]
    master_db.to_excel(MASTER_DB_FILE, index=False)

    print(f"[TMI] Master DB rows: {len(master_db):,}")

    # -------------------------------------------------
    # STEP 3: APPEND UNIQUE_TICKERS.XLSX
    # -------------------------------------------------

    all_tickers = sorted(set(master_db["Ticker"]))

    if UNIQUE_TICKERS_FILE.exists():
        ut = pd.read_excel(UNIQUE_TICKERS_FILE)
        existing = set(ut["Ticker"])
    else:
        ut = pd.DataFrame(columns=["Ticker"])
        existing = set()

    new_tickers = sorted(set(all_tickers) - existing)

    if new_tickers:
        print(f"[TMI] Adding {len(new_tickers)} new tickers")
        new_df = pd.DataFrame({"Ticker": new_tickers})
        ut = pd.concat([ut, new_df], ignore_index=True)
        ut.to_excel(UNIQUE_TICKERS_FILE, index=False)
    else:
        print("[TMI] No new tickers found")

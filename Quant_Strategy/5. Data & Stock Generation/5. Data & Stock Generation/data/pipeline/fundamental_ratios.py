# data/pipeline/fundamental_ratios.py

from __future__ import annotations

import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config.settings import DATA_DIR, DATA_START_YEAR

# =================================================
# PATHS
# =================================================

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UNIQUE_TICKERS_FILE = RAW_DIR / "unique_tickers.xlsx"
OUT_PARQUET = PROCESSED_DIR / "fundamental_ratios.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# MONEYCONTROL CONFIG
# =================================================

AUTOSUGGEST_URL = "https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}
RATIO_PAGES = range(1, 7)

# Moneycontrol → internal metric mapping
METRICS = {
    "Diluted EPS (Rs.)": "EPS",
    "Book Value [InclRevalReserve]/Share (Rs.)": "BookValue",
    "Revenue from Operations/Share (Rs.)": "RevenuePerShare",
    "PBDIT/Share (Rs.)": "EBITDA",
    "Return on Networth / Equity (%)": "ROE",
    "Return on Capital Employed (%)": "ROCE",
    "Return on Assets (%)": "ROA",
}

# =================================================
# HELPERS
# =================================================


def _load_universe() -> List[str]:
    df = pd.read_excel(UNIQUE_TICKERS_FILE)
    return sorted(df["Ticker"].dropna().unique().tolist())


def _load_existing_data() -> pd.DataFrame:
    if OUT_PARQUET.exists():
        return pd.read_parquet(OUT_PARQUET)
    return pd.DataFrame(columns=["Ticker", "Metric", "FiscalYear", "Value"])


def _get_mc_slug_and_code(ticker: str) -> Tuple[str, str] | None:
    try:
        r = requests.get(
            AUTOSUGGEST_URL,
            params={
                "classic": "true",
                "query": ticker.replace(".NS", ""),
                "type": "1",
                "format": "json",
            },
            timeout=10,
        )
        r.raise_for_status()

        link = r.json()[0]["link_src"]
        slug = urlparse(link).path.rstrip("/").split("/")[-2]
        code = link.split("/")[-1].upper()

        return slug, code
    except Exception:
        return None


def _normalize_fy(col: str) -> int | None:
    try:
        yy = int(col.split()[-1])
        return 2000 + yy if yy < 50 else 1900 + yy
    except Exception:
        return None


def _scrape_ratios(slug: str, code: str) -> List[pd.DataFrame]:
    tables = []

    for page in RATIO_PAGES:
        url = f"https://www.moneycontrol.com/financials/{slug}/ratiosVI/{code}/{page}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break
        except Exception:
            break

        soup = BeautifulSoup(r.content, "lxml")
        table = soup.find("table", class_="mctable1")
        if table is None:
            break

        rows = [
            [td.text.strip() for td in tr.find_all("td")]
            for tr in table.find_all("tr")
            if tr.find_all("td")
        ]

        if not rows:
            break

        df = pd.DataFrame(rows)
        df.columns = df.iloc[0]
        df = df.iloc[2:]
        df.rename(columns={df.columns[0]: "Metric"}, inplace=True)
        df.reset_index(drop=True, inplace=True)

        tables.append(df)
        time.sleep(0.15)

    return tables


def _process_ticker(ticker: str, existing: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []

    res = _get_mc_slug_and_code(ticker)
    if res is None:
        return rows

    slug, code = res
    tables = _scrape_ratios(slug, code)
    if not tables:
        return rows

    existing_years = (
        existing[existing["Ticker"] == ticker]
        .groupby("Metric")["FiscalYear"]
        .apply(set)
        .to_dict()
    )

    for table in tables:
        for mc_label, metric in METRICS.items():
            if mc_label not in table["Metric"].values:
                continue

            row = table.loc[table["Metric"] == mc_label].iloc[0]

            for col in table.columns[1:]:
                fy = _normalize_fy(col)
                if fy is None or fy < DATA_START_YEAR:
                    continue

                if fy in existing_years.get(metric, set()):
                    continue

                try:
                    val = float(row[col])
                except Exception:
                    continue

                rows.append(
                    {
                        "Ticker": ticker,
                        "Metric": metric,
                        "FiscalYear": fy,
                        "Value": val,
                    }
                )

    return rows


def _materialize_excel_views(df: pd.DataFrame) -> None:
    """
    Rebuilds factor input Excel files from parquet (source of truth).
    """

    for metric in sorted(df["Metric"].unique()):
        pivot = (
            df[df["Metric"] == metric]
            .pivot(index="FiscalYear", columns="Ticker", values="Value")
            .sort_index()
        )

        out_file = RAW_DIR / f"{metric}.xlsx"
        pivot.to_excel(out_file)

    print("[FUNDAMENTALS] Excel factor views refreshed")


# =================================================
# PIPELINE
# =================================================


def fundamental_ratios() -> None:
    print("=" * 70)
    print("[FUNDAMENTALS] Incremental download (Moneycontrol)")
    print("=" * 70)

    try:
        tickers = _load_universe()
        existing = _load_existing_data()

        records: List[Dict] = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_process_ticker, ticker, existing): ticker
                for ticker in tickers
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[FUNDAMENTALS] Processing",
                ncols=100,
            ):
                records.extend(future.result())

        if records:
            new_df = pd.DataFrame(records)
            combined = pd.concat([existing, new_df], ignore_index=True)

            combined.drop_duplicates(
                subset=["Ticker", "Metric", "FiscalYear"],
                keep="last",
                inplace=True,
            )

            combined.sort_values(
                ["Metric", "Ticker", "FiscalYear"],
                inplace=True,
            )

            combined.to_parquet(OUT_PARQUET, index=False)
        else:
            combined = existing

        _materialize_excel_views(combined)

        print("=" * 70)
        print("[FUNDAMENTALS] SUMMARY")
        print(f"Universe tickers : {len(tickers)}")
        print(f"New rows added   : {len(records)}")
        print(f"Total rows      : {len(combined)}")
        print(f"Stored at       : {OUT_PARQUET}")
        print("=" * 70)
        print("[FUNDAMENTALS] Step completed successfully")
        print("=" * 70)

    except Exception:
        print("=" * 70)
        print("[FUNDAMENTALS] FAILED")
        print("=" * 70)
        raise

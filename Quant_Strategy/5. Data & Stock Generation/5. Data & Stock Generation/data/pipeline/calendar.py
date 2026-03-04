# data/pipeline/calendar.py

from datetime import date
from typing import List

import pandas as pd
import pandas_market_calendars as mcal

from config.settings import DATA_DIR, DATA_START_YEAR

# =================================================
# PATHS
# =================================================

CALENDAR_DIR = DATA_DIR / "processed" / "calendar"
CALENDAR_DIR.mkdir(parents=True, exist_ok=True)

PAST_CALENDAR_FILE = CALENDAR_DIR / "trading_days_past.parquet"
FUTURE_CALENDAR_FILE = CALENDAR_DIR / "trading_days_future.parquet"

# =================================================
# NSE Calendar
# =================================================

_NSE_CALENDAR = mcal.get_calendar("NSE")

# =================================================
# HELPERS
# =================================================


def _get_trading_days(start: date, end: date) -> List[date]:
    schedule = _NSE_CALENDAR.schedule(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
    )
    return [d.date() for d in schedule.index]


def _last_completed_trading_day(today: date) -> date:
    lookback_start = today - pd.Timedelta(days=10)
    days = _get_trading_days(lookback_start, today)

    if not days:
        raise RuntimeError("[CALENDAR] No recent NSE trading days found")

    return days[-1]


def _first_trading_day_of_month(year: int, month: int) -> date:
    start = date(year, month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).date()
    days = _get_trading_days(start, end)

    if not days:
        raise RuntimeError(f"[CALENDAR] No trading days for {year}-{month:02d}")

    return days[0]


# =================================================
# PUBLIC API
# =================================================


def trading_calendar_past() -> pd.DataFrame:
    """
    Cache all NSE trading days from DATA_START_YEAR
    till last completed trading day.
    """

    print("[CALENDAR] Building historical trading calendar")

    today = date.today()
    start_date = date(DATA_START_YEAR, 1, 1)
    end_date = _last_completed_trading_day(today)

    trading_days = _get_trading_days(start_date, end_date)

    df = pd.DataFrame({"date": trading_days})
    df["year"] = df["date"].apply(lambda d: d.year)
    df["month"] = df["date"].apply(lambda d: d.month)

    df.to_parquet(PAST_CALENDAR_FILE, index=False)

    print(f"[CALENDAR] Cached {len(df)} trading days ({start_date} → {end_date})")

    return df


def trading_calendar_future() -> None:
    """
    Identify future monthly rebalancing dates.

    Rules:
    - Show ONLY current year
    - If current month is December, also show next year
    - Show ONLY dates strictly AFTER today
    """

    print("[CALENDAR] Building future rebalancing calendar")

    today = date.today()

    # -------------------------------------------------
    # Decide which years to include
    # -------------------------------------------------
    years = [today.year]

    if today.month == 12:
        years.append(today.year + 1)

    records = []

    for year in years:
        for month in range(1, 13):
            first_day = _first_trading_day_of_month(year, month)

            # Skip past rebalancing dates
            if first_day <= today:
                continue

            records.append(
                {
                    "year": year,
                    "month": month,
                    "first_trading_day": first_day,
                    "weekday": first_day.strftime("%A"),
                }
            )

    if not records:
        print("[CALENDAR] No future rebalancing dates found")
        return

    df = pd.DataFrame(records)
    df.to_parquet(FUTURE_CALENDAR_FILE, index=False)

    # -------------------------------------------------
    # Print + Telegram (DATA channel)
    # -------------------------------------------------

    lines = ["📅 <b>Upcoming Rebalancing Dates</b>\n"]

    print("-" * 60)
    for _, row in df.iterrows():
        label = date(row["year"], row["month"], 1).strftime("%b %Y")
        day = row["first_trading_day"].strftime("%d %b %Y")
        line = f"{label:<10} → {day} ({row['weekday']})"
        print(line)
        lines.append(line)

    print("-" * 60)
    print("[CALENDAR] Step completed successfully")
    print("=" * 60)

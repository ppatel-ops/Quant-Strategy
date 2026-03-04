# data/pipeline/run_pipeline.py

from datetime import date
import traceback

from data.pipeline.master_universe import master_universe
from data.pipeline.calendar import (
    trading_calendar_future,
    trading_calendar_past,
)
from data.pipeline.sector_industry import sector_industry
from data.pipeline.stocks_prices import stocks_prices
from data.pipeline.stocks_prices import stocks_prices_KITE_without_multithread
from data.pipeline.indices_prices import indices_prices
from data.pipeline.fundamental_ratios import fundamental_ratios

# =================================================
# PIPELINE ORCHESTRATOR
# =================================================


def run_data_pipeline() -> None:
    """
    Data Pipeline

    Characteristics:
    - Idempotent
    - Incremental
    - Crash-safe
    """

    today = date.today()

    print("=" * 70)
    print("Data Pipeline")
    print("=" * 70)
    print(f"Run Date: {today.isoformat()}")

    try:
        # -------------------------------------------------
        # STEP 1: MASTER UNIVERSE
        # -------------------------------------------------

        master_universe()

        # -------------------------------------------------
        # STEP 2: SECTOR & INDUSTRY DATA
        # -------------------------------------------------

        sector_industry()

        # -------------------------------------------------
        # STEP 3: TRADING CALENDARS
        # -------------------------------------------------

        trading_calendar_future()
        trading_calendar_past()

        # -------------------------------------------------
        # STEP 4: STOCK PRICES
        # -------------------------------------------------

        stocks_prices()
        stocks_prices_KITE_without_multithread()
        # -------------------------------------------------
        # STEP 5: INDICES PRICES
        # -------------------------------------------------

        indices_prices()

        # -------------------------------------------------
        # STEP 6: FUNDAMENTAL RATIOS
        # -------------------------------------------------

        fundamental_ratios()

        # -------------------------------------------------
        print("=" * 70)
        print("DATA PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception:
        print("=" * 70)
        print("DATA PIPELINE FAILED")
        print("=" * 70)
        traceback.print_exc()

        raise

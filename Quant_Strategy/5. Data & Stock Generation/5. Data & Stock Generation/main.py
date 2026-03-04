# main.py

from enum import Enum
import traceback

from config.settings import ensure_directories

# -----------------------------
# PIPELINES
# -----------------------------
from data.pipeline.run_pipeline import run_data_pipeline
from signals.generator import generate_investment_lists

# ==========================================================
# MAIN ORCHESTRATOR
# ==========================================================


def main() -> None:
    """
    Master Orchestrator

    Flow:
    1. Run data pipeline
    2. Generate investment lists (signals)
    """

    print("=" * 70)
    print("Strategy Orchestrator ")
    print("=" * 70)

    ensure_directories()

    try:
        # --------------------------------------------------
        # STEP 1: DATA PIPELINE
        # --------------------------------------------------

        run_data_pipeline()

        # --------------------------------------------------
        # STEP 2: SIGNAL GENERATION
        # --------------------------------------------------

        generate_investment_lists()

    except Exception as exc:
        print("=" * 70)
        print(" STRATEGY PIPELINE FAILED ")
        print("=" * 70)
        traceback.print_exc()
        raise exc

    finally:
        print("=" * 70)
        print(" Strategy run completed ")
        print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

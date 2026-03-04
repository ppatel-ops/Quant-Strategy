# signals/generator.py

import traceback

# -------------------------------------------------
# FACTORS
# -------------------------------------------------
from factors.value import run_value_factor
from factors.quality import run_quality_factor
from factors.momentum import run_momentum_factor
from optimization.optimization import portfolio_optimization


# =================================================
# SIGNAL GENERATOR
# =================================================


def generate_investment_lists() -> None:
    """
    TMI → Momentum → Quality → Value → Optimization
    """

    print("=" * 70)
    print("Signal Generation Pipeline")
    print("=" * 70)

    try:
        # ---------------------------------------------
        # MOMENTUM
        # ---------------------------------------------

        run_momentum_factor()

        # ---------------------------------------------
        # QUALITY
        # ---------------------------------------------

        run_quality_factor()

        # ---------------------------------------------
        # VALUE
        # ---------------------------------------------

        run_value_factor()

        # ---------------------------------------------
        # OPTIMIZATION
        # ---------------------------------------------

        portfolio_optimization()

    except Exception:
        traceback.print_exc()
        raise

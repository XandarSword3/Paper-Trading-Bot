import pytest
import pandas as pd
import numpy as np

from quant_research.core.walk_forward import stitch_walk_forward_trades


def test_fold_returns_stitching_and_ruin_floor():
    """
    Unit test verifying:
    1. Fold returns of +10% and -10% stitch to exactly -1.0% total return:
       (1.0 + 0.10) * (1.0 - 0.10) - 1.0 = -0.01 = -1.0%.
    2. Equity can never go below zero (hard ruin floor), even with catastrophic negative returns.
    3. Maximum drawdown is strictly bounded in [0%, 100%].
    """
    # 1. Fold 1 (+10%) and Fold 2 (-10%)
    fold1_trades = pd.DataFrame([
        {"entry_time": "2025-01-01", "exit_time": "2025-01-02", "return_pct": 0.10, "net_pnl": 1000.0}
    ])
    fold2_trades = pd.DataFrame([
        {"entry_time": "2025-02-01", "exit_time": "2025-02-02", "return_pct": -0.10, "net_pnl": -1000.0}
    ])

    result = stitch_walk_forward_trades([fold1_trades, fold2_trades], initial_capital=10_000.0)

    # Must stitch to exactly -1.0%
    assert abs(result["total_return_pct"] - (-1.0)) < 1e-6, (
        f"Expected -1.0% return, got {result['total_return_pct']}"
    )

    # Max Drawdown from peak (11,000 -> 9,900 = 1,100 / 11,000 = 10%)
    assert abs(result["max_drawdown_pct"] - 10.0) < 1e-6, (
        f"Expected 10.0% max drawdown, got {result['max_drawdown_pct']}"
    )

    # 2. Hard ruin floor at zero: catastrophic -200% loss trade
    catastrophic_trades = pd.DataFrame([
        {"entry_time": "2025-03-01", "exit_time": "2025-03-02", "return_pct": -2.0, "net_pnl": -20000.0},
        {"entry_time": "2025-03-03", "exit_time": "2025-03-04", "return_pct": 0.50, "net_pnl": 5000.0},
    ])

    cat_result = stitch_walk_forward_trades([catastrophic_trades], initial_capital=10_000.0)

    # Equity must floor at zero (-100.0% return), cannot go negative
    assert cat_result["total_return_pct"] == -100.0, (
        f"Expected -100.0% floor, got {cat_result['total_return_pct']}"
    )
    # Max drawdown must never exceed 100.0%
    assert cat_result["max_drawdown_pct"] <= 100.0, (
        f"Expected max drawdown <= 100%, got {cat_result['max_drawdown_pct']}"
    )

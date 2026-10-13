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


def test_stitching_accepts_tz_aware_exit_times_and_reports_daily_sharpe():
    """
    Regression test for the daily-Sharpe stitched equity curve:
    exit_time values that are ALREADY tz-aware must not be re-localised
    (pd.Timestamp(value, tz='UTC') raises ValueError on a tz-aware input), and the
    stitched result must expose the canonical daily Sharpe alongside the legacy
    trade Sharpe.
    """
    idx = pd.date_range("2025-01-01", periods=40, freq="6h", tz="UTC")
    trades = pd.DataFrame(
        {
            "entry_time": idx,
            "exit_time": idx + pd.Timedelta(hours=6),
            "return_pct": [0.01, -0.005] * 20,
            "net_pnl": [100.0, -50.0] * 20,
        }
    )

    result = stitch_walk_forward_trades([trades], initial_capital=10_000.0)

    assert result["total_trades"] == 40
    # 20 x (+1%) and 20 x (-0.5%), fully compounded in exit order
    expected = ((1.01 ** 20) * (0.995 ** 20) - 1.0) * 100.0
    assert abs(result["total_return_pct"] - expected) < 1e-6, (
        f"Expected {expected}%, got {result['total_return_pct']}"
    )
    assert "daily_sharpe" in result
    assert isinstance(result["daily_sharpe"], float)
    assert abs(result["daily_sharpe"]) > 0.0, "daily Sharpe of a trending equity curve must be non-zero"

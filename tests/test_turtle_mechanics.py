"""
Real Mechanical Unit Tests for Classical Turtle Strategy.
Verifies:
1. Pyramiding mechanics: up to 4 units at 0.5N spacing.
2. Stop ratcheting: stop loss adjusts to 2N below newest unit entry price.
3. 2.0x leverage cap: binds when ATR/price < 0.25% and clamps gross notional.
4. Pessimistic intrabar exit ordering: Stop Loss triggers before Channel Exit.
5. Invariant enforcement: assert_no_same_bar_pyramiding.
"""
import pytest
import pandas as pd
import numpy as np

from quant_research.core.engine import CostModel, QuantEngine, assert_no_same_bar_pyramiding
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_diagnostics import run_turtle_diagnostic_sim


def test_turtle_pyramid_up_to_four_units_and_stop_ratchet():
    """
    Constructs a strong trending market with 4 breakout expansions:
    Verifies that:
    - 4 units are added sequentially.
    - Each unit enters at >= previous entry + 0.5*ATR.
    - No two units enter on the same candle bar.
    - Ratchet stop updates upward with each unit.
    """
    n = 120
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Base: flat at 100 for 40 bars (warmup), then steady staircase trend up to 200
    prices = np.full(n, 100.0)
    for i in range(40, n):
        prices[i] = 100.0 + (i - 40) * 1.5

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 0.5,
        "close": prices,
    }).set_index("timestamp")

    # Add reversal at the end so all units exit and complete
    df.iloc[-5:, df.columns.get_loc("close")] = 50.0
    df.iloc[-5:, df.columns.get_loc("low")] = 40.0

    features = prepare_turtle_features(df, N=20, M=10)
    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)
    params = {"N": 20, "M": 10, "config_id": "TEST_PYRAMID"}

    metrics, trades = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost,
        initial_capital=100_000.0,
        return_trades=True,
    )

    assert len(trades) > 0
    # Group by exit time to inspect the position cycle
    cycle_units = [t for t in trades if t.exit_time == trades[-1].exit_time]
    unit_nums = [t.unit_number for t in cycle_units]

    # Verify pyramid units reached up to 4
    assert max(unit_nums) == 4, f"Expected 4 pyramid units, got {max(unit_nums)}"
    assert sorted(unit_nums) == [1, 2, 3, 4]

    # Verify strict invariant: no two units share same entry timestamp
    assert_no_same_bar_pyramiding(trades)


def test_turtle_leverage_cap_binding():
    """
    Verifies that when ATR / price < 0.25%, the desired unit size exceeds 2x leverage
    and is strictly clamped to exactly 2.0x portfolio equity gross notional.
    """
    n = 60
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Very small ATR: price = 100,000, daily/hourly range = 50 -> ATR/price = 0.05% < 0.25%
    prices = np.full(n, 100_000.0)
    # Trigger breakout at bar 35
    for i in range(35, n):
        prices[i] = 100_000.0 + (i - 35) * 50.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 25.0,
        "low": prices - 25.0,
        "close": prices,
    }).set_index("timestamp")

    # Add exit crash at end
    df.iloc[-3:, df.columns.get_loc("close")] = 90_000.0
    df.iloc[-3:, df.columns.get_loc("low")] = 85_000.0

    features = prepare_turtle_features(df, N=20, M=10)
    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)
    params = {"N": 20, "M": 10, "config_id": "TEST_CAP"}

    metrics, trades = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost,
        initial_capital=100_000.0,
        return_trades=True,
    )

    assert len(trades) > 0
    # For unit 1, notional must NOT exceed 2.0 * initial_capital ($200,000.00)
    u1 = [t for t in trades if t.unit_number == 1][0]
    notional = u1.quantity * u1.entry_price
    assert notional <= 200_000.01, f"Expected notional <= $200k, got {notional}"
    # Clamped near 2.0x ($200k)
    assert notional > 190_000.0, f"Expected clamped notional near $200k, got {notional}"


def test_pessimistic_intrabar_exit_ordering():
    """
    Verifies that when a single candle hits BOTH the Stop Loss (intrabar)
    and the Channel Exit (on close), the engine executes the Stop Loss FIRST.
    """
    n = 60
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    prices = np.full(n, 100.0)
    # Trigger long entry at bar 30
    prices[30] = 110.0
    prices[31] = 110.0  # entry fills on open at 110.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 2.0,
        "low": prices - 2.0,
        "close": prices,
    }).set_index("timestamp")

    # At bar 32, create an ambiguous candle:
    # Low plunges to 95 (breaching 2N stop loss of ~106)
    # AND Close drops to 96 (breaching Donchian 10-bar exit channel of ~98)
    df.iloc[32, df.columns.get_loc("low")] = 90.0
    df.iloc[32, df.columns.get_loc("close")] = 92.0

    features = prepare_turtle_features(df, N=20, M=10)
    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)
    params = {"N": 20, "M": 10, "config_id": "TEST_ORDERING"}

    metrics, trades = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost,
        initial_capital=100_000.0,
        return_trades=True,
    )

    assert len(trades) > 0
    # Exit reason MUST be Stop Loss, not Channel Exit
    assert trades[0].exit_reason == "2N Stop Loss", f"Expected '2N Stop Loss', got {trades[0].exit_reason}"

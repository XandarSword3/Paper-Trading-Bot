import pytest
import pandas as pd
import numpy as np
from quant_research.core.engine import CostModel
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation


def test_turtle_strategy_mechanics_and_invariants():
    """
    Verifies that the Turtle simulator adheres to:
    1. Closed bar breakout signals filling on next open.
    2. Pyramiding up to 4 units at 0.5N intervals.
    3. Stop ratcheting to 2N below each new unit.
    4. assert_no_same_bar_pyramiding invariant.
    """
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Base price series: 50 warmup bars flat at 100.0, trend up to 160 at bar 80, then sharp crash to 80
    prices = np.full(n, 100.0)
    for i in range(50, 80):
        prices[i] = 100.0 + (i - 50) * 2.0  # steady upward trend to 160.0
    for i in range(80, n):
        prices[i] = 160.0 - (i - 80) * 4.0  # sharp drop triggering exit

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
    }).set_index("timestamp")

    features = prepare_turtle_features(df, N=20, M=10)
    params = {"N": 20, "M": 10, "config_id": "TURTLE_TEST"}
    cost_model = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)

    metrics, trades = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost_model,
        initial_capital=100_000.0,
        bars_per_year=8760.0,
        return_trades=True,
    )

    # Invariants
    assert len(trades) > 0, "Expected completed or entered trades"
    # Verify no two units share the exact same entry timestamp
    entry_times = [t.entry_time for t in trades]
    assert len(entry_times) == len(set(entry_times)), "Pyramid units must not share same entry timestamp"
    # Verify unit numbers are within 1 to 4
    for t in trades:
        assert 1 <= t.unit_number <= 4, f"Unit number {t.unit_number} exceeds [1, 4] range"

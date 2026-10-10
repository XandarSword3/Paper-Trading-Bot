"""
Unit test proving that same-bar pyramid duplication cannot recur in any new or existing strategy code.

Background:
In early versions of the backtester, a breakout candle that triggered an initial entry
could also immediately trigger pyramid add(s) on the exact same bar, because the pyramid
check used the candle's high/low against last_add_price. On large breakout candles or with
tight pyramid spacing, this silently doubled or quadrupled position size and risk on bar 0,
using intraday information that was not actionable in real-time.

This test suite:
1. Tests against synthetic adversarial candles designed to provoke the bug (explosive 100-ATR breakout candles).
2. Tests both Long and Short pyramiding with hyper-aggressive spacing.
3. Tests a universal trade invariant validator that can be applied to any trade log.
"""
import pytest
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Any

from config import StrategyParams, DEFAULT_PARAMS
from strategy import TurtleDonchianStrategy, TradeRecord


def assert_no_same_bar_pyramiding(trades: List[Any]):
    """
    Universal invariant check: For any list of trades, no two trades/units
    may ever share the same entry_time (i.e., at most 1 unit entered or added per bar).
    """
    if not trades:
        return
    entry_times = [
        t.entry_time if hasattr(t, "entry_time") else t["entry_time"]
        for t in trades
    ]
    counts = Counter(entry_times)
    duplicates = {ts: c for ts, c in counts.items() if c > 1}
    assert not duplicates, (
        f"INVARIANT VIOLATION: Same-bar pyramid duplication detected on {len(duplicates)} bar(s): {duplicates}"
    )


def _generate_adversarial_breakout_data(direction: str = "long") -> pd.DataFrame:
    """
    Generates synthetic price data:
    - 60 bars of low-volatility flat consolidation (to let 20-bar Donchian & ATR stabilize).
    - 1 explosive breakout bar spanning 100x the typical ATR.
    - 10 continuation bars.
    """
    n_pre = 220
    dates = pd.date_range("2024-01-01", periods=n_pre + 15, freq="1h")
    
    # Flat baseline
    base_price = 100.0
    opens = [base_price] * n_pre
    highs = [base_price + 1.0] * n_pre
    lows = [base_price - 1.0] * n_pre
    closes = [base_price] * n_pre
    volumes = [1000.0] * n_pre

    if direction == "long":
        # Explosive breakout bar at index 60:
        # ATR is ~2.0. High is +100 above baseline!
        opens.append(base_price)
        highs.append(base_price + 100.0)
        lows.append(base_price - 0.5)
        closes.append(base_price + 90.0)
        volumes.append(50000.0)

        # Continuation bars
        for k in range(1, 15):
            opens.append(base_price + 90.0 + k)
            highs.append(base_price + 95.0 + k)
            lows.append(base_price + 89.0 + k)
            closes.append(base_price + 92.0 + k)
            volumes.append(5000.0)
    else:
        # Explosive breakdown bar at index 60
        opens.append(base_price)
        highs.append(base_price + 0.5)
        lows.append(base_price - 80.0)
        closes.append(base_price - 75.0)
        volumes.append(50000.0)

        # Continuation bars
        for k in range(1, 15):
            opens.append(base_price - 75.0 - k)
            highs.append(base_price - 74.0 - k)
            lows.append(base_price - 80.0 - k)
            closes.append(base_price - 77.0 - k)
            volumes.append(5000.0)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_adversarial_long_breakout_candle_never_duplicates_pyramid():
    """
    Stress test: A 100-ATR breakout bar with hyper-aggressive pyramid spacing (0.01 ATR)
    MUST NEVER open more than 1 unit on that breakout bar.
    """
    df = _generate_adversarial_breakout_data(direction="long")

    params = StrategyParams(
        entry_len=20,
        exit_len=10,
        atr_len=14,
        trail_mult=10.0,
        risk_percent=1.0,
        pyramid_spacing_n=0.01,  # Ultra-tight spacing to aggressively provoke same-bar trigger
        max_units=4,
        long_only=True,
        use_regime_filter=False,
    )

    strat = TurtleDonchianStrategy(params)
    strat.run_backtest(df, initial_capital=100_000.0, verbose=False)

    assert len(strat.trades) > 0, "Strategy should have entered the breakout trade"
    
    # Assert universal invariant: no duplicate entry times
    assert_no_same_bar_pyramiding(strat.trades)

    # Specific check on the breakout bar (index 60)
    breakout_ts = df.index[220]
    trades_on_breakout = [t for t in strat.trades if t.entry_time == breakout_ts]
    assert len(trades_on_breakout) == 1, (
        f"Expected exactly 1 unit on the breakout candle, found {len(trades_on_breakout)}"
    )
    assert trades_on_breakout[0].unit_number == 1, "The single trade must be Unit #1"


def test_adversarial_short_breakdown_candle_never_duplicates_pyramid():
    """
    Stress test: A massive short breakdown bar with hyper-aggressive pyramid spacing (0.01 ATR)
    MUST NEVER open more than 1 unit on that breakdown bar.
    """
    df = _generate_adversarial_breakout_data(direction="short")

    params = StrategyParams(
        entry_len=20,
        exit_len=10,
        atr_len=14,
        trail_mult=10.0,
        risk_percent=1.0,
        pyramid_spacing_n=0.01,
        max_units=4,
        long_only=False,
        use_regime_filter=False,
    )

    strat = TurtleDonchianStrategy(params)
    strat.run_backtest(df, initial_capital=100_000.0, verbose=False)

    assert len(strat.trades) > 0, "Strategy should have entered the short trade"
    assert_no_same_bar_pyramiding(strat.trades)

    breakout_ts = df.index[220]
    trades_on_breakout = [t for t in strat.trades if t.entry_time == breakout_ts]
    assert len(trades_on_breakout) == 1, (
        f"Expected exactly 1 unit on the short breakdown candle, found {len(trades_on_breakout)}"
    )
    assert trades_on_breakout[0].unit_number == 1


def test_universal_invariant_catches_synthetic_violation():
    """
    Validates that the invariant check itself correctly fails when a bug produces
    duplicate entry timestamps.
    """
    ts = pd.Timestamp("2024-01-01 12:00:00")
    dummy_trades = [
        TradeRecord(
            entry_time=ts,
            exit_time=None,
            direction="long",
            entry_price=100.0,
            exit_price=None,
            quantity=1.0,
            pnl=None,
            exit_reason=None,
            unit_number=1,
        ),
        TradeRecord(
            entry_time=ts,  # Same entry time: duplicate!
            exit_time=None,
            direction="long",
            entry_price=101.0,
            exit_price=None,
            quantity=1.0,
            pnl=None,
            exit_reason=None,
            unit_number=2,
        ),
    ]

    with pytest.raises(AssertionError, match="INVARIANT VIOLATION"):
        assert_no_same_bar_pyramiding(dummy_trades)

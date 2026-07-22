"""
Regression test for a bug found during the July 2026 walk-forward audit:
pyramiding could fire on the exact same bar as a position's initial entry
(or its most recent add), because last_add_price was set to that bar's own
close while the trigger check used that same bar's high/low. On a big
breakout candle this let a second unit through with an identical
entry_time, entry_price, and quantity to the first — silently doubling
risk on the entry bar for ~8% of trades across the 9-fold OOS suite,
using information (that bar's own high) that isn't actually actionable
twice within one bar in live/paper trading.

Fix: gate both pyramid branches with `timestamp > last_add_time`, and set
last_add_time alongside last_add_price on every fresh entry and every add.

This test runs the real fold-0 window (same data, same winning params as
walk_forward_results_v1.json) and asserts no two trades ever share an
entry_time — i.e. at most one unit is ever added per bar.
"""
import warnings
warnings.filterwarnings("ignore")

from collections import Counter

import pandas as pd

from config import DEFAULT_PARAMS, StrategyParams
from strategy import TurtleDonchianStrategy


def _load_btc_4h():
    return pd.read_csv(
        "data/BTCUSDT_4h.csv", parse_dates=["timestamp"], index_col="timestamp"
    )


def test_no_two_units_share_an_entry_bar():
    df = _load_btc_4h()

    # Fold 0's winning params (walk_forward_results_v1.json / fold_0.json) —
    # chosen because it's the fold where the bug was originally reproduced.
    train_start = pd.Timestamp("2018-05-15 04:00:00")
    test_end = pd.Timestamp("2020-11-15 04:00:00")
    combined = df[(df.index >= train_start) & (df.index < test_end)]

    params = StrategyParams(
        entry_len=24, exit_len=7, atr_len=17, trail_mult=4.0,
        risk_percent=1.0, pyramid_spacing_n=0.75,
        size_stop_mult=DEFAULT_PARAMS.size_stop_mult,
        max_units=DEFAULT_PARAMS.max_units,
        long_only=DEFAULT_PARAMS.long_only,
        use_regime_filter=DEFAULT_PARAMS.use_regime_filter,
        lot_step=DEFAULT_PARAMS.lot_step,
        commission_pct=DEFAULT_PARAMS.commission_pct,
        slippage_pct=DEFAULT_PARAMS.slippage_pct,
    )
    strat = TurtleDonchianStrategy(params)
    strat.run_backtest(combined, initial_capital=100_000.0, verbose=False)

    assert len(strat.trades) > 20, "sanity check: expected a meaningful number of trades"

    entry_time_counts = Counter(t.entry_time for t in strat.trades)
    dup_bars = {t: c for t, c in entry_time_counts.items() if c > 1}

    assert not dup_bars, (
        f"found {len(dup_bars)} bar(s) with more than one unit opened at the "
        f"same entry_time (same-bar pyramid duplication bug has regressed): "
        f"{dup_bars}"
    )


def test_pyramid_add_requires_a_later_bar_than_last_add():
    """More direct unit check: units_count should never exceed 1 unless at
    least one bar has elapsed since the position (or its last add) opened."""
    df = _load_btc_4h()
    train_start = pd.Timestamp("2018-05-15 04:00:00")
    test_end = pd.Timestamp("2020-11-15 04:00:00")
    combined = df[(df.index >= train_start) & (df.index < test_end)]

    # A tight pyramid_spacing_n makes same-bar re-triggering easy to hit if
    # the guard regresses — this is deliberately more aggressive than any
    # winning fold's parameters to stress-test the gate itself.
    params = StrategyParams(
        entry_len=24, exit_len=7, atr_len=17, trail_mult=4.0,
        risk_percent=1.0, pyramid_spacing_n=0.1,
        size_stop_mult=DEFAULT_PARAMS.size_stop_mult,
        max_units=DEFAULT_PARAMS.max_units,
        long_only=DEFAULT_PARAMS.long_only,
        use_regime_filter=DEFAULT_PARAMS.use_regime_filter,
        lot_step=DEFAULT_PARAMS.lot_step,
        commission_pct=DEFAULT_PARAMS.commission_pct,
        slippage_pct=DEFAULT_PARAMS.slippage_pct,
    )
    strat = TurtleDonchianStrategy(params)
    strat.run_backtest(combined, initial_capital=100_000.0, verbose=False)

    entry_time_counts = Counter(t.entry_time for t in strat.trades)
    assert all(c <= 1 for c in entry_time_counts.values()), (
        "a spacing_n aggressive enough to invite same-bar re-triggering still "
        "produced a bar with >1 unit opened — the timestamp > last_add_time "
        "guard is not holding"
    )

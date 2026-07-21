"""
Shared metric-calculation helpers with no dependency on strategy.py or
walk_forward.py, so both can import from here without a circular import.

infer_bars_per_year() was originally written inside walk_forward.py
(as _infer_bars_per_year) for Phase 2 of VALIDATION_REMEDIATION_PLAN.md,
specifically because strategy.py's own get_equity_stats() inferred
annualization from equity.index.freq — which is None for essentially all
real fetched OHLCV data (pandas does not set .freq just because bars happen
to be evenly spaced; it has to be constructed via date_range or explicitly
assigned). That silently defaulted every backtest to 365*24 bars/year
(hourly) regardless of the actual timeframe, inflating Sharpe/CAGR/Sortino/
Calmar by ~2x for any 4h backtest — which includes the primary V1 BTC
strategy, robustness_test.py's whole grid, and anything else that calls
strategy.get_equity_stats() directly.

walk_forward.py, monte_carlo.py, and edge_validation_test.py already worked
around this with their own correct inference and were unaffected. This
module makes that the one canonical implementation instead of a fork, and
strategy.py now uses it too.
"""

import pandas as pd


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Median bar spacing, not the (often-unset) .freq attribute, since real
    fetched data frequently has freq=None even when regularly spaced."""
    if len(index) < 3:
        return 365.0
    diffs = index.to_series().diff().dropna()
    median_seconds = diffs.dt.total_seconds().median()
    if not median_seconds or median_seconds <= 0:
        return 365.0
    return (365.25 * 86400) / median_seconds

"""Regression tests for funding_rate_carry.py."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import DEFAULT_SPLIT
from funding_rate_carry import compute_stats, run_backtest


def _synthetic_funding(rates, start="2020-01-01", freq="8h"):
    idx = pd.date_range(start, periods=len(rates), freq=freq)
    return pd.Series(rates, index=idx, name="funding_rate")


def test_positive_funding_compounds_up():
    fr = _synthetic_funding([0.0003] * 300)  # ~0.03%/8h, realistic typical rate
    equity = run_backtest(fr, initial_capital=100_000.0)
    assert equity.iloc[-1] > 100_000.0
    stats = compute_stats(equity, 100_000.0)
    assert stats["max_drawdown_pct"] == 0.0  # monotonically up, no drawdown at all
    assert stats["cagr_pct"] > 0


def test_negative_funding_costs_money_but_stays_bounded():
    fr = _synthetic_funding([-0.0005] * 300)
    equity = run_backtest(fr, initial_capital=100_000.0)
    assert equity.iloc[-1] < 100_000.0
    assert equity.min() > 0  # never wipes out at realistic rates


def test_extreme_bad_data_point_is_clipped_not_catastrophic():
    # A single glitched/bad row (e.g. a parsing error yielding -1.0, "100%
    # of notional paid in one settlement") shouldn't be able to send the
    # whole curve to zero or negative.
    fr = _synthetic_funding([0.0003] * 50 + [-5.0] + [0.0003] * 50)
    equity = run_backtest(fr, initial_capital=100_000.0)
    assert equity.min() > 0


def test_holdout_boundary_is_respected():
    holdout_start = pd.Timestamp(DEFAULT_SPLIT.holdout_start)
    fr = _synthetic_funding([0.0002] * 700, start="2024-06-01", freq="8h")
    usable = fr[fr.index < holdout_start]
    assert len(usable) < len(fr), "test fixture should actually straddle the holdout boundary"
    assert usable.index[-1] < holdout_start

"""
Regression test for the drawdown bug found in the July 2026 V4 walk-forward
run: the aggregated OOS report showed oos_max_drawdown_pct=-121.3%, which is
not a physically possible number — you cannot lose more than 100% of a
non-margin-called account.

Root cause: run_backtest() tracked `equity` as an unbounded running sum of
realized PnL with no floor at zero. calculate_unit_size() then sized new
positions off `equity * risk_percent`, but its `max(lot_step, ...)` clamp
guaranteed a nonzero position size regardless of whether `equity` was
positive, zero, or negative — so a wiped-out account kept opening new,
appreciably-sized positions using leftover leverage, and could compound
losses arbitrarily far past -100%. A real leveraged account gets liquidated
at (or before) equity=0 and stops trading; nothing enforced that here.

Fix: strategy.py now clamps equity to 0.0 the moment it would go
non-positive, marks the backtest "ruined", and halts all further trading
(no exits/entries/pyramiding) for the remainder of that run.
calculate_unit_size() also refuses to size anything off equity <= 0 as a
second line of defense.

This test doesn't try to reproduce the exact V4/fold-2 numbers (that needs
four years of real 1h data and the original grid-search winning params) —
it exercises the guard directly with a small synthetic series engineered to
force a wipeout, which is deterministic and fast, and is the right level of
test for a boundary condition like this.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import StrategyParams
from strategy import TurtleDonchianStrategy


def _build_ruinous_series() -> pd.DataFrame:
    """Flat warmup, then a modest breakout (just enough to trigger one long
    entry), then an immediate violent crash, then a recovery rally that -
    under the old, unguarded code - would have re-triggered a fresh entry
    off a already-ruined account. Steering the *organic* Donchian/pyramid
    logic into an exact wipeout is fiddly (trailing-stop exits cap the loss
    on any single unit to roughly trail_mult*atr), so this scenario is
    paired with `_OverExposedStrategy` below, which fixes the position size
    at an unrealistically large, equity-agnostic value - reproducing the
    actual bug's effect (sizing that ignores whether equity can support it)
    without needing years of real data to organically compound into ruin.
    """
    n_warmup = 260
    rng = np.random.default_rng(7)

    # Quiet, low-volatility chop so ATR settles small and the 200-EMA regime
    # filter has something stable to sit on top of.
    base = 20_000 + rng.normal(0, 5, n_warmup).cumsum() * 0.05
    warmup_close = base - base.min() + 20_000

    breakout = [20_300, 20_320]          # clears the entry channel
    crash = [15_000, 9_000, 4_000]       # blows straight through any stop
    recovery = [20_500, 20_600, 20_700]  # would re-trigger a fresh breakout

    close = np.concatenate([warmup_close, breakout, crash, recovery])
    high = close + 5
    low = close - 5

    idx = pd.date_range("2020-01-01", periods=len(close), freq="4h")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def _ruin_prone_params() -> StrategyParams:
    return StrategyParams(
        entry_len=20, exit_len=10, atr_len=14, trail_mult=1.0,
        size_stop_mult=1.0,
        risk_percent=1.0,
        max_units=4, pyramid_spacing_n=0.25,
        long_only=True, use_regime_filter=False,
        lot_step=0.001, commission_pct=0.08, slippage_pct=0.05,
    )


class _OverExposedStrategy(TurtleDonchianStrategy):
    """Reproduces the bug's effect directly: a position size that ignores
    equity entirely (the old calculate_unit_size's `max(lot_step, ...)`
    floor did exactly this once `equity` went non-positive). Everything
    else - entries, exits, pyramiding, the ruin guard - is the real,
    unmodified strategy loop."""

    def calculate_unit_size(self, equity: float, atr: float, price: float) -> float:
        return 1_000.0  # a fixed, oversized BTC position regardless of equity


def test_equity_never_goes_negative_and_halts_after_ruin():
    df = _build_ruinous_series()
    strat = _OverExposedStrategy(_ruin_prone_params())
    results = strat.run_backtest(df, initial_capital=100_000.0, verbose=False)

    equity = results["equity"]

    # The core fix: equity is never allowed below zero.
    assert equity.min() >= 0.0, f"equity went negative: {equity.min()}"

    # Sanity check that this scenario actually exercises ruin (otherwise the
    # test isn't testing anything) — the crash should have driven equity to
    # exactly 0 at some point.
    ruin_idx = equity[equity == 0.0].index
    assert len(ruin_idx) > 0, "scenario didn't reach ruin — test needs a harsher crash"
    first_ruin = ruin_idx[0]

    # Once ruined, equity must stay flat at 0 for every remaining bar —
    # no rebound, no further compounding either direction.
    assert (equity.loc[first_ruin:] == 0.0).all(), \
        "equity moved after hitting 0 — ruin should be terminal for the run"

    # No trade may have been opened after the account was already ruined.
    late_trades = [t for t in strat.trades if t.entry_time > first_ruin]
    assert not late_trades, f"opened {len(late_trades)} trade(s) after ruin: {late_trades}"


def test_calculate_unit_size_refuses_nonpositive_equity():
    strat = TurtleDonchianStrategy(_ruin_prone_params())
    assert strat.calculate_unit_size(equity=0.0, atr=100.0, price=30_000.0) == 0.0
    assert strat.calculate_unit_size(equity=-5_000.0, atr=100.0, price=30_000.0) == 0.0

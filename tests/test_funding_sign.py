"""
Funding-direction unit tests (PREREG section 3):

    long  + positive rate -> PAYS
    short + positive rate -> RECEIVES
    (both reversed for a negative rate)

These lock in the fix to the defect where turtle_strategy.run_turtle_simulation and
turtle_diagnostics.run_turtle_diagnostic_sim charged -qty * close * rate for every
position regardless of direction.  Values are hand-computed and asserted to 1e-9,
both through QuantEngine.funding_payment (the single source of truth) and end to
end through both Turtle simulators on a synthetic frame with one known settlement.
"""
import numpy as np
import pandas as pd
import pytest

from quant_research.core.engine import CostModel, QuantEngine
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_diagnostics import run_turtle_diagnostic_sim

QTY = 2.5
CLOSE = 83_000.0
RATE = 0.00025
# hand computation: 2.5 * 83,000 * 0.00025 = 51.875 USDT
HAND = 51.875


def test_quant_engine_funding_payment_directions():
    eng = QuantEngine(CostModel())

    # long pays on a positive rate; short receives
    assert eng.funding_payment("long", QTY, CLOSE, RATE) == pytest.approx(-HAND, abs=1e-9)
    assert eng.funding_payment("short", QTY, CLOSE, RATE) == pytest.approx(+HAND, abs=1e-9)

    # negative rate reverses both
    assert eng.funding_payment("long", QTY, CLOSE, -RATE) == pytest.approx(+HAND, abs=1e-9)
    assert eng.funding_payment("short", QTY, CLOSE, -RATE) == pytest.approx(-HAND, abs=1e-9)

    # numeric directions agree with the string forms (both simulators use ints)
    assert eng.funding_payment(1, QTY, CLOSE, RATE) == pytest.approx(-HAND, abs=1e-9)
    assert eng.funding_payment(-1, QTY, CLOSE, RATE) == pytest.approx(+HAND, abs=1e-9)
    assert eng.funding_payment(0, QTY, CLOSE, RATE) == 0.0

    # explicit formula check
    assert eng.funding_payment("short", QTY, CLOSE, RATE) == pytest.approx(QTY * CLOSE * RATE, abs=1e-9)


def _build_breakout_frame(direction: str, n: int = 60) -> pd.DataFrame:
    """
    Synthetic 1h frame: 35 flat bars, one breakout bar that signals a long/short
    entry at the next open, ten flat bars (the position stays open, no pyramid add),
    then one extreme bar that is guaranteed to touch the 2N stop so exactly one
    single-unit cycle is completed.
    """
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    for i in range(35):
        close[i] = 100.0 + 0.5 * np.sin(i / 2.0)
    entry = 95.0 if direction == "short" else 105.0
    close[35] = entry
    for i in range(36, 46):
        close[i] = entry + 0.1 * np.sin(i)
    close[46] = entry
    for i in range(47, n):
        close[i] = entry

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    high[35] = max(open_[35], close[35]) + 0.2
    low[35] = min(open_[35], close[35]) - 0.2
    if direction == "short":
        high[46] = entry + 5.0   # far above any plausible 2N stop
        low[46] = entry - 0.2
    else:
        high[46] = entry + 0.2
        low[46] = entry - 5.0    # far below any plausible 2N stop

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=ts)


SETTLEMENT_BAR = 40
SETTLEMENT_RATE = 0.00025


def _settlement_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([SETTLEMENT_RATE], index=pd.DatetimeIndex([frame.index[SETTLEMENT_BAR]]))


def test_turtle_sim_short_receives_positive_funding():
    frame = _build_breakout_frame("short")
    features = prepare_turtle_features(frame, N=20, M=10)
    costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    metrics, trades = run_turtle_simulation(
        features, _settlement_series(frame), {"N": 20, "M": 10}, costs, 100_000.0, return_trades=True
    )

    shorts = [t for t in trades if t.direction == "short"]
    assert len(shorts) == 1, f"expected exactly one completed short unit, got {[ (t.direction, t.exit_reason) for t in trades ]}"
    unit = shorts[0]
    close_at_settlement = float(frame["close"].iloc[SETTLEMENT_BAR])
    expected = +unit.quantity * close_at_settlement * SETTLEMENT_RATE
    assert unit.funding_pnl == pytest.approx(expected, abs=1e-9)
    assert unit.funding_pnl > 0.0, "a short must RECEIVE funding on a positive rate"
    assert unit.exit_reason.startswith("2N Stop Loss")


def test_turtle_sim_long_pays_positive_funding():
    frame = _build_breakout_frame("long")
    features = prepare_turtle_features(frame, N=20, M=10)
    costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    metrics, trades = run_turtle_simulation(
        features, _settlement_series(frame), {"N": 20, "M": 10}, costs, 100_000.0, return_trades=True
    )

    longs = [t for t in trades if t.direction == "long"]
    assert len(longs) == 1, f"expected exactly one completed long unit, got {[ (t.direction, t.exit_reason) for t in trades ]}"
    unit = longs[0]
    close_at_settlement = float(frame["close"].iloc[SETTLEMENT_BAR])
    expected = -unit.quantity * close_at_settlement * SETTLEMENT_RATE
    assert unit.funding_pnl == pytest.approx(expected, abs=1e-9)
    assert unit.funding_pnl < 0.0, "a long must PAY funding on a positive rate"


def test_diagnostic_sim_short_receives_positive_funding():
    """The diagnostics simulator is the second module where the sign was wrong."""
    frame = _build_breakout_frame("short")
    features = prepare_turtle_features(frame, N=20, M=10)
    costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    diag = run_turtle_diagnostic_sim(features, _settlement_series(frame), 20, 10, costs)

    assert diag["n_trades"] >= 1
    # net = gross - fee + funding  =>  funding bps per unit is implied by the three means
    funding_bps = diag["net_bps"] - (diag["gross_bps"] - diag["fee_bps"])
    assert funding_bps > 0.0, f"short cycle must show positive funding bps, got {funding_bps}"


def test_diagnostic_sim_long_pays_positive_funding():
    frame = _build_breakout_frame("long")
    features = prepare_turtle_features(frame, N=20, M=10)
    costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    diag = run_turtle_diagnostic_sim(features, _settlement_series(frame), 20, 10, costs)

    assert diag["n_trades"] >= 1
    funding_bps = diag["net_bps"] - (diag["gross_bps"] - diag["fee_bps"])
    assert funding_bps < 0.0, f"long cycle must show negative funding bps, got {funding_bps}"

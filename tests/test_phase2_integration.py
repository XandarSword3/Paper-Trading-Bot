"""Phase 2 Task 2 regression tests â€” synthetic fixtures only.

Covers: daily-filter parity with the audited candidate_runner
implementation, 1m sub-bar exit resolution (engine + simulation level),
cost-model accounting vs PREREG conventions, and the append-only trial
registry budget enforcement. NO real market data, NO strategy runs on
historical prices.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from v4_system.data_engine import manifest as manifest_mod
from v4_system.data_engine import store
from v4_system.adapters import quant_research_data as qr


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    return tmp_path


def _candles(start, periods, tf="1h", base=100.0):
    idx = pd.date_range(start, periods=periods, freq=tf, tz="UTC")
    return pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1,
         "close": base + 0.5, "volume": 10.0,
         "close_time": idx + pd.Timedelta(hours=1),
         "quote_volume": 1000.0, "trades": 5}, index=idx)


# --------------------------------------------------------------------------- #
# 1. Daily trend filter â€” parity with the single audited implementation
# --------------------------------------------------------------------------- #

def test_daily_trend_delegates_to_candidate_runner():
    """The adapter must use candidate_runner's implementation, not a copy."""
    import inspect
    from quant_research import candidate_runner
    src = inspect.getsource(qr.daily_trend_map)
    assert "daily_trend_from_daily_close" in src
    assert "candidate_runner" in src
    # the audited function itself is unchanged in behaviour
    daily = pd.Series(np.linspace(100, 200, 60),
                      index=pd.date_range("2024-01-01", periods=60, freq="D",
                                          tz="UTC"))
    t = candidate_runner.daily_trend_from_daily_close(daily)
    assert set(t.unique()) <= {-1.0, 1.0}
    assert t.iloc[-1] == 1.0  # monotonic uptrend -> long bias


def test_daily_trend_parity_on_v4_data(store_dir):
    """Adapter output == candidate_runner function on identical daily closes."""
    from quant_research.candidate_runner import daily_trend_from_daily_close
    idx = pd.date_range("2024-01-01", periods=40 * 24 * 60, freq="1min",
                        tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 0.05,
                       "low": close - 0.05, "close": close,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1m", source="test", symbol="BTCUSDT",
               timeframe="1m")
    trend = qr.daily_trend_map("BTCUSDT", "2024-02-15T00:00:00Z",
                                strict=False)
    daily = df["close"].resample("1D").last().dropna()
    expected = daily_trend_from_daily_close(daily)
    pd.testing.assert_series_equal(trend, expected)


def test_apply_daily_mask_is_the_audited_function():
    """qr.apply_daily_mask must BE candidate_runner.apply_daily_mask."""
    from quant_research import candidate_runner
    feats = {"timestamps": pd.DatetimeIndex(["2024-06-03T10:00:00Z"]),
             "entry_high": np.array([100.0]), "entry_low": np.array([90.0])}
    trend = pd.Series([1.0], index=pd.DatetimeIndex(["2024-06-02"], tz="UTC"))
    out = qr.apply_daily_mask(feats, trend)
    # trend=+1 on previous day -> longs allowed, shorts blocked (-inf)
    assert out["entry_low"][0] == -np.inf
    assert out["entry_high"][0] == 100.0
    assert feats["entry_low"][0] == 90.0  # input untouched (copy)


def test_load_turtle_bundle_keys_and_shared_cutoff(store_dir):
    store.save(_candles("2024-01-01", 48), "BTCUSDT_perp_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    store.save(_candles("2024-01-01", 48), "BTCUSDT_perp_1m",
               source="test", symbol="BTCUSDT", timeframe="1m")
    store.save(pd.DataFrame({"fundingRate": [0.0001, 0.0002]},
                            index=pd.DatetimeIndex(
                                ["2024-01-01T00:00:00Z",
                                 "2024-01-01T08:00:00Z"])),
               "BTCUSDT_funding", source="test", symbol="BTCUSDT",
               timeframe="funding")
    b = qr.load_turtle_bundle("BTCUSDT", "1h", "2024-01-03T00:00:00Z",
                              strict=False)
    assert set(b) == {"df", "funding", "df_1m", "daily_trend"}
    assert b["df"].index[-1] <= pd.Timestamp("2024-01-03T00:00:00Z", tz="UTC")
    assert b["df_1m"] is not None and len(b["df_1m"]) > 0
    assert b["funding"].name == "funding_rate"
    assert b["daily_trend"].attrs["pit_label"] == qr.PIT_LABEL


# --------------------------------------------------------------------------- #
# 2. 1m sub-bar execution â€” engine level
# --------------------------------------------------------------------------- #

def _engine():
    from quant_research.core.engine import QuantEngine, CostModel
    return QuantEngine(CostModel())


def test_subbar_level_resolves_before_stop():
    """Long: level (99) above stop (98); first sub-bar touches level only."""
    eng = _engine()
    sub = pd.DataFrame(
        {"open": [100.0], "high": [100.5], "low": [98.5], "close": [99.0]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"]))
    price, side, reason = eng.resolve_intrabar_exit_levels(
        stop_price=98.0, level_price=99.0, direction="long",
        bar_high=101.0, bar_low=97.0, sub_bars_df=sub)
    assert price == 99.0 and side == "level"
    assert "1m resolved" in reason


def test_subbar_stop_resolves_first_when_reached_first():
    eng = _engine()
    sub = pd.DataFrame(
        {"open": [100.0], "high": [100.5], "low": [97.5], "close": [98.0]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"]))
    price, side, reason = eng.resolve_intrabar_exit_levels(
        stop_price=98.0, level_price=99.0, direction="long",
        bar_high=101.0, bar_low=97.0, sub_bars_df=sub)
    assert price == 98.0 and side == "stop"


def test_subbar_same_bar_both_levels_pessimistic_stop():
    eng = _engine()
    sub = pd.DataFrame(
        {"open": [100.0], "high": [100.5], "low": [97.0], "close": [98.0]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"]))
    price, side, _ = eng.resolve_intrabar_exit_levels(
        stop_price=98.0, level_price=99.0, direction="long",
        bar_high=101.0, bar_low=97.0, sub_bars_df=sub)
    assert price == 98.0 and side == "stop"  # pessimistic fallback


def test_subbar_missing_falls_back_pessimistic():
    eng = _engine()
    price, side, _ = eng.resolve_intrabar_exit_levels(
        stop_price=98.0, level_price=99.0, direction="long",
        bar_high=101.0, bar_low=97.0, sub_bars_df=None)
    assert price == 98.0 and side == "stop"
    price2, side2, _ = eng.resolve_intrabar_exit_levels(
        stop_price=98.0, level_price=99.0, direction="long",
        bar_high=101.0, bar_low=97.0,
        sub_bars_df=pd.DataFrame())  # empty -> same fallback
    assert price2 == 98.0 and side2 == "stop"


def test_subbar_window_is_signal_bar_only():
    """Sub-bars are sliced [bar_open, next_bar_open) â€” no lookahead."""
    from quant_research.strategies.turtle_strategy import (
        run_turtle_simulation)
    # Crafted features: long entry bar0->fill bar1; ambiguous bar at idx 20.
    n = 45
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    o = np.full(n, 100.0); h = np.full(n, 101.0)
    l = np.full(n, 99.0); c = np.full(n, 100.0)
    o[1] = 100.0
    # ambiguous bar 20: stop=98 (open[1]-2*atr), level=exit_low=99
    l[35] = 97.0; c[35] = 96.0; h[35] = 101.0
    exit_low = np.full(n, 90.0); exit_low[35] = 99.0
    feats = {
        "open": o, "high": h, "low": l, "close": c,
        "entry_high": c - 0.01, "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 110.0), "exit_low": exit_low,
        "atr": np.full(n, 1.0), "timestamps": idx,
    }
    params = {"N": 2, "M": 2, "max_units": 1, "asset": "T", "timeframe": "1h",
              "arm": "t", "config_id": "t"}
    from quant_research.core.engine import CostModel
    # 1m sub-bars inside bar 20 only: first touches level(99) not stop(98)
    m1_idx = pd.date_range(idx[35], periods=60, freq="1min", tz="UTC")
    m1 = pd.DataFrame({"open": 100.0, "high": 100.5,
                       "low": np.where(m1_idx == m1_idx[0], 98.5, 99.5),
                       "close": 99.5, "volume": 1.0}, index=m1_idx)
    metrics, trades = run_turtle_simulation(
        feats, None, params, CostModel(), bars_per_year=8760.0,
        df_1m=m1, return_trades=True, max_units=1)
    assert metrics["ambiguous_bars"] >= 1
    assert len(trades) == 1
    assert "1m resolved" in trades[0].exit_reason
    assert "Donchian" in trades[0].exit_reason  # level won, not stop


def test_subbar_absent_pessimistic_in_sim():
    from quant_research.strategies.turtle_strategy import (
        run_turtle_simulation)
    from quant_research.core.engine import CostModel
    n = 45
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    o = np.full(n, 100.0); h = np.full(n, 101.0)
    l = np.full(n, 99.0); c = np.full(n, 100.0)
    l[35] = 97.0; c[35] = 96.0; h[35] = 101.0
    exit_low = np.full(n, 90.0); exit_low[35] = 99.0
    feats = {
        "open": o, "high": h, "low": l, "close": c,
        "entry_high": c - 0.01, "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 110.0), "exit_low": exit_low,
        "atr": np.full(n, 1.0), "timestamps": idx,
    }
    params = {"N": 2, "M": 2, "max_units": 1, "asset": "T", "timeframe": "1h",
              "arm": "t", "config_id": "t"}
    metrics, trades = run_turtle_simulation(
        feats, None, params, CostModel(), bars_per_year=8760.0,
        df_1m=None, return_trades=True, max_units=1)
    assert metrics["ambiguous_bars"] >= 1
    assert len(trades) == 1
    assert trades[0].exit_reason == "2N Stop Loss"  # pessimistic fallback


# --------------------------------------------------------------------------- #
# 3. Cost model vs PREREG II.3
# --------------------------------------------------------------------------- #

def test_costmodel_round_trip_matches_prereg():
    from quant_research.core.engine import CostModel
    base = CostModel(taker_fee_pct=0.05, slippage_pct=0.02,
                     apply_funding=True, fill_on_next_open=True)
    stress = CostModel(taker_fee_pct=0.05, slippage_pct=0.04,
                       apply_funding=True, fill_on_next_open=True)
    assert base.round_trip_bps == pytest.approx(14.0)    # 5+2 bps per side x2
    assert stress.round_trip_bps == pytest.approx(18.0)  # 5+4 bps per side x2


def test_execution_costs_long_and_short():
    from quant_research.core.engine import CostModel, QuantEngine
    eng = QuantEngine(CostModel())
    # friction per side = 0.05% + 0.02% = 0.07%
    assert eng.apply_execution_costs(100.0, "long", True) == pytest.approx(100.07)
    assert eng.apply_execution_costs(100.0, "long", False) == pytest.approx(99.93)
    assert eng.apply_execution_costs(100.0, "short", True) == pytest.approx(99.93)
    assert eng.apply_execution_costs(100.0, "short", False) == pytest.approx(100.07)


def test_funding_direction_aware():
    from quant_research.core.engine import CostModel, QuantEngine
    eng = QuantEngine(CostModel())
    # positive rate: long PAYS, short RECEIVES
    assert eng.funding_payment("long", 100.0, 100.0, 0.0001) == pytest.approx(-1.0)
    assert eng.funding_payment("short", 100.0, 100.0, 0.0001) == pytest.approx(1.0)
    # negative rate reverses
    assert eng.funding_payment("long", 100.0, 100.0, -0.0001) == pytest.approx(1.0)
    assert eng.funding_payment("short", 100.0, 100.0, -0.0001) == pytest.approx(-1.0)


def test_trade_pnl_no_double_counting():
    from quant_research.core.engine import CostModel, QuantEngine
    eng = QuantEngine(CostModel())
    ee, ex, gross, cost, net, bps = eng.calculate_trade_pnl(
        entry_price=100.0, exit_price=110.0, quantity=1.0,
        direction="long", funding_pnl=0.5)
    assert gross == pytest.approx(10.0)
    assert ee == pytest.approx(100.07)
    assert ex == pytest.approx(109.923)
    assert cost == pytest.approx(gross - (ex - ee) * 1.0)
    assert net == pytest.approx((ex - ee) + 0.5)
    # friction charged exactly once per side (entry+exit), not twice
    assert cost == pytest.approx(0.147, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4. Trial registry â€” append-only + budget enforcement
# --------------------------------------------------------------------------- #

@pytest.fixture
def trials_path(tmp_path):
    return str(tmp_path / "trials_log.csv")


def test_registry_register_and_assert(trials_path):
    from v4_system import trials_registry as tr
    tid = tr.register_trial(family="family1_turtle_dt", description="t",
                            params={"N": 20, "M": 10}, n_configs=8,
                            path=trials_path)
    assert tid.startswith("family1_turtle_dt-")
    tr.assert_registered(tid, path=trials_path)  # no raise
    with pytest.raises(tr.UnregisteredTrialError):
        tr.assert_registered("nonexistent", path=trials_path)


def test_registry_append_only_result(trials_path):
    from v4_system import trials_registry as tr
    tid = tr.register_trial(family="f", description="d",
                            params={"a": 1}, n_configs=1, path=trials_path)
    tr.record_result(trial_id=tid, result="NO_EDGE",
                     net_expectancy="-3.2", path=trials_path)
    rows = tr._read_rows(trials_path)
    assert len(rows) == 2
    assert rows[0]["result"] == "REGISTERED"   # original untouched
    assert rows[1]["result"] == "NO_EDGE"
    assert rows[1]["trial_id"] == tid
    # recording for unregistered id fails
    with pytest.raises(tr.UnregisteredTrialError):
        tr.record_result(trial_id="ghost", result="X", path=trials_path)


def test_registry_budget_limits(trials_path):
    from v4_system import trials_registry as tr
    # per-family limit = 12
    for i in range(tr.MAX_TRIALS_PER_FAMILY):
        tr.register_trial(family="fam", description=f"t{i}",
                          params={"i": i}, n_configs=1, path=trials_path)
    with pytest.raises(tr.TrialBudgetError):
        tr.register_trial(family="fam", description="overflow",
                          params={"i": 99}, n_configs=1, path=trials_path)
    # other family still allowed until global 40
    tr.register_trial(family="other", description="ok",
                      params={"x": 1}, n_configs=1, path=trials_path)


def test_registry_repeat_limit(trials_path):
    from v4_system import trials_registry as tr
    params = {"N": 20, "M": 10, "tf": "1h"}
    for _ in range(tr.MAX_REPEATS):
        tr.register_trial(family="fam", description="r",
                          params=params, n_configs=1, path=trials_path)
    with pytest.raises(tr.TrialBudgetError):
        tr.register_trial(family="fam", description="r4",
                          params=params, n_configs=1, path=trials_path)


def test_registry_config_count_limit(trials_path):
    from v4_system import trials_registry as tr
    with pytest.raises(tr.TrialBudgetError):
        tr.register_trial(family="fam", description="too many",
                          params={}, n_configs=201, path=trials_path)


def test_registry_preserves_existing_header(tmp_path):
    """The committed header-only trials_log.csv must remain valid."""
    from v4_system import trials_registry as tr
    real = os.path.join(os.path.dirname(tr.__file__), "trials_log.csv")
    rows = tr._read_rows(real)
    assert rows == []  # header only, zero registrations
    status = tr.budget_status(real)
    assert status["total_registered"] == 0
    assert status["max_total"] == 40

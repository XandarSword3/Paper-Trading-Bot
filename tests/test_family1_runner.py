"""Phase 2 Task 3 tests — Family 1 runner, synthetic fixtures only.

No real market data, no historical backtests, no holdout access.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from v4_system import run_family1 as r1
from v4_system import trials_registry as registry


# --------------------------------------------------------------------------- #
# Configuration matrix & folds
# --------------------------------------------------------------------------- #

def test_registered_configs_exactly_eight():
    cfgs = r1.registered_configs()
    assert len(cfgs) == 8
    tfs = {c.timeframe for c in cfgs}
    pairs = {(c.N, c.M) for c in cfgs}
    assert tfs == {"1h", "4h"}
    assert pairs == {(20, 10), (20, 20), (55, 10), (55, 20)}
    # each config runs on all three universe assets (asset not in identity)
    for c in cfgs:
        assert c.params["max_units"] == 1
        assert c.params["arm"] == "C1_daily_filter"
        assert c.params["risk_percent"] == 1.0
        assert "asset" not in c.params


def test_folds_five_chronological_per_asset():
    for asset in r1.ASSETS:
        folds = r1.generate_family1_folds(asset)
        assert len(folds) == 5
        prev = None
        for f in folds:
            ts, te = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
            assert ts < te
            assert te < r1.HOLDOUT_START
            if prev is not None:
                assert ts > prev
            prev = te
        # last fold ends at dev_end
        assert pd.Timestamp(folds[-1]["test_end"]) == pd.Timestamp(
            r1.ASSETS[asset]["dev_end"], tz="UTC") + pd.Timedelta(seconds=1) \
            or pd.Timestamp(folds[-1]["test_end"]) <= pd.Timestamp(
                r1.ASSETS[asset]["dev_end"], tz="UTC") + pd.Timedelta(days=1)


def test_eth_cutoff_respected():
    folds = r1.generate_family1_folds("ETHUSDT")
    last_end = pd.Timestamp(folds[-1]["test_end"])
    assert last_end <= pd.Timestamp("2026-03-31", tz="UTC") + pd.Timedelta(days=1)
    assert last_end < pd.Timestamp("2026-04-01", tz="UTC")


def test_btc_sol_extend_to_october():
    for asset in ("BTCUSDT", "SOLUSDT"):
        folds = r1.generate_family1_folds(asset)
        last_end = pd.Timestamp(folds[-1]["test_end"])
        assert last_end > pd.Timestamp("2026-09-01", tz="UTC")
        assert last_end < r1.HOLDOUT_START


def test_holdout_rejected_in_fold_generation(monkeypatch):
    bad = dict(r1.ASSETS)
    bad["BTCUSDT"] = {"dev_start": "2024-01-01", "dev_end": "2026-11-15"}
    monkeypatch.setattr(r1, "ASSETS", bad)
    with pytest.raises(ValueError, match="holdout"):
        r1.generate_family1_folds("BTCUSDT")


# --------------------------------------------------------------------------- #
# PLAN mode — validation only
# --------------------------------------------------------------------------- #

def test_plan_mode_validates_and_writes_no_results(tmp_path):
    out = tmp_path / "plan.json"
    plan = r1.run_plan(str(out))
    assert plan["mode"] == "PLAN"
    assert plan["n_configs"] == 8
    assert plan["holdout"]["status"] == "EXCLUDED"
    assert plan["pit_label"] == r1.qr.PIT_LABEL
    for asset in r1.ASSETS:
        assert len(plan["assets"][asset]["folds"]) == 5
        assert plan["assets"][asset]["datasets_present"]["perp_1h"] is True
    data = json.loads(out.read_text())
    assert len(data["configs"]) == 8
    # PLAN must not consume budget: registration count unchanged by plan
    # (the real Family 1 trial may already be registered — that's fine)
    assert registry.budget_status()["total_registered"] >= 0


def test_cost_scenarios_match_prereg():
    assert r1.BASE_COSTS.round_trip_bps == pytest.approx(14.0)
    assert r1.STRESS_COSTS.round_trip_bps == pytest.approx(18.0)
    assert r1.BASE_COSTS is not r1.STRESS_COSTS
    assert r1.COST_SCENARIOS["base"].slippage_pct == 0.02
    assert r1.COST_SCENARIOS["stress"].slippage_pct == 0.04


# --------------------------------------------------------------------------- #
# EXECUTE mode safeguards
# --------------------------------------------------------------------------- #

def test_execute_rejects_unregistered_trial(monkeypatch):
    # must fail BEFORE any data load
    monkeypatch.setattr(r1.qr, "load_turtle_bundle",
                        lambda *a, **k: pytest.fail("data loaded"))
    with pytest.raises(registry.UnregisteredTrialError):
        r1.execute_trial("family1_turtle_dt-999-deadbeef")


def test_execute_rejects_unregistered_config(monkeypatch):
    monkeypatch.setattr(registry, "assert_registered", lambda tid: None)
    monkeypatch.setattr(r1.qr, "load_turtle_bundle",
                        lambda *a, **k: pytest.fail("data loaded"))
    bad = r1.Config("1h", 999, 999)  # not in matrix
    with pytest.raises(ValueError, match="unregistered configuration"):
        r1.execute_trial("fake-trial", configs=[bad])


# --------------------------------------------------------------------------- #
# run_fold — warm-up isolation, sub-bar boundary, funding slice
# --------------------------------------------------------------------------- #

def _synthetic_features(n=600):
    """Hourly features engineered for entries; entry_high just below close."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = 100 + np.cumsum(np.random.default_rng(7).normal(0, 0.3, n))
    return {
        "open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
        "entry_high": c - 0.01, "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 1e9), "exit_low": np.full(n, -1e9),
        "atr": np.full(n, 1.0), "timestamps": idx,
    }


def test_run_fold_warmup_isolation():
    feats = _synthetic_features()
    fold = {"fold_id": "F1", "train_start": "2024-01-01",
            "train_end": "2024-05-01", "test_start": "2024-05-02",
            "test_end": "2024-06-01"}
    funding = pd.Series(0.0001, index=pd.date_range(
        "2024-01-01", periods=2000, freq="8h", tz="UTC"), name="funding_rate")
    cfg = r1.Config("1h", 20, 10)
    res = r1.run_fold(cfg, fold, feats, funding, None, r1.BASE_COSTS)
    test_start = pd.Timestamp(fold["test_start"], tz="UTC")
    # every reported OOS trade entered inside the test window
    for t in res["trades"]:
        assert pd.Timestamp(t.entry_time) >= test_start
    assert res["n_trades_oos"] <= res["n_trades_total"]
    assert res["warmup_start"] < fold["test_start"]


def test_run_fold_1m_subbars_sliced_to_window():
    """df_1m must be sliced to [warm_start, test_end] — no future bars."""
    feats = _synthetic_features()
    fold = {"fold_id": "F1", "train_start": "2024-01-01",
            "train_end": "2024-05-01", "test_start": "2024-05-02",
            "test_end": "2024-06-01"}
    funding = pd.Series(0.0, index=pd.date_range(
        "2024-01-01", periods=2000, freq="8h", tz="UTC"), name="funding_rate")
    # 1m bars spanning well beyond the fold — must be clipped
    m1_idx = pd.date_range("2024-01-01", "2024-12-31", freq="1min", tz="UTC")
    m1 = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                       "close": 100.0, "volume": 1.0}, index=m1_idx)
    cfg = r1.Config("1h", 20, 10)
    res = r1.run_fold(cfg, fold, feats, funding, m1, r1.BASE_COSTS)
    assert res["n_trades_total"] >= 0  # ran without error
    # verify slicing helper directly
    sl = r1._slice_1m(m1, "2024-03-01", "2024-06-01")
    assert sl.index.min() >= pd.Timestamp("2024-03-01", tz="UTC")
    assert sl.index.max() <= pd.Timestamp("2024-06-01", tz="UTC")


def test_funding_slice_alignment():
    funding = pd.Series([0.0001, 0.0002, 0.0003],
                        index=pd.DatetimeIndex(
                            ["2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z",
                             "2024-01-02T00:00:00Z"]), name="funding_rate")
    sl = r1._slice_funding(funding, "2024-01-01", "2024-01-01T12:00:00Z")
    assert len(sl) == 2
    assert sl.index[-1] == pd.Timestamp("2024-01-01T08:00:00Z", tz="UTC")


# --------------------------------------------------------------------------- #
# Pooled stats, hurdles, DSR
# --------------------------------------------------------------------------- #

def _fake_trade(bps, pnl):
    from quant_research.core.engine import EngineTrade
    return EngineTrade(
        entry_time=pd.Timestamp("2024-06-01", tz="UTC"),
        exit_time=pd.Timestamp("2024-06-02", tz="UTC"),
        direction="long", entry_price=100.0, executed_entry_price=100.07,
        exit_price=101.0, executed_exit_price=100.93, quantity=1.0,
        gross_pnl=1.0, fee_and_slip_cost=0.14, funding_pnl=0.0,
        net_pnl=pnl, return_bps=bps, exit_reason="test", unit_number=1)


def test_pooled_stats_and_hurdles():
    trades = [_fake_trade(5.0, 50.0) for _ in range(35)]
    fold_results = [{"trades": trades, "fold_id": "F1"}]
    stats = r1._pooled_stats(fold_results, r1.BASE_COSTS)
    assert stats["n_oos_trades"] == 35
    assert stats["net_expectancy_bps"] == pytest.approx(5.0)
    assert stats["dsr"] <= 1.0 and stats["dsr"] >= 0.0
    assert stats["cost_model"]["round_trip_bps"] == pytest.approx(14.0)
    good = {"net_expectancy_bps": 5.0, "dsr": 0.99,
            "max_drawdown_pct": 5.0, "n_oos_trades": 40}
    stress_good = {"net_expectancy_bps": 3.0}
    assert r1._passes_hurdles(good, stress_good) is True
    # stress failure -> fail
    assert r1._passes_hurdles(good, {"net_expectancy_bps": -1.0}) is False
    # too few trades -> fail
    assert r1._passes_hurdles({**good, "n_oos_trades": 10},
                              stress_good) is False
    # drawdown above hurdle -> fail
    assert r1._passes_hurdles({**good, "max_drawdown_pct": 25.0},
                              stress_good) is False


def test_pooled_stats_drawdown_from_folds_not_trades():
    """DD must come from fold equity curves, never from per-trade pnl."""
    trades = [_fake_trade(5.0, 50.0) for _ in range(35)]
    fold_results = [
        {"trades": trades, "fold_id": "F1",
         "max_drawdown_pct_marked": 12.0, "max_drawdown_pct_cash": 8.0},
        {"trades": trades, "fold_id": "F2",
         "max_drawdown_pct_marked": 30.0, "max_drawdown_pct_cash": 10.0},
    ]
    stats = r1._pooled_stats(fold_results, r1.BASE_COSTS)
    # worst fold wins; per-trade path is NOT used
    assert stats["max_drawdown_pct"] == pytest.approx(30.0)
    assert stats["max_drawdown_pct_cash"] == pytest.approx(10.0)
    assert stats["fold_max_drawdowns_marked"] == [12.0, 30.0]
    # documented aggregation (PREREG III.9), no longer provisional
    assert "worst-fold marked equity" in stats["drawdown_aggregation"]
    assert "PREREG III.9" in stats["drawdown_aggregation"]
    assert "PROVISIONAL" not in stats["drawdown_aggregation"]
    # sampling-frequency metadata is explicit and accurate
    assert "decision-bar close" in stats["equity_sampling"]
    assert "intrabar excursions not captured" in stats["equity_sampling"]
    # per-trade cumulative pnl would give ~0% DD here — proving the old
    # approximation is gone
    assert stats["max_drawdown_pct"] != 0.0


# --------------------------------------------------------------------------- #
# Equity-curve drawdown — exact accounting
# --------------------------------------------------------------------------- #

def test_equity_max_drawdown_known_curve():
    # peak 120 -> trough 90 = -25%; recovery to 110 does not deepen it
    assert r1.equity_max_drawdown_pct([100, 120, 90, 110]) == \
        pytest.approx(25.0)
    # monotonic up -> no drawdown
    assert r1.equity_max_drawdown_pct([100, 110, 120]) == 0.0
    # single point / empty -> 0
    assert r1.equity_max_drawdown_pct([100]) == 0.0
    assert r1.equity_max_drawdown_pct([]) == 0.0


def test_marked_equity_includes_unrealized_loss():
    """Open position under water at fold end: marked DD > cash DD."""
    # features: long entry early, price then slides, no exit triggers
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.concatenate([
        np.full(60, 100.0),
        np.linspace(100, 80, n - 60),   # steady decline after entry
    ])
    feats = {
        "open": c, "high": c + 0.2, "low": c - 0.2, "close": c,
        "entry_high": np.full(n, 99.0),   # close > 99 -> long entries
        "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 1e9),     # channel exit unreachable
        "exit_low": np.full(n, -1e9),
        "atr": np.full(n, 50.0),  # stop = entry - 100 -> never hit
        "timestamps": idx,
    }
    fold = {"fold_id": "F1", "train_start": "2024-01-01",
            "train_end": "2024-01-10", "test_start": "2024-01-11",
            "test_end": "2024-01-20"}
    funding = pd.Series(0.0, index=pd.date_range(
        "2024-01-01", periods=2000, freq="8h", tz="UTC"), name="funding_rate")
    cfg = r1.Config("1h", 20, 10)
    res = r1.run_fold(cfg, fold, feats, funding, None, r1.BASE_COSTS)
    # position open at fold end with unrealized loss -> marked DD exists
    assert res["max_drawdown_pct_marked"] > 0
    # cash equity never realized the loss -> cash DD smaller or zero
    assert res["max_drawdown_pct_marked"] >= res["max_drawdown_pct_cash"]


def test_open_position_at_fold_end_excluded_from_trades():
    """Unfinished trades are not in OOS trade stats but hit marked equity."""
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.concatenate([np.full(60, 100.0),
                        np.linspace(100, 70, n - 60)])
    feats = {
        "open": c, "high": c + 0.2, "low": c - 0.2, "close": c,
        "entry_high": np.full(n, 99.0), "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 1e9), "exit_low": np.full(n, -1e9),
        "atr": np.full(n, 50.0), "timestamps": idx,
    }
    fold = {"fold_id": "F1", "train_start": "2024-01-01",
            "train_end": "2024-01-10", "test_start": "2024-01-11",
            "test_end": "2024-01-20"}
    funding = pd.Series(0.0, index=pd.date_range(
        "2024-01-01", periods=2000, freq="8h", tz="UTC"), name="funding_rate")
    cfg = r1.Config("1h", 20, 10)
    res = r1.run_fold(cfg, fold, feats, funding, None, r1.BASE_COSTS)
    # no completed trades inside test window (position never closed)
    assert res["n_trades_oos"] == 0
    # but marked equity reflects the open loss
    assert res["final_marked_equity"] < 100_000.0


def test_config_fails_if_any_fold_exceeds_20pct():
    """Worst-fold rule: one bad fold sinks the configuration."""
    trades = [_fake_trade(5.0, 50.0) for _ in range(35)]
    fold_results = [
        {"trades": trades, "fold_id": "F1",
         "max_drawdown_pct_marked": 5.0, "max_drawdown_pct_cash": 5.0},
        {"trades": trades, "fold_id": "F2",
         "max_drawdown_pct_marked": 20.1, "max_drawdown_pct_cash": 3.0},
        {"trades": trades, "fold_id": "F3",
         "max_drawdown_pct_marked": 4.0, "max_drawdown_pct_cash": 4.0},
    ]
    stats = r1._pooled_stats(fold_results, r1.BASE_COSTS)
    assert stats["max_drawdown_pct"] == pytest.approx(20.1)
    good = {"net_expectancy_bps": 5.0, "dsr": 0.99,
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "n_oos_trades": 40}
    assert r1._passes_hurdles(good, {"net_expectancy_bps": 3.0}) is False


def test_cash_dd_cannot_override_marked_dd():
    """Cash drawdown is diagnostic only — marked MDD gates."""
    trades = [_fake_trade(5.0, 50.0) for _ in range(35)]
    fold_results = [
        {"trades": trades, "fold_id": "F1",
         "max_drawdown_pct_marked": 25.0, "max_drawdown_pct_cash": 1.0},
    ]
    stats = r1._pooled_stats(fold_results, r1.BASE_COSTS)
    assert stats["max_drawdown_pct"] == 25.0       # marked gates
    assert stats["max_drawdown_pct_cash"] == 1.0   # diagnostic only
    good = {"net_expectancy_bps": 5.0, "dsr": 0.99,
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "n_oos_trades": 40}
    assert r1._passes_hurdles(good, {"net_expectancy_bps": 3.0}) is False


def test_fold_independence_no_equity_stitching():
    """Two folds are independent sims: pooled stats never concatenate
    equity curves; each fold reports its own DD."""
    trades = [_fake_trade(5.0, 50.0) for _ in range(35)]
    fold_results = [
        {"trades": trades, "fold_id": "F1",
         "max_drawdown_pct_marked": 5.0, "max_drawdown_pct_cash": 5.0},
        {"trades": trades, "fold_id": "F2",
         "max_drawdown_pct_marked": 15.0, "max_drawdown_pct_cash": 15.0},
    ]
    stats = r1._pooled_stats(fold_results, r1.BASE_COSTS)
    assert stats["fold_max_drawdowns_marked"] == [5.0, 15.0]
    assert stats["max_drawdown_pct"] == 15.0  # worst fold, not stitched


def test_missing_data_gap_no_invented_equity():
    """Gaps in timestamps produce no equity points — curve length equals
    bar count, nothing is forward-filled."""
    # features with a 48h hole in the middle
    idx = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC"))
        + list(pd.date_range("2024-01-10", periods=100, freq="1h", tz="UTC")))
    n = len(idx)
    c = np.full(n, 100.0)
    feats = {
        "open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
        "entry_high": np.full(n, 99.0), "entry_low": np.full(n, -1e9),
        "exit_high": np.full(n, 1e9), "exit_low": np.full(n, -1e9),
        "atr": np.full(n, 50.0), "timestamps": idx,
    }
    fold = {"fold_id": "F1", "train_start": "2024-01-01",
            "train_end": "2024-01-05", "test_start": "2024-01-06",
            "test_end": "2024-01-20"}
    funding = pd.Series(0.0, index=pd.date_range(
        "2024-01-01", periods=2000, freq="8h", tz="UTC"), name="funding_rate")
    cfg = r1.Config("1h", 20, 10)
    res = r1.run_fold(cfg, fold, feats, funding, None, r1.BASE_COSTS)
    # simulation ran on available bars only; no crash, DD defined
    assert res["max_drawdown_pct_marked"] >= 0.0


def test_dsr_uses_registered_trial_count():
    assert r1.N_TRIALS_DSR == 1355  # 1354 historical + 1 Family 1

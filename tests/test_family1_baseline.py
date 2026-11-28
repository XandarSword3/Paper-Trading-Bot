"""Tests for the Family 1 random-entry baseline (synthetic only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4_system import random_baseline_family1 as rb
from v4_system import run_family1 as r1


def _feat(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.full(n, 100.0)
    return {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
            "entry_high": np.full(n, 1e9), "entry_low": np.full(n, -1e9),
            "exit_high": np.full(n, 1e9), "exit_low": np.full(n, -1e9),
            "atr": np.full(n, 1.0), "timestamps": idx}


def test_random_features_only_in_test_window():
    feat = _feat(300)
    rng = np.random.default_rng(0)
    out = rb._random_entry_features(feat, rng, test_start_idx=200,
                                    n_signals=10, long_share=1.0)
    triggered = np.where(np.isfinite(out["entry_high"]))[0]
    assert len(triggered) == 10
    assert triggered.min() >= 200
    assert triggered.max() < 300
    # warm-up region untouched
    assert np.all(np.isinf(out["entry_high"][:200]))
    # exit channels and OHLCV untouched
    assert np.array_equal(out["close"], feat["close"])
    assert np.array_equal(out["exit_high"], feat["exit_high"])


def test_random_features_direction_mix():
    feat = _feat(300)
    rng = np.random.default_rng(1)
    out = rb._random_entry_features(feat, rng, test_start_idx=50,
                                    n_signals=100, long_share=0.0)
    longs = np.isfinite(out["entry_high"]).sum()
    shorts = np.isfinite(out["entry_low"]).sum()
    assert longs == 0 and shorts == 100


def test_random_features_deterministic_seed():
    feat = _feat(300)
    a = rb._random_entry_features(feat, np.random.default_rng(42), 100, 20, 0.5)
    b = rb._random_entry_features(feat, np.random.default_rng(42), 100, 20, 0.5)
    assert np.array_equal(a["entry_high"], b["entry_high"])
    assert np.array_equal(a["entry_low"], b["entry_low"])


def test_fold_refs_from_strategy_trades():
    from quant_research.core.engine import EngineTrade
    trades = [
        EngineTrade(pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-02", tz="UTC"), "long",
                    100, 100.07, 101, 100.93, 1, 1, 0.14, 0, 0.86, 8.6,
                    "x", 1),
        EngineTrade(pd.Timestamp("2024-06-03", tz="UTC"),
                    pd.Timestamp("2024-06-04", tz="UTC"), "short",
                    100, 99.93, 99, 100.07, 1, -1, 0.14, 0, 0.86, 8.6,
                    "x", 1),
    ]
    fold_results = [{"fold_id": "F1", "test_start": "2024-06-01",
                     "test_end": "2024-06-30", "trades": trades}]
    refs = rb._fold_refs(fold_results)
    assert refs[0]["n_signals"] == 2
    assert refs[0]["long_share"] == pytest.approx(0.5)


def test_p_beats_logic():
    """p_beats = fraction of null >= strategy; hurdle 0.05."""
    # emulate run_baselines post-processing
    null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strat = 2.5
    p = float((null >= strat).mean())
    assert p == pytest.approx(0.6)
    assert p > rb.P_BEATS_HURDLE
    strat_high = 10.0
    p2 = float((null >= strat_high).mean())
    assert p2 == 0.0
    assert p2 <= rb.P_BEATS_HURDLE


def test_hurdles_require_baseline_pass():
    good = {"net_expectancy_bps": 5.0, "dsr": 0.99,
            "max_drawdown_pct": 5.0, "n_oos_trades": 40}
    stress = {"net_expectancy_bps": 3.0}
    assert r1._passes_hurdles(good, stress, baseline_passed=True) is True
    assert r1._passes_hurdles(good, stress, baseline_passed=False) is False

"""Compatibility tests for v4_system.adapters.quant_research_data.

Two layers:
1. Synthetic store fixtures â€” schema mapping, unified as_of cutoff,
   unfinished-candle exclusion, future-event rejection, PIT labels,
   funding alignment. No real market data, no strategy runs.
2. Stored-data smoke checks â€” read-only loads of real V4 datasets,
   skipped when the (gitignored) data directory is absent.
"""
from __future__ import annotations

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


def _candles(start: str, periods: int, tf: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=tf, tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 10.0, "close_time": idx + pd.Timedelta(hours=1),
         "quote_volume": 1000.0, "trades": 5},
        index=idx)


def _funding(times, rates=None) -> pd.DataFrame:
    rates = rates if rates is not None else [0.0001] * len(times)
    return pd.DataFrame({"fundingRate": rates},
                        index=pd.DatetimeIndex(times, tz="UTC"))


# --------------------------------------------------------------------------- #
# Synthetic-fixture tests
# --------------------------------------------------------------------------- #

def test_dataset_name_mapping():
    assert qr.dataset_name("BTCUSDT", "1h") == "BTCUSDT_perp_1h"
    assert qr.dataset_name("ETHUSDT", "4h", market="spot") == "ETHUSDT_spot_4h"
    with pytest.raises(ValueError):
        qr.dataset_name("BTCUSDT", "2h")
    with pytest.raises(ValueError):
        qr.dataset_name("BTCUSDT", "1h", market="futures")


def test_load_klines_schema_and_label(store_dir):
    df = _candles("2024-01-01", 10)
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    out = qr.load_klines("BTCUSDT", "1h", "2024-01-01T12:00:00Z", strict=False)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"
    for c in qr.OHLCV_COLS:
        assert c in out.columns
    assert out.attrs["pit_label"] == qr.PIT_LABEL
    assert out.attrs["dataset"] == "BTCUSDT_perp_1h"


def test_unified_as_of_cutoff(store_dir):
    store.save(_candles("2024-01-01", 48), "BTCUSDT_perp_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    store.save(_candles("2024-01-01", 48, "1h"), "BTCUSDT_spot_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    store.save(_funding(pd.date_range("2024-01-01", periods=3, freq="8h",
                                      tz="UTC")),
               "BTCUSDT_funding", source="test", symbol="BTCUSDT",
               timeframe="funding")
    as_of = "2024-01-03T00:00:00Z"
    uni = qr.load_universe(["BTCUSDT"], "1h", as_of)
    k = uni["klines"]["BTCUSDT"]
    # 1h candle opening at 2024-01-02 23:00 closes at 2024-01-03 00:00 -> kept;
    # candle opening 2024-01-03 00:00 closes 01:00 -> excluded (unfinished).
    assert k.index[-1] == pd.Timestamp("2024-01-02 23:00", tz="UTC")
    assert len(k) == 48


def test_future_events_rejected(store_dir):
    store.save(_candles("2024-01-01", 10), "BTCUSDT_perp_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    out = qr.load_klines("BTCUSDT", "1h", "2024-01-01T05:30:00Z", strict=False)
    # candle opening 05:00 closes 06:00 > as_of -> excluded
    assert out.index[-1] == pd.Timestamp("2024-01-01 04:00", tz="UTC")


def test_strict_mode_rejects_unknown_availability(store_dir):
    store.save(_candles("2024-01-01", 5), "BTCUSDT_perp_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    strict_out = qr.load_klines("BTCUSDT", "1h", "2024-01-02T00:00:00Z",
                                strict=True)
    assert len(strict_out) == 0  # no available_at -> strict rejects all
    nonstrict = qr.load_klines("BTCUSDT", "1h", "2024-01-02T00:00:00Z",
                               strict=False)
    assert len(nonstrict) == 5


def test_funding_alignment_exact_timestamps(store_dir):
    times = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    store.save(_funding(times, [0.0001, -0.0002, 0.0003]),
               "BTCUSDT_funding", source="test", symbol="BTCUSDT",
               timeframe="funding")
    s = qr.load_funding("BTCUSDT", "2024-01-02T00:00:00Z", strict=False)
    assert s.name == "funding_rate"
    assert list(s.index) == list(times)
    assert s.iloc[1] == -0.0002
    assert s.attrs["pit_label"] == qr.PIT_LABEL


def test_no_invented_rows_across_gaps(store_dir):
    df = _candles("2024-01-01", 10)
    df = df.drop(df.index[5])  # exchange-side gap
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    out = qr.load_klines("BTCUSDT", "1h", "2024-01-02T00:00:00Z", strict=False)
    assert len(out) == 9
    assert pd.Timestamp("2024-01-01 05:00", tz="UTC") not in out.index


def test_as_of_must_be_timezone_aware(store_dir):
    store.save(_candles("2024-01-01", 3), "BTCUSDT_perp_1h",
               source="test", symbol="BTCUSDT", timeframe="1h")
    with pytest.raises(ValueError):
        qr.load_klines("BTCUSDT", "1h", "2024-01-01 00:00")  # naive
    with pytest.raises(ValueError):
        qr.load_klines("BTCUSDT", "1h", None)


def test_daily_trend_map_construction(store_dir):
    # 30 days of 1m closes trending up -> trend should be +1 late in sample
    idx = pd.date_range("2024-01-01", periods=30 * 24 * 60, freq="1min",
                        tz="UTC")
    close = 100 + np.arange(len(idx)) * 0.001
    df = pd.DataFrame({"open": close, "high": close + 0.1,
                       "low": close - 0.1, "close": close,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1m", source="test", symbol="BTCUSDT",
               timeframe="1m")
    trend = qr.daily_trend_map("BTCUSDT", "2024-01-31T00:00:00Z", ema_span=20,
                                strict=False)
    assert isinstance(trend, pd.Series)
    assert trend.index.tz is not None
    assert set(trend.unique()) <= {-1.0, 1.0}
    assert trend.iloc[-1] == 1.0  # uptrend -> long bias
    assert trend.attrs["pit_label"] == qr.PIT_LABEL


def test_compatibility_with_qr_indicators(store_dir):
    """Adapter output must feed quant_research.core.indicators directly."""
    from quant_research.core.indicators import compute_indicators
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 0.5,
                       "low": close - 0.5, "close": close,
                       "volume": rng.uniform(1, 10, len(idx))}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    out = qr.load_klines("BTCUSDT", "1h", "2024-01-20T00:00:00Z", strict=False)
    ind = compute_indicators(out)
    for col in ("bb_upper", "bb_lower", "rsi6", "atr14", "adx14", "ema50"):
        assert col in ind.columns, f"missing indicator column {col}"


def test_compatibility_with_prepare_turtle_features(store_dir):
    """Adapter output must feed turtle_strategy.prepare_turtle_features."""
    from quant_research.strategies.turtle_strategy import (
        prepare_turtle_features)
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 1,
                       "low": close - 1, "close": close,
                       "volume": rng.uniform(1, 10, len(idx))}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    out = qr.load_klines("BTCUSDT", "1h", "2024-01-10T00:00:00Z", strict=False)
    feats = prepare_turtle_features(out, N=20, M=10)
    for key in ("open", "high", "low", "close", "entry_high", "entry_low",
                "exit_high", "exit_low", "atr", "timestamps"):
        assert key in feats
    assert len(feats["timestamps"]) == len(out)


# --------------------------------------------------------------------------- #
# Stored-data smoke checks (read-only; skipped without the data dir)
# --------------------------------------------------------------------------- #

_DATA_PRESENT = os.path.exists(store.dataset_path("BTCUSDT_perp_1h"))


@pytest.mark.skipif(not _DATA_PRESENT, reason="V4 data directory absent")
def test_real_dataset_loads_and_labels():
    out = qr.load_klines("BTCUSDT", "1h", "2026-06-01T00:00:00Z", strict=False)
    assert len(out) > 1000
    assert out.index[-1] <= pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC")
    assert out.attrs["pit_label"] == qr.PIT_LABEL


@pytest.mark.skipif(not _DATA_PRESENT, reason="V4 data directory absent")
def test_real_funding_loads():
    s = qr.load_funding("BTCUSDT", "2026-06-01T00:00:00Z", strict=False)
    assert s.name == "funding_rate"
    assert s.index.tz is not None
    assert len(s) > 100


@pytest.mark.skipif(not _DATA_PRESENT, reason="V4 data directory absent")
def test_real_daily_trend_map():
    trend = qr.daily_trend_map("BTCUSDT", "2026-06-01T00:00:00Z", strict=False)
    assert len(trend) > 300  # ~3.4 years of daily bars from 2023-01
    assert set(trend.dropna().unique()) <= {-1.0, 1.0}


@pytest.mark.skipif(not _DATA_PRESENT, reason="V4 data directory absent")
def test_manifest_unchanged_by_loads():
    """Loading must not mutate datasets or the manifest."""
    import hashlib
    import json
    mpath = store.MANIFEST_PATH
    before = hashlib.sha256(open(mpath, "rb").read()).hexdigest()
    qr.load_klines("BTCUSDT", "1h", "2026-06-01T00:00:00Z", strict=False)
    qr.load_funding("BTCUSDT", "2026-06-01T00:00:00Z", strict=False)
    after = hashlib.sha256(open(mpath, "rb").read()).hexdigest()
    assert before == after

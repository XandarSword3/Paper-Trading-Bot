"""Regression tests for item 3: point-in-time data integrity.

Covers candle closure for every supported timeframe, available_at /
unified as_of cutoff semantics, atomic storage, and safe incremental
tail refresh. All fixtures are deterministic — no exchange connectivity.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from v4_system.data_engine import manifest as manifest_mod
from v4_system.data_engine import pit, refresh, store
from v4_system.data_engine.klines import _INTERVAL_MS

T0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Redirect the parquet store and manifest into a temp directory."""
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    return tmp_path


def _candle_df(opens, tf="1h", start_price=100.0):
    """Minimal kline-shaped DataFrame indexed by candle OPEN time."""
    idx = pd.DatetimeIndex(opens, tz="UTC") if not isinstance(opens, pd.DatetimeIndex) else opens
    n = len(idx)
    return pd.DataFrame({
        "open": [start_price + i for i in range(n)],
        "high": [start_price + i + 1 for i in range(n)],
        "low": [start_price + i - 1 for i in range(n)],
        "close": [start_price + i + 0.5 for i in range(n)],
        "volume": [1.0] * n,
    }, index=idx)


def _series_df(times, value=1.0):
    idx = pd.DatetimeIndex(times, tz="UTC") if not isinstance(times, pd.DatetimeIndex) else times
    return pd.DataFrame({"value": [value] * len(idx)}, index=idx)


# ---------------------------------------------------------------------------
# 1. Candle validity — every supported timeframe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tf", pit.CANDLE_TFS)
def test_unfinished_candle_excluded_all_timeframes(tf):
    delta = pit.CANDLE_DELTA[tf]
    opens = [T0, T0 + delta, T0 + 2 * delta]
    df = _candle_df(opens, tf)
    # as_of is 1ms before the last candle closes -> last candle unfinished.
    as_of = T0 + 3 * delta - pd.Timedelta(milliseconds=1)
    out = pit.drop_unfinished_candles(df, tf, as_of)
    assert list(out.index) == [T0, T0 + delta]


@pytest.mark.parametrize("tf", pit.CANDLE_TFS)
def test_closed_candles_remain_available_all_timeframes(tf):
    delta = pit.CANDLE_DELTA[tf]
    opens = [T0, T0 + delta, T0 + 2 * delta]
    df = _candle_df(opens, tf)
    as_of = T0 + 3 * delta  # last candle closes exactly at as_of
    out = pit.drop_unfinished_candles(df, tf, as_of)
    assert list(out.index) == opens


@pytest.mark.parametrize("tf", pit.CANDLE_TFS)
def test_boundary_exact_close_is_usable(tf):
    """A candle is usable AT its close time (close <= as_of)."""
    delta = pit.CANDLE_DELTA[tf]
    df = _candle_df([T0], tf)
    out = pit.drop_unfinished_candles(df, tf, T0 + delta)
    assert len(out) == 1
    out = pit.drop_unfinished_candles(df, tf, T0 + delta - pd.Timedelta(milliseconds=1))
    assert len(out) == 0


def test_1030_decision_cannot_access_1100_candle():
    """1h candle opening 10:00 closes 11:00; a 10:30 decision must not see it."""
    open_10 = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    df = _candle_df([open_10], "1h")
    at_1030 = pit.drop_unfinished_candles(df, "1h", pd.Timestamp("2026-01-01 10:30:00", tz="UTC"))
    assert len(at_1030) == 0
    at_1100 = pit.drop_unfinished_candles(df, "1h", pd.Timestamp("2026-01-01 11:00:00", tz="UTC"))
    assert len(at_1100) == 1


def test_utc_and_day_boundary_1d():
    """1d candles open at UTC midnight and close at the next midnight."""
    day = pd.Timestamp("2026-01-05", tz="UTC")
    df = _candle_df([day, day + pd.Timedelta(days=1)], "1d")
    # 23:59:59 on Jan 6 -> Jan 6 candle (closes Jan 7) unfinished, Jan 5 closed.
    out = pit.drop_unfinished_candles(df, "1d", pd.Timestamp("2026-01-06 23:59:59", tz="UTC"))
    assert list(out.index) == [day]
    out = pit.drop_unfinished_candles(df, "1d", pd.Timestamp("2026-01-07", tz="UTC"))
    assert len(out) == 2


def test_naive_as_of_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        pit.normalize_as_of("2026-01-01 10:30:00")
    with pytest.raises(ValueError):
        pit.apply_cutoff(_candle_df([T0]), "2026-01-01")


# ---------------------------------------------------------------------------
# 2. available_at / unified as_of cutoff
# ---------------------------------------------------------------------------

def test_available_at_after_as_of_rejected_both_modes():
    df = _series_df([T0, T0 + pd.Timedelta(hours=1)])
    df["available_at"] = pd.to_datetime([T0, T0 + pd.Timedelta(hours=2)], utc=True)
    as_of = T0 + pd.Timedelta(hours=1, minutes=30)
    for strict in (True, False):
        out = pit.apply_cutoff(df, as_of, strict=strict)
        assert list(out.index) == [T0], f"strict={strict}"


def test_unknown_availability_strict_vs_non_strict():
    df = _series_df([T0, T0 + pd.Timedelta(hours=1)])
    df["available_at"] = pd.NaT
    as_of = T0 + pd.Timedelta(hours=2)
    assert len(pit.apply_cutoff(df, as_of, strict=True)) == 0
    assert len(pit.apply_cutoff(df, as_of, strict=False)) == 2


def test_missing_available_at_column_strict_rejects_all():
    df = _series_df([T0])
    as_of = T0 + pd.Timedelta(hours=1)
    assert len(pit.apply_cutoff(df, as_of, strict=True)) == 0
    assert len(pit.apply_cutoff(df, as_of, strict=False)) == 1


def test_future_event_timestamps_rejected():
    df = _series_df([T0, T0 + pd.Timedelta(days=1)])
    as_of = T0 + pd.Timedelta(hours=1)
    out = pit.apply_cutoff(df, as_of, strict=False)
    assert list(out.index) == [T0]


def test_common_cutoff_across_datasets(store_dir):
    """One as_of filters candles AND funding identically."""
    candles = _candle_df([T0, T0 + pd.Timedelta(hours=1)], "1h")
    candles["available_at"] = pd.Timestamp("2026-01-01 00:00:30", tz="UTC")
    store.save(candles, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    funding = _series_df([T0, T0 + pd.Timedelta(hours=8)])
    funding["available_at"] = pd.Timestamp("2026-01-01 00:00:30", tz="UTC")
    store.save(funding, "BTCUSDT_funding", source="test", symbol="BTCUSDT",
               timeframe="funding")

    as_of = T0 + pd.Timedelta(hours=1)  # 1h candle at T0+1h closes exactly then
    out = store.load_many(["BTCUSDT_perp_1h", "BTCUSDT_funding"], as_of,
                          strict=True)
    # Candle: T0 closed (01:00 <= 01:00), T0+1h closes 02:00 -> excluded.
    assert list(out["BTCUSDT_perp_1h"].index) == [T0]
    # Funding: T0+8h is a FUTURE event relative to as_of -> excluded;
    # T0 is kept. Same cutoff as the candles.
    assert list(out["BTCUSDT_funding"].index) == [T0]


def test_historical_availability_never_fabricated(store_dir):
    """Saving without available_at records nothing; strict load rejects."""
    df = _candle_df([T0, T0 + pd.Timedelta(hours=1)], "1h")
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    raw = pd.read_parquet(store.dataset_path("BTCUSDT_perp_1h"))
    assert "available_at" not in raw.columns or raw["available_at"].isna().all()
    as_of = T0 + pd.Timedelta(hours=3)
    assert len(store.load("BTCUSDT_perp_1h", as_of=as_of, strict=True)) == 0
    # Non-strict keeps closed candles (backward compatibility).
    assert len(store.load("BTCUSDT_perp_1h", as_of=as_of, strict=False)) == 2


def test_live_record_available_at():
    rec = {"_recv_ts": "2026-01-01T10:00:00.123+00:00"}
    ts = pit.live_record_available_at(rec)
    assert ts == pd.Timestamp("2026-01-01 10:00:00.123", tz="UTC")
    assert pit.live_record_available_at({}) is None
    assert pit.live_record_available_at({"_recv_ts": "not-a-date"}) is None


# ---------------------------------------------------------------------------
# 3. Atomic storage
# ---------------------------------------------------------------------------

def test_interrupted_write_preserves_previous_dataset(store_dir, monkeypatch):
    df1 = _candle_df([T0], "1h")
    store.save(df1, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    path = store.dataset_path("BTCUSDT_perp_1h")
    before = open(path, "rb").read()

    df2 = _candle_df([T0, T0 + pd.Timedelta(hours=1)], "1h")

    real_to_parquet = pd.DataFrame.to_parquet

    def boom(self, *a, **k):
        # Simulate a crash mid-write: partial file, then failure.
        target = a[0] if a else k.get("path")
        with open(target, "wb") as f:
            f.write(b"partial-garbage")
        raise OSError("simulated disk failure")

    pd.DataFrame.to_parquet = boom
    try:
        with pytest.raises(OSError):
            store.save(df2, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
                       timeframe="1h")
    finally:
        pd.DataFrame.to_parquet = real_to_parquet
    assert open(path, "rb").read() == before, "previous dataset was corrupted"
    assert store.load("BTCUSDT_perp_1h").shape == df1.shape
    # No temporary files left behind.
    leftovers = [f for f in os.listdir(store.DATA_DIR) if ".tmp-" in f]
    assert leftovers == []


def test_failed_validation_prevents_replacement(store_dir, monkeypatch):
    df1 = _candle_df([T0], "1h")
    store.save(df1, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    path = store.dataset_path("BTCUSDT_perp_1h")
    before = open(path, "rb").read()

    df2 = _candle_df([T0, T0 + pd.Timedelta(hours=1)], "1h")
    real_read = pd.read_parquet

    def bad_read(p, *a, **k):
        out = real_read(p, *a, **k)
        if ".tmp-" in str(p):
            return out.iloc[:1]  # wrong row count -> validation fails
        return out

    pd.read_parquet = bad_read
    try:
        with pytest.raises(ValueError, match="validation failed"):
            store.save(df2, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
                       timeframe="1h")
    finally:
        pd.read_parquet = real_read
    assert open(path, "rb").read() == before
    assert store.load("BTCUSDT_perp_1h").shape == df1.shape


def test_tmp_files_never_loaded_as_datasets(store_dir):
    df = _candle_df([T0], "1h")
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    # Stray temporary file with garbage content.
    with open(store.dataset_path("BTCUSDT_perp_1h") + ".tmp-999", "wb") as f:
        f.write(b"not a parquet file")
    loaded = store.load("BTCUSDT_perp_1h")
    assert len(loaded) == 1
    with pytest.raises(FileNotFoundError):
        store.load("BTCUSDT_perp_1h.parquet.tmp-999")


def test_manifest_consistent_with_dataset(store_dir):
    df = _candle_df([T0, T0 + pd.Timedelta(hours=1)], "1h")
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    manifest = json.loads(open(store.MANIFEST_PATH, encoding="utf-8").read())
    entry = manifest["BTCUSDT_perp_1h"]
    import hashlib
    with open(store.dataset_path("BTCUSDT_perp_1h"), "rb") as f:
        assert entry["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert entry["rows"] == 2 and entry["timeframe"] == "1h"
    # Failed save must not touch the manifest.
    monkey_df = _candle_df([T0], "1h")
    real_to_parquet = pd.DataFrame.to_parquet

    def boom(self, *a, **k):
        raise OSError("disk full")
    pd.DataFrame.to_parquet = boom
    try:
        with pytest.raises(OSError):
            store.save(monkey_df, "BTCUSDT_perp_1h", source="test",
                       symbol="BTCUSDT", timeframe="1h")
    finally:
        pd.DataFrame.to_parquet = real_to_parquet
    manifest2 = json.loads(open(store.MANIFEST_PATH, encoding="utf-8").read())
    assert manifest2["BTCUSDT_perp_1h"]["rows"] == 2


# ---------------------------------------------------------------------------
# 4. Incremental tail refresh
# ---------------------------------------------------------------------------

class FakeKlinesClient:
    """Deterministic /fapi/v1/klines fake.

    Generates candles for [startTime, endTime] at the requested interval,
    INCLUDING the still-open candle (open <= endTime) so the refresh's
    unfinished-candle filtering is exercised. ``drop_after`` (ms) simulates
    an exchange that stops returning recent candles (missing tail).
    """

    def __init__(self, drop_after_ms=None):
        self.drop_after_ms = drop_after_ms
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(dict(params or {}))
        interval_ms = _INTERVAL_MS[params["interval"]]
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        first_open = -(-start_ms // interval_ms) * interval_ms  # ceil
        rows = []
        open_ms = first_open
        while open_ms <= end_ms:
            if self.drop_after_ms is not None and open_ms >= self.drop_after_ms:
                break
            close_ms = open_ms + interval_ms - 1
            rows.append([open_ms, "100", "101", "99", "100.5", "10",
                         close_ms, "1000", 5, "5", "500", "0"])
            open_ms += interval_ms
        return rows


def _seed_klines(store_dir, name="BTCUSDT_perp_1h", tf="1h", n=6):
    delta = pit.CANDLE_DELTA[tf]
    df = _candle_df([T0 + i * delta for i in range(n)], tf)
    store.save(df, name, source="seed", symbol="BTCUSDT", timeframe=tf)
    return df


def test_refresh_adds_new_closed_candles(store_dir):
    existing = _seed_klines(store_dir, n=6)
    as_of = T0 + 8 * pit.CANDLE_DELTA["1h"]  # candles 6,7 closed by now
    client = FakeKlinesClient()
    report = refresh.refresh_klines_dataset(
        "BTCUSDT_perp_1h", "BTCUSDT", "1h", as_of=as_of, client=client)
    out = store.load("BTCUSDT_perp_1h")
    assert len(out) == 8, f"expected 8 candles, got {len(out)}"
    assert report["tail_complete"] is True
    assert report["missing_bars"] == 0
    # Historical rows OUTSIDE the overlap window are unchanged. The overlap
    # window (last 3 stored candles) is legitimately re-fetched, so its
    # values may be refreshed — dedupe keeps the newest copy.
    fetch_start = existing.index[-1] - 3 * pit.CANDLE_DELTA["1h"]
    untouched = existing.index < fetch_start
    pd.testing.assert_frame_equal(
        out.loc[existing.index[untouched]][existing.columns],
        existing.loc[existing.index[untouched]])


def test_refresh_drops_unfinished_candle(store_dir):
    _seed_klines(store_dir, n=6)
    # as_of is mid-candle: candle opening at T0+6h closes T0+7h -> unfinished.
    as_of = T0 + 6 * pit.CANDLE_DELTA["1h"] + pd.Timedelta(minutes=30)
    client = FakeKlinesClient()
    refresh.refresh_klines_dataset("BTCUSDT_perp_1h", "BTCUSDT", "1h",
                                   as_of=as_of, client=client)
    out = store.load("BTCUSDT_perp_1h")
    assert out.index[-1] == T0 + 5 * pit.CANDLE_DELTA["1h"]
    assert len(out) == 6


def test_refresh_idempotent_no_duplicates(store_dir):
    _seed_klines(store_dir, n=6)
    as_of = T0 + 8 * pit.CANDLE_DELTA["1h"]
    client = FakeKlinesClient()
    refresh.refresh_klines_dataset("BTCUSDT_perp_1h", "BTCUSDT", "1h",
                                   as_of=as_of, client=client)
    first = store.load("BTCUSDT_perp_1h")
    refresh.refresh_klines_dataset("BTCUSDT_perp_1h", "BTCUSDT", "1h",
                                   as_of=as_of, client=client)
    second = store.load("BTCUSDT_perp_1h")
    assert len(second) == len(first) == 8
    assert not second.index.duplicated().any()
    assert second.index.is_monotonic_increasing


def test_refresh_overlap_dedupes(store_dir):
    """The overlap window re-fetches stored candles; dedupe keeps one copy."""
    _seed_klines(store_dir, n=6)
    as_of = T0 + 7 * pit.CANDLE_DELTA["1h"]
    client = FakeKlinesClient()
    refresh.refresh_klines_dataset("BTCUSDT_perp_1h", "BTCUSDT", "1h",
                                   overlap_bars=3, as_of=as_of, client=client)
    out = store.load("BTCUSDT_perp_1h")
    assert len(out) == 7  # candles 0..6, candle 6 newly closed
    assert not out.index.duplicated().any()


def test_refresh_detects_missing_recent_candles(store_dir):
    """Exchange stops returning the newest closed candle -> reported, not hidden."""
    _seed_klines(store_dir, n=6)
    as_of = T0 + 8 * pit.CANDLE_DELTA["1h"]
    # Exchange data ends at candle 6 (T0+6h); candles 7 missing.
    drop_after = int((T0 + 7 * pit.CANDLE_DELTA["1h"]).timestamp() * 1000)
    client = FakeKlinesClient(drop_after_ms=drop_after)
    report = refresh.refresh_klines_dataset(
        "BTCUSDT_perp_1h", "BTCUSDT", "1h", as_of=as_of, client=client)
    assert report["tail_complete"] is False
    assert report["missing_bars"] >= 1
    out = store.load("BTCUSDT_perp_1h")
    assert out.index[-1] == T0 + 6 * pit.CANDLE_DELTA["1h"]


def test_interrupted_refresh_recovers_after_restart(store_dir, monkeypatch):
    _seed_klines(store_dir, n=6)
    as_of = T0 + 8 * pit.CANDLE_DELTA["1h"]
    path = store.dataset_path("BTCUSDT_perp_1h")
    before = open(path, "rb").read()

    real_save = store.save
    monkeypatch.setattr(store, "save",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        refresh.refresh_klines_dataset("BTCUSDT_perp_1h", "BTCUSDT", "1h",
                                       as_of=as_of, client=FakeKlinesClient())
    monkeypatch.setattr(store, "save", real_save)
    # Previous valid dataset untouched.
    assert open(path, "rb").read() == before
    # "Restart": a fresh refresh recomputes from the preserved dataset.
    report = refresh.refresh_klines_dataset(
        "BTCUSDT_perp_1h", "BTCUSDT", "1h", as_of=as_of,
        client=FakeKlinesClient())
    assert report["tail_complete"] is True
    assert len(store.load("BTCUSDT_perp_1h")) == 8


def test_refresh_dataset_dispatch(store_dir):
    _seed_klines(store_dir, n=6)
    as_of = T0 + 7 * pit.CANDLE_DELTA["1h"]
    report = refresh.refresh_dataset("BTCUSDT_perp_1h", "BTCUSDT",
                                     as_of=as_of, client=FakeKlinesClient())
    assert report["dataset"] == "BTCUSDT_perp_1h"
    assert len(store.load("BTCUSDT_perp_1h")) == 7


# ---------------------------------------------------------------------------
# 5. Funding / OI tail refresh
# ---------------------------------------------------------------------------

class FakeFundingClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(dict(params or {}))
        start_ms = int(params["startTime"])
        end_ms = int(params["endTime"])
        rows = []
        t = start_ms
        while t <= end_ms:
            rows.append({"symbol": "BTCUSDT", "fundingTime": t,
                         "fundingRate": "0.0001"})
            t += 8 * 3600 * 1000
        return rows


def test_refresh_funding_tail(store_dir):
    t0 = int(T0.timestamp() * 1000)
    idx = pd.to_datetime([t0, t0 + 8 * 3600 * 1000], unit="ms", utc=True)
    df = pd.DataFrame({"fundingRate": [0.0001, 0.0002]}, index=idx)
    store.save(df, "BTCUSDT_funding", source="seed", symbol="BTCUSDT",
               timeframe="funding")
    as_of = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    report = refresh.refresh_funding_dataset("BTCUSDT_funding", "BTCUSDT",
                                             as_of=as_of,
                                             client=FakeFundingClient())
    out = store.load("BTCUSDT_funding")
    assert not out.index.duplicated().any()
    assert out.index.is_monotonic_increasing
    assert len(out) >= 2
    assert report["added_rows"] >= 0


class FakeOIClient:
    def get(self, path, params=None):
        end_ms = int(params["endTime"])
        rows = []
        t = end_ms - 5 * 3600 * 1000
        while t <= end_ms:
            rows.append({"symbol": "BTCUSDT", "timestamp": t,
                         "sumOpenInterest": "100", "sumOpenInterestValue": "5000"})
            t += 3600 * 1000
        return rows


def test_refresh_oi_tail(store_dir):
    t0 = int(T0.timestamp() * 1000)
    idx = pd.to_datetime([t0], unit="ms", utc=True)
    df = pd.DataFrame({"sumOpenInterest": [100.0],
                       "sumOpenInterestValue": [5000.0]}, index=idx)
    store.save(df, "BTCUSDT_oi_1h", source="seed", symbol="BTCUSDT",
               timeframe="1h")
    as_of = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    report = refresh.refresh_oi_dataset("BTCUSDT_oi_1h", "BTCUSDT", period="1h",
                                        as_of=as_of, client=FakeOIClient())
    out = store.load("BTCUSDT_oi_1h")
    assert not out.index.duplicated().any()
    assert out.index.is_monotonic_increasing
    assert report["rows"] == len(out)

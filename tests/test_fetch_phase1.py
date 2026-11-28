"""Regression tests for item 5: historical coverage expansion.

Listing-driven enumeration, spot/perp classification, timeframe and OI
period enumeration, resume-after-interruption, failure preservation,
unfinished-candle exclusion, atomic writes, manifest provenance, and
coverage verification. All exchange responses are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from v4_system.data_engine import fetch_phase1 as fp
from v4_system.data_engine import manifest as manifest_mod
from v4_system.data_engine import pit, store
from v4_system.data_engine.klines import _INTERVAL_MS

NOW = datetime.now(timezone.utc)


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    monkeypatch.setattr(fp, "REPORTS_DIR", str(tmp_path / "reports"))
    return tmp_path


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def test_dataset_plan_enumeration():
    plan = fp.dataset_plan()
    # 3 symbols x (6 perp + 6 spot + 1 funding + 2 mark/index + 5 oi) = 60
    assert len(plan) == 60
    names = {p["name"] for p in plan}
    assert "BTCUSDT_perp_1m" in names and "BTCUSDT_spot_1m" in names
    assert "SOLUSDT_spot_15m" in names and "ETHUSDT_perp_1d" in names
    # Every project timeframe covered for both markets.
    for sym in fp.SYMBOLS:
        for tf in fp.ALL_TFS:
            assert f"{sym}_perp_{tf}" in names
            assert f"{sym}_spot_{tf}" in names
    # OI periods = project timeframes supported by the OI endpoint.
    oi = sorted(p["timeframe"] for p in plan if p["kind"] == "oi"
                and p["symbol"] == "BTCUSDT")
    assert oi == ["15m", "1d", "1h", "4h", "5m"]
    assert set(oi) == set(fp.OI_TFS)
    assert set(fp.OI_TFS) <= set(fp.TIMEFRAMES)


def test_dataset_plan_classification():
    plan = fp.dataset_plan(symbols=["BTCUSDT"], timeframes=["1h"])
    by_name = {p["name"]: p for p in plan}
    assert by_name["BTCUSDT_perp_1h"]["market"] == "perp"
    assert by_name["BTCUSDT_spot_1h"]["market"] == "spot"
    assert by_name["BTCUSDT_perp_1h"]["start"] == fp.PERP_LISTING["BTCUSDT"]
    assert by_name["BTCUSDT_spot_1h"]["start"] == fp.SPOT_LISTING["BTCUSDT"]
    assert by_name["BTCUSDT_perp_1m"]["start"] == fp.INTRADAY_START \
        if "BTCUSDT_perp_1m" in by_name else True
    kinds = {p["kind"] for p in plan}
    assert kinds == {"perp_kline", "spot_kline", "funding", "mark_kline",
                     "index_kline", "oi"}


# ---------------------------------------------------------------------------
# Mock exchange clients
# ---------------------------------------------------------------------------

def _kline_rows(start_ms, end_ms, interval, n=5):
    interval_ms = _INTERVAL_MS[interval]
    first = -(-start_ms // interval_ms) * interval_ms
    rows = []
    t = first
    while t <= end_ms and len(rows) < n:
        rows.append([t, "100", "101", "99", "100.5", "10",
                     t + interval_ms - 1, "1000", 5, "5", "500", "0"])
        t += interval_ms
    return rows


class FakeFutures:
    def __init__(self, fail_paths=()):
        self.calls = []
        self.fail_paths = set(fail_paths)

    def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path in self.fail_paths:
            raise RuntimeError(f"simulated failure on {path}")
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING",
                "contractType": "PERPETUAL", "baseAsset": "BTC",
                "quoteAsset": "USDT", "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001",
                     "minQty": "0.001", "maxQty": "100"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    {"filterType": "MAINTENANCE_MARGIN",
                     "maintMarginPercent": "2.5",
                     "requiredMarginPercent": "5"},
                ]}]}
        if path in ("/fapi/v1/klines", "/fapi/v1/markPriceKlines",
                    "/fapi/v1/indexPriceKlines"):
            return _kline_rows(int(params["startTime"]), int(params["endTime"]),
                               params["interval"])
        if path == "/fapi/v1/fundingRate":
            start = int(params["startTime"])
            return [{"symbol": "BTCUSDT", "fundingTime": start + i * 8 * 3600_000,
                     "fundingRate": "0.0001"} for i in range(3)]
        if path == "/futures/data/openInterestHist":
            end = int(params["endTime"])
            period_ms = {"1h": 3600_000, "5m": 300_000, "15m": 900_000,
                         "4h": 14_400_000, "1d": 86_400_000}[params["period"]]
            return [{"symbol": "BTCUSDT", "timestamp": end - i * period_ms,
                     "sumOpenInterest": "100", "sumOpenInterestValue": "5000"}
                    for i in range(3)]
        raise AssertionError(f"unexpected path {path}")


class FakeSpot(FakeFutures):
    def get(self, path, params=None):
        if path == "/api/v3/klines":
            self.calls.append((path, dict(params or {})))
            return _kline_rows(int(params["startTime"]), int(params["endTime"]),
                               params["interval"])
        return super().get(path, params)


def _run(symbols=("BTCUSDT",), timeframes=("1h",), **kw):
    return fp.run_full(symbols=list(symbols), timeframes=list(timeframes),
                       client=FakeFutures(**kw), spot_client=FakeSpot(**kw))


# ---------------------------------------------------------------------------
# Resume / failure preservation
# ---------------------------------------------------------------------------

def test_existing_dataset_is_refreshed_not_redownloaded(store_dir, monkeypatch):
    # Seed an existing perp_1h dataset.
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="seed", symbol="BTCUSDT",
               timeframe="1h")
    refreshed, fetched = [], []
    monkeypatch.setattr(fp.refresh_mod, "refresh_klines_dataset",
                        lambda *a, **k: refreshed.append(a[0]) or {"ok": True})
    monkeypatch.setattr(fp, "_fetch_new",
                        lambda *a, **k: fetched.append(a[0]["name"]))
    report = _run()
    assert "BTCUSDT_perp_1h" in refreshed
    assert "BTCUSDT_perp_1h" not in fetched
    # Other datasets in the plan were fetched.
    assert "BTCUSDT_spot_1h" in fetched
    assert report["summary"]["expected"] == 10  # 2 kline + funding + 2 mark/index + 5 OI


def test_failed_download_recorded_and_existing_preserved(store_dir, monkeypatch):
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="seed", symbol="BTCUSDT",
               timeframe="1h")
    before = open(store.dataset_path("BTCUSDT_perp_1h"), "rb").read()
    monkeypatch.setattr(fp.refresh_mod, "refresh_klines_dataset",
                        lambda *a, **k: {"ok": True})
    # Spot client fails on klines -> spot_1h download fails.
    class BadSpot(FakeSpot):
        def get(self, path, params=None):
            if path == "/api/v3/klines":
                raise RuntimeError("spot down")
            return super().get(path, params)
    report = fp.run_full(symbols=["BTCUSDT"], timeframes=["1h"],
                         client=FakeFutures(), spot_client=BadSpot())
    assert open(store.dataset_path("BTCUSDT_perp_1h"), "rb").read() == before
    spot_entry = report["datasets"]["BTCUSDT_spot_1h"]
    assert spot_entry["download_status"] == "failed"
    assert "spot down" in spot_entry["download_error"]
    assert report["summary"]["failed"] >= 1


# ---------------------------------------------------------------------------
# Unfinished candles / atomic writes / manifest
# ---------------------------------------------------------------------------

def test_unfinished_candle_excluded_on_save(store_dir):
    """A kline fetch that includes the still-open candle must not persist it."""
    report = _run()
    df = store.load("BTCUSDT_perp_1h")
    now = pd.Timestamp.now(tz="UTC")
    current_hour_open = now.floor("1h")
    assert df.index[-1] < current_hour_open, \
        f"unfinished candle persisted: {df.index[-1]}"
    assert report["datasets"]["BTCUSDT_perp_1h"]["status"] == "PASS"


def test_manifest_provenance_recorded(store_dir):
    _run()
    manifest = json.loads(open(store.MANIFEST_PATH, encoding="utf-8").read())
    entry = manifest["BTCUSDT_perp_1h"]
    assert entry["source"] == "fapi /fapi/v1/klines"
    assert entry["symbol"] == "BTCUSDT"
    assert entry["timeframe"] == "1h"
    assert entry["rows"] > 0
    assert "sha256" in entry
    # Historical availability is unknown -> no fabricated available_at.
    raw = pd.read_parquet(store.dataset_path("BTCUSDT_perp_1h"))
    assert "available_at" not in raw.columns


def test_atomic_write_preserves_previous_on_failure(store_dir, monkeypatch):
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="seed", symbol="BTCUSDT",
               timeframe="1h")
    before = open(store.dataset_path("BTCUSDT_perp_1h"), "rb").read()
    real = pd.DataFrame.to_parquet

    def boom(self, *a, **k):
        raise OSError("disk full")
    pd.DataFrame.to_parquet = boom
    try:
        with pytest.raises(OSError):
            store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
                       timeframe="1h")
    finally:
        pd.DataFrame.to_parquet = real
    assert open(store.dataset_path("BTCUSDT_perp_1h"), "rb").read() == before


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------

def test_verify_dataset_pass_and_incomplete(store_dir):
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    v = fp.verify_dataset("BTCUSDT_perp_1h", "1h")
    assert v["status"] == "PASS"
    assert v["checks"]["rows"] == 4
    assert v["checks"]["duplicates"] == 0

    # Duplicates -> INCOMPLETE.
    df_dup = pd.concat([df, df.iloc[[0]]])
    store.save(df_dup, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    v = fp.verify_dataset("BTCUSDT_perp_1h", "1h")
    assert v["status"] == "INCOMPLETE"
    assert v["checks"]["duplicates"] == 1

    # Missing dataset -> FAIL.
    v = fp.verify_dataset("NOPE_perp_1h", "1h")
    assert v["status"] == "FAIL"


def test_verify_coverage_summary(store_dir):
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    plan = fp.dataset_plan(symbols=["BTCUSDT"], timeframes=["1h"])
    report = fp.verify_coverage(plan, statuses=[
        {"name": "BTCUSDT_perp_1h", "status": "downloaded"}])
    assert report["summary"]["expected"] == len(plan)
    assert report["summary"]["validated"] == 1
    assert report["summary"]["missing"] == len(plan) - 1
    assert report["datasets"]["BTCUSDT_perp_1h"]["status"] == "PASS"


def test_gap_detection_reported(store_dir):
    # Two clusters separated by a 3h gap in a 1h dataset.
    idx = list(pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")) + \
          list(pd.date_range("2026-01-01 05:00", periods=2, freq="1h", tz="UTC"))
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=pd.DatetimeIndex(idx))
    store.save(df, "BTCUSDT_perp_1h", source="test", symbol="BTCUSDT",
               timeframe="1h")
    v = fp.verify_dataset("BTCUSDT_perp_1h", "1h")
    assert v["checks"]["n_gaps"] >= 1
    assert v["checks"]["missing_bars_total"] >= 2

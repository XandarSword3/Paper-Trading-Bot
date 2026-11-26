"""Tests for research.trendlab.download_resumable failure semantics.

* HTTP 404 -> month is absent (permanent), NOT a failure; run exits 0.
* Network failure -> recorded, run exits non-zero, and combined parquets
  are NOT overwritten (a previous complete dataset is preserved).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research.trendlab import download_resumable as dr


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect all cache/output paths into tmp_path and stub the fetcher."""
    months = tmp_path / "months"
    klines_dir = tmp_path / "klines"
    funding_dir = tmp_path / "funding"
    monkeypatch.setattr(dr, "MONTHS_DIR", months)
    monkeypatch.setattr(dr, "KLINE_MONTHS", months / "klines")
    monkeypatch.setattr(dr, "FUNDING_MONTHS", months / "funding")
    monkeypatch.setattr(dr, "FAILED_PATH", months / "FAILED.json")
    monkeypatch.setattr(dr, "KLINE_DIR", klines_dir)
    monkeypatch.setattr(dr, "FUNDING_DIR", funding_dir)
    return {"months": months, "klines": klines_dir, "funding": funding_dir,
            "failed": months / "FAILED.json"}


def _failed_entries(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def test_404_is_absent_not_failure(sandbox, monkeypatch):
    monkeypatch.setattr(dr, "_fetch_status", lambda url, **kw: (None, "missing"))
    sym, mkey, err, fatal = dr._fetch_kline_month("BTCUSDT", "1h", (2020, 1))
    assert (sym, mkey, err, fatal) == ("BTCUSDT", "2020-01", None, False)

    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=4)
    assert rc == 0
    # 404s are not recorded as failures.
    assert _failed_entries(sandbox["failed"]) == {}


def test_network_failure_aborts_and_preserves_combined(sandbox, monkeypatch):
    # Pre-existing combined parquet that must survive the aborted run.
    sentinel = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01"], utc=True),
                             "open": [1.0]})
    combined = sandbox["klines"] / "BTCUSDT_1h.parquet"
    sandbox["klines"].mkdir(parents=True, exist_ok=True)
    sentinel.to_parquet(combined, index=False)

    monkeypatch.setattr(dr, "_fetch_status", lambda url, **kw: (None, "neterr"))
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=4)
    assert rc == 2

    # Combined file untouched.
    out = pd.read_parquet(combined)
    assert len(out) == 1 and out["open"].iloc[0] == 1.0

    # Failures recorded for retry.
    failed = _failed_entries(sandbox["failed"])
    assert failed and failed.get("klines") and failed.get("funding")


def test_network_failure_does_not_create_combined(sandbox, monkeypatch):
    monkeypatch.setattr(dr, "_fetch_status", lambda url, **kw: (None, "neterr"))
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=4)
    assert rc == 2
    assert not (sandbox["klines"] / "BTCUSDT_1h.parquet").exists()
    assert not (sandbox["funding"] / "BTCUSDT.parquet").exists()


def test_successful_month_is_cached_and_assembled(sandbox, monkeypatch, tmp_path):
    # Build a real monthly kline zip in memory is overkill; instead pre-seed
    # the month cache and use --assemble-only semantics via run(assemble_only).
    month_dir = sandbox["months"] / "klines" / "BTCUSDT_1h"
    month_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
        "close": [1.0, 2.0], "volume": [1.0, 1.0],
        "quote_volume": [1.0, 1.0], "trades": [1, 1],
        "taker_buy_volume": [0.5, 0.5], "taker_buy_quote_volume": [0.5, 0.5],
    })
    df.to_parquet(month_dir / "2020-01.parquet", index=False)

    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=1, assemble_only=True)
    assert rc == 0
    combined = sandbox["klines"] / "BTCUSDT_1h.parquet"
    assert combined.exists()
    out = pd.read_parquet(combined)
    assert len(out) == 2

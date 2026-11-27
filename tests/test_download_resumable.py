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


def _month_df(day: str, open_: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime([day], utc=True),
        "open": [open_], "high": [open_], "low": [open_], "close": [open_],
        "volume": [1.0], "quote_volume": [1.0], "trades": [1],
        "taker_buy_volume": [0.5], "taker_buy_quote_volume": [0.5],
    })


def _seed_kline_month(sandbox, ym: str, day: str, open_: float = 1.0) -> None:
    d = sandbox["months"] / "klines" / "BTCUSDT_1h"
    d.mkdir(parents=True, exist_ok=True)
    _month_df(day, open_).to_parquet(d / f"{ym}.parquet", index=False)


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


def test_corrupt_zip_is_fatal(sandbox, monkeypatch):
    """A payload that is not a valid zip must abort the run (fatal)."""
    monkeypatch.setattr(dr, "_fetch_status", lambda url, **kw: (b"this is not a zip", "ok"))
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=4)
    assert rc == 2
    assert not (sandbox["klines"] / "BTCUSDT_1h.parquet").exists()
    failed = _failed_entries(sandbox["failed"])
    assert failed.get("klines")  # parse failures are recorded


def test_interior_404_is_fatal_in_assemble_only(sandbox, monkeypatch):
    """A missing month BETWEEN cached months is a hole, not pre-listing absence."""
    _seed_kline_month(sandbox, "2020-01", "2020-01-01")
    _seed_kline_month(sandbox, "2020-03", "2020-03-01")
    # 2020-02 is missing -> interior gap.
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=1, assemble_only=True)
    assert rc == 2
    assert not (sandbox["klines"] / "BTCUSDT_1h.parquet").exists()


def test_assemble_only_recovers_after_network_failure(sandbox, monkeypatch):
    """After a neterr abort, --assemble-only is the recovery path — but only
    when the cached months are complete (no interior gaps)."""
    _seed_kline_month(sandbox, "2020-01", "2020-01-01")
    _seed_kline_month(sandbox, "2020-02", "2020-02-01")
    monkeypatch.setattr(dr, "_fetch_status", lambda url, **kw: (None, "neterr"))
    assert dr.run(symbols=["BTCUSDT"], start="2020-01", workers=4) == 2
    assert not (sandbox["klines"] / "BTCUSDT_1h.parquet").exists()

    # Recovery: assemble-only with the two cached months (rest are tail).
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=1, assemble_only=True)
    assert rc == 0
    out = pd.read_parquet(sandbox["klines"] / "BTCUSDT_1h.parquet")
    assert len(out) == 2


def test_combined_file_never_loses_timestamps(sandbox, monkeypatch):
    """A combined file with timestamps the new data lacks must be preserved."""
    # Old combined has Jan + Feb timestamps.
    old = pd.concat([_month_df("2020-01-01"), _month_df("2020-02-01")], ignore_index=True)
    sandbox["klines"].mkdir(parents=True, exist_ok=True)
    old.to_parquet(sandbox["klines"] / "BTCUSDT_1h.parquet", index=False)

    # Month cache only has Jan -> new frame would miss Feb's timestamp.
    _seed_kline_month(sandbox, "2020-01", "2020-01-01")
    rc = dr.run(symbols=["BTCUSDT"], start="2020-01", workers=1, assemble_only=True)
    assert rc == 2
    out = pd.read_parquet(sandbox["klines"] / "BTCUSDT_1h.parquet")
    assert len(out) == 2  # old file untouched

"""Regression tests for Integrity Correction 3 — item-8 dataset validator.

Covers the report-generation bug (fields read from ``v["checks"]``, not
top-level ``v``), the PASS/INCOMPLETE/FAIL classification rules, gap
detection/classification, OI coverage limits, available_at status, and
metadata snapshot validity.
"""
import pandas as pd
import pytest

from v4_system.data_engine import final_validation as fv
from v4_system.data_engine import store


def _entry(name="BTCUSDT_perp_1h", market="perp", timeframe="1h"):
    return {"name": name, "symbol": "BTCUSDT", "market": market,
            "timeframe": timeframe, "kind": "candle", "source": "test"}


def _verify(rows=100, dupes=0, unfinished=0, n_gaps=0, missing=0,
            avail_frac=0.0, status="PASS", error=None):
    """Synthetic verify_dataset output — metrics nested under 'checks'."""
    checks = {
        "rows": rows,
        "start": "2024-01-01 00:00:00+00:00",
        "end": "2024-01-05 03:00:00+00:00",
        "duplicates": dupes,
        "n_gaps": n_gaps,
        "missing_bars_total": missing,
        "unfinished_candles": unfinished,
        "available_at_known_frac": avail_frac,
    }
    out = {"name": "x", "status": status, "checks": checks}
    if error:
        out["error"] = error
    return out


# --------------------------------------------------------------------------- #
# Nested-field extraction (the original bug: v["rows"] etc. were always None)
# --------------------------------------------------------------------------- #

def test_market_row_reads_nested_checks_not_top_level():
    v = _verify(rows=1234, dupes=0, unfinished=0, avail_frac=0.0)
    # Top-level must NOT carry the metrics (that was the bug)
    assert "rows" not in v or v.get("rows") is None
    row = fv._market_row(_entry(), v, {"sha256": "abc", "fetched_at": "t",
                                       "source": "src"})
    assert row["rows"] == 1234
    assert row["first_ts"] is not None and row["last_ts"] is not None
    assert row["dupes"] == 0 and row["unfinished"] == 0
    assert row["checksum_sha256"] == "abc"
    assert row["manifest_fetched_at"] == "t"
    assert row["freshly_refreshed"] is False


def test_market_row_date_range_and_counts():
    v = _verify(rows=500, n_gaps=2, missing=7, avail_frac=1.0)
    row = fv._market_row(_entry(), v, {})
    assert row["rows"] == 500
    assert row["gaps"] == 2 and row["missing_bars"] == 7
    assert row["first_ts"].startswith("2024-01-01")
    assert row["last_ts"].startswith("2024-01-05")


# --------------------------------------------------------------------------- #
# Classification rules
# --------------------------------------------------------------------------- #

def test_fail_on_duplicates():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(dupes=3))
    assert status == "FAIL" and structural == "FAIL"


def test_fail_on_unfinished_candles():
    status, structural, _ = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(unfinished=2))
    assert status == "FAIL" and structural == "FAIL"


def test_fail_on_load_error():
    status, structural, _ = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(),
        {"name": "x", "status": "FAIL", "error": "empty dataset",
         "checks": {}})
    assert status == "FAIL" and structural == "FAIL"


def test_incomplete_on_unexplained_gaps():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(n_gaps=1, missing=5))
    assert status == "INCOMPLETE" and structural == "PASS"
    assert any("UNEXPLAINED" in x for x in lim)


def test_incomplete_on_spot_gaps_documented_exchange_side():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_spot_1h", _entry(market="spot"),
        _verify(n_gaps=1, missing=5))
    assert status == "INCOMPLETE" and structural == "PASS"
    assert any("documented exchange-side" in x for x in lim)
    assert not any("UNEXPLAINED" in x for x in lim)


def test_incomplete_on_oi_coverage_cap():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_perp_oi_1h", _entry(), _verify(avail_frac=1.0))
    assert status == "INCOMPLETE" and structural == "PASS"
    assert any("30 days" in x for x in lim)


def test_incomplete_on_unknown_available_at():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(avail_frac=0.0))
    assert status == "INCOMPLETE" and structural == "PASS"
    assert any("available_at unknown" in x for x in lim)


def test_pass_requires_clean_structure_and_coverage():
    status, structural, lim = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(avail_frac=1.0))
    assert status == "PASS" and structural == "PASS" and lim == []


def test_partial_available_at_still_incomplete():
    status, _, lim = fv._classify_market(
        "BTCUSDT_perp_1h", _entry(), _verify(avail_frac=0.5))
    assert status == "INCOMPLETE"
    assert any("50%" in x for x in lim)


# --------------------------------------------------------------------------- #
# Gap detection (store.detect_gaps) — known vs unexplained is a
# classification concern; detection itself must be exact.
# --------------------------------------------------------------------------- #

def test_detect_gaps_finds_missing_bars():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    # remove 3 bars to create one gap of 3 missing bars
    keep = [i for i in idx if i not in idx[4:7]]
    df = pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(keep))
    gaps = store.detect_gaps(df, "1h")
    assert len(gaps) == 1
    start, end, missing = gaps[0]
    assert missing == 3


def test_detect_gaps_ignores_small_jitter():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": 1.0}, index=idx)
    assert store.detect_gaps(df, "1h") == []


# --------------------------------------------------------------------------- #
# Metadata snapshot classification (via _classify helpers is market-only;
# metadata rows are built inline — test the status logic indirectly by
# checking the regenerated report artifact instead).
# --------------------------------------------------------------------------- #

def test_regenerated_report_has_no_null_core_fields():
    """The corrected report must populate rows/first/last for every
    market dataset — the original bug left them all null."""
    import json
    from pathlib import Path
    p = Path(fv.__file__).resolve().parents[1] / "reports" / \
        "item8_final_validation.json"
    if not p.exists():
        pytest.skip("report not yet regenerated")
    data = json.loads(p.read_text())
    market = [d for d in data["datasets"] if d["kind"] == "market_data"]
    assert market, "no market datasets in report"
    for d in market:
        assert d["rows"] is not None, d["dataset"]
        assert d["first_ts"] is not None, d["dataset"]
        assert d["last_ts"] is not None, d["dataset"]
        assert d["dupes"] is not None, d["dataset"]
        assert "structural_status" in d
        assert "coverage_limitations" in d
    # Aggregate counts must be internally consistent
    counts = {"PASS": 0, "INCOMPLETE": 0, "FAIL": 0}
    for d in data["datasets"]:
        counts[d["status"]] += 1
    assert counts == data["counts"]
    assert data["expected_total"] == len(data["datasets"])


def test_regenerated_report_classification_honest():
    """The corrected report must not claim 64/64 PASS: unknown
    available_at alone forces INCOMPLETE on historical market data."""
    import json
    from pathlib import Path
    p = Path(fv.__file__).resolve().parents[1] / "reports" / \
        "item8_final_validation.json"
    if not p.exists():
        pytest.skip("report not yet regenerated")
    data = json.loads(p.read_text())
    assert data["counts"]["PASS"] < data["expected_total"]
    # Every market dataset with unknown available_at is INCOMPLETE
    for d in data["datasets"]:
        if d["kind"] == "market_data" and \
                d.get("available_at_known_frac", 0.0) < 1.0:
            assert d["status"] == "INCOMPLETE", d["dataset"]

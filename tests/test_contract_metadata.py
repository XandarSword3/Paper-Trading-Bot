"""Regression tests for item 6: contract metadata, funding intervals,
dated margin tiers. All exchange responses are mocked.
"""
from __future__ import annotations

import pandas as pd
import pytest

from v4_system.data_engine import contract_specs as cs
from v4_system.data_engine import funding_intervals as fi
from v4_system.data_engine import margin_tiers as mt
from v4_system.data_engine import manifest as manifest_mod
from v4_system.data_engine import store


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    return tmp_path


# ---------------------------------------------------------------------------
# Contract metadata
# ---------------------------------------------------------------------------

def _perp_symbol(sym="BTCUSDT"):
    return {
        "symbol": sym, "status": "TRADING", "contractType": "PERPETUAL",
        "baseAsset": "BTC", "quoteAsset": "USDT",
        "pricePrecision": 2, "quantityPrecision": 3,
        "baseAssetPrecision": 8, "quotePrecision": 8,
        "maintMarginPercent": "2.5000", "requiredMarginPercent": "5.0000",
        "marginAsset": "USDT", "liquidationFee": "0.015000",
        "onboardDate": 1569398400000, "deliveryDate": 4133404800000,
        "underlyingType": "COIN", "underlyingSubType": ["PoW"],
        "fundingIntervalHours": None,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001",
             "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "notional": "50"},
        ],
    }


class FakeFutures:
    def __init__(self, symbols):
        self.symbols = symbols
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(path)
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": self.symbols}
        raise AssertionError(path)


class FakeSpot:
    def get(self, path, params=None):
        if path == "/api/v3/exchangeInfo":
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING",
                "baseAsset": "BTC", "quoteAsset": "USDT",
                "pricePrecision": 2, "quantityPrecision": 6,
                "baseAssetPrecision": 8, "quotePrecision": 8,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.000001",
                     "minQty": "0.000001", "maxQty": "9000"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                ]}]}
        raise AssertionError(path)


def test_perp_specs_top_level_margin_fields_populated():
    df = cs.fetch_exchange_specs(["BTCUSDT"], client=FakeFutures([_perp_symbol()]))
    row = df.loc["BTCUSDT"]
    assert row["market"] == "perp"
    assert row["maint_margin_percent"] == "2.5000"
    assert row["required_margin_percent"] == "5.0000"
    assert row["margin_asset"] == "USDT"
    assert row["liquidation_fee"] == "0.015000"
    assert row["onboard_date"] == 1569398400000
    assert row["tick_size"] == "0.10" and row["step_size"] == "0.001"
    assert row["min_notional"] == "50"
    # fundingIntervalHours is NOT in exchangeInfo -> stays null.
    assert pd.isna(row["funding_interval_hours"])


def test_perp_specs_missing_fields_stay_null():
    sym = _perp_symbol()
    sym["maintMarginPercent"] = None
    sym["requiredMarginPercent"] = None
    df = cs.fetch_exchange_specs(["BTCUSDT"], client=FakeFutures([sym]))
    row = df.loc["BTCUSDT"]
    assert pd.isna(row["maint_margin_percent"])
    assert pd.isna(row["required_margin_percent"])


def test_spot_specs_classification_and_notional():
    df = cs.fetch_spot_specs(["BTCUSDT"], client=FakeSpot())
    row = df.loc["BTCUSDT"]
    assert row["market"] == "spot"
    assert row["min_notional"] == "5.00000000"
    assert pd.isna(row["maint_margin_percent"])
    assert pd.isna(row["funding_interval_hours"])


def test_multiple_instruments():
    syms = [_perp_symbol("BTCUSDT"), _perp_symbol("ETHUSDT")]
    df = cs.fetch_exchange_specs(None, client=FakeFutures(syms))
    assert set(df.index) == {"BTCUSDT", "ETHUSDT"}
    assert (df["market"] == "perp").all()


# ---------------------------------------------------------------------------
# Funding intervals
# ---------------------------------------------------------------------------

def _funding_df(hours_list):
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    idx = [t0 + pd.Timedelta(hours=h) for h in hours_list]
    return pd.DataFrame({"fundingRate": 0.0001}, index=pd.DatetimeIndex(idx))


def test_infer_regular_8h():
    df = _funding_df([i * 8 for i in range(20)])
    obs = fi.infer_intervals(df)
    assert obs["dominant_interval_hours"] == 8.0
    assert obs["changes_detected"] is False
    assert obs["confidence"] in ("high", "medium")


def test_infer_4h_interval():
    df = _funding_df([i * 4 for i in range(20)])
    obs = fi.infer_intervals(df)
    assert obs["dominant_interval_hours"] == 4.0


def test_frequency_change_detected():
    # 8h for 10 settlements, then 4h for 10 settlements.
    hours = [i * 8 for i in range(10)]
    hours += [hours[-1] + 4 * (i + 1) for i in range(10)]
    df = _funding_df(hours)
    obs = fi.infer_intervals(df, min_run=3)
    assert obs["changes_detected"] is True
    intervals = {s["interval_hours"] for s in obs["observed_intervals"]}
    assert 8.0 in intervals and 4.0 in intervals


def test_isolated_gap_not_a_frequency_change():
    # One missing settlement (16h gap) in an otherwise regular 8h series.
    hours = [i * 8 for i in range(10)]
    hours[5] = hours[4] + 16  # skip one settlement
    hours = hours[:5] + [hours[4] + 8 * (i + 1) for i in range(5)]
    df = _funding_df(hours)
    obs = fi.infer_intervals(df, min_run=3)
    assert obs["changes_detected"] is False
    assert obs["dominant_interval_hours"] == 8.0


def test_millisecond_jitter_not_a_frequency_change():
    """Funding timestamps carry ms jitter; it must not look like a change."""
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    idx = [t0 + pd.Timedelta(hours=i * 8, milliseconds=i % 3)
           for i in range(20)]
    df = pd.DataFrame({"fundingRate": 0.0001}, index=pd.DatetimeIndex(idx))
    obs = fi.infer_intervals(df)
    assert obs["changes_detected"] is False
    assert obs["dominant_interval_hours"] == 8.0


def test_unknown_interval_for_too_few_observations():
    df = _funding_df([0, 8])
    obs = fi.infer_intervals(df)
    assert obs["n_settlements"] == 2
    df_empty = pd.DataFrame()
    obs2 = fi.infer_intervals(df_empty)
    assert obs2["confidence"] == "unknown"


def test_funding_info_mapping():
    class FakeInfo:
        def get(self, path, params=None):
            assert path == "/fapi/v1/fundingInfo"
            return [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.003",
                     "adjustedFundingRateFloor": "-0.003",
                     "fundingIntervalHours": 8, "disclaimer": True,
                     "updateTime": None}]
    info = fi.fetch_funding_info(["BTCUSDT"], client=FakeInfo())
    assert info.loc["BTCUSDT", "fundingIntervalHours"] == 8
    assert info.loc["BTCUSDT", "adjustedFundingRateCap"] == 0.003


def test_current_config_not_presented_as_historical():
    """build_funding_interval_table separates current vs observed."""
    funding = {"BTCUSDT": _funding_df([i * 8 for i in range(12)])}
    info = pd.DataFrame({"fundingIntervalHours": [8],
                         "adjustedFundingRateCap": [0.003],
                         "adjustedFundingRateFloor": [-0.003]},
                        index=pd.Index(["BTCUSDT"], name="symbol"))
    table = fi.build_funding_interval_table(funding, info,
                                            retrieved_at="2026-11-28T00:00:00+00:00")
    row = table.loc["BTCUSDT"]
    assert row["current_interval_hours"] == 8
    assert row["current_source"] == "fapi /fapi/v1/fundingInfo"
    assert row["observed_dominant_interval_hours"] == 8.0
    assert not row["changes_detected"]
    assert row["retrieved_at"] == "2026-11-28T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Margin tiers
# ---------------------------------------------------------------------------

def _brackets(sym="BTCUSDT"):
    return {sym: [
        {"bracket": 1, "initialLeverage": 125, "notionalCap": 50000,
         "notionalFloor": 0, "maintMarginRatio": "0.004", "cum": "0.000"},
        {"bracket": 2, "initialLeverage": 100, "notionalCap": 250000,
         "notionalFloor": 50000, "maintMarginRatio": "0.005", "cum": "0.000"},
        {"bracket": 3, "initialLeverage": 50, "notionalCap": 1000000,
         "notionalFloor": 250000, "maintMarginRatio": "0.01", "cum": "0.000"},
    ]}


def test_brackets_to_frame_mapping():
    snap = mt.brackets_to_frame(_brackets(),
                                snapshot_time="2026-11-28T10:00:00+00:00")
    assert len(snap) == 3
    row = snap[snap["bracket"] == 1].iloc[0]
    assert row["symbol"] == "BTCUSDT"
    assert row["initial_leverage"] == 125
    assert row["notional_floor"] == 0 and row["notional_cap"] == 50000
    assert row["maint_margin_ratio"] == 0.004
    assert row["effective_date"] is None or pd.isna(row["effective_date"])
    assert row["snapshot_time"] == pd.Timestamp("2026-11-28T10:00:00+00:00")


def test_validate_brackets_valid():
    snap = mt.brackets_to_frame(_brackets())
    assert mt.validate_brackets(snap) == []


def test_validate_brackets_overlap_detected():
    bad = _brackets()
    bad["BTCUSDT"][1]["notionalFloor"] = 40000  # overlaps bracket 1 cap
    snap = mt.brackets_to_frame(bad)
    problems = mt.validate_brackets(snap)
    assert any("overlap" in p for p in problems)


def test_validate_brackets_floor_ge_cap():
    bad = _brackets()
    bad["BTCUSDT"][0]["notionalCap"] = 0  # floor == cap
    snap = mt.brackets_to_frame(bad)
    problems = mt.validate_brackets(snap)
    assert any("floor >= cap" in p for p in problems)


def test_save_snapshot_dedupe_and_append(store_dir):
    snap1 = mt.brackets_to_frame(_brackets(),
                                 snapshot_time="2026-11-28T10:00:00+00:00")
    mt.save_snapshot(snap1)
    # Same snapshot again -> no duplicates.
    mt.save_snapshot(snap1)
    df = store.load(mt.DATASET)
    assert len(df) == 3
    # New snapshot at a later time -> appended.
    snap2 = mt.brackets_to_frame(_brackets(),
                                 snapshot_time="2026-11-29T10:00:00+00:00")
    mt.save_snapshot(snap2)
    df = store.load(mt.DATASET)
    assert len(df) == 6
    assert df["snapshot_time"].nunique() == 2


def test_margin_tiers_as_of_pit(store_dir):
    snap1 = mt.brackets_to_frame(_brackets(),
                                 snapshot_time="2026-11-28T10:00:00+00:00")
    snap2 = mt.brackets_to_frame(_brackets(),
                                 snapshot_time="2026-11-29T10:00:00+00:00")
    mt.save_snapshot(snap1)
    mt.save_snapshot(snap2)
    # Decision at 2026-11-28 12:00 -> only snapshot 1 visible.
    out = mt.margin_tiers_as_of("BTCUSDT", "2026-11-28T12:00:00+00:00")
    assert out["snapshot_time"].nunique() == 1
    assert out["snapshot_time"].iloc[0] == pd.Timestamp("2026-11-28T10:00:00+00:00")
    # Decision before any snapshot -> empty.
    out2 = mt.margin_tiers_as_of("BTCUSDT", "2026-11-27T00:00:00+00:00")
    assert len(out2) == 0
    # Decision after both -> latest snapshot.
    out3 = mt.margin_tiers_as_of("BTCUSDT", "2026-11-30T00:00:00+00:00")
    assert out3["snapshot_time"].iloc[0] == pd.Timestamp("2026-11-29T10:00:00+00:00")


def test_snapshot_failure_preserves_previous(store_dir, monkeypatch):
    snap1 = mt.brackets_to_frame(_brackets(),
                                 snapshot_time="2026-11-28T10:00:00+00:00")
    mt.save_snapshot(snap1)
    before = open(store.dataset_path(mt.DATASET), "rb").read()
    real = pd.DataFrame.to_parquet

    def boom(self, *a, **k):
        raise OSError("disk full")
    pd.DataFrame.to_parquet = boom
    try:
        snap2 = mt.brackets_to_frame(_brackets(),
                                     snapshot_time="2026-11-29T10:00:00+00:00")
        with pytest.raises(OSError):
            mt.save_snapshot(snap2)
    finally:
        pd.DataFrame.to_parquet = real
    assert open(store.dataset_path(mt.DATASET), "rb").read() == before


def test_manifest_provenance_for_margin_tiers(store_dir):
    import json
    snap = mt.brackets_to_frame(_brackets())
    mt.save_snapshot(snap)
    manifest = json.loads(open(store.MANIFEST_PATH, encoding="utf-8").read())
    entry = manifest[mt.DATASET]
    assert "leverageBracket" in entry["source"]
    assert entry["rows"] == 3

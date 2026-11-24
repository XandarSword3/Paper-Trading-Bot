"""Tests for the remote RGMT runner's market-data adapters."""

from __future__ import annotations

import json

import pytest

from scripts.run_rgmt_remote import MIN_TURNOVER_24H, _okx_turnover_usd


def test_okx_turnover_is_converted_from_base_currency_to_usd() -> None:
    # OKX reports BTC volume in BTC, not USDT. Raw 125,947 must not be compared
    # directly to a USD threshold, or BTC gets filtered out of the universe.
    item = {"volCcy24h": "125947.1138", "last": "61563"}
    assert _okx_turnover_usd(item) == pytest.approx(125947.1138 * 61563)
    assert _okx_turnover_usd(item) >= MIN_TURNOVER_24H


def test_okx_turnover_handles_missing_or_invalid_fields() -> None:
    assert _okx_turnover_usd({}) == 0.0
    assert _okx_turnover_usd({"volCcy24h": "n/a", "last": "1"}) == 0.0
    assert _okx_turnover_usd({"last": "61563"}) == 0.0


def test_load_state_normalises_legacy_nulls(tmp_path, monkeypatch) -> None:
    from scripts import run_rgmt_remote

    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "cash": None,
                "equity": None,
                "peak_equity": None,
                "realized_pnl": None,
                "total_fees_paid": None,
                "total_funding_cost": None,
                "runs": "4",
                "positions": {"BTCUSDT": "1.5", "BROKEN": None},
                "entry_prices": {"BTCUSDT": "100"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_rgmt_remote, "STATE_PATH", path)

    state = run_rgmt_remote.load_state(10_000.0)

    assert state["cash"] == 10_000.0
    assert state["equity"] == 10_000.0
    assert state["peak_equity"] == 10_000.0
    assert state["realized_pnl"] == 0.0
    assert state["total_fees_paid"] == 0.0
    assert state["total_funding_cost"] == 0.0
    assert state["runs"] == 4
    assert state["positions"] == {"BTCUSDT": 1.5}
    assert state["entry_prices"] == {"BTCUSDT": 100.0}

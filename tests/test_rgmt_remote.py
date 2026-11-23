"""Tests for the remote RGMT runner's market-data adapters."""

from __future__ import annotations

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

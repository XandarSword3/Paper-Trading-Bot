"""Tests for the RGMT mainnet DSR gate in scripts/run_rgmt_live.py.

The gate is fail-closed: mainnet new exposure is refused unless the committed
gate file parses and shows DSR >= 0.95. Flattening / reduce-only orders must
always survive.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_rgmt_live import apply_dsr_gate, dsr_gate_allows_mainnet


def _write_gate(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_gate_file_blocks(tmp_path):
    assert dsr_gate_allows_mainnet(tmp_path / "nope.json") is False


def test_unreadable_gate_file_blocks(tmp_path):
    bad = tmp_path / "gate.json"
    bad.write_text("{not json", encoding="utf-8")
    assert dsr_gate_allows_mainnet(bad) is False


def test_gate_file_without_dsr_blocks(tmp_path):
    p = tmp_path / "gate.json"
    _write_gate(p, {"metric": "dsr"})
    assert dsr_gate_allows_mainnet(p) is False


def test_dsr_below_threshold_blocks(tmp_path):
    p = tmp_path / "gate.json"
    _write_gate(p, {"dsr": 0.659, "threshold": 0.95})
    assert dsr_gate_allows_mainnet(p) is False


def test_dsr_at_threshold_allows(tmp_path):
    p = tmp_path / "gate.json"
    _write_gate(p, {"dsr": 0.95})
    assert dsr_gate_allows_mainnet(p) is True


def test_dsr_above_threshold_allows(tmp_path):
    p = tmp_path / "gate.json"
    _write_gate(p, {"dsr": 0.97})
    assert dsr_gate_allows_mainnet(p) is True


def test_committed_gate_file_is_below_threshold():
    """The committed gate must currently BLOCK mainnet (DSR 0.659 < 0.95)."""
    from scripts.run_rgmt_live import DSR_GATE_PATH
    assert dsr_gate_allows_mainnet(DSR_GATE_PATH) is False


def test_apply_dsr_gate_keeps_only_reduce_only():
    orders = [
        {"symbol": "BTCUSDT", "side": "BUY", "quantity": 1.0, "reduce_only": False},
        {"symbol": "ETHUSDT", "side": "SELL", "quantity": 2.0, "reduce_only": True},
        {"symbol": "SOLUSDT", "side": "SELL", "quantity": 3.0, "reduce_only": True},
    ]
    kept = apply_dsr_gate(orders, allowed=False)
    assert kept == orders[1:]
    assert all(o["reduce_only"] for o in kept)


def test_apply_dsr_gate_passthrough_when_allowed():
    orders = [{"symbol": "BTCUSDT", "reduce_only": False}]
    assert apply_dsr_gate(orders, allowed=True) == orders

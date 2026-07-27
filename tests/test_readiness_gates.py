"""
Tests for build_readiness_gates.py (Phase 5) against synthetic trade logs —
same pattern as the rest of the remediation plan's phases (e.g. Kraken
fetcher tested against a mocked response, cross_market_validation tested with
synthetic trending/flat/short-history legs), since this sandbox has no
Binance/Kraken egress to pull real data with.

Run with: pytest test_readiness_gates.py -v
"""
from datetime import datetime, timedelta, timezone

from research.strategies.config import GateThresholds
from research.validation.build_readiness_gates import compute_paper_metrics, evaluate_gate, build_gate_file


def _make_exit(t_offset_days, pnl, equity):
    return {
        "type": "EXIT",
        "reason": "test",
        "time": (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=t_offset_days)).isoformat(),
        "price": 100.0,
        "size": 1.0,
        "pnl": pnl,
        "equity": equity,
    }


def test_zero_trades_is_not_ready():
    metrics = compute_paper_metrics([], initial_capital=1000.0)
    assert metrics["trade_count"] == 0
    ready, reasons = evaluate_gate(metrics, GateThresholds(), None)
    assert ready is False
    assert any("completed paper trades" in r for r in reasons)


def test_winning_track_record_passes():
    # 40 trades over ~1 year, small consistent wins, no drawdown of note.
    trades = []
    equity = 1000.0
    for i in range(40):
        pnl = 5.0  # steady small wins
        equity += pnl
        trades.append(_make_exit(i * 9, pnl, equity))
    metrics = compute_paper_metrics(trades, initial_capital=1000.0)
    ready, reasons = evaluate_gate(metrics, GateThresholds(), None)
    assert metrics["trade_count"] == 40
    assert metrics["total_return_pct"] > 0
    assert ready is True, reasons


def test_losing_track_record_fails_on_return_and_drawdown():
    # Mirrors the real V4 situation: enough trades, but net negative and a
    # real drawdown — must fail even though trade_count clears the bar.
    trades = []
    equity = 1000.0
    for i in range(40):
        pnl = -8.0
        equity += pnl
        trades.append(_make_exit(i * 9, pnl, equity))
    metrics = compute_paper_metrics(trades, initial_capital=1000.0)
    ready, reasons = evaluate_gate(metrics, GateThresholds(), None)
    assert metrics["total_return_pct"] < 0
    assert ready is False
    assert any("net negative" in r or "below the" in r for r in reasons)
    assert any("drawdown" in r for r in reasons)


def test_insufficient_trades_fails_even_if_profitable():
    trades = []
    equity = 1000.0
    for i in range(5):
        pnl = 20.0
        equity += pnl
        trades.append(_make_exit(i * 9, pnl, equity))
    metrics = compute_paper_metrics(trades, initial_capital=1000.0)
    ready, reasons = evaluate_gate(metrics, GateThresholds(), None)
    assert ready is False
    assert any("only 5 completed" in r for r in reasons)


def test_build_gate_file_is_fail_closed_on_bad_input(tmp_path):
    bad_trades_path = tmp_path / "not_json.json"
    bad_trades_path.write_text("{ this is not valid json")
    output_path = tmp_path / "readiness_test.json"
    gate = build_gate_file("test", str(bad_trades_path), str(output_path))
    assert gate["ready_for_live"] is False
    assert "failed to load" in gate["reasons"][0]


def test_real_v4_trades_file_is_not_ready():
    # Sanity check against the actual repo file: real recorded loss should
    # not clear the gate. Skips quietly if trades_v4.json isn't present
    # (e.g. running this file outside the repo root).
    import json
    from pathlib import Path
    p = Path("trades_v4.json")
    if not p.exists():
        return
    trades = json.loads(p.read_text())
    metrics = compute_paper_metrics(trades, initial_capital=1000.0)
    ready, reasons = evaluate_gate(metrics, GateThresholds(), None)
    assert metrics["total_return_pct"] < 0
    assert ready is False

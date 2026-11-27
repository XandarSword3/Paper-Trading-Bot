"""Tests for v4_system.data_engine.live_watchdog."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from v4_system.data_engine import live_watchdog as lw


def _write_state(data_dir, stream, age_seconds):
    state_dir = data_dir / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    (state_dir / f"{stream}.json").write_text(json.dumps({
        "last_msg_ts": updated.isoformat(),
        "updated": updated.isoformat(),
    }), encoding="utf-8")


def test_healthy_streams_produce_no_alerts(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    assert lw.check_once(tmp_path) == []


def test_stale_stream_alerts(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_state(tmp_path, "binance_bookticker", age_seconds=400)
    alerts = lw.check_once(tmp_path)
    assert len(alerts) == 1
    assert alerts[0]["stream"] == "binance_bookticker"
    assert alerts[0]["kind"] == "stale"


def test_quiet_liquidation_stream_uses_connection_liveness(tmp_path):
    # A liquidation stream with NO data rows but a fresh state file (connected,
    # quiet market) must NOT alert.
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_state(tmp_path, "bybit_liquidations", age_seconds=250)  # < 300s
    assert lw.check_once(tmp_path) == []
    _write_state(tmp_path, "bybit_liquidations", age_seconds=400)  # > 300s
    alerts = lw.check_once(tmp_path)
    assert [a["stream"] for a in alerts] == ["bybit_liquidations"]


def test_missing_state_file_alerts(tmp_path):
    alerts = lw.check_once(tmp_path)
    assert len(alerts) == len(lw.STREAM_THRESHOLDS)
    assert all(a["kind"] == "no_state" for a in alerts)

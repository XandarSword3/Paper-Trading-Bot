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


def test_watchdog_uses_recorder_data_dir_by_default():
    # The watchdog must watch the SAME directory the recorder writes to.
    from v4_system.data_engine import live_recorder as lr
    assert lw.DATA_DIR == lr.DATA_DIR, \
        f"watchdog DATA_DIR {lw.DATA_DIR} != recorder DATA_DIR {lr.DATA_DIR}"
    assert lw.DATA_DIR.parent.parent.name == "v4_system", lw.DATA_DIR


def _write_rows(data_dir, stream="binance_bookticker"):
    data_dir = data_dir / stream / "2026-11-27"
    data_dir.mkdir(parents=True, exist_ok=True)
    row_file = data_dir / "12-00-00-000000.jsonl"
    row_file.write_text(
        json.dumps({"symbol": "BTCUSDT", "_recv_ts": "2026-11-27T12:00:00+00:00"}) + "\n",
        encoding="utf-8")
    import os
    os.utime(row_file, None)  # freshly written


def test_healthy_streams_produce_no_alerts(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_rows(tmp_path)
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
    _write_rows(tmp_path)
    _write_state(tmp_path, "bybit_liquidations", age_seconds=250)  # < 300s
    assert lw.check_once(tmp_path) == []
    _write_state(tmp_path, "bybit_liquidations", age_seconds=400)  # > 300s
    alerts = lw.check_once(tmp_path)
    assert [a["stream"] for a in alerts] == ["bybit_liquidations"]


def test_missing_state_file_alerts(tmp_path):
    alerts = lw.check_once(tmp_path)
    assert len(alerts) == len(lw.STREAM_THRESHOLDS)
    assert all(a["kind"] == "no_state" for a in alerts)


# -- fix 2 (round 3): missing data, not just disconnects -------------------

def test_bookticker_no_rows_for_3_minutes_alerts(tmp_path):
    # Connection alive (fresh state) but no data rows written for 3+ minutes.
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    alerts = lw.check_once(tmp_path)
    kinds = {(a["stream"], a["kind"]) for a in alerts}
    assert ("binance_bookticker", "no_rows") in kinds, alerts


def test_bookticker_recent_rows_no_alert(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_rows(tmp_path)
    alerts = lw.check_once(tmp_path)
    assert not any(a["kind"] == "no_rows" for a in alerts), alerts


# -- fix 3 (round 3): Telegram alerts ---------------------------------------

def test_telegram_config_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_TOKEN=tok123\nTELEGRAM_CHAT_ID=42\n", encoding="utf-8")
    monkeypatch.setattr(lw, "ENV_PATH", env)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert lw.telegram_config() == ("tok123", "42")


def test_telegram_config_env_vars_win(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_TOKEN=fromfile\nTELEGRAM_CHAT_ID=1\n", encoding="utf-8")
    monkeypatch.setattr(lw, "ENV_PATH", env)
    monkeypatch.setenv("TELEGRAM_TOKEN", "fromenv")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "99")
    assert lw.telegram_config() == ("fromenv", "99")


def test_send_telegram_posts_to_bot_api(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["body"] = req.data.decode("utf-8")
        return FakeResp()

    monkeypatch.setattr(lw.urllib.request, "urlopen", fake_urlopen)
    assert lw.send_telegram("hello", config=("tok", "42")) is True
    assert captured["url"].endswith("/bottok/sendMessage"), captured["url"]
    assert '"chat_id": "42"' in captured["body"]
    assert "hello" in captured["body"]


def test_emit_sends_telegram_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(lw, "ALERTS_FILE", tmp_path / "_alerts.jsonl")
    sent = []
    monkeypatch.setattr(lw, "send_telegram", lambda text: sent.append(text) or True)
    lw.emit([{"stream": "binance_bookticker", "kind": "no_rows"}])
    assert len(sent) == 1
    assert "no_rows" in sent[0]

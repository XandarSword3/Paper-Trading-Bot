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


def test_send_telegram_logs_http_error_body(monkeypatch, caplog):
    import urllib.error

    def fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden",
            hdrs=None, fp=__import__("io").BytesIO(
                b'{"ok":false,"description":"Forbidden: bot was blocked by the user"}'))

    monkeypatch.setattr(lw.urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level("WARNING", logger="live_watchdog"):
        assert lw.send_telegram("hello", config=("tok", "42")) is False
    assert any("bot was blocked by the user" in rec.message for rec in caplog.records), \
        "Telegram's error description was not logged"
    assert all("tok" not in rec.message for rec in caplog.records), "token leaked into logs"


def test_emit_sends_telegram_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(lw, "ALERTS_FILE", tmp_path / "_alerts.jsonl")
    sent = []
    monkeypatch.setattr(lw, "send_telegram", lambda text: sent.append(text) or True)
    lw.emit([{"stream": "binance_bookticker", "kind": "no_rows"}])
    assert len(sent) == 1
    assert "no_rows" in sent[0]


# -- fix 3 (round 4): alert once, hourly reminder, once on recovery ---------

def test_recorder_reported_error_alerts(tmp_path):
    # A rejected Bybit subscription is persisted by the recorder as state.error;
    # the watchdog must raise an alert for it.
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_rows(tmp_path)
    state_dir = tmp_path / "_state"
    state = json.loads((state_dir / "bybit_liquidations.json").read_text())
    state["error"] = "subscribe rejected: invalid topic"
    (state_dir / "bybit_liquidations.json").write_text(json.dumps(state))
    alerts = lw.check_once(tmp_path)
    kinds = {(a["stream"], a["kind"]) for a in alerts}
    assert ("bybit_liquidations", "error") in kinds, alerts


def test_alert_state_sends_once_on_start():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "no_rows"}
    first = st.process([alert], now_ts=1000.0)
    assert len(first) == 1 and first[0]["event"] == "start"
    second = st.process([alert], now_ts=1060.0)
    assert second == [], "same problem alerted twice within the hour"


def test_alert_state_reminds_hourly():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "no_rows"}
    st.process([alert], now_ts=1000.0)   # start
    assert st.process([alert], now_ts=2000.0) == []   # 1000s later: no reminder
    reminder = st.process([alert], now_ts=4601.0)     # 3601s after last send
    assert len(reminder) == 1 and reminder[0]["event"] == "reminder", reminder
    assert st.process([alert], now_ts=5000.0) == []   # reminder resets the clock


def test_alert_state_sends_once_on_recovery():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "no_rows"}
    st.process([alert], now_ts=1000.0)
    # Brief health is NOT a recovery: 60s of quiet changes nothing.
    assert st.process([], now_ts=1060.0) == []
    # Recovery only after 5 continuous healthy minutes (1060 + 300 = 1360).
    recovery = st.process([], now_ts=1361.0)
    assert len(recovery) == 1
    assert recovery[0]["event"] == "recovery"
    assert recovery[0]["stream"] == "binance_bookticker"
    assert st.process([], now_ts=1420.0) == [], "recovery sent twice"


# -- incident tracking: flapping must not fake recovery --------------------

def _alert():
    return {"stream": "binance_bookticker", "kind": "error"}


def test_flapping_reconnects_stay_one_incident():
    st = lw.AlertState()
    # start, then alternating present/absent every 60s for 10 minutes
    assert st.process([_alert()], now_ts=0.0)[0]["event"] == "start"
    t = 60.0
    events = []
    while t <= 600.0:
        present = (int(t // 60) % 2 == 0)
        events += st.process([_alert()] if present else [], now_ts=t)
        t += 60.0
    assert not any(e["event"] == "recovery" for e in events), \
        "brief reconnections triggered a recovery"
    assert not any(e["event"] == "start" for e in events[1:]), \
        "flapping re-opened the incident"


def test_recovery_after_5_minutes_stable():
    st = lw.AlertState()
    st.process([_alert()], now_ts=0.0)
    st.process([], now_ts=60.0)    # healthy_since = 60
    assert st.process([], now_ts=359.0) == []   # 299s stable: no recovery
    rec = st.process([], now_ts=360.0)          # 300s stable: recovery
    assert len(rec) == 1 and rec[0]["event"] == "recovery"


def test_recovery_timer_resets_when_problem_returns():
    st = lw.AlertState()
    st.process([_alert()], now_ts=0.0)
    st.process([], now_ts=60.0)    # healthy_since = 60
    st.process([], now_ts=200.0)   # 140s stable
    st.process([_alert()], now_ts=250.0)  # problem back: timer cancelled
    assert st.process([], now_ts=300.0) == []  # healthy_since = 300 now
    assert st.process([], now_ts=599.0) == []  # only 299s stable
    rec = st.process([], now_ts=600.0)         # 300s stable -> recovery
    assert len(rec) == 1 and rec[0]["event"] == "recovery"


def test_hourly_reminders_continue_during_flapping():
    st = lw.AlertState()
    st.process([_alert()], now_ts=0.0)          # start
    # flap every 60s; reminders must still fire hourly from last send
    t = 60.0
    reminders = []
    while t <= 3700.0:
        present = (int(t // 60) % 2 == 0)
        for e in st.process([_alert()] if present else [], now_ts=t):
            if e["event"] == "reminder":
                reminders.append(t)
        t += 60.0
    assert reminders, "no hourly reminder during a prolonged incident"
    assert all(b - a >= 3600.0 for a, b in zip(reminders, reminders[1:]))


def test_watchdog_restart_preserves_incident(tmp_path):
    path = tmp_path / "incidents.json"
    st1 = lw.AlertState(state_path=path)
    st1.process([_alert()], now_ts=1000.0)
    # "Restart": brand-new AlertState loading the same file.
    st2 = lw.AlertState(state_path=path)
    out = st2.process([_alert()], now_ts=1060.0)
    assert out == [], "restart re-sent start for an ongoing incident"
    # And it must not fake a recovery either while the problem persists.
    out = st2.process([], now_ts=1100.0)
    assert out == []
    # Recovery still requires 5 stable minutes after the restart.
    out = st2.process([], now_ts=1401.0)
    assert len(out) == 1 and out[0]["event"] == "recovery"


# -- stale-timeout observability: unhealthy_since + recorder-confirmed health

def _write_state_with_health(data_dir, stream, unhealthy_since=None, healthy=False,
                             error=None, age_seconds=10):
    state_dir = data_dir / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    state = {
        "last_msg_ts": updated.isoformat(),
        "updated": updated.isoformat(),
        "healthy": healthy,
    }
    if unhealthy_since:
        state["unhealthy_since"] = unhealthy_since
    if error:
        state["error"] = error
    (state_dir / f"{stream}.json").write_text(json.dumps(state), encoding="utf-8")


def test_unhealthy_since_raises_error_alert(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_rows(tmp_path)
    # bookTicker stale-timeout marker, no explicit error field.
    _write_state_with_health(tmp_path, "binance_bookticker",
                             unhealthy_since="2026-11-27T20:00:00+00:00")
    alerts = lw.check_once(tmp_path)
    err = [a for a in alerts if a["stream"] == "binance_bookticker"
           and a["kind"] == "error"]
    assert len(err) == 1, alerts
    assert "unhealthy since" in err[0]["detail"]


def test_healthy_flag_alone_produces_no_alert(tmp_path):
    for stream in lw.STREAM_THRESHOLDS:
        _write_state(tmp_path, stream, age_seconds=10)
    _write_rows(tmp_path)
    _write_state_with_health(tmp_path, "binance_bookticker", healthy=True)
    assert lw.check_once(tmp_path) == []


def test_confirmed_healthy_streams_reads_state(tmp_path):
    _write_state_with_health(tmp_path, "binance_bookticker", healthy=True)
    _write_state_with_health(tmp_path, "binance_liquidations", healthy=False)
    assert lw.confirmed_healthy_streams(tmp_path) == {"binance_bookticker"}


def test_recovery_ready_recovers_without_second_300s_wait():
    """Recorder confirmed 300s healthy -> recovery at once, not +300s more."""
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "error"}
    st.process([alert], now_ts=0.0)          # incident starts
    # Flapping continues: alert present at t=60, 120, ... (no recovery).
    assert st.process([alert], now_ts=60.0) == []
    # At t=300 the recorder has confirmed health: alert absent + ready.
    out = st.process([], now_ts=300.0,
                     recovery_ready={"binance_bookticker|error"})
    assert len(out) == 1 and out[0]["event"] == "recovery", out
    # Exactly one recovery.
    assert st.process([], now_ts=360.0,
                      recovery_ready={"binance_bookticker|error"}) == []


def test_without_recovery_ready_old_300s_rule_still_applies():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "error"}
    st.process([alert], now_ts=0.0)
    assert st.process([], now_ts=300.0) == []    # healthy_since = 300
    assert st.process([], now_ts=599.0) == []    # 299s alert-free: no recovery
    rec = st.process([], now_ts=600.0)           # 300s alert-free: recovery
    assert len(rec) == 1 and rec[0]["event"] == "recovery"


def test_recovery_ready_ignored_while_alert_present():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "error"}
    st.process([alert], now_ts=0.0)
    # Recorder says healthy but the watchdog still sees the alert -> no recovery.
    out = st.process([alert], now_ts=300.0,
                     recovery_ready={"binance_bookticker|error"})
    assert out == []


def test_new_failure_during_recovery_period_keeps_incident():
    st = lw.AlertState()
    alert = {"stream": "binance_bookticker", "kind": "error"}
    st.process([alert], now_ts=0.0)
    st.process([], now_ts=100.0)              # healthy_since = 100
    st.process([alert], now_ts=200.0)         # failure again: timer cancelled
    assert st.process([], now_ts=400.0) == []  # healthy_since = 400
    assert st.process([], now_ts=699.0) == []  # 299s stable
    rec = st.process([], now_ts=700.0)         # 300s stable -> recovery
    assert len(rec) == 1 and rec[0]["event"] == "recovery"


def test_watchdog_restart_preserves_incident_for_recovery_ready(tmp_path):
    path = tmp_path / "incidents.json"
    st1 = lw.AlertState(state_path=path)
    st1.process([{"stream": "binance_bookticker", "kind": "error"}], now_ts=1000.0)
    # Restart: new AlertState loads the incident; recorder now confirms health.
    st2 = lw.AlertState(state_path=path)
    out = st2.process([], now_ts=1300.0,
                      recovery_ready={"binance_bookticker|error"})
    assert len(out) == 1 and out[0]["event"] == "recovery"
    assert st2.process([], now_ts=1360.0,
                       recovery_ready={"binance_bookticker|error"}) == []

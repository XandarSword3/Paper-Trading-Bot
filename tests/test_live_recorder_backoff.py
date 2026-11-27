"""Regression tests for the bookTicker reconnect-storm fix.

Deterministic simulated WebSocket failures: no live Binance connection is
used. A fake monotonic clock drives connection uptime; time.sleep is
recorded (and advances the fake clock) instead of really sleeping.
"""

from __future__ import annotations

import json
import time as _time

import pytest
import websocket

from v4_system.data_engine import live_recorder as lr

_REAL_SLEEP = _time.sleep  # saved before monkeypatching lr.time.sleep


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeWS:
    """Scripted websocket for run(): each item is consumed per recv_data call.

    ("msg",)      -> one text message, clock advances 1s
    ("msgx", n)   -> n messages, clock advances 1s each
    ("fail",)     -> raise WebSocketConnectionClosedException immediately
    """

    def __init__(self, script, clock):
        self.script = list(script)
        self.clock = clock
        self.closed = False

    def settimeout(self, _t):
        pass

    def send(self, _data):
        pass

    def ping(self, _data=b""):
        pass

    def pong(self, _data):
        pass

    def close(self, timeout=None):
        self.closed = True

    def recv_data(self, control_frame=False):
        if not self.script:
            raise websocket.WebSocketConnectionClosedException("lost")
        item = self.script.pop(0)
        kind = item[0]
        if kind == "msg":
            self.clock.now += 1.0
            return (0x1, json.dumps({"e": "bookTicker", "s": "BTCUSDT"}).encode())
        if kind == "msgx":
            # n one-second messages, then continue with the rest of the script
            n = item[1]
            self.clock.now += 1.0
            if n > 1:
                self.script.insert(0, ("msgx", n - 1))
            return (0x1, json.dumps({"e": "bookTicker", "s": "BTCUSDT"}).encode())
        if kind == "fail":
            raise websocket.WebSocketConnectionClosedException("lost")
        if kind == "timeout":
            raise websocket.WebSocketTimeoutException("timed out")
        if kind == "sleep":
            _REAL_SLEEP(item[1])
            return self.recv_data(control_frame=control_frame)
        raise AssertionError(f"unknown script item {item!r}")


def _make_recorder(tmp_path, monkeypatch, clock, scripts, max_attempts=None, **kw):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lr.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(lr.time, "sleep", clock.sleep)
    attempts = {"n": 0}
    holder = {}

    def factory(url, timeout=20):
        idx = attempts["n"]
        attempts["n"] += 1
        script = scripts[min(idx, len(scripts) - 1)]
        ws = FakeWS(script, clock)
        if max_attempts is not None and attempts["n"] >= max_attempts:
            holder["rec"].stop_event.set()
        return ws

    monkeypatch.setattr(lr.websocket, "create_connection", factory)
    params = dict(flush_seconds=9999, backoff_initial=5.0, backoff_max=120.0,
                  healthy_reset_seconds=300.0, require_data_for_reset=True)
    params.update(kw)
    r = lr.StreamRecorder("binance_bookticker", "ws://x",
                          lambda m: [{"symbol": "BTCUSDT"}], **params)
    holder["rec"] = r
    return r, attempts


def _run_until_attempts(r, attempts, max_attempts):
    """Run the reconnect loop in a thread; the factory stops it after
    max_attempts connections (see _make_recorder)."""
    import threading
    t = threading.Thread(target=r.run, daemon=True)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "recorder run loop did not stop"
    assert attempts["n"] >= max_attempts, f"only {attempts['n']} attempts"


# -- backoff policy ---------------------------------------------------------

def test_repeated_short_connections_increase_backoff(tmp_path, monkeypatch):
    clock = FakeClock()
    # Each connection: handshake OK, 3s of data, then dropped.
    script = [("msgx", 3), ("fail",)]
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [script], max_attempts=6)
    _run_until_attempts(r, attempts, 6)
    sleeps = clock.sleeps
    assert len(sleeps) >= 5, f"expected >=5 reconnect delays, got {sleeps}"
    bases = [5, 10, 20, 40, 80, 120]
    for i, s in enumerate(sleeps[:6]):
        lo = 0.75 * bases[i]
        hi = min(1.25 * bases[i], 120.0)  # hard cap AFTER jitter
        assert lo <= s <= hi, f"delay #{i}={s} outside [{lo}, {hi}]"
    assert r._backoff_failures == 6


def test_handshake_alone_does_not_reset_backoff(tmp_path, monkeypatch):
    clock = FakeClock()
    # Connection handshakes but drops after 2s with NO data at all.
    script = [("fail",)]
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [script], max_attempts=4)
    _run_until_attempts(r, attempts, 4)
    sleeps = clock.sleeps
    bases = [5, 10, 20]
    for i, s in enumerate(sleeps[:3]):
        lo, hi = 0.75 * bases[i], 1.25 * bases[i]
        assert lo <= s <= hi, f"delay #{i}={s} outside [{lo}, {hi}]"
    assert r._backoff_failures == 4


def test_stable_connection_resets_backoff(tmp_path, monkeypatch):
    clock = FakeClock()
    flap = [("msgx", 3), ("fail",)]
    stable = [("msgx", 310), ("fail",)]  # 310s uptime >= 300s healthy reset
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [flap, flap, stable, flap],
                                 max_attempts=4)
    _run_until_attempts(r, attempts, 4)
    sleeps = clock.sleeps
    # delays: 5, 10, then after the stable connection the counter resets -> 5
    assert len(sleeps) >= 3
    assert 0.75 * 5 <= sleeps[0] <= 1.25 * 5
    assert 0.75 * 10 <= sleeps[1] <= 1.25 * 10
    assert 0.75 * 5 <= sleeps[2] <= 1.25 * 5, \
        f"backoff did not reset after 5 healthy minutes: {sleeps[2]}"
    assert r._backoff_failures == 1  # only the final flap counts


def test_backoff_never_exceeds_max(tmp_path, monkeypatch):
    clock = FakeClock()
    script = [("fail",)]
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [script], max_attempts=12)
    _run_until_attempts(r, attempts, 12)
    assert clock.sleeps, "no reconnect delays recorded"
    assert max(clock.sleeps) <= 120.0, \
        f"delay exceeded the 120s hard cap: {max(clock.sleeps)}"
    # and it actually reaches the cap region
    assert any(s > 100 for s in clock.sleeps), clock.sleeps


def test_jitter_within_bounds_and_hard_cap():
    r = lr.StreamRecorder("s", "ws://x", lambda m: [], backoff_initial=5.0,
                          backoff_max=120.0)
    for failures, base in [(1, 5), (2, 10), (3, 20), (4, 40), (5, 80), (6, 120), (9, 120)]:
        r._backoff_failures = failures
        hi = min(1.25 * base, 120.0)
        for _ in range(500):
            d = r._next_delay()
            assert 0.75 * base <= d <= hi, (failures, d)
            assert d <= 120.0, f"delay {d} exceeded the hard cap"
    # At maximum jitter the delay is clamped to exactly the cap.
    r._backoff_failures = 9
    monkey_vals = [r._next_delay() for _ in range(500)]
    assert max(monkey_vals) <= 120.0


def test_no_overlapping_reconnect_attempts(tmp_path, monkeypatch):
    clock = FakeClock()
    script = [("fail",)]
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [script], max_attempts=5)
    _run_until_attempts(r, attempts, 5)
    # Every attempt is followed by exactly one sleep -> strictly sequential.
    assert attempts["n"] == 5
    assert len(clock.sleeps) == 5


def test_disconnects_still_record_gaps(tmp_path, monkeypatch):
    clock = FakeClock()
    # Real 1.2s pause after the messages so the gap exceeds the 1s logging
    # threshold (gap duration uses the real wall clock).
    script = [("msgx", 2), ("sleep", 1.2), ("fail",)]
    r, attempts = _make_recorder(tmp_path, monkeypatch, clock, [script], max_attempts=2)
    _run_until_attempts(r, attempts, 2)
    gaps = list((tmp_path / "_gaps").rglob("*.jsonl"))
    assert gaps, "no gap records written"
    records = [json.loads(line) for f in gaps for line in f.read_text().splitlines()]
    assert any(rec["stream"] == "binance_bookticker" and rec["reason"] == "disconnect"
               for rec in records), records


def test_other_streams_keep_default_policy():
    recs = lr.build_recorders(60.0)
    by_name = {r.name: r for r in recs}
    bt = by_name["binance_bookticker"]
    assert bt.backoff_initial == 5.0 and bt.backoff_max == 120.0
    assert bt.require_data_for_reset is True
    for name in ("binance_liquidations", "bybit_liquidations"):
        r = by_name[name]
        assert r.backoff_initial == lr.RECONNECT_BASE_DELAY
        assert r.backoff_max == lr.RECONNECT_MAX_DELAY
        assert r.require_data_for_reset is False


# -- health tracking (stale-timeout observability fix) ---------------------

def _health_recorder(tmp_path, monkeypatch, clock, scripts, max_attempts, **kw):
    params = dict(track_health=True, staleness_timeout=0.02,
                  healthy_reset_seconds=300.0, require_data_for_reset=True,
                  flush_seconds=9999, backoff_initial=5.0, backoff_max=120.0)
    params.update(kw)
    return _make_recorder(tmp_path, monkeypatch, clock, scripts,
                          max_attempts=max_attempts, **params)


def _state_file(tmp_path):
    p = tmp_path / "_state" / "binance_bookticker.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def test_stale_timeout_marks_unhealthy_and_persists(tmp_path, monkeypatch):
    clock = FakeClock()
    # One message, then silence -> staleness timeout (0.02s real) -> reconnect.
    script = [("msg",), ("timeout",)]
    r, attempts = _health_recorder(tmp_path, monkeypatch, clock, [script], 2)
    _run_until_attempts(r, attempts, 2)
    assert r._unhealthy_since is not None, "stale timeout did not mark unhealthy"
    assert r._confirmed_healthy is False
    state = _state_file(tmp_path)
    assert state is not None and state.get("unhealthy_since"), state
    assert state.get("healthy") is False


def test_handshake_and_single_message_do_not_clear_unhealthy(tmp_path, monkeypatch):
    clock = FakeClock()
    stale = [("msg",), ("timeout",)]
    # Attempt 2: handshake OK + exactly ONE message, then dropped.
    one_msg = [("msg",), ("fail",)]
    r, attempts = _health_recorder(tmp_path, monkeypatch, clock, [stale, one_msg], 2)
    _run_until_attempts(r, attempts, 2)
    assert r._unhealthy_since is not None, "handshake/one message cleared unhealthy"
    assert r._confirmed_healthy is False
    # (streak is reset when the connection ends; unhealthy persists)


def test_300s_continuous_data_confirms_health(tmp_path, monkeypatch):
    """Unit-level: handshake and single messages never clear unhealthy;
    only 300 continuous seconds of data does."""
    clock = FakeClock()
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lr.time, "monotonic", clock.monotonic)
    r = lr.StreamRecorder("binance_bookticker", "ws://x",
                          lambda m: [{"symbol": "BTCUSDT"}],
                          track_health=True, healthy_reset_seconds=300.0,
                          flush_seconds=9999)
    r._mark_unhealthy("stale timeout")
    assert r._unhealthy_since is not None
    # Handshake alone (no messages) must not clear it.
    assert r._unhealthy_since is not None and r._confirmed_healthy is False
    # One message starts the streak but does not confirm.
    r._note_message()
    assert r._confirmed_healthy is False and r._unhealthy_since is not None
    # 299 continuous seconds: still not confirmed.
    clock.now += 299.0
    r._note_message()
    assert r._confirmed_healthy is False
    # At 300s: confirmed, unhealthy cleared.
    clock.now += 1.0
    r._note_message()
    assert r._confirmed_healthy is True
    assert r._unhealthy_since is None
    r.last_msg_ts = _time.time()  # so _persist_state does not early-return
    r._flush()
    state = _state_file(tmp_path)
    assert state is not None and state.get("healthy") is True
    assert "unhealthy_since" not in state


def test_new_failure_after_confirmation_resets_health(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lr.time, "monotonic", clock.monotonic)
    r = lr.StreamRecorder("binance_bookticker", "ws://x",
                          lambda m: [{"symbol": "BTCUSDT"}],
                          track_health=True, healthy_reset_seconds=300.0,
                          flush_seconds=9999)
    r._note_message()
    clock.now += 300.0
    r._note_message()
    assert r._confirmed_healthy is True
    # A new stale timeout must reset the interval and re-open the incident.
    r._mark_unhealthy("stale timeout")
    assert r._confirmed_healthy is False
    assert r._unhealthy_since is not None
    assert r._streak_start_mono is None


def test_new_stale_timeout_resets_healthy_interval(tmp_path, monkeypatch):
    clock = FakeClock()
    stale = [("msg",), ("timeout",)]
    # 200s of data, then stale; then only 150s of data -> never confirmed.
    partial = [("msgx", 200), ("timeout",)]
    short = [("msgx", 150), ("fail",)]
    r, attempts = _health_recorder(tmp_path, monkeypatch, clock,
                                    [stale, partial, short], 3)
    _run_until_attempts(r, attempts, 3)
    assert r._confirmed_healthy is False
    assert r._unhealthy_since is not None


def test_repeated_stale_reconnects_keep_one_unhealthy_since(tmp_path, monkeypatch):
    clock = FakeClock()
    stale = [("msg",), ("timeout",)]
    r, attempts = _health_recorder(tmp_path, monkeypatch, clock, [stale], 4)
    _run_until_attempts(r, attempts, 4)
    assert r._unhealthy_since is not None
    assert r._confirmed_healthy is False

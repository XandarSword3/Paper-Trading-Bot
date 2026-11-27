"""Tests for v4_system.data_engine.live_recorder.

Each test targets one defect found in review of 7284c4c and fails on that
commit. FakeWS mimics websocket-client: recv() returns "" on a close frame,
recv_data() returns (opcode, payload).
"""

from __future__ import annotations

import json
import struct
import time

import pytest
import websocket

from v4_system.data_engine import live_recorder as lr


class FakeWS:
    """Scripted websocket stand-in.

    Script items:
      ("msg", raw_str)     -> one text message
      ("sleep", seconds)   -> pause, then continue with next item
      ("timeout",)         -> WebSocketTimeoutException
      ("close", code, reason) -> close frame
      ("error", exc)       -> raise exc
    """

    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.closed = False

    def settimeout(self, _t):
        pass

    def send(self, data):
        self.sent.append(data)

    def pong(self, _data):
        pass

    def ping(self, _data=b""):
        pass

    def close(self):
        self.closed = True

    def _next(self):
        while self.script:
            item = self.script.pop(0)
            kind = item[0]
            if kind == "sleep":
                time.sleep(item[1])
                continue
            if kind == "msg":
                return (0x1, item[1].encode("utf-8"))
            if kind == "ping":
                return (0x9, item[1] if len(item) > 1 else b"")
            if kind == "pong":
                return (0xA, item[1] if len(item) > 1 else b"")
            if kind == "timeout":
                raise websocket.WebSocketTimeoutException("timed out")
            if kind == "close":
                code, reason = item[1], item[2]
                payload = struct.pack(">H", code) + reason.encode("utf-8")
                return (0x8, payload)
            if kind == "error":
                raise item[1]
            raise AssertionError(f"unknown script item {item!r}")
        raise websocket.WebSocketTimeoutException("timed out")

    def recv_data(self, control_frame=False):
        return self._next()

    def recv(self):
        opcode, data = self._next()
        if opcode == 0x8:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return data


def _rows_on_disk(tmp_path, stream):
    rows = []
    for f in _jsonl_files(tmp_path, stream):
        rows.extend(json.loads(line) for line in f.read_text().splitlines())
    return rows


def _bybit_liq(symbol="BTCUSDT", side="Buy", v="0.5", p="50000.1", ts=1795753643175):
    return {
        "topic": f"allLiquidation.{symbol}",
        "type": "snapshot",
        "ts": ts,
        "data": [{"T": ts - 175, "s": symbol, "S": side, "v": v, "p": p}],
    }


# -- fix 1: Bybit data is a list; parser must see the full message ---------

def test_bybit_liquidation_list_data_produces_one_row_per_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("bybit_liquidations", "ws://x", lr.parse_bybit_liquidation,
                          flush_seconds=9999)
    payload = _bybit_liq()
    payload["data"].append({"T": 1795753643001, "s": "BTCUSDT", "S": "Sell",
                            "v": "0.25", "p": "50001.0"})
    ws = FakeWS([("msg", json.dumps(payload)), ("timeout",)])
    r._receive_loop(ws, time.time() + 0.3)
    rows = _rows_on_disk(tmp_path, "bybit_liquidations")
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    row = rows[0]
    assert row["s"] == "BTCUSDT"
    assert row["S"] == "Buy"          # position side: Buy = long liquidated
    assert row["v"] == "0.5"
    assert row["p"] == "50000.1"
    assert row["ts"] == 1795753643175
    assert row["T"] == 1795753643175 - 175
    assert rows[1]["S"] == "Sell"


# -- fix 2: sampling must be per-second in the receive loop, not per-flush --

def _bookticker(symbol="BTCUSDT", price="50000.0"):
    return {"stream": f"{symbol.lower()}@bookTicker", "data": {
        "e": "bookTicker", "E": 1, "u": 1, "s": symbol,
        "b": price, "B": "1.0", "a": "50001.0", "A": "1.0", "T": 1}}


def test_bookticker_sampling_is_per_second_not_per_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_bookticker", "ws://x", lr.parse_binance_bookticker,
                          flush_seconds=9999, sample_seconds=0.05,
                          sample_key=lambda rec: rec.get("symbol"))
    script = []
    for i in range(3):
        script.append(("msg", json.dumps(_bookticker(price=f"5000{i}.0"))))
        script.append(("sleep", 0.06))
    script.append(("timeout",))
    ws = FakeWS(script)
    r._receive_loop(ws, time.time() + 2.0)
    # flush_seconds=9999: rows must have been sampled into the buffer during
    # the receive loop, not at flush time. They land on disk at loop exit.
    rows = _rows_on_disk(tmp_path, "binance_bookticker")
    assert len(rows) == 3, f"expected 3 sampled rows, got {len(rows)}"
    assert [row["bid_price"] for row in rows] == ["50000.0", "50001.0", "50002.0"]


def test_bookticker_sampling_drops_within_window(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_bookticker", "ws://x", lr.parse_binance_bookticker,
                          flush_seconds=9999, sample_seconds=10.0,
                          sample_key=lambda rec: rec.get("symbol"))
    script = [("msg", json.dumps(_bookticker(price="1.0"))),
              ("msg", json.dumps(_bookticker(price="2.0"))),
              ("timeout",)]
    ws = FakeWS(script)
    r._receive_loop(ws, time.time() + 0.3)
    rows = _rows_on_disk(tmp_path, "binance_bookticker")
    assert len(rows) == 1
    assert rows[0]["bid_price"] == "1.0"


# -- fix 3: flush timer belongs to the stream; flush on disconnect ---------

def _liq_msg():
    return json.dumps({"e": "forceOrder", "E": 1, "o": {
        "s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "1",
        "p": "1", "ap": "1", "X": "FILLED", "l": "1", "z": "1", "T": 1}})


def _jsonl_files(tmp_path, stream):
    return list((tmp_path / stream).rglob("*.jsonl"))


def test_flush_on_disconnect_writes_buffered_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=9999)
    ws = FakeWS([("msg", _liq_msg()), ("error", ConnectionError("boom"))])
    r._receive_loop(ws, time.time() + 5.0)
    files = _jsonl_files(tmp_path, "binance_liquidations")
    assert files, "disconnect did not flush the buffer to disk"
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["symbol"] == "BTCUSDT"


def test_flush_timer_survives_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=0.1)
    r.buffer.append({"symbol": "BTCUSDT"})
    # Two short receive loops (as after a reconnect): neither alone reaches
    # flush_seconds, but together they exceed it. The timer must not reset.
    r._receive_loop(FakeWS([("timeout",)]), time.time() + 0.08)
    r._receive_loop(FakeWS([("timeout",)]), time.time() + 0.08)
    files = _jsonl_files(tmp_path, "binance_liquidations")
    assert files, "flush timer reset on reconnect; buffered row never written"


# -- fix 4: staleness timeout reconnects a silent socket -------------------

def test_staleness_timeout_reconnects_silent_socket(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_bookticker", "ws://x", lr.parse_binance_bookticker,
                          flush_seconds=9999, staleness_timeout=0.2)
    connections = []

    def fake_connect(*a, **k):
        connections.append(1)
        return FakeWS([("timeout",)])

    monkeypatch.setattr(lr.websocket, "create_connection", fake_connect)
    r.run(duration=2.5)
    assert len(connections) >= 2, \
        f"silent socket never reconnected ({len(connections)} connection(s))"


def test_liquidation_stream_staleness_policy():
    # Binance liquidations are quiet by nature: no data-staleness timeout;
    # protocol ping/pong detects a dead socket instead.
    # Bybit reconnects after ~60s with no message (pongs count as activity).
    # bookTicker is high-frequency: a few seconds of silence means a dead socket.
    recs = {r.name: r for r in lr.build_recorders(60.0)}
    assert recs["binance_liquidations"].staleness_timeout is None
    assert recs["bybit_liquidations"].staleness_timeout == 60.0
    assert recs["binance_bookticker"].staleness_timeout is not None
    assert recs["binance_bookticker"].staleness_timeout <= 10.0


# -- fix 5: gaps start at last message, close codes, persisted state --------

def _gap_rows(tmp_path):
    rows = []
    for f in (tmp_path / "_gaps").rglob("*.jsonl"):
        rows.extend(json.loads(line) for line in f.read_text().splitlines())
    return rows


def test_close_frame_records_close_code_and_gap_from_last_message(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=9999)
    ws1 = FakeWS([("msg", _liq_msg()), ("sleep", 0.05), ("close", 1000, "bye")])
    rest = FakeWS([("timeout",)])
    conns = [ws1] + [rest] * 50
    monkeypatch.setattr(lr.websocket, "create_connection",
                        lambda *a, **k: conns.pop(0))
    r.run(duration=1.5)
    gaps = _gap_rows(tmp_path)
    assert gaps, "close frame did not produce a gap record"
    g = gaps[0]
    assert g.get("close_code") == 1000, f"close code missing: {g}"
    assert g.get("close_reason") == "bye"
    # Gap must start at the last received message, not at disconnect time.
    last_msg_dt = __import__("datetime").datetime.fromtimestamp(
        r.last_msg_ts, tz=__import__("datetime").timezone.utc)
    gap_start = __import__("datetime").datetime.fromisoformat(g["gap_start"])
    assert abs((gap_start - last_msg_dt).total_seconds()) < 0.5, \
        f"gap_start {g['gap_start']} != last message {last_msg_dt.isoformat()}"


def test_last_message_time_persisted_for_restart_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=9999)
    ws = FakeWS([("msg", _liq_msg()), ("timeout",)])
    monkeypatch.setattr(lr.websocket, "create_connection", lambda *a, **k: ws)
    r.run(duration=0.4)
    state_file = tmp_path / "_state" / "binance_liquidations.json"
    assert state_file.exists(), "last-message time was not persisted"
    state = json.loads(state_file.read_text())
    assert state.get("last_msg_ts"), f"state missing last_msg_ts: {state}"


# -- fix 1 (round 2): real protocol ping/pong keepalive --------------------

def _silent_ws_server():
    """Minimal WebSocket server that completes the handshake then stays silent."""
    import base64
    import hashlib
    import socket
    import threading

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]
    count = [0]

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            count[0] += 1
            try:
                req = b""
                while b"\r\n\r\n" not in req:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    req += chunk
                key = ""
                for line in req.decode("utf-8", "replace").split("\r\n"):
                    if line.lower().startswith("sec-websocket-key:"):
                        key = line.split(":", 1)[1].strip()
                accept = base64.b64encode(hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                ).digest()).decode()
                conn.sendall((
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
                conn.settimeout(30)
                while True:  # silent: read and discard everything
                    if not conn.recv(4096):
                        break
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    threading.Thread(target=serve, daemon=True).start()
    return port, count


def test_ping_pong_keepalive_reconnects_when_no_pong(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    port, count = _silent_ws_server()
    r = lr.StreamRecorder("binance_liquidations", f"ws://127.0.0.1:{port}",
                          lr.parse_binance_liquidation, flush_seconds=9999,
                          ws_ping_interval=0.2, ws_ping_timeout=0.5,
                          ws_recv_timeout=0.1)
    r.run(duration=4.0)
    assert count[0] >= 2, \
        f"silent server never triggered a pong-timeout reconnect ({count[0]} connection(s))"


# -- fix 2 (round 2): Bybit staleness ~60s, pongs count as activity --------

def test_ping_interval_must_be_shorter_than_pong_timeout():
    # Otherwise the pong timeout fires before the first ping is even sent.
    rec = [r for r in lr.build_recorders(60.0) if r.name == "binance_liquidations"][0]
    assert rec.ws_ping_interval is not None
    assert rec.ws_ping_timeout is not None
    assert rec.ws_ping_interval < rec.ws_ping_timeout, \
        f"ping every {rec.ws_ping_interval}s but pong timeout {rec.ws_ping_timeout}s"


def test_bybit_staleness_timeout_is_about_60s():
    rec = [r for r in lr.build_recorders(60.0) if r.name == "bybit_liquidations"][0]
    assert rec.staleness_timeout is not None
    assert 30.0 <= rec.staleness_timeout <= 90.0, rec.staleness_timeout


def test_pong_frames_count_as_activity_for_staleness(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("bybit_liquidations", "ws://x", lr.parse_bybit_liquidation,
                          flush_seconds=9999, staleness_timeout=0.3)
    # Pongs every 0.2s with recv timeouts in between: if pongs count as
    # activity the loop survives to the deadline; otherwise it goes stale.
    script = []
    for _ in range(8):
        script.append(("pong", b""))
        script.append(("sleep", 0.2))
        script.append(("timeout",))
    ws = FakeWS(script)
    start = time.time()
    r._receive_loop(ws, time.time() + 1.2)
    elapsed = time.time() - start
    assert elapsed > 1.0, \
        f"pongs did not count as activity; loop exited after {elapsed:.2f}s"


# -- fix 3 (round 2): never overwrite a data file ---------------------------

def test_flush_never_overwrites_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=9999)
    r.buffer.append({"symbol": "BTCUSDT", "n": 1})
    r._flush()
    r.buffer.append({"symbol": "BTCUSDT", "n": 2})
    r._flush()
    files = _jsonl_files(tmp_path, "binance_liquidations")
    assert len(files) == 2, f"second flush overwrote the first ({len(files)} file(s))"
    rows = _rows_on_disk(tmp_path, "binance_liquidations")
    assert [row["n"] for row in rows] == [1, 2]


# -- fix 4 (round 2): restart gaps of any length ----------------------------

def test_restart_gap_logged_for_any_length(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    state_dir = tmp_path / "_state"
    state_dir.mkdir(parents=True)
    last = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    (state_dir / "binance_liquidations.json").write_text(
        json.dumps({"last_msg_ts": last}), encoding="utf-8")
    r = lr.StreamRecorder("binance_liquidations", "ws://x", lr.parse_binance_liquidation,
                          flush_seconds=9999)
    ws = FakeWS([("timeout",)])
    monkeypatch.setattr(lr.websocket, "create_connection", lambda *a, **k: ws)
    r.run(duration=0.3)
    gaps = _gap_rows(tmp_path)
    assert any(g["reason"] == "restart_or_crash" for g in gaps), \
        f"5s restart gap was not logged: {gaps}"


# -- fix 5 (round 2): bookTicker sampled once per clock second --------------

def test_bookticker_sampling_is_per_clock_second(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("binance_bookticker", "ws://x", lr.parse_binance_bookticker,
                          flush_seconds=9999, sample_seconds=1.0,
                          sample_key=lambda rec: rec.get("symbol"))
    # 100.9 and 100.95 are the same clock second -> one row.
    # 101.0 is the next clock second (only 0.1s later) -> a sliding 1s window
    # would drop it, clock-second sampling must keep it.
    times = iter([100.9, 100.95, 101.0])
    monkeypatch.setattr(lr.time, "time", lambda: next(times, 101.0))
    assert r._sample_emit({"symbol": "BTCUSDT", "bid_price": "1"}) is True
    assert r._sample_emit({"symbol": "BTCUSDT", "bid_price": "2"}) is False
    assert r._sample_emit({"symbol": "BTCUSDT", "bid_price": "3"}) is True
    assert [row["bid_price"] for row in r.buffer] == ["1", "3"]


def test_bybit_unrecognised_topic_is_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lr, "DATA_DIR", tmp_path)
    r = lr.StreamRecorder("bybit_liquidations", "ws://x", lr.parse_bybit_liquidation,
                          flush_seconds=9999)
    payload = {"topic": "orderbook.1.BTCUSDT", "type": "snapshot", "ts": 1, "data": {}}
    ws = FakeWS([("msg", json.dumps(payload)), ("timeout",)])
    with caplog.at_level("INFO", logger="live_recorder"):
        r._receive_loop(ws, time.time() + 0.3)
    assert any("orderbook.1.BTCUSDT" in rec.message for rec in caplog.records), \
        "unrecognised Bybit topic was not logged"
    assert _rows_on_disk(tmp_path, "bybit_liquidations") == []

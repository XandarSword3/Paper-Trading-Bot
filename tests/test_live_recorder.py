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

    def recv_data(self):
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


def test_liquidation_stream_has_no_data_staleness_timeout():
    # Quiet is normal for liquidation feeds: they rely on ping/pong, not data.
    for rec in lr.build_recorders(60.0):
        if rec.name in ("binance_liquidations", "bybit_liquidations"):
            assert rec.staleness_timeout is None, rec.name
        else:
            assert rec.staleness_timeout is not None, rec.name


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

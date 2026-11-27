"""Live recorder for Binance liquidations + order book, and Bybit liquidations.

Purpose
-------
``research/new_edge/FINDINGS.md`` lists "no historical liquidations / order
book" as the data gap that blocks several hypotheses. Binance only *pushes*
these over WebSocket — there is no historical endpoint — so the only way to
get them is to record them live, starting now.

Streams recorded
----------------
* ``binance_liquidations`` — ``!forceOrder@arr`` on
  ``wss://fstream.binance.com/market/ws``. **SAMPLED**: Binance pushes at most
  the largest liquidation per symbol per second, so this is NOT the complete
  liquidation flow. Every row carries ``sampled=true``.
* ``binance_bookticker`` — ``btcusdt/ethusdt/solusdt@bookTicker`` on
  ``wss://fstream.binance.com/public/stream`` (top-of-book; full depth diffs
  are a later upgrade).
* ``bybit_liquidations`` — ``allLiquidation`` on
  ``wss://stream.bybit.com/v5/public/linear``. Bybit pushes every liquidation
  (fuller coverage than Binance's sampled feed), stored separately so the two
  sources are never mixed.

Durability
----------
* Messages are buffered and flushed every ``--flush-seconds`` (default 60) to
  one JSONL file per stream per minute, written via tmp-file + ``os.replace``
  (atomic — a crash never leaves a half-written file).
* Every disconnect/reconnect is appended to ``gaps.jsonl`` with the gap
  duration, so missing data is always visible instead of silent.
* On startup the recorder prints the last recorded timestamp per stream so a
  restart gap is obvious.

Usage
-----
Run forever::

    python -m v4_system.data_engine.live_recorder

One-hour self-test with a forced disconnect at the 15-minute mark::

    python -m v4_system.data_engine.live_recorder --duration 3600 --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import websocket

logger = logging.getLogger("live_recorder")

V4_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = V4_ROOT / "data" / "live"

BINANCE_LIQ_URL = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
BINANCE_BOOK_URL = (
    "wss://fstream.binance.com/public/stream?streams="
    "btcusdt@bookTicker/ethusdt@bookTicker/solusdt@bookTicker"
)
BYBIT_LIQ_URL = "wss://stream.bybit.com/v5/public/linear"
# Bybit has no all-symbol liquidation topic; subscribe per symbol.
BYBIT_SUBSCRIBE = json.dumps({"op": "subscribe", "args": [
    "allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT", "allLiquidation.SOLUSDT",
]})
BYBIT_PING = json.dumps({"op": "ping"})

RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


class StreamRecorder:
    """One WebSocket stream: reconnect loop, gap log, atomic minute files."""

    def __init__(self, name: str, url: str, parse, subscribe_msg: str | None = None,
                 flush_seconds: float = 60.0, sample_seconds: float | None = None,
                 sample_key=None, ping_msg: str | None = None,
                 ping_interval: float = 20.0,
                 staleness_timeout: float | None = None,
                 ws_ping_interval: float | None = None,
                 ws_ping_timeout: float | None = None,
                 ws_recv_timeout: float = 5.0) -> None:
        self.name = name
        self.url = url
        self.parse = parse            # raw json dict -> list[dict]
        self.subscribe_msg = subscribe_msg
        self.ping_msg = ping_msg      # app-level keepalive (Bybit requires it)
        self.ping_interval = ping_interval
        self.staleness_timeout = staleness_timeout  # None = quiet is normal
        # Protocol-level ping/pong (websocket-client has no built-in keepalive:
        # create_connection ignores ping_interval/ping_timeout).
        self.ws_ping_interval = ws_ping_interval
        self.ws_ping_timeout = ws_ping_timeout
        self.ws_recv_timeout = ws_recv_timeout
        self._last_pong_ts: float | None = None
        self.flush_seconds = flush_seconds
        self.sample_seconds = sample_seconds
        self.sample_key = sample_key  # rec -> hashable key for sampling
        self._last_emit: dict = {}
        self.last_flush = time.time()  # stream-level: survives reconnects
        self.buffer: list[dict] = []
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.msg_count = 0
        self.last_msg_ts: float | None = None
        self._forced_disconnect_at: float | None = None
        self._pending_gap_start: datetime | None = None
        self._last_close: dict | None = None
        self._last_error: str | None = None

    # -- storage ---------------------------------------------------------
    def _flush_key(self) -> str:
        # Unique per flush (microsecond precision): a data file is NEVER
        # overwritten, even by two flushes inside the same second.
        return _utcnow().strftime("%Y-%m-%d/%H-%M-%S-%f")

    def _flush(self) -> None:
        with self.buffer_lock:
            records = list(self.buffer)
            self.buffer = []
        # Persist liveness state even when there is nothing to write: the
        # watchdog uses it to tell "connected but quiet" from "disconnected".
        self._persist_state()
        if not records:
            return
        flush_key = self._flush_key()
        out_dir = DATA_DIR / self.name
        final = out_dir / f"{flush_key}.jsonl"
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        logger.debug("%s: flushed %d records -> %s", self.name, len(records), final)

    def _state_path(self) -> Path:
        return DATA_DIR / "_state" / f"{self.name}.json"

    def _persist_state(self) -> None:
        """Persist last-message time so restarts/crashes are visible."""
        if self.last_msg_ts is None:
            return
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_msg_ts": _iso(datetime.fromtimestamp(self.last_msg_ts, tz=timezone.utc)),
            "updated": _iso(_utcnow()),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)

    def _report_startup(self) -> None:
        """Log the last recorded message time (docstring startup claim)."""
        path = self._state_path()
        if not path.exists():
            logger.info("%s: no previous state (first run)", self.name)
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            last = datetime.fromisoformat(state["last_msg_ts"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: unreadable state file: %s", self.name, exc)
            return
        age = (_utcnow() - last).total_seconds()
        logger.info("%s: last recorded message at %s (%.0fs ago)",
                    self.name, state["last_msg_ts"], age)
        # Any restart gap is logged, however short.
        self._log_gap(last, _utcnow(), "restart_or_crash")

    def _sample_emit(self, rec: dict) -> bool:
        """Sample at most once per clock bucket per key; append to buffer.

        Returns True if the record was emitted. Clock-second buckets mean
        bookTicker yields exactly one row per symbol per clock second.
        """
        if not (self.sample_seconds and self.sample_key):
            with self.buffer_lock:
                self.buffer.append(rec)
            return True
        key = self.sample_key(rec)
        bucket = int(time.time() // self.sample_seconds)
        with self.buffer_lock:
            if self._last_emit.get(key) == bucket:
                return False
            self._last_emit[key] = bucket
            self.buffer.append(rec)
        return True

    def _maybe_flush(self) -> None:
        if time.time() - self.last_flush >= self.flush_seconds:
            self._flush()
            self.last_flush = time.time()

    def _log_gap(self, gap_start: datetime, gap_end: datetime, reason: str,
                 details: dict | None = None) -> None:
        gaps_dir = DATA_DIR / "_gaps"
        gaps_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "stream": self.name,
            "gap_start": _iso(gap_start),
            "gap_end": _iso(gap_end),
            "seconds": round((gap_end - gap_start).total_seconds(), 3),
            "reason": reason,
        }
        if details:
            rec.update(details)
        path = gaps_dir / f"{_utcnow().strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        logger.warning("%s: GAP %.1fs (%s) %s -> %s", self.name, rec["seconds"],
                       reason, rec["gap_start"], rec["gap_end"])

    # -- main loop -------------------------------------------------------
    def run(self, duration: float | None = None, self_test: bool = False) -> None:
        self._report_startup()
        deadline = time.time() + duration if duration else None
        if self_test and duration:
            self._forced_disconnect_at = time.time() + duration * 0.25
        delay = RECONNECT_BASE_DELAY
        while not self.stop_event.is_set():
            if deadline and time.time() >= deadline:
                break
            attempt_start = _utcnow()
            connected = False
            try:
                ws = websocket.create_connection(self.url, timeout=20)
                ws.settimeout(self.ws_recv_timeout)
                if self.subscribe_msg:
                    ws.send(self.subscribe_msg)
                connected = True
                delay = RECONNECT_BASE_DELAY
                logger.info("%s: connected to %s", self.name, self.url)
                if self._pending_gap_start is not None:
                    gap_end = _utcnow()
                    if (gap_end - self._pending_gap_start).total_seconds() > 1.0:
                        details = {}
                        if self._last_close:
                            details["close_code"] = self._last_close.get("code")
                            details["close_reason"] = self._last_close.get("reason")
                        if self._last_error:
                            details["error"] = self._last_error
                        self._log_gap(self._pending_gap_start, gap_end, "disconnect",
                                      details or None)
                    self._pending_gap_start = None
                    self._last_close = None
                    self._last_error = None
                self._last_pong_ts = time.time()  # grace period after connect
                pinger_stop = threading.Event()
                pinger = None
                if self.ping_msg:
                    pinger = threading.Thread(
                        target=self._ping_loop, args=(ws, pinger_stop),
                        name=f"{self.name}-ping", daemon=True)
                    pinger.start()
                ws_pinger_stop = threading.Event()
                ws_pinger = None
                if self.ws_ping_interval:
                    ws_pinger = threading.Thread(
                        target=self._ws_ping_loop, args=(ws, ws_pinger_stop),
                        name=f"{self.name}-ws-ping", daemon=True)
                    ws_pinger.start()
                try:
                    self._receive_loop(ws, deadline)
                finally:
                    pinger_stop.set()
                    ws_pinger_stop.set()
                    if pinger:
                        pinger.join(timeout=2)
                    if ws_pinger:
                        ws_pinger.join(timeout=2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: connection error: %s", self.name, exc)
            finally:
                try:
                    # Short timeout: a silent peer must not stall the reconnect.
                    ws.close(timeout=1)  # type: ignore[name-defined]
                except Exception:  # noqa: BLE001
                    pass
            # Gap starts at the last received message (not at disconnect);
            # a healthy connection's lifetime is NOT a gap.
            if self.last_msg_ts is not None:
                self._pending_gap_start = datetime.fromtimestamp(
                    self.last_msg_ts, tz=timezone.utc)
            elif connected:
                self._pending_gap_start = _utcnow()
            elif self._pending_gap_start is None:
                self._pending_gap_start = attempt_start
            if deadline and time.time() >= deadline:
                break
            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)
        self._flush()

    def _ws_ping_loop(self, ws, stop: threading.Event) -> None:
        """Protocol-level keepalive: ws.ping() every ws_ping_interval."""
        while not stop.wait(self.ws_ping_interval):
            if stop.is_set():
                return
            try:
                ws.ping()
            except Exception:  # noqa: BLE001 — connection already gone
                return

    def _ping_loop(self, ws, stop: threading.Event) -> None:
        while not stop.is_set():
            stop.wait(self.ping_interval)
            if stop.is_set():
                return
            try:
                ws.send(self.ping_msg)
            except Exception:  # noqa: BLE001 — connection already gone
                return

    def _receive_loop(self, ws, deadline: float | None) -> None:
        try:
            self._receive_loop_inner(ws, deadline)
        finally:
            # Flush on ANY exit (error, close frame, forced disconnect,
            # deadline) so buffered rows are never stranded across a reconnect.
            self._flush()

    def _receive_loop_inner(self, ws, deadline: float | None) -> None:
        conn_last_msg = time.time()  # per-connection: staleness grace period
        while not self.stop_event.is_set():
            if deadline and time.time() >= deadline:
                return
            if self._forced_disconnect_at and time.time() >= self._forced_disconnect_at:
                logger.warning("%s: SELF-TEST forced disconnect", self.name)
                self._forced_disconnect_at = None
                return
            try:
                opcode, data = ws.recv_data(control_frame=True)
            except websocket.WebSocketTimeoutException:
                self._maybe_flush()
                if (self.staleness_timeout and conn_last_msg
                        and time.time() - conn_last_msg > self.staleness_timeout):
                    logger.warning("%s: stale (no message for %.1fs), reconnecting",
                                   self.name, time.time() - conn_last_msg)
                    return
                if (self.ws_ping_timeout and self._last_pong_ts
                        and time.time() - self._last_pong_ts > self.ws_ping_timeout):
                    logger.warning("%s: no pong for %.1fs, reconnecting",
                                   self.name, time.time() - self._last_pong_ts)
                    return
                continue
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("%s: recv error: %s", self.name, exc)
                return
            if opcode == 0x8:  # close frame: 2-byte code + reason
                code = None
                reason = ""
                if isinstance(data, (bytes, bytearray)) and len(data) >= 2:
                    code = struct.unpack(">H", data[:2])[0]
                    reason = data[2:].decode("utf-8", errors="replace")
                self._last_close = {"code": code, "reason": reason}
                logger.warning("%s: close frame code=%s reason=%r", self.name, code, reason)
                return
            if opcode == 0xA:  # pong: proof of life for the ping keepalive
                self._last_pong_ts = time.time()
                conn_last_msg = time.time()
                continue
            if opcode in (0x9,):  # ping
                try:
                    ws.pong(data)
                except Exception:  # noqa: BLE001
                    pass
                conn_last_msg = time.time()
                continue
            if not data:
                continue
            if isinstance(data, (bytes, bytearray)):
                raw = data.decode("utf-8", errors="replace")
            else:
                raw = data
            self.last_msg_ts = time.time()
            conn_last_msg = time.time()
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            # Binance combined streams wrap payloads in {"stream": ..., "data": {...}}.
            # Bybit messages carry topic/ts/data themselves and must be passed whole.
            if isinstance(msg, dict) and "stream" in msg:
                payload = msg.get("data", msg)
            else:
                payload = msg
            try:
                recs = self.parse(payload)
            except Exception as exc:  # noqa: BLE001 — a bad message must never kill the stream
                logger.warning("%s: parse error (skipped): %s", self.name, exc)
                continue
            if not recs:
                continue
            for rec in recs:
                rec["_recv_ts"] = _iso(_utcnow())
                self._sample_emit(rec)
                self.msg_count += 1
            self._maybe_flush()


# -- parsers -------------------------------------------------------------

def parse_binance_liquidation(msg: dict) -> list[dict]:
    """Binance forceOrder. SAMPLED feed: largest liquidation per symbol/sec."""
    if not isinstance(msg, dict) or msg.get("e") != "forceOrder":
        return []
    o = msg.get("o", {})
    return [{
        "source": "binance",
        "sampled": True,
        "event_type": "liquidation",
        "event_time": msg.get("E"),
        "symbol": o.get("s"),
        "side": o.get("S"),
        "order_type": o.get("o"),
        "time_in_force": o.get("f"),
        "orig_qty": o.get("q"),
        "price": o.get("p"),
        "avg_price": o.get("ap"),
        "status": o.get("X"),
        "last_filled_qty": o.get("l"),
        "filled_qty": o.get("z"),
        "trade_time": o.get("T"),
    }]


def parse_binance_bookticker(msg: dict) -> list[dict]:
    if not isinstance(msg, dict) or msg.get("e") != "bookTicker":
        return []
    return [{
        "source": "binance",
        "sampled": False,
        "event_type": "book_ticker",
        "event_time": msg.get("E"),
        "update_id": msg.get("u"),
        "symbol": msg.get("s"),
        "bid_price": msg.get("b"),
        "bid_qty": msg.get("B"),
        "ask_price": msg.get("a"),
        "ask_qty": msg.get("A"),
        "transaction_time": msg.get("T"),
    }]


def parse_bybit_liquidation(msg: dict) -> list[dict]:
    """Bybit v5 allLiquidation.{SYMBOL} — full coverage (not sampled).

    Bybit's ``data`` is a LIST of liquidation entries; one row is emitted per
    entry. ``S`` is the POSITION side: ``Buy`` means a long was liquidated.
    """
    if not isinstance(msg, dict):
        return []
    topic = msg.get("topic", "")
    if not topic.startswith("allLiquidation"):
        if topic:
            logger.info("bybit: unrecognised topic %r (skipped)", topic)
        return []
    data = msg.get("data", [])
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    rows = []
    for d in data:
        if not isinstance(d, dict):
            continue
        rows.append({
            "source": "bybit",
            "sampled": False,
            "event_type": "liquidation",
            "event_time": msg.get("ts"),
            "T": d.get("T"),
            "s": d.get("s"),
            "S": d.get("S"),
            "v": d.get("v"),
            "p": d.get("p"),
            "ts": msg.get("ts"),
        })
    return rows


def build_recorders(flush_seconds: float) -> list[StreamRecorder]:
    return [
        # Liquidation feeds are quiet by nature: no data-staleness timeout;
        # protocol ping/pong detects a dead socket instead.
        StreamRecorder("binance_liquidations", BINANCE_LIQ_URL,
                       parse_binance_liquidation, flush_seconds=flush_seconds,
                       ws_ping_interval=20.0, ws_ping_timeout=30.0),
        # bookTicker is high-frequency: a few seconds of silence means a dead
        # socket, so use a data-staleness timeout.
        StreamRecorder("binance_bookticker", BINANCE_BOOK_URL,
                       parse_binance_bookticker, flush_seconds=flush_seconds,
                       sample_seconds=1.0, sample_key=lambda r: r.get("symbol"),
                       staleness_timeout=5.0),
        StreamRecorder("bybit_liquidations", BYBIT_LIQ_URL,
                       parse_bybit_liquidation, BYBIT_SUBSCRIBE,
                       flush_seconds=flush_seconds,
                       ping_msg=BYBIT_PING, ping_interval=20.0,
                       staleness_timeout=60.0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Live liquidation/order-book recorder")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run, then exit (default: forever)")
    ap.add_argument("--self-test", action="store_true",
                    help="force a disconnect at 25%% of --duration to test reconnect")
    ap.add_argument("--flush-seconds", type=float, default=60.0)
    ap.add_argument("--streams", nargs="*", default=None,
                    help="subset of: binance_liquidations binance_bookticker bybit_liquidations")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    recorders = build_recorders(args.flush_seconds)
    if args.streams:
        wanted = set(args.streams)
        recorders = [r for r in recorders if r.name in wanted]
        unknown = wanted - {r.name for r in recorders}
        if unknown:
            raise SystemExit(f"unknown streams: {unknown}")

    threads = [threading.Thread(target=r.run, args=(args.duration, args.self_test),
                                name=r.name, daemon=True) for r in recorders]
    for t in threads:
        t.start()
    logger.info("Recorder started: %s -> %s", [r.name for r in recorders], DATA_DIR)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("Stopping (KeyboardInterrupt)")
        for r in recorders:
            r.stop_event.set()
        for t in threads:
            t.join(timeout=10)

    for r in recorders:
        logger.info("%s: %d messages recorded", r.name, r.msg_count)


if __name__ == "__main__":
    main()

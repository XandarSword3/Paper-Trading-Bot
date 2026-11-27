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
                 ping_interval: float = 20.0) -> None:
        self.name = name
        self.url = url
        self.parse = parse            # raw json dict -> record dict | None
        self.subscribe_msg = subscribe_msg
        self.ping_msg = ping_msg      # app-level keepalive (Bybit requires it)
        self.ping_interval = ping_interval
        self.flush_seconds = flush_seconds
        self.sample_seconds = sample_seconds
        self.sample_key = sample_key  # rec -> hashable key for sampling
        self._pending: dict = {}
        self._last_emit: dict = {}
        self.buffer: list[dict] = []
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.msg_count = 0
        self.last_msg_ts: float | None = None
        self._forced_disconnect_at: float | None = None
        self._pending_gap_start: datetime | None = None

    # -- storage ---------------------------------------------------------
    def _flush_key(self) -> str:
        # One file per flush (second precision): consecutive flushes inside the
        # same minute must never overwrite each other.
        return _utcnow().strftime("%Y-%m-%d/%H-%M-%S")

    def _flush(self) -> None:
        with self.buffer_lock:
            records = list(self.buffer)
            self.buffer = []
            if self.sample_seconds and self._pending:
                now = time.time()
                for key, rec in self._pending.items():
                    if now - self._last_emit.get(key, 0.0) >= self.sample_seconds:
                        records.append(rec)
                        self._last_emit[key] = now
                self._pending = {}
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

    def _log_gap(self, gap_start: datetime, gap_end: datetime, reason: str) -> None:
        gaps_dir = DATA_DIR / "_gaps"
        gaps_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "stream": self.name,
            "gap_start": _iso(gap_start),
            "gap_end": _iso(gap_end),
            "seconds": round((gap_end - gap_start).total_seconds(), 3),
            "reason": reason,
        }
        path = gaps_dir / f"{_utcnow().strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        logger.warning("%s: GAP %.1fs (%s) %s -> %s", self.name, rec["seconds"],
                       reason, rec["gap_start"], rec["gap_end"])

    # -- main loop -------------------------------------------------------
    def run(self, duration: float | None = None, self_test: bool = False) -> None:
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
                ws.settimeout(5.0)
                if self.subscribe_msg:
                    ws.send(self.subscribe_msg)
                connected = True
                delay = RECONNECT_BASE_DELAY
                logger.info("%s: connected to %s", self.name, self.url)
                if self._pending_gap_start is not None:
                    gap_end = _utcnow()
                    if (gap_end - self._pending_gap_start).total_seconds() > 1.0:
                        self._log_gap(self._pending_gap_start, gap_end, "disconnect")
                    self._pending_gap_start = None
                pinger_stop = threading.Event()
                pinger = None
                if self.ping_msg:
                    pinger = threading.Thread(
                        target=self._ping_loop, args=(ws, pinger_stop),
                        name=f"{self.name}-ping", daemon=True)
                    pinger.start()
                try:
                    self._receive_loop(ws, deadline)
                finally:
                    pinger_stop.set()
                    if pinger:
                        pinger.join(timeout=2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: connection error: %s", self.name, exc)
            finally:
                try:
                    ws.close()  # type: ignore[name-defined]
                except Exception:  # noqa: BLE001
                    pass
            # Gap starts at disconnect (or failed connect attempt) and ends at
            # the next successful connect. A healthy connection's lifetime is
            # NOT a gap.
            if connected:
                self._pending_gap_start = _utcnow()
            elif self._pending_gap_start is None:
                self._pending_gap_start = attempt_start
            if deadline and time.time() >= deadline:
                break
            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)
        self._flush()

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
        last_flush = time.time()
        while not self.stop_event.is_set():
            if deadline and time.time() >= deadline:
                return
            if self._forced_disconnect_at and time.time() >= self._forced_disconnect_at:
                logger.warning("%s: SELF-TEST forced disconnect", self.name)
                self._forced_disconnect_at = None
                return
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                if time.time() - last_flush >= self.flush_seconds:
                    self._flush()
                    last_flush = time.time()
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: recv error: %s", self.name, exc)
                return
            self.last_msg_ts = time.time()
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            # Combined streams wrap payloads in {"stream": ..., "data": {...}}
            data = msg.get("data", msg) if isinstance(msg, dict) else msg
            try:
                rec = self.parse(data)
            except Exception as exc:  # noqa: BLE001 — a bad message must never kill the stream
                logger.warning("%s: parse error (skipped): %s", self.name, exc)
                continue
            if rec is None:
                continue
            rec["_recv_ts"] = _iso(_utcnow())
            if self.sample_seconds and self.sample_key:
                with self.buffer_lock:
                    self._pending[self.sample_key(rec)] = rec
            else:
                with self.buffer_lock:
                    self.buffer.append(rec)
            self.msg_count += 1
            if time.time() - last_flush >= self.flush_seconds:
                self._flush()
                last_flush = time.time()


# -- parsers -------------------------------------------------------------

def parse_binance_liquidation(msg: dict) -> dict | None:
    """Binance forceOrder. SAMPLED feed: largest liquidation per symbol/sec."""
    if msg.get("e") != "forceOrder":
        return None
    o = msg.get("o", {})
    return {
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
    }


def parse_binance_bookticker(msg: dict) -> dict | None:
    if msg.get("e") != "bookTicker":
        return None
    return {
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
    }


def parse_bybit_liquidation(msg: dict) -> dict | None:
    """Bybit v5 allLiquidation.{SYMBOL} — full coverage (not sampled)."""
    if not isinstance(msg, dict):
        return None
    topic = msg.get("topic", "")
    if not topic.startswith("allLiquidation"):
        return None
    d = msg.get("data", {})
    return {
        "source": "bybit",
        "sampled": False,
        "event_type": "liquidation",
        "event_time": msg.get("ts"),
        "symbol": d.get("s"),
        "side": d.get("S"),
        "size": d.get("v"),
        "price": d.get("p"),
    }


def build_recorders(flush_seconds: float) -> list[StreamRecorder]:
    return [
        StreamRecorder("binance_liquidations", BINANCE_LIQ_URL,
                       parse_binance_liquidation, flush_seconds=flush_seconds),
        StreamRecorder("binance_bookticker", BINANCE_BOOK_URL,
                       parse_binance_bookticker, flush_seconds=flush_seconds,
                       sample_seconds=1.0, sample_key=lambda r: r.get("symbol")),
        StreamRecorder("bybit_liquidations", BYBIT_LIQ_URL,
                       parse_bybit_liquidation, BYBIT_SUBSCRIBE,
                       flush_seconds=flush_seconds,
                       ping_msg=BYBIT_PING, ping_interval=20.0),
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

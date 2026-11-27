"""Watchdog for the live recorder: alert when a stream stops writing.

High-frequency streams (bookTicker) must keep producing data files.
Quiet liquidation streams are judged by CONNECTION liveness instead: the
recorder persists ``_state/<stream>.json`` on every flush (every
``flush_seconds``) while connected, so a stale state file means the stream
is disconnected or the recorder is dead.

Alerts are appended to ``v4_system/data/live/_alerts.jsonl`` and printed.
Set ``LIVE_ALERT_WEBHOOK`` to a URL to also POST each alert as JSON.

Run standalone (checks every 60s):

    python -m v4_system.data_engine.live_watchdog
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("live_watchdog")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "live"
ALERTS_FILE = DATA_DIR / "_alerts.jsonl"
CHECK_INTERVAL = 60.0

# stream -> max seconds since the recorder last flushed state while connected
STREAM_THRESHOLDS = {
    "binance_bookticker": 180.0,
    "binance_liquidations": 300.0,
    "bybit_liquidations": 300.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _state_path(data_dir: Path, stream: str) -> Path:
    return data_dir / "_state" / f"{stream}.json"


def check_once(data_dir: Path = DATA_DIR,
               now: datetime | None = None) -> list[dict]:
    """Return a list of alert records (empty when everything is healthy)."""
    now = now or _utcnow()
    alerts = []
    for stream, threshold in STREAM_THRESHOLDS.items():
        path = _state_path(data_dir, stream)
        if not path.exists():
            alerts.append({
                "stream": stream,
                "kind": "no_state",
                "detail": f"no state file at {path}",
            })
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            updated = datetime.fromisoformat(state["updated"])
        except Exception as exc:  # noqa: BLE001
            alerts.append({
                "stream": stream,
                "kind": "bad_state",
                "detail": f"unreadable state file: {exc}",
            })
            continue
        age = (now - updated).total_seconds()
        if age > threshold:
            alerts.append({
                "stream": stream,
                "kind": "stale",
                "age_seconds": round(age, 1),
                "threshold_seconds": threshold,
                "detail": f"no flush for {age:.0f}s (threshold {threshold:.0f}s); "
                          "stream disconnected or recorder dead",
            })
    return alerts


def emit(alerts: list[dict]) -> None:
    if not alerts:
        return
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    for alert in alerts:
        rec = {"ts": _iso(_utcnow()), **alert}
        line = json.dumps(rec, separators=(",", ":"))
        with open(ALERTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.critical("ALERT %s", line)
        webhook = os.environ.get("LIVE_ALERT_WEBHOOK")
        if webhook:
            try:
                req = urllib.request.Request(
                    webhook, data=line.encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10)  # noqa: S310
            except Exception as exc:  # noqa: BLE001
                logger.warning("webhook alert failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("live watchdog started (data dir %s)", DATA_DIR)
    while True:
        try:
            emit(check_once())
        except Exception:  # noqa: BLE001 — watchdog must never die
            logger.exception("watchdog check failed")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

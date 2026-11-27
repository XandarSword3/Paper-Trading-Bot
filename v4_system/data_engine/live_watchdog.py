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

from v4_system.data_engine.live_recorder import DATA_DIR  # single source of truth

logger = logging.getLogger("live_watchdog")

ALERTS_FILE = DATA_DIR / "_alerts.jsonl"
CHECK_INTERVAL = 60.0

# stream -> max seconds since the recorder last flushed state while connected
STREAM_THRESHOLDS = {
    "binance_bookticker": 180.0,
    "binance_liquidations": 300.0,
    "bybit_liquidations": 300.0,
}

# High-frequency streams must also keep producing ROWS, not just flushes:
# stream -> max seconds since the newest data file was written.
NO_ROWS_THRESHOLDS = {
    "binance_bookticker": 180.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _state_path(data_dir: Path, stream: str) -> Path:
    return data_dir / "_state" / f"{stream}.json"


def _latest_data_mtime(data_dir: Path, stream: str) -> float | None:
    """Newest mtime among the stream's data files, or None if there are none."""
    stream_dir = data_dir / stream
    if not stream_dir.exists():
        return None
    mtimes = [p.stat().st_mtime for p in stream_dir.rglob("*.jsonl")]
    return max(mtimes) if mtimes else None


def check_once(data_dir: Path = DATA_DIR,
               now: datetime | None = None) -> list[dict]:
    """Return a list of alert records (empty when everything is healthy)."""
    now = now or _utcnow()
    alerts = []
    fresh_streams = set()
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
        else:
            fresh_streams.add(stream)
    # Missing data, not just disconnects: a connected high-frequency stream
    # that writes no rows for NO_ROWS_THRESHOLDS seconds is broken.
    for stream, threshold in NO_ROWS_THRESHOLDS.items():
        if stream not in fresh_streams:
            continue  # already alerted (stale / no_state / bad_state)
        latest = _latest_data_mtime(data_dir, stream)
        age = (now.timestamp() - latest) if latest is not None else None
        if age is None or age > threshold:
            alerts.append({
                "stream": stream,
                "kind": "no_rows",
                "age_seconds": round(age, 1) if age is not None else None,
                "threshold_seconds": threshold,
                "detail": "no data rows written recently although the stream "
                          "is connected (missing data, not just disconnects)",
            })
    return alerts


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def _load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, # comments, optional quotes)."""
    path = path or ENV_PATH
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def telegram_config() -> tuple[str, str] | None:
    """(token, chat_id) from env vars or .env; None when not configured."""
    env = _load_dotenv()
    token = os.environ.get("TELEGRAM_TOKEN") or env.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or env.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        return token, chat_id
    return None


def send_telegram(text: str, config: tuple[str, str] | None = None) -> bool:
    """Send a Telegram message; returns True on HTTP success."""
    config = config or telegram_config()
    if not config:
        logger.warning("Telegram not configured (TELEGRAM_TOKEN/TELEGRAM_CHAT_ID)")
        return False
    token, chat_id = config
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            logger.warning("Telegram API error: %s", body)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: %s", exc)
        return False


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
        send_telegram(f"V4 live recorder ALERT\n{line}")


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

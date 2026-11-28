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
        # Health incident: an explicit recorder error OR an unhealthy_since
        # marker (set on stale timeouts, close frames, connect failures —
        # paths that previously left state.error empty and let the watchdog
        # declare a false recovery). Both map to the SAME "error" kind so
        # repeated failures stay one incident.
        unhealthy_since = state.get("unhealthy_since")
        if state.get("error") or unhealthy_since:
            detail = (f"recorder reported error: {state['error']}"
                      if state.get("error")
                      else f"recorder unhealthy since {unhealthy_since}")
            alerts.append({
                "stream": stream,
                "kind": "error",
                "detail": detail,
            })
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
    except urllib.error.HTTPError as exc:
        # Telegram's error body carries the human-readable description
        # (e.g. "Forbidden: bot was blocked by the user"). Never contains the token.
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            detail = "<unreadable body>"
        logger.warning("Telegram HTTP %s: %s", exc.code, detail)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: %s", exc)
        return False


REMINDER_INTERVAL = 3600.0  # remind at most hourly while a problem persists
RECOVERY_STABLE_SECONDS = 300.0  # 5 continuous healthy minutes before recovery
INCIDENTS_PATH = DATA_DIR / "_watchdog_incidents.json"
PENDING_TG_PATH = DATA_DIR / "_telegram_pending.json"


def _load_pending_telegram() -> list[str]:
    try:
        raw = json.loads(PENDING_TG_PATH.read_text(encoding="utf-8"))
        return [str(t) for t in raw] if isinstance(raw, list) else []
    except Exception:  # noqa: BLE001 — missing/corrupt outbox starts empty
        return []


def _save_pending_telegram(items: list[str]) -> None:
    try:
        PENDING_TG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PENDING_TG_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items), encoding="utf-8")
        os.replace(tmp, PENDING_TG_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not persist telegram outbox: %s", exc)


def _queue_telegram(text: str) -> None:
    items = _load_pending_telegram()
    items.append(text)
    _save_pending_telegram(items)


def flush_pending_telegram() -> None:
    """Retry queued Telegram sends until they succeed (called every check)."""
    items = _load_pending_telegram()
    if not items:
        return
    remaining = [t for t in items if not send_telegram(t)]
    if len(remaining) != len(items):
        _save_pending_telegram(remaining)


class AlertState:
    """Incident-based alerting.

    An incident starts when a (stream, kind) problem first appears and ends
    only after RECOVERY_STABLE_SECONDS of continuous health — brief
    reconnections must NOT trigger recovery. While active, reminders are
    sent at most once per hour. State is persisted to disk so a watchdog
    restart neither re-alerts an ongoing incident nor fakes a recovery.
    """

    def __init__(self, reminder_interval: float = REMINDER_INTERVAL,
                 recovery_stable_seconds: float = RECOVERY_STABLE_SECONDS,
                 state_path: Path | None = None):
        self.reminder_interval = reminder_interval
        self.recovery_stable_seconds = recovery_stable_seconds
        self.state_path = state_path
        # key "stream|kind" -> {"start": ts, "last_sent": ts, "healthy_since": ts|None}
        self.incidents: dict[str, dict] = {}
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.incidents = {k: dict(v) for k, v in raw.items()}
        except Exception:  # noqa: BLE001 — corrupt state must not kill the watchdog
            self.incidents = {}

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.incidents), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not persist watchdog incident state: %s", exc)

    # -- alerting ----------------------------------------------------------
    @staticmethod
    def _key(stream: str, kind: str) -> str:
        return f"{stream}|{kind}"

    def process(self, alerts: list[dict], now_ts: float,
                recovery_ready: set[str] | None = None) -> list[dict]:
        current = {self._key(a["stream"], a["kind"]) for a in alerts}
        to_send: list[dict] = []
        for alert in alerts:
            key = self._key(alert["stream"], alert["kind"])
            inc = self.incidents.get(key)
            if inc is None:
                self.incidents[key] = {"start": now_ts, "last_sent": now_ts,
                                       "healthy_since": None}
                to_send.append({**alert, "event": "start"})
            else:
                inc["healthy_since"] = None  # problem is present again
                if now_ts - inc["last_sent"] >= self.reminder_interval:
                    to_send.append({**alert, "event": "reminder"})
                    inc["last_sent"] = now_ts
        for key in list(self.incidents):
            if key in current:
                continue
            inc = self.incidents[key]
            # The recorder has independently confirmed 300s of continuous
            # healthy ingestion: recover immediately instead of waiting a
            # second consecutive 300s alert-free period.
            if recovery_ready and key in recovery_ready:
                stream, kind = key.split("|", 1)
                to_send.append({
                    "stream": stream, "kind": kind, "event": "recovery",
                    "detail": "recorder confirmed 300s of healthy ingestion",
                })
                del self.incidents[key]
                continue
            if inc.get("healthy_since") is None:
                inc["healthy_since"] = now_ts
            elif now_ts - inc["healthy_since"] >= self.recovery_stable_seconds:
                stream, kind = key.split("|", 1)
                to_send.append({
                    "stream": stream, "kind": kind, "event": "recovery",
                    "detail": "problem resolved",
                })
                del self.incidents[key]
        self._save()
        return to_send


def confirmed_healthy_streams(data_dir: Path = DATA_DIR) -> set[str]:
    """Streams whose recorder has confirmed HEALTHY_CONFIRM_SECONDS of
    continuous connected data ingestion (state.healthy == true).

    The recorder owns the 300-second health proof; the watchdog uses this to
    recover a health incident WITHOUT imposing a second consecutive 300s
    wait on top of the recorder's own.
    """
    out = set()
    state_dir = data_dir / "_state"
    if not state_dir.exists():
        return out
    for path in sorted(state_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if state.get("healthy"):
            out.add(path.stem)
    return out


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
        text = f"V4 live recorder ALERT\n{line}"
        if not send_telegram(text):
            _queue_telegram(text)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("live watchdog started (data dir %s)", DATA_DIR)
    state = AlertState(state_path=INCIDENTS_PATH)
    while True:
        try:
            alerts = check_once()
            # Streams the recorder has proven healthy for 300s may recover
            # their health incident immediately (no second 300s wait).
            ready = {f"{s}|error" for s in confirmed_healthy_streams()}
            emit(state.process(alerts, time.time(), recovery_ready=ready))
            flush_pending_telegram()
        except Exception:  # noqa: BLE001 — watchdog must never die
            logger.exception("watchdog check failed")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

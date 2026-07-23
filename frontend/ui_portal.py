from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
LH_RESULTS_ROOT = RESULTS_ROOT / "liquidation_hunter"

KRAKEN_PRICE_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

GITHUB_USER = "XandarSword3"
GITHUB_REPO = "Paper-Trading-Bot"
GITHUB_BRANCH = "master"

STATE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/bot_state.json"
TRADES_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/trades.json"
STATE_V4_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/bot_state_v4.json"
TRADES_V4_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/trades_v4.json"


def apply_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-top: #f3f9f7;
  --bg-mid: #eef5ff;
  --bg-bottom: #ffffff;
  --ink: #152238;
  --muted: #5b6b86;
  --card: rgba(255, 255, 255, 0.85);
  --line: rgba(38, 66, 120, 0.16);
  --accent: #1f8f74;
  --accent-2: #2f6fe4;
  --danger: #c73636;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 800px at 100% -10%, rgba(47,111,228,0.10), transparent 50%),
    radial-gradient(900px 600px at -10% -20%, rgba(31,143,116,0.16), transparent 45%),
    linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 55%, var(--bg-bottom) 100%);
}

.block-container {
  padding-top: 1.3rem;
  padding-bottom: 2rem;
}

h1, h2, h3 {
  color: var(--ink);
  letter-spacing: -0.02em;
}

code, pre, .stCode {
  font-family: 'IBM Plex Mono', monospace;
}

div[data-testid="metric-container"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--card);
  box-shadow: 0 10px 30px rgba(18, 32, 61, 0.05);
  padding: 14px 16px;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
}

.portal-panel {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
  background: var(--card);
  box-shadow: 0 10px 24px rgba(18, 32, 61, 0.06);
}

.portal-kicker {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.76rem;
  font-weight: 700;
}

.portal-title {
  color: var(--ink);
  font-weight: 700;
  font-size: 1.1rem;
  margin-top: 6px;
  margin-bottom: 6px;
}

.portal-subtle {
  color: var(--muted);
  font-size: 0.92rem;
}

.portal-pill {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 10px;
  background: rgba(255,255,255,0.95);
  color: var(--ink);
  font-size: 0.78rem;
}

.portal-ok { color: var(--accent); font-weight: 700; }
.portal-warn { color: var(--danger); font-weight: 700; }

section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.82);
  border-right: 1px solid var(--line);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=25)
def load_remote_json(url: str) -> dict[str, Any] | list[Any] | None:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=20)
def get_kraken_price() -> float | None:
    try:
        response = requests.get(KRAKEN_PRICE_URL, timeout=8)
        if response.status_code != 200:
            return None
        data = response.json()
        key = [k for k in data.get("result", {}).keys() if k != "last"]
        if not key:
            return None
        return float(data["result"][key[0]]["c"][0])
    except Exception:
        return None


@st.cache_data(ttl=45)
def get_kraken_candles(interval_minutes: int, count: int = 300) -> pd.DataFrame:
    try:
        response = requests.get(
            KRAKEN_OHLC_URL,
            params={"pair": "XBTUSD", "interval": int(interval_minutes)},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        key = [k for k in payload.get("result", {}).keys() if k != "last"]
        if not key:
            return pd.DataFrame()

        rows = payload["result"][key[0]]
        frame = pd.DataFrame(
            rows,
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
        )
        frame["time"] = pd.to_datetime(frame["time"].astype(int), unit="s", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["time", "open", "high", "low", "close"]).tail(max(int(count), 50))
        return frame.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_state(local_name: str, remote_url: str) -> dict[str, Any]:
    local = _load_json(PROJECT_ROOT / local_name)
    if isinstance(local, dict):
        return local
    remote = load_remote_json(remote_url)
    if isinstance(remote, dict):
        return remote
    return {}


def load_trades(local_name: str, remote_url: str) -> list[dict[str, Any]]:
    local = _load_json(PROJECT_ROOT / local_name)
    if isinstance(local, list):
        return [x for x in local if isinstance(x, dict)]
    remote = load_remote_json(remote_url)
    if isinstance(remote, list):
        return [x for x in remote if isinstance(x, dict)]
    return []


def load_liquidation_hunter_artifacts() -> dict[str, Any]:
    summary = _load_json(LH_RESULTS_ROOT / "last_sync_summary.json")
    readiness = _load_json(LH_RESULTS_ROOT / "paper_readiness.json")
    robust = _load_json(LH_RESULTS_ROOT / "robustness_report.json")

    return {
        "summary": summary if isinstance(summary, dict) else {},
        "paper_readiness": readiness if isinstance(readiness, dict) else {},
        "robustness_report": robust if isinstance(robust, dict) else {},
        "artifacts_dir": str(LH_RESULTS_ROOT),
    }


def trade_stats(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, float]:
    exits = [t for t in trades if str(t.get("type", "")).upper() == "EXIT"]
    pnls = [float(t.get("pnl", 0.0) or 0.0) for t in exits]

    if not pnls:
        return {
            "closed_trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "final_equity": float(initial_capital),
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls)) * 100.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "closed_trades": float(len(pnls)),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "total_pnl": float(sum(pnls)),
        "max_drawdown": float(max_dd),
        "final_equity": float(equity),
    }


def equity_curve(trades: list[dict[str, Any]], initial_capital: float) -> pd.DataFrame:
    points: list[dict[str, Any]] = []
    equity = float(initial_capital)
    points.append({"time": pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30), "equity": equity})

    for t in trades:
        t_time = pd.to_datetime(t.get("time"), utc=True, errors="coerce")
        if pd.isna(t_time):
            continue
        if "equity" in t:
            equity = float(t.get("equity") or equity)
        elif str(t.get("type", "")).upper() == "EXIT":
            equity += float(t.get("pnl", 0.0) or 0.0)
        points.append({"time": t_time, "equity": equity})

    return pd.DataFrame(points).dropna(subset=["time", "equity"]).sort_values("time")


def fmt_ts(ts_text: str | None) -> str:
    if not ts_text:
        return "n/a"
    ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(ts_text)
    delta = pd.Timestamp.now(tz="UTC") - ts
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def now_utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

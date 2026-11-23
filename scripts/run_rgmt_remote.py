"""Remote daily execution runner for Regime-Gated Multi-Horizon Trend (RGMT).

Designed to run in GitHub Actions on a daily cron (00:05 UTC) or locally.
Updates paper trading state in data/rgmt_paper_state.json and writes targets to data/rgmt_targets.csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.strategies.trend_portfolio import TrendConfig, TrendPortfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rgmt_remote")

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DATA_DIR / "rgmt_paper_state.json"
TARGETS_PATH = DATA_DIR / "rgmt_targets.csv"

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT",
    "ICPUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
    "TIAUSDT", "SEIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT",
    "TONUSDT", "ENAUSDT", "AAVEUSDT", "UNIUSDT", "RENDERUSDT", "FETUSDT",
    "ARUSDT",
]

MIN_TURNOVER_24H = 5_000_000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 3.0
MAX_DRAWDOWN_GUARD = 0.30


def _okx_turnover_usd(item: dict) -> float:
    """Convert an OKX ticker's 24h volume to USD notional.

    OKX reports ``volCcy24h`` in base-currency units (BTC, PEPE, ...), not USDT.
    Comparing that raw number to a USD threshold dropped BTC from the universe,
    which in turn silently disabled the regime gate.
    """
    try:
        return float(item.get("volCcy24h") or 0.0) * float(item.get("last") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class ResilientMarketClient:
    """Direct HTTP market client fetching from Bybit with automatic OKX fallback for cloud/US runners."""

    def __init__(self, bybit_url: str = "https://api.bybit.com", okx_url: str = "https://www.okx.com") -> None:
        self.bybit_url = bybit_url
        self.okx_url = okx_url
        self.session = requests.Session()
        self.provider = "bybit"
        try:
            r = self.session.get(f"{self.bybit_url}/v5/market/tickers?category=linear&limit=1", timeout=5)
            if r.status_code == 200 and "result" in r.json():
                self.provider = "bybit"
                logger.info("Market data provider: Bybit Linear API")
            else:
                self.provider = "okx"
                logger.info("Market data provider: OKX API (Bybit returned %s)", r.status_code)
        except Exception as e:
            self.provider = "okx"
            logger.info("Market data provider: OKX API (Bybit probe failed: %s)", e)

    def _to_okx_inst(self, symbol: str) -> str:
        base = symbol.replace("USDT", "")
        if base.startswith("1000"):
            base = base[4:]
        return f"{base}-USDT-SWAP"

    def get_tickers(self) -> Dict[str, dict]:
        if self.provider == "bybit":
            try:
                url = f"{self.bybit_url}/v5/market/tickers"
                params = {"category": "linear"}
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json().get("result", {}).get("list", [])
                if data:
                    return {item["symbol"]: item for item in data if "symbol" in item}
            except Exception as e:
                logger.warning("Bybit tickers failed (%s), falling back to OKX...", e)
                self.provider = "okx"

        # OKX fallback
        url = f"{self.okx_url}/api/v5/market/tickers"
        params = {"instType": "SWAP"}
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        result = {}
        for item in data:
            inst = item.get("instId", "")
            if inst.endswith("-USDT-SWAP"):
                base = inst.replace("-USDT-SWAP", "")
                sym = f"{base}USDT"
                result[sym] = {
                    "symbol": sym,
                    "lastPrice": item.get("last", "0.0"),
                    "turnover24h": _okx_turnover_usd(item),
                }
                if base in ("PEPE", "SHIB", "FLOKI", "BONK", "SATS"):
                    result[f"1000{base}USDT"] = result[sym]
        return result

    def get_daily_klines(self, symbol: str, limit: int = 500) -> pd.Series:
        if self.provider == "bybit":
            try:
                url = f"{self.bybit_url}/v5/market/kline"
                params = {"category": "linear", "symbol": symbol, "interval": "D", "limit": limit}
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                raw = resp.json().get("result", {}).get("list", [])
                if raw:
                    df = pd.DataFrame(raw, columns=["start", "open", "high", "low", "close", "volume", "turnover"])
                    df["timestamp"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df = df.dropna(subset=["close"]).sort_values("timestamp")
                    today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
                    df = df[df["timestamp"] < today_utc]
                    if not df.empty:
                        return df.set_index("timestamp")["close"]
            except Exception as e:
                logger.warning("Bybit klines failed for %s (%s), trying OKX...", symbol, e)

        # OKX fallback
        try:
            inst_id = self._to_okx_inst(symbol)
            url = f"{self.okx_url}/api/v5/market/candles"
            params = {"instId": inst_id, "bar": "1D", "limit": min(limit, 300)}
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json().get("data", [])
            if not raw:
                return pd.Series(dtype=float)
            df = pd.DataFrame(raw, columns=["start", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df["timestamp"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp")
            today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
            df = df[df["timestamp"] < today_utc]
            if df.empty:
                return pd.Series(dtype=float)
            return df.set_index("timestamp")["close"]
        except Exception as e:
            logger.warning("OKX klines also failed for %s: %s", symbol, e)
            return pd.Series(dtype=float)


def fetch_panel(client: ResilientMarketClient, symbols: List[str]) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            s = client.get_daily_klines(sym)
            if not s.empty:
                series_map[sym] = s
        except Exception as err:
            logger.warning("Failed kline fetch for %s: %s", sym, err)

    if not series_map:
        raise RuntimeError("No candle data fetched from market data client")

    panel = pd.DataFrame(series_map).sort_index()
    logger.info("Panel fetched: %d symbols x %d daily bars (last close %s)",
                panel.shape[1], panel.shape[0], panel.index[-1].strftime("%Y-%m-%d"))
    return panel


def load_state(initial_capital: float = 10000.0) -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "strategy": "RGMT",
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "equity": initial_capital,
        "peak_equity": initial_capital,
        "realized_pnl": 0.0,
        "total_fees_paid": 0.0,
        "positions": {},
        "entry_prices": {},
        "runs": 0,
        "history": [],
        "halted": False,
        "last_run": None,
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as exc:
        logger.warning("Telegram notification failed: %s", exc)


def write_github_summary(report_md: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(report_md)
        except Exception as exc:
            logger.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Daily RGMT Runner")
    parser.add_argument("--mode", choices=["signal", "paper"], default="paper")
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()

    client = ResilientMarketClient()
    all_tickers = client.get_tickers()

    # Filter universe by liquidity
    liquid = [
        s for s in UNIVERSE
        if s in all_tickers and float(all_tickers[s].get("turnover24h", 0.0) or 0.0) >= MIN_TURNOVER_24H
    ]
    logger.info("Filtered universe: %d / %d liquid contracts (> $5M turnover)", len(liquid), len(UNIVERSE))

    # Fetch panel and compute targets
    panel = fetch_panel(client, liquid)
    if "BTCUSDT" not in panel.columns:
        logger.error(
            "BTCUSDT is missing from the fetched panel; the regime gate cannot be "
            "evaluated. Forcing a flat book (no new risk)."
        )
    cfg = TrendConfig(lookbacks=(60, 120), target_vol=0.25, max_gross=3.0)
    portfolio = TrendPortfolio(cfg)
    regime_on = portfolio.regime_active(panel)
    detail = portfolio.describe(panel)
    if "BTCUSDT" not in panel.columns and float(detail["target_weight"].abs().sum()) > 1e-12:
        raise RuntimeError(
            "Regime reference missing but non-zero targets were produced; refusing to trade"
        )
    detail.to_csv(TARGETS_PATH)

    gross_exposure = float(detail["target_weight"].abs().sum())
    net_exposure = float(detail["target_weight"].sum())
    if "BTCUSDT" not in panel.columns:
        regime_status = "OFF (BTCUSDT unavailable, forced flat)"
    else:
        regime_status = "ON (Risk Active)" if regime_on else "OFF (Cash Preservation)"

    state = load_state(args.capital)
    prices = {s: float(all_tickers[s]["lastPrice"]) for s in liquid if s in all_tickers}

    # Mark to market current positions
    unrealized = 0.0
    for sym, qty in state["positions"].items():
        px = prices.get(sym, state["entry_prices"].get(sym, 0.0))
        if qty != 0 and px > 0:
            unrealized += qty * (px - state["entry_prices"].get(sym, px))

    current_equity = state["cash"] + unrealized
    state["peak_equity"] = max(state.get("peak_equity", current_equity), current_equity)
    dd_pct = (1.0 - current_equity / max(state["peak_equity"], 1.0)) * 100.0

    # Execute paper rebalance orders
    orders: List[Tuple[str, float, float, float]] = []
    cost_rate = (FEE_BPS + SLIPPAGE_BPS) / 10000.0

    if not state.get("halted", False):
        all_syms = sorted(set(detail.index) | set(state["positions"].keys()))
        for sym in all_syms:
            target_w = float(detail.loc[sym, "target_weight"]) if sym in detail.index else 0.0
            px = prices.get(sym)
            if not px or px <= 0:
                continue

            target_qty = (target_w * current_equity) / px
            old_qty = float(state["positions"].get(sym, 0.0))
            delta_qty = target_qty - old_qty
            notional = abs(delta_qty) * px

            # Ignore rebalances under $10 or 0.05% of portfolio
            if notional < max(10.0, 0.0005 * current_equity):
                continue

            # Apply fill & fee
            fee = notional * cost_rate
            state["cash"] -= fee
            state["total_fees_paid"] += fee

            # Calculate realized PnL on reduction/flip
            old_entry = float(state["entry_prices"].get(sym, px))
            if old_qty != 0 and (old_qty > 0) != (target_qty > 0):
                realized = old_qty * (px - old_entry)
                state["realized_pnl"] += realized
                state["cash"] += realized
                state["entry_prices"][sym] = px
            elif abs(target_qty) < abs(old_qty):
                closed_qty = old_qty - target_qty
                realized = closed_qty * (px - old_entry)
                state["realized_pnl"] += realized
                state["cash"] += realized
            elif old_qty == 0:
                state["entry_prices"][sym] = px

            state["positions"][sym] = target_qty
            orders.append((sym, delta_qty, target_qty, px))

    # Recalculate equity after rebalance
    final_unrealized = sum(
        qty * (prices.get(sym, px) - state["entry_prices"].get(sym, px))
        for sym, qty in state["positions"].items()
        if qty != 0 and (px := prices.get(sym)) is not None
    )
    final_equity = state["cash"] + final_unrealized
    state["equity"] = final_equity
    state["runs"] = int(state.get("runs", 0)) + 1
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["history"].append({
        "time": state["last_run"],
        "equity": final_equity,
        "drawdown_pct": dd_pct,
        "orders": len(orders),
    })
    save_state(state)

    # Print summary
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"""
### 📊 RGMT Daily Paper Trading Report ({now_str})

* **BTC Regime Gate:** `{regime_status}`
* **Portfolio Equity:** `${final_equity:,.2f}` (Initial: `${state['initial_capital']:,.2f}`)
* **Drawdown from Peak:** `{dd_pct:.2f}%`
* **Target Gross Leverage:** `{gross_exposure:.2f}x` | **Net:** `{net_exposure:+.2f}x`
* **Rebalance Orders Placed:** `{len(orders)}`

| Metric | Value |
| :--- | :--- |
| **Realized PnL** | `${state['realized_pnl']:+,.2f}` |
| **Total Fees Paid** | `${state['total_fees_paid']:,.2f}` |
| **Active Positions** | `{len([q for q in state['positions'].values() if abs(q) > 1e-6])}` |
"""
    print(report)
    write_github_summary(report)

    # Telegram notification
    tg_msg = (
        f"📊 *RGMT Daily Rebalance Update*\n"
        f"• Status: *{regime_status}*\n"
        f"• Equity: *${final_equity:,.2f}* (DD: {dd_pct:.1f}%)\n"
        f"• Gross Leverage: *{gross_exposure:.2f}x*\n"
        f"• Orders: *{len(orders)}*\n"
        f"• Time: {now_str}"
    )
    send_telegram(tg_msg)


if __name__ == "__main__":
    main()

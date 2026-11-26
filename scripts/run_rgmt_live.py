"""Live/testnet execution adapter for Regime-Gated Multi-Horizon Trend (RGMT).

This script is deliberately hard to fire by accident:

* it is a dry run unless ``--execute`` is passed;
* even with ``--execute`` it refuses to trade unless ``ALLOW_LIVE=true``;
* it targets Binance Futures **testnet** unless ``BINANCE_TESTNET=false``;
* mainnet additionally requires ``ALLOW_MAINNET_LIVE=true``;
* a kill-switch file (``data/RGMT_LIVE_HALT``) flattens the book and stops;
* the 30% drawdown guard flattens the book and latches a halt;
* it never exceeds the frozen research limits (3x gross, 15% per name).

Usage
-----
Check credentials without trading::

    python scripts/run_rgmt_live.py --check

Print the order plan without sending anything::

    python scripts/run_rgmt_live.py

Execute on testnet::

    set ALLOW_LIVE=true
    python scripts/run_rgmt_live.py --execute

The account should be dedicated to this strategy: positions in symbols that
are not part of the target set are flattened, because a leftover directional
position would corrupt the risk budget.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.rgmt_paper import (  # noqa: E402
    MAX_DRAWDOWN_GUARD,
    MAX_GROSS,
    flatten_targets,
    validate_targets,
)
from research.strategies.trend_portfolio import TrendConfig, TrendPortfolio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rgmt_live")

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "rgmt_live_state.json"
HALT_PATH = DATA_DIR / "RGMT_LIVE_HALT"
DSR_GATE_PATH = ROOT_DIR / "data" / "RGMT_DSR_GATE.json"
DSR_MIN_THRESHOLD = 0.95

TESTNET_BASE = "https://testnet.binancefuture.com"
MAINNET_BASE = "https://fapi.binance.com"
LEVERAGE = 3
MAX_ORDERS_PER_RUN = 40
MAX_ORDER_NOTIONAL_FRACTION = 0.20
MIN_ORDER_NOTIONAL = 10.0

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT",
    "ICPUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
    "TIAUSDT", "SEIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT",
    "TONUSDT", "ENAUSDT", "AAVEUSDT", "UNIUSDT", "RENDERUSDT", "FETUSDT",
    "ARUSDT",
]


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from a local ``.env`` without overwriting.

    Keeps the live executor runnable from the repo root without exporting
    secrets by hand.  The file is gitignored.
    """
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def check_execution_gates(
    *, testnet: bool, allow_live: bool, allow_mainnet: bool, execute: bool
) -> None:
    """Fail closed unless every explicit opt-in is present.

    Kept separate from ``main`` so the gate logic can be unit-tested without a
    network or credentials.
    """
    if not testnet and not allow_mainnet:
        raise SystemExit("Refusing mainnet: set ALLOW_MAINNET_LIVE=true to override")
    if execute and not allow_live:
        raise SystemExit("Refusing to trade: set ALLOW_LIVE=true to override")


def dsr_gate_allows_mainnet(path: Path = DSR_GATE_PATH,
                           threshold: float = DSR_MIN_THRESHOLD) -> bool:
    """Fail-closed DSR gate for mainnet new exposure.

    Reads the committed gate file ``data/RGMT_DSR_GATE.json``. Returns True
    only if it parses and ``dsr >= threshold``. Missing, unreadable, or
    malformed files all return False — mainnet must never open new risk on an
    absent or stale gate.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dsr = float(payload["dsr"])
    except Exception:
        return False
    return dsr >= threshold


def apply_dsr_gate(orders: List[dict], allowed: bool) -> List[dict]:
    """When the DSR gate blocks mainnet, keep only reduce-only orders.

    Flattening and de-risking must always remain possible; only NEW exposure
    is refused.
    """
    if allowed:
        return orders
    return [order for order in orders if order.get("reduce_only")]


class BinanceFuturesClient:
    """Minimal signed REST client for Binance USDT-M futures."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = TESTNET_BASE if testnet else MAINNET_BASE
        self.testnet = testnet
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _request(self, method: str, path: str, params: Optional[dict] = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            query = urlencode(params, doseq=True)
            signature = hmac.new(
                self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
        url = f"{self.base}{path}"
        response = self.session.request(method, url, params=params, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} failed ({response.status_code}): {response.text[:500]}"
            )
        return response.json()

    def account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def positions(self) -> List[dict]:
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)

    def exchange_info(self) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request(
            "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True
        )

    def place_market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.12f}".rstrip("0").rstrip("."),
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def klines(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[list]:
        return self._request(
            "GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit}
        )


def load_state(initial_equity: float) -> dict:
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("peak_equity", initial_equity)
                payload.setdefault("halted", False)
                payload.setdefault("orders", [])
                return payload
        except Exception as exc:
            logger.warning("Could not read live state: %s", exc)
    return {
        "strategy": "RGMT-live",
        "peak_equity": float(initial_equity),
        "equity": float(initial_equity),
        "halted": False,
        "halt_reason": None,
        "orders": [],
        "last_run": None,
    }


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def fetch_panel(client: BinanceFuturesClient, symbols: List[str]) -> pd.DataFrame:
    series: Dict[str, pd.Series] = {}
    today = pd.Timestamp.now(tz="UTC").normalize()
    for symbol in symbols:
        try:
            raw = client.klines(symbol)
        except Exception as exc:
            logger.warning("Klines failed for %s: %s", symbol, exc)
            continue
        if not raw:
            continue
        frame = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["close"])
        frame = frame[frame["timestamp"] < today]
        if not frame.empty:
            series[symbol] = frame.set_index("timestamp")["close"]
    if not series:
        raise RuntimeError("no kline data fetched")
    return pd.DataFrame(series).sort_index()


def compute_targets(panel: pd.DataFrame) -> Tuple[Dict[str, float], bool, pd.DataFrame]:
    cfg = TrendConfig(
        lookbacks=(60, 120),
        target_vol=0.25,
        max_gross=MAX_GROSS,
        max_asset_weight=0.15,
        min_asset_vol=0.15,
    )
    portfolio = TrendPortfolio(cfg)
    regime_on = portfolio.regime_active(panel)
    detail = portfolio.describe(panel)
    targets = {str(sym): float(val) for sym, val in detail["target_weight"].items()}
    if not regime_on:
        targets = flatten_targets(targets)
    return targets, regime_on, detail


def parse_filters(exchange_info: dict) -> Dict[str, dict]:
    filters: Dict[str, dict] = {}
    for item in exchange_info.get("symbols", []):
        symbol = item.get("symbol")
        if not symbol:
            continue
        step = 0.001
        min_qty = 0.001
        min_notional = 5.0
        for entry in item.get("filters", []):
            kind = entry.get("filterType")
            if kind == "LOT_SIZE":
                step = float(entry.get("stepSize", step) or step)
                min_qty = float(entry.get("minQty", min_qty) or min_qty)
            elif kind in {"MIN_NOTIONAL", "NOTIONAL"}:
                min_notional = float(entry.get("notional", entry.get("minNotional", min_notional)) or min_notional)
        filters[symbol] = {"step": step, "min_qty": min_qty, "min_notional": min_notional}
    return filters


def round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def plan_orders(
    targets: Dict[str, float],
    equity: float,
    positions: Dict[str, float],
    marks: Dict[str, float],
    filters: Dict[str, dict],
    *,
    max_orders: int = MAX_ORDERS_PER_RUN,
) -> List[dict]:
    """Return a list of market orders that move the book to ``targets``."""
    orders: List[dict] = []
    for symbol in sorted(set(targets) | set(positions)):
        mark = marks.get(symbol)
        if not mark or mark <= 0:
            logger.warning("No mark price for %s; skipping", symbol)
            continue
        target_weight = float(targets.get(symbol, 0.0) or 0.0)
        target_qty = target_weight * equity / mark
        current_qty = float(positions.get(symbol, 0.0) or 0.0)
        delta = target_qty - current_qty
        if abs(delta) * mark < MIN_ORDER_NOTIONAL:
            continue

        spec = filters.get(symbol, {"step": 0.001, "min_qty": 0.001, "min_notional": 5.0})
        step = float(spec["step"])
        min_qty = float(spec["min_qty"])
        min_notional = float(spec["min_notional"])

        flip = (
            current_qty != 0.0
            and target_qty != 0.0
            and (current_qty > 0.0) != (target_qty > 0.0)
        )
        if flip:
            close_qty = round_down(abs(current_qty), step)
            open_qty = round_down(abs(target_qty), step)
            if close_qty >= min_qty and close_qty * mark >= min_notional:
                orders.append(
                    {
                        "symbol": symbol,
                        "side": "SELL" if current_qty > 0 else "BUY",
                        "quantity": close_qty,
                        "reduce_only": True,
                        "reason": "flip_close",
                    }
                )
            if open_qty >= min_qty and open_qty * mark >= min_notional:
                orders.append(
                    {
                        "symbol": symbol,
                        "side": "BUY" if target_qty > 0 else "SELL",
                        "quantity": open_qty,
                        "reduce_only": False,
                        "reason": "flip_open",
                    }
                )
            continue

        quantity = round_down(abs(delta), step)
        if quantity < min_qty or quantity * mark < min_notional:
            continue
        orders.append(
            {
                "symbol": symbol,
                "side": "BUY" if delta > 0 else "SELL",
                "quantity": quantity,
                "reduce_only": abs(target_qty) < abs(current_qty),
                "reason": "rebalance",
            }
        )

    if len(orders) > max_orders:
        raise RuntimeError(f"order plan has {len(orders)} orders, limit is {max_orders}")
    return orders


def safety_check(targets: Dict[str, float], equity: float, orders: List[dict], marks: Dict[str, float]) -> None:
    validate_targets(targets)
    gross_target = sum(abs(float(v)) for v in targets.values())
    if gross_target > MAX_GROSS + 1e-9:
        raise RuntimeError(f"target gross {gross_target:.4f} exceeds {MAX_GROSS}")
    total_notional = 0.0
    for order in orders:
        # Reduce-only orders lower risk and must never be blocked by a
        # notional cap; otherwise a small/empty account could not de-risk.
        if order.get("reduce_only"):
            continue
        mark = marks.get(order["symbol"], 0.0)
        notional = order["quantity"] * mark
        total_notional += notional
        if notional > MAX_ORDER_NOTIONAL_FRACTION * equity + 10.0:
            raise RuntimeError(
                f"order {order['symbol']} notional {notional:.2f} exceeds "
                f"{MAX_ORDER_NOTIONAL_FRACTION:.0%} of equity"
            )
    if total_notional > 4.0 * equity + 10.0:
        raise RuntimeError(f"total order notional {total_notional:.2f} exceeds 4x equity")


def _send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Telegram failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="RGMT live/testnet executor")
    parser.add_argument("--execute", action="store_true", help="Actually send orders")
    parser.add_argument("--check", action="store_true", help="Check credentials and print the account")
    parser.add_argument("--resume", action="store_true", help="Clear a latched drawdown halt")
    parser.add_argument("--max-orders", type=int, default=MAX_ORDERS_PER_RUN)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET are not set")

    testnet = _env_bool("BINANCE_TESTNET", True)
    allow_live = _env_bool("ALLOW_LIVE", False)
    allow_mainnet = _env_bool("ALLOW_MAINNET_LIVE", False)
    check_execution_gates(
        testnet=testnet,
        allow_live=allow_live,
        allow_mainnet=allow_mainnet,
        execute=args.execute,
    )

    client = BinanceFuturesClient(api_key, api_secret, testnet=testnet)
    account = client.account()
    equity = float(account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0)
    logger.info(
        "Account: %s equity=%.2f wallet=%.2f available=%.2f",
        "TESTNET" if testnet else "MAINNET",
        equity,
        float(account.get("totalWalletBalance", 0.0) or 0.0),
        float(account.get("availableBalance", 0.0) or 0.0),
    )

    positions: Dict[str, float] = {}
    marks: Dict[str, float] = {}
    for row in client.positions():
        symbol = row.get("symbol")
        if not symbol:
            continue
        if str(row.get("positionSide", "BOTH")).upper() != "BOTH":
            raise SystemExit("Hedge mode is not supported; switch the account to one-way mode")
        qty = float(row.get("positionAmt", 0.0) or 0.0)
        if abs(qty) > 1e-12:
            positions[symbol] = qty
            marks[symbol] = float(row.get("markPrice", 0.0) or 0.0)

    if args.check:
        print(json.dumps({"equity": equity, "positions": positions, "marks": marks}, indent=2))
        return

    state = load_state(equity)
    if args.resume:
        state["halted"] = False
        state["halt_reason"] = None
        state["peak_equity"] = equity
        logger.warning("Manual resume: halt cleared, peak reset to %.2f", equity)

    peak = max(float(state.get("peak_equity", equity)), equity)
    drawdown = 0.0 if peak <= 0 else max(1.0 - equity / peak, 0.0)
    if drawdown >= MAX_DRAWDOWN_GUARD and not state.get("halted"):
        state["halted"] = True
        state["halt_reason"] = f"max_drawdown_guard:{drawdown:.4f}"
        logger.error("DRAWDOWN GUARD BREACHED: %.2f%%; flattening", drawdown * 100)

    panel = fetch_panel(client, UNIVERSE)
    targets, regime_on, detail = compute_targets(panel)

    if HALT_PATH.exists() or state.get("halted"):
        reason = "kill_switch_file" if HALT_PATH.exists() else state.get("halt_reason")
        logger.error("HALT active (%s); flattening", reason)
        targets = flatten_targets(set(targets) | set(positions))
        state["halted"] = True
        state["halt_reason"] = reason

    # Prices for planning: prefer exchange mark prices, fall back to the last close.
    last_close = panel.ffill().iloc[-1]
    for symbol in set(targets) | set(positions):
        if symbol not in marks or marks[symbol] <= 0:
            if symbol in last_close and pd.notna(last_close[symbol]):
                marks[symbol] = float(last_close[symbol])

    filters = parse_filters(client.exchange_info())
    orders = plan_orders(targets, equity, positions, marks, filters, max_orders=args.max_orders)

    # DSR gate: mainnet may only de-risk while the committed gate file shows
    # DSR below threshold (or is missing/unreadable). Testnet is unaffected.
    if not testnet:
        dsr_ok = dsr_gate_allows_mainnet()
        if not dsr_ok:
            logger.error(
                "DSR GATE BLOCKED mainnet new exposure (need DSR >= %.2f in %s); "
                "only reduce-only orders will be sent",
                DSR_MIN_THRESHOLD, DSR_GATE_PATH,
            )
            orders = apply_dsr_gate(orders, allowed=False)

    safety_check(targets, equity, orders, marks)

    gross = sum(abs(v) for v in targets.values())
    logger.info(
        "Regime=%s gross_target=%.2fx equity=%.2f orders=%d",
        "ON" if regime_on else "OFF",
        gross,
        equity,
        len(orders),
    )
    for order in orders:
        notional = order["quantity"] * marks.get(order["symbol"], 0.0)
        logger.info(
            "%s %s qty=%.8f (~$%.2f) reduce_only=%s (%s)",
            order["side"],
            order["symbol"],
            order["quantity"],
            notional,
            order["reduce_only"],
            order["reason"],
        )

    if not args.execute:
        print(json.dumps({"dry_run": True, "orders": orders, "targets": targets}, indent=2))
        return

    # Set leverage only for symbols we are about to trade.
    for symbol in sorted({order["symbol"] for order in orders}):
        try:
            client.set_leverage(symbol, LEVERAGE)
        except Exception as exc:
            logger.warning("Could not set leverage for %s: %s", symbol, exc)

    filled: List[dict] = []
    for order in orders:
        try:
            response = client.place_market_order(
                order["symbol"],
                order["side"],
                order["quantity"],
                reduce_only=order["reduce_only"],
            )
            filled.append({**order, "response": response})
            logger.info("FILLED %s %s qty=%.8f", order["side"], order["symbol"], order["quantity"])
        except Exception as exc:
            logger.error("ORDER FAILED %s %s: %s", order["side"], order["symbol"], exc)
            # Do not continue opening new risk after a failure.
            break

    state["equity"] = equity
    state["peak_equity"] = max(float(state.get("peak_equity", equity)), equity)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["orders"].extend(
        {
            "time": state["last_run"],
            "symbol": item["symbol"],
            "side": item["side"],
            "quantity": item["quantity"],
            "reduce_only": item["reduce_only"],
            "reason": item["reason"],
        }
        for item in filled
    )
    state["orders"] = state["orders"][-2_000:]
    save_state(state)

    _send_telegram(
        f"📊 *RGMT LIVE rebalance*\n"
        f"• Mode: *{'TESTNET' if testnet else 'MAINNET'}*\n"
        f"• Regime: *{'ON' if regime_on else 'OFF'}*\n"
        f"• Equity: *${equity:,.2f}*\n"
        f"• Orders filled: *{len(filled)}/{len(orders)}*\n"
        f"• Gross target: *{gross:.2f}x*"
    )
    print(json.dumps({"executed": True, "filled": len(filled), "planned": len(orders)}, indent=2))


if __name__ == "__main__":
    main()

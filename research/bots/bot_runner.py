"""
Unified Bot Runner for Paper-Trading-Bot
Supports V1 Turtle (4H) and V4 Fast (1H) strategies.
Simulation Mode via Kraken Public API + readiness gate checks + Telegram notifications.
"""
import os
import sys
import json
import argparse
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

from research.validation.readiness_gate import check_gate

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("bot_runner")

KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Configuration specs for registered strategies
STRATEGY_SPECS = {
    "v1": {
        "display_name": "V1 Turtle-Donchian Strategy (4H)",
        "gate_key": "v1",
        "interval_min": 240,
        "entry_len": 40,
        "exit_len": 16,
        "atr_len": 20,
        "trail_mult": 4.0,
        "risk_pct": 0.01,
        "max_units": 4,
        "state_file": DATA_DIR / "bot_state.json",
        "trades_file": DATA_DIR / "trades.json"
    },
    "v4": {
        "display_name": "V4 High-Frequency Strategy (1H)",
        "gate_key": "v4",
        "interval_min": 60,
        "entry_len": 8,
        "exit_len": 16,
        "atr_len": 14,
        "trail_mult": 3.5,
        "risk_pct": 0.01,
        "max_units": 4,
        "state_file": DATA_DIR / "bot_state_v4.json",
        "trades_file": DATA_DIR / "trades_v4.json"
    }
}


def send_telegram(message: str):
    """Send message to Telegram"""
    if not TELEGRAM_CHAT_ID or not TELEGRAM_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def load_state(spec: dict) -> dict:
    """Load bot state from state file"""
    state_file = spec["state_file"]
    default_state = {
        "strategy": spec["gate_key"].upper(),
        "initial_capital": 1000.0,
        "equity": 1000.0,
        "position_size": 0.0,
        "position_units": [],
        "trade_count": 0,
        "last_run": None
    }
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                for key, val in default_state.items():
                    if key not in state:
                        state[key] = val
                return state
        except Exception as e:
            logger.error(f"Error loading state from {state_file}: {e}")
    return default_state


def save_state(spec: dict, state: dict):
    """Save bot state to state file and database"""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state_file = spec["state_file"]
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    # Persist snapshot to DB
    try:
        from backend.models import SessionLocal, EquitySnapshot
        db = SessionLocal()
        snap = EquitySnapshot(
            strategy_id=spec["gate_key"],
            equity=float(state.get("equity", 1000.0)),
            position_size=float(state.get("position_size", 0.0)),
            position_units_json=json.dumps(state.get("position_units", [])),
            timestamp=datetime.now(timezone.utc)
        )
        db.add(snap)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Error persisting state to DB: {e}")


def load_trades(spec: dict) -> list:
    """Load trade history"""
    trades_file = spec["trades_file"]
    if trades_file.exists():
        try:
            with open(trades_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trades from {trades_file}: {e}")
    return []


def record_trade(spec: dict, trade: dict):
    """Record a trade to the trades file and database"""
    trades_file = spec["trades_file"]
    trades = load_trades(spec)
    trades.append(trade)
    with open(trades_file, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

    # Persist trade to DB
    try:
        from backend.models import SessionLocal, Trade
        db = SessionLocal()
        trade_obj = Trade(
            strategy_id=spec["gate_key"],
            trade_type=trade.get("type", "BUY_ENTRY"),
            price=float(trade.get("price", 0.0)),
            quantity=float(trade.get("size", trade.get("quantity", 0.0))),
            pnl=float(trade["pnl"]) if "pnl" in trade else None,
            reason=trade.get("reason"),
            timestamp=datetime.now(timezone.utc)
        )
        db.add(trade_obj)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Error persisting trade to DB: {e}")



def get_candles(interval_min: int, limit: int = 720) -> list:
    """Fetch candles from Kraken API"""
    try:
        response = requests.get(
            f"{KRAKEN_API}/OHLC",
            params={"pair": PAIR, "interval": interval_min},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error") and len(data["error"]) > 0:
            logger.error(f"Kraken API error: {data['error']}")
            return []
        
        result_key = list(data["result"].keys())[0]
        if result_key == "last":
            result_key = list(data["result"].keys())[1]
        
        raw_candles = data["result"][result_key]
        candles = []
        for k in raw_candles:
            candles.append({
                'time': int(k[0]) * 1000,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[6])
            })
        return candles
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        return []


def calculate_indicators(candles: list, spec: dict) -> dict:
    """Calculate Donchian channels and ATR"""
    entry_len = spec["entry_len"]
    exit_len = spec["exit_len"]
    atr_len = spec["atr_len"]

    if len(candles) < max(entry_len, atr_len) + 1:
        return None

    entry_highs = [c['high'] for c in candles[-(entry_len + 1):-1]]
    entry_high = max(entry_highs)

    exit_lows = [c['low'] for c in candles[-(exit_len + 1):-1]]
    exit_low = min(exit_lows)

    trs = []
    for i in range(len(candles) - atr_len, len(candles)):
        if i < 1:
            continue
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    atr = sum(trs) / len(trs) if trs else 0

    return {
        'entry_high': entry_high,
        'exit_low': exit_low,
        'atr': atr,
        'current_price': candles[-1]['close'],
        'prev_high': candles[-2]['high'],
        'prev_low': candles[-2]['low']
    }


def simulate_order(side: str, quantity: float, current_price: float) -> dict:
    """Simulate order execution with 0.05% slippage"""
    slippage = 0.0005
    fill_price = current_price * (1 + slippage) if side == "BUY" else current_price * (1 - slippage)
    logger.info(f"[SIMULATED] {side} {quantity:.5f} BTC @ ${fill_price:,.2f}")
    return {'price': fill_price, 'quantity': quantity}


def run_strategy(strat_key: str):
    """Run bot cycle for a single strategy"""
    if strat_key not in STRATEGY_SPECS:
        logger.error(f"Unknown strategy key: {strat_key}")
        return

    spec = STRATEGY_SPECS[strat_key]
    now = datetime.now(timezone.utc)

    logger.info("=" * 60)
    logger.info(f"RUNNING BOT: {spec['display_name']} - {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    ready, gate_reason = check_gate(spec["gate_key"])
    if not ready:
        logger.info(f"[{spec['gate_key'].upper()}] GATE BLOCKED — log-only mode: {gate_reason}")
    else:
        logger.info(f"[{spec['gate_key'].upper()}] Gate passed: {gate_reason}")

    state = load_state(spec)
    candles = get_candles(spec["interval_min"])
    if not candles:
        logger.error("No candle data received. Aborting strategy run.")
        return

    ind = calculate_indicators(candles, spec)
    if not ind:
        logger.error("Failed to calculate indicators.")
        return

    current_price = ind['current_price']
    atr = ind['atr']
    entry_high = ind['entry_high']
    exit_low = ind['exit_low']

    position_size = state["position_size"]
    units = state["position_units"]
    equity = state["equity"]

    # Market state summary
    logger.info(f"Price: ${current_price:,.2f} | Entry High: ${entry_high:,.2f} | Exit Low: ${exit_low:,.2f} | ATR: ${atr:,.2f}")

    if not ready:
        save_state(spec, state)
        return

    # Entry / Pyramiding logic
    if position_size == 0 and current_price > entry_high:
        risk_amount = equity * spec["risk_pct"]
        stop_dist = spec["trail_mult"] * atr
        size = risk_amount / stop_dist if stop_dist > 0 else 0
        
        if size > 0:
            order = simulate_order("BUY", size, current_price)
            state["position_size"] = order["quantity"]
            state["position_units"] = [{
                "price": order["price"],
                "size": order["quantity"],
                "time": now.isoformat(),
                "stop": order["price"] - (spec["trail_mult"] * atr)
            }]
            trade = {
                "id": len(load_trades(spec)) + 1,
                "type": "BUY_ENTRY",
                "price": order["price"],
                "size": order["quantity"],
                "time": now.isoformat(),
                "reason": f"Breakout above ${entry_high:,.2f}"
            }
            record_trade(spec, trade)
            send_telegram(f"🚀 <b>[{spec['gate_key'].upper()}] BUY ENTRY</b>\nSize: {order['quantity']:.5f} BTC @ ${order['price']:,.2f}")

    elif position_size > 0:
        # Check Trailing Stop or Channel Exit
        highest_stop = max([u.get("stop", 0) for u in units]) if units else 0
        
        if current_price < exit_low or current_price < highest_stop:
            reason = "Channel Exit" if current_price < exit_low else "Trailing Stop"
            order = simulate_order("SELL", position_size, current_price)
            
            pnl = sum([(order["price"] - u["price"]) * u["size"] for u in units])
            state["equity"] += pnl
            state["position_size"] = 0
            state["position_units"] = []
            state["trade_count"] += 1

            trade = {
                "id": len(load_trades(spec)) + 1,
                "type": "SELL_EXIT",
                "price": order["price"],
                "size": order["quantity"],
                "pnl": pnl,
                "time": now.isoformat(),
                "reason": f"{reason} triggered"
            }
            record_trade(spec, trade)
            send_telegram(f"🛑 <b>[{spec['gate_key'].upper()}] EXIT</b>\nSize: {order['quantity']:.5f} BTC @ ${order['price']:,.2f}\nPnL: ${pnl:+,.2f}")

    save_state(spec, state)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Bot Runner")
    parser.add_argument("--strategy", type=str, choices=["v1", "v4", "all"], default="all",
                        help="Strategy to run (v1, v4, or all)")
    args = parser.parse_args()

    if args.strategy == "all":
        run_strategy("v4")
        run_strategy("v1")
    else:
        run_strategy(args.strategy)


if __name__ == "__main__":
    main()

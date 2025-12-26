"""
GitHub Actions Trading Bot - V4 High-Frequency Strategy
SIMULATION MODE: Uses Kraken public API (1H candles, ~1 trade/day)
V4 Optimal: Entry=8, Exit=16, Trail=3.5x ATR → 1572% backtest return
"""
import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# === CONFIGURATION ===
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"

# Telegram Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8489311506:AAGyZli23sqDU6D8_VD_TJw6cq_XT0EdgL0")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7599276205")

# V4 Strategy Parameters (optimized - 1572% backtest return)
ENTRY_LEN = 8       # 8-hour breakout (vs V1's 40 on 4H)
EXIT_LEN = 16       # 16-hour exit channel
ATR_LEN = 14        # ATR period
TRAIL_MULT = 3.5    # Trailing stop multiplier (vs V1's 4.0)
RISK_PCT = 0.01     # 1% risk per trade
MAX_UNITS = 4       # Max pyramid units

# File paths (separate from V1)
STATE_FILE = Path("bot_state_v4.json")
TRADES_FILE = Path("trades_v4.json")


# === TELEGRAM NOTIFICATIONS ===
def send_telegram(message):
    """Send message to Telegram"""
    if not TELEGRAM_CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")


# === STATE MANAGEMENT ===
def load_state():
    """Load bot state from JSON file"""
    default_state = {
        "strategy": "V4",
        "initial_capital": 1000.0,
        "equity": 1000.0,
        "position_size": 0.0,
        "position_units": [],
        "trade_count": 0,
        "last_run": None
    }
    
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for key, val in default_state.items():
                    if key not in state:
                        state[key] = val
                return state
        except Exception as e:
            print(f"Error loading state: {e}")
    
    return default_state


def save_state(state):
    """Save bot state to JSON file"""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_trades():
    """Load trade history"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def save_trade(trade):
    """Append trade to history"""
    trades = load_trades()
    trades.append(trade)
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


# === MARKET DATA (Kraken Public API - 1H candles) ===
def get_candles(limit=720):
    """
    Fetch 1H candles from Kraken public API.
    Interval 60 = 1 hour (in minutes).
    """
    try:
        response = requests.get(
            f"{KRAKEN_API}/OHLC",
            params={
                "pair": PAIR,
                "interval": 60  # 1 hour in minutes (vs V1's 240 for 4H)
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("error") and len(data["error"]) > 0:
            print(f"Kraken API error: {data['error']}")
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
        print(f"Failed to fetch candles: {e}")
        return []


def get_price():
    """Get current BTC price from Kraken"""
    try:
        response = requests.get(
            f"{KRAKEN_API}/Ticker",
            params={"pair": PAIR},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("error") and len(data["error"]) > 0:
            return None
        
        result_key = list(data["result"].keys())[0]
        return float(data["result"][result_key]["c"][0])
    except Exception as e:
        print(f"Failed to get price: {e}")
        return None


# === SIMULATED TRADING ===
def simulate_order(side, quantity, current_price):
    """Simulate a market order with 0.05% slippage"""
    slippage = 0.0005
    if side == "BUY":
        fill_price = current_price * (1 + slippage)
    else:
        fill_price = current_price * (1 - slippage)
    
    print(f"[V4 SIMULATED] {side} {quantity:.5f} BTC @ ${fill_price:,.2f}")
    
    return {'price': fill_price, 'quantity': quantity}


# === INDICATORS ===
def calculate_indicators(candles):
    """Calculate Donchian channels and ATR for V4 (shorter periods)"""
    if len(candles) < max(ENTRY_LEN, EXIT_LEN, ATR_LEN) + 5:
        return None
    
    # Entry high (8-period max of highs) - use previous bars
    entry_highs = [c['high'] for c in candles[-(ENTRY_LEN+1):-1]]
    entry_high = max(entry_highs)
    
    # Exit low (16-period min of lows) - use previous bars
    exit_lows = [c['low'] for c in candles[-(EXIT_LEN+1):-1]]
    exit_low = min(exit_lows)
    
    # ATR calculation
    trs = []
    for i in range(len(candles) - ATR_LEN, len(candles)):
        if i < 1:
            continue
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i-1]['close']
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    
    atr = sum(trs) / len(trs) if trs else 0
    
    return {
        'entry_high': entry_high,
        'exit_low': exit_low,
        'atr': atr,
        'current_price': candles[-1]['close'],
        'current_high': candles[-1]['high'],
        'current_low': candles[-1]['low'],
        'prev_high': candles[-2]['high'],
        'prev_low': candles[-2]['low']
    }


# === MAIN BOT LOGIC ===
def run_bot():
    """Main V4 bot execution"""
    now = datetime.now(timezone.utc)
    print("=" * 70)
    print(f"V4 HIGH-FREQUENCY BOT - {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print("[V4] 1H timeframe | Entry=8 | Exit=16 | Trail=3.5x ATR")
    print("[SIMULATION MODE] Using Kraken prices, simulated trades")
    
    # Load state
    state = load_state()
    print(f"\nLoaded: Equity=${state['equity']:,.2f}, Position={state['position_size']:.5f} BTC")
    
    # Fetch candles
    print("\nFetching 1H data from Kraken...")
    candles = get_candles()
    if not candles:
        print("ERROR: Failed to fetch candles")
        save_state(state)
        return
    
    print(f"Got {len(candles)} hourly candles")
    
    # Calculate indicators
    indicators = calculate_indicators(candles)
    if not indicators:
        print("ERROR: Not enough data for indicators")
        save_state(state)
        return
    
    print(f"\nMarket:")
    print(f"  Price: ${indicators['current_price']:,.2f}")
    print(f"  ATR: ${indicators['atr']:,.2f}")
    print(f"  Entry High ({ENTRY_LEN}): ${indicators['entry_high']:,.2f}")
    print(f"  Exit Low ({EXIT_LEN}): ${indicators['exit_low']:,.2f}")
    
    # Trading logic
    position_size = state['position_size']
    position_units = state['position_units']
    equity = state['equity']
    price = indicators['current_price']
    
    # === CHECK EXITS ===
    if position_size > 0:
        exit_triggered = False
        exit_reason = ""
        
        # Donchian exit
        if indicators['current_low'] < indicators['exit_low']:
            exit_triggered = True
            exit_reason = "Donchian Exit"
        
        # Trailing stop
        if not exit_triggered:
            for unit in position_units:
                if price <= unit.get('trailing_stop', 0):
                    exit_triggered = True
                    exit_reason = "Trailing Stop"
                    break
        
        if exit_triggered:
            print(f"\n>>> V4 EXIT: {exit_reason}")
            order = simulate_order("SELL", position_size, price)
            
            total_cost = sum(u['entry_price'] * u['size'] for u in position_units)
            pnl = order['price'] * position_size - total_cost
            equity += pnl
            
            print(f"P&L: ${pnl:,.2f}")
            
            save_trade({
                'type': 'EXIT', 'reason': exit_reason,
                'time': now.isoformat(), 'price': order['price'],
                'size': position_size, 'pnl': pnl, 'equity': equity,
                'strategy': 'V4'
            })
            
            # Telegram notification
            emoji = "🟢" if pnl > 0 else "🔴"
            send_telegram(
                f"{emoji} <b>[V4] EXIT - {exit_reason}</b>\n\n"
                f"📍 Price: ${order['price']:,.2f}\n"
                f"📊 Size: {position_size:.5f} BTC\n"
                f"💰 P&L: ${pnl:+,.2f}\n"
                f"💼 Equity: ${equity:,.2f}"
            )
            
            position_size = 0.0
            position_units = []
            state['trade_count'] += 1
    
    # === CHECK ENTRIES ===
    if len(position_units) < MAX_UNITS:
        entry_signal = False
        entry_reason = ""
        
        # Breakout: current high > entry_high
        if indicators['current_high'] > indicators['entry_high']:
            if position_units:
                last_entry = position_units[-1]['entry_price']
                if price >= last_entry + indicators['atr']:
                    entry_signal = True
                    entry_reason = "Pyramid"
            else:
                entry_signal = True
                entry_reason = "Breakout"
        
        if entry_signal:
            print(f"\n>>> V4 ENTRY: {entry_reason}")
            
            risk = equity * RISK_PCT
            stop_dist = TRAIL_MULT * indicators['atr']
            unit_size = max(0.0001, min(risk / stop_dist, 0.1))
            
            order = simulate_order("BUY", unit_size, price)
            trailing_stop = order['price'] - stop_dist
            
            position_units.append({
                'entry_price': order['price'],
                'size': order['quantity'],
                'trailing_stop': trailing_stop,
                'time': now.isoformat()
            })
            position_size += order['quantity']
            state['trade_count'] += 1
            
            print(f"Stop: ${trailing_stop:,.2f}")
            
            save_trade({
                'type': 'ENTRY', 'reason': entry_reason,
                'time': now.isoformat(), 'price': order['price'],
                'size': order['quantity'], 'trailing_stop': trailing_stop,
                'equity': equity, 'strategy': 'V4'
            })
            
            # Telegram notification
            send_telegram(
                f"📈 <b>[V4] ENTRY - {entry_reason}</b>\n\n"
                f"📍 Price: ${order['price']:,.2f}\n"
                f"📊 Size: {order['quantity']:.5f} BTC\n"
                f"🛑 Stop: ${trailing_stop:,.2f}\n"
                f"📦 Units: {len(position_units)}/{MAX_UNITS}\n"
                f"💼 Equity: ${equity:,.2f}"
            )
    
    # Update trailing stops
    for unit in position_units:
        new_stop = price - (TRAIL_MULT * indicators['atr'])
        if new_stop > unit.get('trailing_stop', 0):
            unit['trailing_stop'] = new_stop
    
    # Save state
    state['position_size'] = position_size
    state['position_units'] = position_units
    state['equity'] = equity
    save_state(state)
    
    print(f"\n{'='*70}")
    print(f"[V4] Equity: ${equity:,.2f} | Position: {position_size:.5f} BTC | Trades: {state['trade_count']}")
    print("=" * 70)


if __name__ == "__main__":
    run_bot()

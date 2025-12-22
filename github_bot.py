"""
GitHub Actions Trading Bot - V1 Turtle-Donchian Strategy
SIMULATION MODE: Uses Kraken public API (works globally, no geo-restrictions)
Simulates trades internally (no exchange account needed)
"""
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# === CONFIGURATION ===
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"  # BTC/USD on Kraken

# V1 Strategy Parameters (optimized from backtesting)
ENTRY_LEN = 40
EXIT_LEN = 16
ATR_LEN = 20
TRAIL_MULT = 4.0
RISK_PCT = 0.01
MAX_UNITS = 4

# File paths
STATE_FILE = Path("bot_state.json")
TRADES_FILE = Path("trades.json")


# === STATE MANAGEMENT ===
def load_state():
    """Load bot state from JSON file"""
    default_state = {
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


# === MARKET DATA (Kraken Public API) ===
def get_candles(limit=720):
    """
    Fetch 4H candles from Kraken public API.
    Kraken returns up to 720 candles per request.
    Interval 240 = 4 hours (in minutes).
    """
    try:
        response = requests.get(
            f"{KRAKEN_API}/OHLC",
            params={
                "pair": PAIR,
                "interval": 240  # 4 hours in minutes
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("error") and len(data["error"]) > 0:
            print(f"Kraken API error: {data['error']}")
            return []
        
        # Kraken returns data in format: [time, open, high, low, close, vwap, volume, count]
        # The key is like "XXBTZUSD" (with X prefix for crypto)
        result_key = list(data["result"].keys())[0]
        if result_key == "last":
            result_key = list(data["result"].keys())[1]
        
        raw_candles = data["result"][result_key]
        
        candles = []
        for k in raw_candles:
            candles.append({
                'time': int(k[0]) * 1000,  # Convert to milliseconds
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
        return float(data["result"][result_key]["c"][0])  # Last trade price
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
    
    print(f"[SIMULATED] {side} {quantity:.5f} BTC @ ${fill_price:,.2f}")
    
    return {'price': fill_price, 'quantity': quantity}


# === INDICATORS ===
def calculate_indicators(candles):
    """Calculate Donchian channels and ATR"""
    if len(candles) < max(ENTRY_LEN, ATR_LEN) + 1:
        return None
    
    # Entry high (40-period max of highs) - use previous bars
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
        'prev_high': candles[-2]['high'],
        'prev_low': candles[-2]['low']
    }


# === MAIN BOT LOGIC ===
def run_bot():
    """Main bot execution"""
    now = datetime.now(timezone.utc)
    print("=" * 70)
    print(f"TRADING BOT - {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print("[SIMULATION MODE] Using Kraken prices, simulated trades")
    
    # Load state
    state = load_state()
    print(f"\nLoaded: Equity=${state['equity']:,.2f}, Position={state['position_size']:.5f} BTC")
    
    # Fetch candles
    print("\nFetching data from Kraken...")
    candles = get_candles()
    if not candles:
        print("ERROR: Failed to fetch candles")
        save_state(state)
        return
    
    print(f"Got {len(candles)} candles")
    
    # Calculate indicators
    indicators = calculate_indicators(candles)
    if not indicators:
        print("ERROR: Not enough data for indicators")
        save_state(state)
        return
    
    print(f"\nMarket:")
    print(f"  Price: ${indicators['current_price']:,.2f}")
    print(f"  ATR: ${indicators['atr']:,.2f}")
    print(f"  Entry High (40): ${indicators['entry_high']:,.2f}")
    print(f"  Exit Low (16): ${indicators['exit_low']:,.2f}")
    
    # Trading logic
    position_size = state['position_size']
    position_units = state['position_units']
    equity = state['equity']
    price = indicators['current_price']
    
    # === CHECK EXITS ===
    if position_size > 0:
        exit_triggered = False
        exit_reason = ""
        
        if indicators['prev_low'] < indicators['exit_low']:
            exit_triggered = True
            exit_reason = "Donchian Exit"
        else:
            for unit in position_units:
                if price <= unit.get('trailing_stop', 0):
                    exit_triggered = True
                    exit_reason = "Trailing Stop"
                    break
        
        if exit_triggered:
            print(f"\n>>> EXIT: {exit_reason}")
            order = simulate_order("SELL", position_size, price)
            
            total_cost = sum(u['entry_price'] * u['size'] for u in position_units)
            pnl = order['price'] * position_size - total_cost
            equity += pnl
            
            print(f"P&L: ${pnl:,.2f}")
            
            save_trade({
                'type': 'EXIT', 'reason': exit_reason,
                'time': now.isoformat(), 'price': order['price'],
                'size': position_size, 'pnl': pnl, 'equity': equity
            })
            
            position_size = 0.0
            position_units = []
            state['trade_count'] += 1
    
    # === CHECK ENTRIES ===
    if len(position_units) < MAX_UNITS:
        entry_signal = False
        entry_reason = ""
        
        if indicators['prev_high'] > indicators['entry_high']:
            if position_units:
                last_entry = position_units[-1]['entry_price']
                if price >= last_entry + indicators['atr']:
                    entry_signal = True
                    entry_reason = "Pyramid"
            else:
                entry_signal = True
                entry_reason = "Breakout"
        
        if entry_signal:
            print(f"\n>>> ENTRY: {entry_reason}")
            
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
                'equity': equity
            })
    
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
    print(f"Equity: ${equity:,.2f} | Position: {position_size:.5f} BTC | Trades: {state['trade_count']}")
    print("=" * 70)


if __name__ == "__main__":
    run_bot()

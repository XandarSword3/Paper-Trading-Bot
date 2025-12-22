"""
GitHub Actions Trading Bot - V1 Turtle-Donchian Strategy
Uses direct HTTP requests to Binance Testnet API (avoids geo-restrictions)
Stateless design: reads state from JSON, executes, saves state back
"""
import os
import json
import hmac
import hashlib
import time
import requests
from datetime import datetime
from pathlib import Path

# === CONFIGURATION ===
TESTNET_BASE_URL = "https://testnet.binance.vision"
SYMBOL = "BTCUSDT"

# V1 Strategy Parameters
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
    state["last_run"] = datetime.utcnow().isoformat()
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


# === BINANCE API (Direct HTTP) ===
def get_signature(query_string, api_secret):
    """Generate HMAC SHA256 signature"""
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def binance_request(endpoint, method="GET", params=None, signed=False):
    """Make request to Binance testnet API"""
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
    
    url = f"{TESTNET_BASE_URL}/api/v3/{endpoint}"
    headers = {"X-MBX-APIKEY": api_key}
    
    if params is None:
        params = {}
    
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        params["signature"] = get_signature(query_string, api_secret)
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, params=params, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, params=params, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def ping():
    """Test connectivity"""
    result = binance_request("ping")
    return result is not None


def get_price():
    """Get current BTC price"""
    result = binance_request("ticker/price", params={"symbol": SYMBOL})
    if result:
        return float(result["price"])
    return None


def get_candles(limit=200):
    """Fetch 4H candles"""
    result = binance_request("klines", params={
        "symbol": SYMBOL,
        "interval": "4h",
        "limit": limit
    })
    
    if not result:
        return []
    
    candles = []
    for k in result:
        candles.append({
            'time': k[0],
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        })
    
    return candles


def get_account():
    """Get account info"""
    return binance_request("account", signed=True)


def place_order(side, quantity):
    """Place market order"""
    quantity = round(quantity, 5)
    
    result = binance_request("order", method="POST", signed=True, params={
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": quantity
    })
    
    if result:
        fill_price = float(result['fills'][0]['price']) if result.get('fills') else 0
        print(f"Order filled: {side} {quantity} BTC @ ${fill_price:,.2f}")
        return {
            'price': fill_price,
            'quantity': float(result['executedQty'])
        }
    
    return None


# === INDICATORS ===
def calculate_indicators(candles):
    """Calculate Donchian and ATR"""
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
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
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
    """Main bot execution - called once per 4H candle"""
    print("=" * 70)
    print(f"TRADING BOT - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    
    # Load state
    state = load_state()
    print(f"Loaded state: Equity=${state['equity']:,.2f}, Position={state['position_size']:.5f} BTC")
    
    # Test connectivity
    print("\nConnecting to Binance TESTNET...")
    if not ping():
        print("ERROR: Failed to connect to Binance")
        save_state(state)
        return
    print("Connected successfully!")
    
    # Fetch candles
    candles = get_candles(limit=200)
    if not candles:
        print("ERROR: Failed to fetch candles")
        save_state(state)
        return
    
    # Calculate indicators
    indicators = calculate_indicators(candles)
    if not indicators:
        print("ERROR: Not enough candles for indicators")
        save_state(state)
        return
    
    print(f"\nMarket Data:")
    print(f"  Price: ${indicators['current_price']:,.2f}")
    print(f"  ATR: ${indicators['atr']:,.2f}")
    print(f"  Entry High: ${indicators['entry_high']:,.2f}")
    print(f"  Exit Low: ${indicators['exit_low']:,.2f}")
    print(f"  Prev High: ${indicators['prev_high']:,.2f}")
    print(f"  Prev Low: ${indicators['prev_low']:,.2f}")
    
    print(f"\nCurrent State:")
    print(f"  Equity: ${state['equity']:,.2f}")
    print(f"  Position: {state['position_size']:.5f} BTC ({len(state['position_units'])} units)")
    
    # Trading logic
    position_size = state['position_size']
    position_units = state['position_units']
    equity = state['equity']
    
    # Check exits first
    if position_size > 0:
        exit_triggered = False
        exit_reason = ""
        
        # Donchian exit
        if indicators['prev_low'] < indicators['exit_low']:
            exit_triggered = True
            exit_reason = "Donchian Exit"
        # Trailing stop check
        else:
            for unit in position_units:
                if indicators['current_price'] <= unit.get('trailing_stop', 0):
                    exit_triggered = True
                    exit_reason = "Trailing Stop"
                    break
        
        if exit_triggered:
            print(f"\n>>> EXIT SIGNAL: {exit_reason}")
            
            # Close position
            order = place_order("SELL", position_size)
            
            if order:
                exit_price = order['price']
                
                # Calculate P&L
                total_cost = sum(u['entry_price'] * u['size'] for u in position_units)
                total_revenue = exit_price * position_size
                pnl = total_revenue - total_cost
                
                # Update equity
                equity += pnl
                
                print(f"Position closed: P&L ${pnl:,.2f}")
                
                # Record trade
                save_trade({
                    'type': 'EXIT',
                    'reason': exit_reason,
                    'time': datetime.utcnow().isoformat(),
                    'price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'equity': equity
                })
                
                # Reset position
                position_size = 0.0
                position_units = []
                state['trade_count'] += 1
    
    # Check entries (if not at max units)
    if len(position_units) < MAX_UNITS:
        entry_signal = False
        entry_reason = ""
        
        # Breakout entry
        if indicators['prev_high'] > indicators['entry_high']:
            # Check pyramid spacing
            if position_units:
                last_entry = position_units[-1]['entry_price']
                if indicators['current_price'] >= last_entry + indicators['atr']:
                    entry_signal = True
                    entry_reason = "Pyramid Add"
            else:
                entry_signal = True
                entry_reason = "Breakout Entry"
        
        if entry_signal:
            print(f"\n>>> ENTRY SIGNAL: {entry_reason}")
            
            # Calculate position size (1% risk)
            risk_amount = equity * RISK_PCT
            unit_size = risk_amount / (TRAIL_MULT * indicators['atr'])
            unit_size = max(0.001, min(unit_size, 0.1))  # Clamp to reasonable range
            
            print(f"Unit size: {unit_size:.5f} BTC (${unit_size * indicators['current_price']:,.2f})")
            
            # Place order
            order = place_order("BUY", unit_size)
            
            if order:
                entry_price = order['price']
                actual_size = order['quantity']
                
                # Add unit with trailing stop
                trailing_stop = entry_price - (TRAIL_MULT * indicators['atr'])
                position_units.append({
                    'entry_price': entry_price,
                    'size': actual_size,
                    'trailing_stop': trailing_stop,
                    'time': datetime.utcnow().isoformat()
                })
                
                position_size += actual_size
                state['trade_count'] += 1
                
                print(f"Entry: {actual_size:.5f} BTC @ ${entry_price:,.2f}")
                print(f"Trailing stop: ${trailing_stop:,.2f}")
                
                # Record trade
                save_trade({
                    'type': 'ENTRY',
                    'reason': entry_reason,
                    'time': datetime.utcnow().isoformat(),
                    'price': entry_price,
                    'size': actual_size,
                    'trailing_stop': trailing_stop,
                    'equity': equity
                })
    
    # Update trailing stops
    if position_units:
        for unit in position_units:
            new_stop = indicators['current_price'] - (TRAIL_MULT * indicators['atr'])
            if new_stop > unit.get('trailing_stop', 0):
                unit['trailing_stop'] = new_stop
    
    # Save state
    state['position_size'] = position_size
    state['position_units'] = position_units
    state['equity'] = equity
    save_state(state)
    
    print(f"\nFinal State:")
    print(f"  Equity: ${equity:,.2f}")
    print(f"  Position: {position_size:.5f} BTC")
    print(f"  Units: {len(position_units)}")
    print("=" * 70)


if __name__ == "__main__":
    run_bot()

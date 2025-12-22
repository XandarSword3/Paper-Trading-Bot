"""
GitHub Actions Trading Bot - V1 Turtle-Donchian Strategy
SIMULATION MODE: Uses real price data from Binance PUBLIC API
Simulates trades internally (no exchange account needed)
Works from any location including GitHub Actions runners
"""
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# === CONFIGURATION ===
# Binance PUBLIC API - works globally, no auth needed
BINANCE_API = "https://api.binance.com/api/v3"
SYMBOL = "BTCUSDT"

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


# === MARKET DATA (Public API - No Auth) ===
def get_price():
    """Get current BTC price from public API"""
    try:
        response = requests.get(
            f"{BINANCE_API}/ticker/price",
            params={"symbol": SYMBOL},
            timeout=10
        )
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        print(f"Failed to get price: {e}")
        return None


def get_candles(limit=200):
    """Fetch 4H candles from public API"""
    try:
        response = requests.get(
            f"{BINANCE_API}/klines",
            params={
                "symbol": SYMBOL,
                "interval": "4h",
                "limit": limit
            },
            timeout=30
        )
        response.raise_for_status()
        
        candles = []
        for k in response.json():
            candles.append({
                'time': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        
        return candles
    except Exception as e:
        print(f"Failed to fetch candles: {e}")
        return []


# === SIMULATED TRADING ===
def simulate_order(side, quantity, current_price):
    """
    Simulate a market order execution.
    Uses current price with small slippage simulation.
    """
    # Simulate 0.05% slippage
    slippage = 0.0005
    if side == "BUY":
        fill_price = current_price * (1 + slippage)
    else:
        fill_price = current_price * (1 - slippage)
    
    print(f"[SIMULATED] Order: {side} {quantity:.5f} BTC @ ${fill_price:,.2f}")
    
    return {
        'price': fill_price,
        'quantity': quantity
    }


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
    now = datetime.now(timezone.utc)
    print("=" * 70)
    print(f"TRADING BOT - {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print("[SIMULATION MODE] Using real prices, simulated trades")
    
    # Load state
    state = load_state()
    print(f"\nLoaded state: Equity=${state['equity']:,.2f}, Position={state['position_size']:.5f} BTC")
    
    # Fetch candles
    print("\nFetching market data from Binance...")
    candles = get_candles(limit=200)
    if not candles:
        print("ERROR: Failed to fetch candles")
        save_state(state)
        return
    
    print(f"Fetched {len(candles)} candles successfully")
    
    # Calculate indicators
    indicators = calculate_indicators(candles)
    if not indicators:
        print("ERROR: Not enough candles for indicators")
        save_state(state)
        return
    
    print(f"\nMarket Data:")
    print(f"  Current Price: ${indicators['current_price']:,.2f}")
    print(f"  ATR (20): ${indicators['atr']:,.2f}")
    print(f"  Entry High (40): ${indicators['entry_high']:,.2f}")
    print(f"  Exit Low (16): ${indicators['exit_low']:,.2f}")
    print(f"  Prev Candle High: ${indicators['prev_high']:,.2f}")
    print(f"  Prev Candle Low: ${indicators['prev_low']:,.2f}")
    
    print(f"\nCurrent Position:")
    print(f"  Equity: ${state['equity']:,.2f}")
    print(f"  Position: {state['position_size']:.5f} BTC ({len(state['position_units'])} units)")
    
    # Trading logic
    position_size = state['position_size']
    position_units = state['position_units']
    equity = state['equity']
    current_price = indicators['current_price']
    
    # === CHECK EXITS FIRST ===
    if position_size > 0:
        exit_triggered = False
        exit_reason = ""
        
        # Donchian exit: previous low broke below exit channel
        if indicators['prev_low'] < indicators['exit_low']:
            exit_triggered = True
            exit_reason = "Donchian Exit (prev low < exit channel)"
        
        # Trailing stop check
        if not exit_triggered:
            for unit in position_units:
                if current_price <= unit.get('trailing_stop', 0):
                    exit_triggered = True
                    exit_reason = f"Trailing Stop (price ${current_price:,.2f} <= stop ${unit['trailing_stop']:,.2f})"
                    break
        
        if exit_triggered:
            print(f"\n>>> EXIT SIGNAL: {exit_reason}")
            
            # Simulate closing position
            order = simulate_order("SELL", position_size, current_price)
            exit_price = order['price']
            
            # Calculate P&L
            total_cost = sum(u['entry_price'] * u['size'] for u in position_units)
            total_revenue = exit_price * position_size
            pnl = total_revenue - total_cost
            pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Update equity
            equity += pnl
            
            print(f"Position closed: P&L ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"New equity: ${equity:,.2f}")
            
            # Record trade
            save_trade({
                'type': 'EXIT',
                'reason': exit_reason,
                'time': now.isoformat(),
                'price': exit_price,
                'size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'equity': equity
            })
            
            # Reset position
            position_size = 0.0
            position_units = []
            state['trade_count'] += 1
    
    # === CHECK ENTRIES ===
    if len(position_units) < MAX_UNITS:
        entry_signal = False
        entry_reason = ""
        
        # Breakout entry: previous high broke above entry channel
        if indicators['prev_high'] > indicators['entry_high']:
            # Check pyramid spacing (need 1 ATR above last entry)
            if position_units:
                last_entry = position_units[-1]['entry_price']
                if current_price >= last_entry + indicators['atr']:
                    entry_signal = True
                    entry_reason = f"Pyramid Add (price >= last entry + ATR)"
            else:
                entry_signal = True
                entry_reason = "Breakout Entry (prev high > entry channel)"
        
        if entry_signal:
            print(f"\n>>> ENTRY SIGNAL: {entry_reason}")
            
            # Calculate position size (1% risk per unit)
            risk_amount = equity * RISK_PCT
            stop_distance = TRAIL_MULT * indicators['atr']
            unit_size = risk_amount / stop_distance
            
            # Clamp to reasonable BTC size
            unit_size = max(0.0001, min(unit_size, 0.1))
            
            # Check if we can afford it
            unit_value = unit_size * current_price
            if unit_value > equity * 0.25:  # Max 25% of equity per unit
                unit_size = (equity * 0.25) / current_price
                unit_size = max(0.0001, unit_size)
            
            print(f"Unit size: {unit_size:.5f} BTC (${unit_size * current_price:,.2f})")
            
            # Simulate order
            order = simulate_order("BUY", unit_size, current_price)
            entry_price = order['price']
            actual_size = order['quantity']
            
            # Calculate trailing stop
            trailing_stop = entry_price - stop_distance
            
            # Add unit
            position_units.append({
                'entry_price': entry_price,
                'size': actual_size,
                'trailing_stop': trailing_stop,
                'time': now.isoformat()
            })
            
            position_size += actual_size
            state['trade_count'] += 1
            
            print(f"Entry: {actual_size:.5f} BTC @ ${entry_price:,.2f}")
            print(f"Trailing stop: ${trailing_stop:,.2f}")
            
            # Record trade
            save_trade({
                'type': 'ENTRY',
                'reason': entry_reason,
                'time': now.isoformat(),
                'price': entry_price,
                'size': actual_size,
                'trailing_stop': trailing_stop,
                'equity': equity
            })
    
    # === UPDATE TRAILING STOPS ===
    if position_units:
        stops_updated = 0
        for unit in position_units:
            new_stop = current_price - (TRAIL_MULT * indicators['atr'])
            if new_stop > unit.get('trailing_stop', 0):
                unit['trailing_stop'] = new_stop
                stops_updated += 1
        
        if stops_updated > 0:
            print(f"\nUpdated {stops_updated} trailing stop(s)")
    
    # === SAVE STATE ===
    state['position_size'] = position_size
    state['position_units'] = position_units
    state['equity'] = equity
    save_state(state)
    
    # Summary
    unrealized_pnl = 0
    if position_size > 0:
        total_cost = sum(u['entry_price'] * u['size'] for u in position_units)
        current_value = current_price * position_size
        unrealized_pnl = current_value - total_cost
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Equity: ${equity:,.2f}")
    print(f"  Position: {position_size:.5f} BTC")
    print(f"  Units: {len(position_units)}/{MAX_UNITS}")
    if position_size > 0:
        print(f"  Unrealized P&L: ${unrealized_pnl:,.2f}")
    print(f"  Total Trades: {state['trade_count']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_bot()

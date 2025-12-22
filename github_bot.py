"""
GitHub Actions Trading Bot - V1 Turtle-Donchian Strategy
Stateless design: reads state from JSON, executes, saves state back
Runs every 4 hours via GitHub Actions cron
"""
import os
import json
from datetime import datetime
from pathlib import Path
from binance.client import Client
from binance.exceptions import BinanceAPIException

# File paths for persistent state
STATE_FILE = Path("bot_state.json")
TRADES_FILE = Path("trades.json")


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
                # Merge with defaults for any missing keys
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


def get_client():
    """Create Binance client from environment variables"""
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    testnet = os.environ.get("BINANCE_TESTNET", "true").lower() in ("true", "1", "yes")
    
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
    
    if testnet:
        print("Connected to Binance TESTNET")
        # Use explicit testnet URLs to avoid geo-restrictions
        return Client(
            api_key, 
            api_secret, 
            testnet=True,
            tld='us'  # Use .us domain which supports testnet globally
        )
    else:
        print("WARNING: Connected to MAINNET!")
        return Client(api_key, api_secret)


def get_candles(client, limit=200):
    """Fetch 4H candles"""
    try:
        klines = client.get_klines(symbol="BTCUSDT", interval="4h", limit=limit)
        
        candles = []
        for k in klines:
            candles.append({
                'time': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        
        return candles
    except BinanceAPIException as e:
        print(f"Failed to fetch candles: {e}")
        return []


def calculate_indicators(candles, entry_len=40, exit_len=16, atr_len=20):
    """Calculate Donchian and ATR"""
    if len(candles) < max(entry_len, atr_len) + 1:
        return None
    
    # Entry high (40-period max of highs) - use previous bars
    entry_highs = [c['high'] for c in candles[-(entry_len+1):-1]]
    entry_high = max(entry_highs)
    
    # Exit low (16-period min of lows) - use previous bars
    exit_lows = [c['low'] for c in candles[-(exit_len+1):-1]]
    exit_low = min(exit_lows)
    
    # ATR calculation
    trs = []
    for i in range(len(candles) - atr_len, len(candles)):
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


def place_order(client, side, quantity):
    """Place market order"""
    try:
        quantity = round(quantity, 5)
        
        order = client.create_order(
            symbol="BTCUSDT",
            side=side,
            type='MARKET',
            quantity=quantity
        )
        
        fill_price = float(order['fills'][0]['price']) if order.get('fills') else 0
        
        print(f"Order filled: {side} {quantity} BTC @ ${fill_price:,.2f}")
        
        return {
            'price': fill_price,
            'quantity': float(order['executedQty'])
        }
    except BinanceAPIException as e:
        print(f"Order failed: {e}")
        return None


def run_bot():
    """Main bot execution - called once per 4H candle"""
    print("=" * 70)
    print(f"TRADING BOT - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    
    # Load state
    state = load_state()
    print(f"Loaded state: Equity=${state['equity']:,.2f}, Position={state['position_size']:.5f} BTC")
    
    # V1 Strategy parameters
    ENTRY_LEN = 40
    EXIT_LEN = 16
    ATR_LEN = 20
    TRAIL_MULT = 4.0
    RISK_PCT = 0.01
    MAX_UNITS = 4
    
    # Connect to Binance
    client = get_client()
    
    # Fetch candles
    candles = get_candles(client, limit=200)
    if not candles:
        print("ERROR: Failed to fetch candles")
        save_state(state)
        return
    
    # Calculate indicators
    indicators = calculate_indicators(candles, ENTRY_LEN, EXIT_LEN, ATR_LEN)
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
            order = place_order(client, 'SELL', position_size)
            
            if order:
                # Calculate P&L
                total_cost = sum(u['entry_price'] * u['quantity'] for u in position_units)
                total_value = order['price'] * order['quantity']
                pnl = total_value - total_cost
                pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0
                
                equity += pnl
                
                print(f"  Exit Price: ${order['price']:,.2f}")
                print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                print(f"  New Equity: ${equity:,.2f}")
                
                # Save trade
                save_trade({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'EXIT',
                    'reason': exit_reason,
                    'price': order['price'],
                    'quantity': order['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'equity': equity
                })
                
                # Reset position
                position_size = 0.0
                position_units = []
        else:
            # Update trailing stops
            print("\n>>> Updating trailing stops...")
            for unit in position_units:
                new_stop = indicators['current_price'] - (TRAIL_MULT * indicators['atr'])
                if new_stop > unit.get('trailing_stop', 0):
                    old_stop = unit.get('trailing_stop', 0)
                    unit['trailing_stop'] = new_stop
                    print(f"  Unit stop: ${old_stop:,.2f} -> ${new_stop:,.2f}")
    
    # Check entries
    if indicators['prev_high'] > indicators['entry_high']:
        if position_size == 0:
            print(f"\n>>> ENTRY SIGNAL: Donchian Breakout")
            
            # Position size: 1% risk / (2*ATR)
            risk_amount = equity * RISK_PCT
            size = risk_amount / (2.0 * indicators['atr'])
            size = max(size, 10.0 / indicators['current_price'])  # Min $10 notional
            size = round(size, 5)
            
            order = place_order(client, 'BUY', size)
            
            if order:
                position_size = order['quantity']
                position_units = [{
                    'entry_price': order['price'],
                    'quantity': order['quantity'],
                    'trailing_stop': order['price'] - (TRAIL_MULT * indicators['atr']),
                    'entry_time': datetime.utcnow().isoformat()
                }]
                
                print(f"  Entry Price: ${order['price']:,.2f}")
                print(f"  Size: {order['quantity']:.5f} BTC")
                print(f"  Trailing Stop: ${position_units[0]['trailing_stop']:,.2f}")
                
                # Save trade
                save_trade({
                    'time': datetime.utcnow().isoformat(),
                    'type': 'ENTRY',
                    'reason': 'Donchian Breakout',
                    'price': order['price'],
                    'quantity': order['quantity'],
                    'trailing_stop': position_units[0]['trailing_stop'],
                    'equity': equity
                })
                
                state['trade_count'] += 1
        
        elif len(position_units) < MAX_UNITS:
            # Pyramid check
            last_entry = position_units[-1]['entry_price']
            pyramid_threshold = last_entry + (1.5 * indicators['atr'])
            
            if indicators['current_price'] >= pyramid_threshold:
                print(f"\n>>> PYRAMID SIGNAL: Price ${indicators['current_price']:,.2f} >= ${pyramid_threshold:,.2f}")
                
                risk_amount = equity * RISK_PCT
                size = risk_amount / (2.0 * indicators['atr'])
                size = max(size, 10.0 / indicators['current_price'])
                size = round(size, 5)
                
                order = place_order(client, 'BUY', size)
                
                if order:
                    position_size += order['quantity']
                    position_units.append({
                        'entry_price': order['price'],
                        'quantity': order['quantity'],
                        'trailing_stop': order['price'] - (TRAIL_MULT * indicators['atr']),
                        'entry_time': datetime.utcnow().isoformat()
                    })
                    
                    print(f"  Entry Price: ${order['price']:,.2f}")
                    print(f"  Size: {order['quantity']:.5f} BTC")
                    print(f"  Total Position: {position_size:.5f} BTC ({len(position_units)} units)")
                    
                    save_trade({
                        'time': datetime.utcnow().isoformat(),
                        'type': 'PYRAMID',
                        'reason': f'Unit #{len(position_units)}',
                        'price': order['price'],
                        'quantity': order['quantity'],
                        'equity': equity
                    })
                    
                    state['trade_count'] += 1
    else:
        print("\n>>> No entry signal")
    
    # Update state
    state['equity'] = equity
    state['position_size'] = position_size
    state['position_units'] = position_units
    
    # Save state
    save_state(state)
    
    print(f"\n" + "=" * 70)
    print(f"Bot run complete. Next run in ~4 hours.")
    print(f"Equity: ${equity:,.2f} | Position: {position_size:.5f} BTC")
    print("=" * 70)


if __name__ == "__main__":
    run_bot()

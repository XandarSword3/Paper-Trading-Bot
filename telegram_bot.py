"""
Telegram Bot for Strategy Monitoring
Commands: /start, /status, /analysis, /trades, /signals, /help
"""
import os
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8489311506:AAGyZli23sqDU6D8_VD_TJw6cq_XT0EdgL0")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Kraken API
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"

# Strategy Parameters
ENTRY_LEN = 40
EXIT_LEN = 16
ATR_LEN = 20
TRAIL_MULT = 4.0

# File paths
STATE_FILE = Path("bot_state.json")
TRADES_FILE = Path("trades.json")


def send_message(chat_id, text, parse_mode="HTML"):
    """Send message to Telegram"""
    try:
        response = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode
            },
            timeout=10
        )
        return response.json()
    except Exception as e:
        print(f"Send error: {e}")
        return None


def get_updates(offset=None):
    """Get updates from Telegram"""
    try:
        params = {"timeout": 30}
        if offset:
            params["offset"] = offset
        
        response = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params=params,
            timeout=35
        )
        return response.json().get("result", [])
    except Exception as e:
        print(f"Get updates error: {e}")
        return []


def get_candles(count=50, interval=240):
    """Fetch candles from Kraken"""
    try:
        since = int((datetime.now(timezone.utc) - timedelta(hours=interval/60 * count)).timestamp())
        response = requests.get(
            f"{KRAKEN_API}/OHLC",
            params={"pair": PAIR, "interval": interval, "since": since},
            timeout=15
        )
        data = response.json()
        
        if data.get("error"):
            return None
        
        result_key = list(data["result"].keys())[0]
        ohlc = data["result"][result_key]
        
        candles = []
        for row in ohlc[-count:]:
            candles.append({
                'time': datetime.fromtimestamp(row[0], tz=timezone.utc),
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[6])
            })
        return candles
    except Exception as e:
        print(f"Candle fetch error: {e}")
        return None


def get_price():
    """Get current BTC price"""
    try:
        response = requests.get(
            f"{KRAKEN_API}/Ticker",
            params={"pair": PAIR},
            timeout=10
        )
        data = response.json()
        result_key = list(data["result"].keys())[0]
        return float(data["result"][result_key]["c"][0])
    except:
        return None


def calculate_indicators(candles):
    """Calculate strategy indicators"""
    if len(candles) < ENTRY_LEN + 1:
        return None
    
    # Entry high (40-period)
    entry_highs = [c['high'] for c in candles[-(ENTRY_LEN+1):-1]]
    entry_high = max(entry_highs)
    
    # Exit low (16-period)
    exit_lows = [c['low'] for c in candles[-(EXIT_LEN+1):-1]]
    exit_low = min(exit_lows)
    
    # ATR
    trs = []
    for i in range(len(candles) - ATR_LEN, len(candles)):
        if i < 1:
            continue
        tr = max(
            candles[i]['high'] - candles[i]['low'],
            abs(candles[i]['high'] - candles[i-1]['close']),
            abs(candles[i]['low'] - candles[i-1]['close'])
        )
        trs.append(tr)
    
    atr = sum(trs) / len(trs) if trs else 0
    
    return {
        'entry_high': entry_high,
        'exit_low': exit_low,
        'atr': atr,
        'current_price': candles[-1]['close']
    }


def analyze_candle(candle, entry_high, exit_low, trend_pct=0):
    """
    Analyze candle and return strategy fit score (0-100%)
    
    Components:
    1. Distance to breakout (0-40 pts)
    2. Bullish candle strength (0-20 pts)
    3. Close position (0-20 pts)
    4. Trend strength (0-20 pts)
    """
    score = 0
    breakdown = []
    
    # Component 1: Distance to Entry (0-40 points)
    if entry_high > 0:
        pct_to_entry = (candle['close'] - entry_high) / entry_high * 100
        
        if pct_to_entry > 0:
            dist_score = 40
            breakdown.append(f"✅ Above breakout (+{pct_to_entry:.1f}%) → 40/40")
        elif pct_to_entry > -2:
            dist_score = int(35 + pct_to_entry * 2.5)
            breakdown.append(f"🟡 Near breakout ({pct_to_entry:.1f}%) → {dist_score}/40")
        elif pct_to_entry > -5:
            dist_score = int(20 + pct_to_entry * 3)
            breakdown.append(f"🟠 Approaching ({pct_to_entry:.1f}%) → {dist_score}/40")
        else:
            dist_score = max(0, int(5 + pct_to_entry))
            breakdown.append(f"⚪ Far from entry ({pct_to_entry:.1f}%) → {dist_score}/40")
        
        score += dist_score
    
    # Component 2: Bullish Candle (0-20 points)
    body = candle['close'] - candle['open']
    range_size = candle['high'] - candle['low'] if candle['high'] != candle['low'] else 1
    body_ratio = body / range_size
    
    if body_ratio > 0.6:
        bull_score = 20
        breakdown.append(f"✅ Strong bullish ({body_ratio*100:.0f}%) → 20/20")
    elif body_ratio > 0.3:
        bull_score = int(10 + body_ratio * 16)
        breakdown.append(f"🟡 Moderate bullish ({body_ratio*100:.0f}%) → {bull_score}/20")
    elif body_ratio > 0:
        bull_score = int(body_ratio * 33)
        breakdown.append(f"🟠 Weak bullish ({body_ratio*100:.0f}%) → {bull_score}/20")
    else:
        bull_score = 0
        breakdown.append(f"❌ Bearish candle → 0/20")
    
    score += bull_score
    
    # Component 3: Close Position (0-20 points)
    if range_size > 0:
        close_pct = (candle['close'] - candle['low']) / range_size
        
        if close_pct > 0.8:
            close_score = 20
            breakdown.append(f"✅ Closed near high → 20/20")
        elif close_pct > 0.5:
            close_score = int(close_pct * 20)
            breakdown.append(f"🟡 Closed upper half → {close_score}/20")
        else:
            close_score = int(close_pct * 10)
            breakdown.append(f"🟠 Closed lower half → {close_score}/20")
        
        score += close_score
    
    # Component 4: Trend (0-20 points)
    if trend_pct > 5:
        trend_score = 20
        breakdown.append(f"✅ Strong uptrend (+{trend_pct:.1f}%) → 20/20")
    elif trend_pct > 0:
        trend_score = int(10 + trend_pct * 2)
        breakdown.append(f"🟡 Uptrend (+{trend_pct:.1f}%) → {trend_score}/20")
    elif trend_pct > -5:
        trend_score = int(5 + trend_pct)
        breakdown.append(f"🟠 Sideways ({trend_pct:+.1f}%) → {trend_score}/20")
    else:
        trend_score = 0
        breakdown.append(f"❌ Downtrend ({trend_pct:.1f}%) → 0/20")
    
    score += trend_score
    
    # Rating
    if score >= 80:
        rating = "🔥 EXCELLENT"
    elif score >= 60:
        rating = "✅ GOOD"
    elif score >= 40:
        rating = "🟡 MODERATE"
    elif score >= 20:
        rating = "🟠 WEAK"
    else:
        rating = "❌ POOR"
    
    return {
        'score': score,
        'rating': rating,
        'breakdown': breakdown
    }


def load_state():
    """Load bot state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"equity": 1000, "position_size": 0, "position_units": []}


def load_trades():
    """Load trade history"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def handle_start(chat_id):
    """Handle /start command"""
    send_message(chat_id, 
        "🤖 <b>V1 Turtle-Donchian Strategy Bot</b>\n\n"
        "Welcome! I monitor BTC using the Turtle trading strategy.\n\n"
        "<b>Commands:</b>\n"
        "/status - Current position & equity\n"
        "/analysis - Live candle analysis with fit %\n"
        "/signals - Check for active signals\n"
        "/trades - Recent trade history\n"
        "/help - Show this help\n\n"
        "<i>Strategy: Entry={}, Exit={}, Trail={}x ATR</i>".format(
            ENTRY_LEN, EXIT_LEN, TRAIL_MULT
        )
    )


def handle_status(chat_id):
    """Handle /status command"""
    state = load_state()
    price = get_price()
    
    position_size = state.get('position_size', 0)
    position_units = state.get('position_units', [])
    equity = state.get('equity', 1000)
    initial = state.get('initial_capital', 1000)
    
    # Calculate unrealized P&L
    unrealized_pnl = 0
    if position_size > 0 and price:
        entry_cost = sum(u.get('entry_price', 0) * u.get('size', 0) for u in position_units)
        unrealized_pnl = price * position_size - entry_cost
    
    total_equity = equity + unrealized_pnl
    return_pct = ((total_equity / initial) - 1) * 100
    
    msg = f"📊 <b>Status</b>\n\n"
    msg += f"💰 Equity: ${equity:,.2f}\n"
    
    if position_size > 0:
        msg += f"\n<b>📈 Open Position:</b>\n"
        msg += f"• Size: {position_size:.5f} BTC\n"
        msg += f"• Units: {len(position_units)}/4\n"
        
        if price:
            msg += f"• Current Price: ${price:,.2f}\n"
            msg += f"• Unrealized P&L: ${unrealized_pnl:+,.2f}\n"
        
        for i, unit in enumerate(position_units, 1):
            msg += f"\n  Unit {i}:\n"
            msg += f"    Entry: ${unit.get('entry_price', 0):,.2f}\n"
            msg += f"    Stop: ${unit.get('trailing_stop', 0):,.2f}\n"
    else:
        msg += f"\n⏳ No open position\n"
        msg += f"💵 Watching for breakout...\n"
    
    msg += f"\n📈 Total Return: {return_pct:+.1f}%"
    
    send_message(chat_id, msg)


def handle_analysis(chat_id):
    """Handle /analysis command - shows current candle fit %"""
    candles = get_candles(count=50)
    
    if not candles:
        send_message(chat_id, "❌ Failed to fetch market data")
        return
    
    indicators = calculate_indicators(candles)
    if not indicators:
        send_message(chat_id, "❌ Not enough data for analysis")
        return
    
    # Current candle
    current = candles[-1]
    
    # Calculate trend
    if len(candles) >= 10:
        trend_pct = (current['close'] - candles[-10]['close']) / candles[-10]['close'] * 100
    else:
        trend_pct = 0
    
    # Analyze
    analysis = analyze_candle(
        current, 
        indicators['entry_high'],
        indicators['exit_low'],
        trend_pct
    )
    
    msg = f"🔍 <b>Current Candle Analysis</b>\n"
    msg += f"<i>{current['time'].strftime('%Y-%m-%d %H:%M')} UTC</i>\n\n"
    
    msg += f"<b>Strategy Fit: {analysis['score']}%</b>\n"
    msg += f"{analysis['rating']}\n\n"
    
    msg += "<b>📊 Price:</b>\n"
    msg += f"• Open: ${current['open']:,.2f}\n"
    msg += f"• High: ${current['high']:,.2f}\n"
    msg += f"• Low: ${current['low']:,.2f}\n"
    msg += f"• Close: ${current['close']:,.2f}\n\n"
    
    msg += "<b>📏 Levels:</b>\n"
    msg += f"• Entry (40H): ${indicators['entry_high']:,.2f}\n"
    msg += f"• Exit (16L): ${indicators['exit_low']:,.2f}\n"
    msg += f"• ATR: ${indicators['atr']:,.2f}\n\n"
    
    msg += "<b>📋 Score Breakdown:</b>\n"
    for line in analysis['breakdown']:
        msg += f"• {line}\n"
    
    send_message(chat_id, msg)


def handle_signals(chat_id):
    """Handle /signals command"""
    candles = get_candles(count=50)
    
    if not candles:
        send_message(chat_id, "❌ Failed to fetch market data")
        return
    
    indicators = calculate_indicators(candles)
    if not indicators:
        send_message(chat_id, "❌ Not enough data for signals")
        return
    
    state = load_state()
    price = indicators['current_price']
    
    msg = f"🚦 <b>Signal Check</b>\n\n"
    msg += f"📍 Price: ${price:,.2f}\n"
    msg += f"📏 Entry Level: ${indicators['entry_high']:,.2f}\n"
    msg += f"📏 Exit Level: ${indicators['exit_low']:,.2f}\n"
    msg += f"📐 ATR: ${indicators['atr']:,.2f}\n\n"
    
    # Check signals
    prev_high = candles[-2]['high']
    prev_low = candles[-2]['low']
    
    has_position = state.get('position_size', 0) > 0
    position_units = state.get('position_units', [])
    
    # Entry signals
    if prev_high > indicators['entry_high']:
        if not has_position:
            msg += "🟢 <b>BREAKOUT SIGNAL!</b>\n"
            msg += f"Previous high ${prev_high:,.2f} > Entry ${indicators['entry_high']:,.2f}\n"
            msg += "→ New entry opportunity\n\n"
        elif len(position_units) < 4:
            last_entry = position_units[-1].get('entry_price', 0)
            if price >= last_entry + indicators['atr']:
                msg += "🟢 <b>PYRAMID SIGNAL!</b>\n"
                msg += f"Price moved 1 ATR above last entry\n"
                msg += "→ Add unit opportunity\n\n"
            else:
                msg += "⏳ Waiting for pyramid level\n"
                msg += f"Need price above ${last_entry + indicators['atr']:,.2f}\n\n"
        else:
            msg += "📦 Max units (4) reached\n\n"
    else:
        pct_to_entry = (price - indicators['entry_high']) / indicators['entry_high'] * 100
        msg += f"⏳ No entry signal\n"
        msg += f"Price is {pct_to_entry:+.1f}% from breakout level\n\n"
    
    # Exit signals
    if has_position:
        if prev_low < indicators['exit_low']:
            msg += "🔴 <b>EXIT SIGNAL!</b>\n"
            msg += f"Previous low ${prev_low:,.2f} < Exit ${indicators['exit_low']:,.2f}\n"
            msg += "→ Close position\n"
        else:
            # Check trailing stops
            for i, unit in enumerate(position_units, 1):
                stop = unit.get('trailing_stop', 0)
                if price <= stop:
                    msg += f"🔴 <b>STOP TRIGGERED on Unit {i}!</b>\n"
                    msg += f"Price ${price:,.2f} <= Stop ${stop:,.2f}\n"
                else:
                    msg += f"✅ Unit {i} stop safe: ${stop:,.2f}\n"
    
    send_message(chat_id, msg)


def handle_trades(chat_id):
    """Handle /trades command"""
    trades = load_trades()
    
    if not trades:
        send_message(chat_id, "📭 No trades recorded yet")
        return
    
    msg = "<b>📜 Recent Trades</b>\n\n"
    
    for trade in trades[-5:]:  # Last 5 trades
        trade_type = trade.get('type', 'UNKNOWN')
        emoji = "📈" if trade_type == "ENTRY" else "📉"
        
        msg += f"{emoji} <b>{trade_type}</b> - {trade.get('reason', '')}\n"
        msg += f"   Time: {trade.get('time', 'N/A')[:16]}\n"
        msg += f"   Price: ${trade.get('price', 0):,.2f}\n"
        msg += f"   Size: {trade.get('size', 0):.5f} BTC\n"
        
        if 'pnl' in trade:
            pnl = trade['pnl']
            pnl_emoji = "🟢" if pnl > 0 else "🔴"
            msg += f"   P&L: {pnl_emoji} ${pnl:+,.2f}\n"
        
        msg += f"   Equity: ${trade.get('equity', 0):,.2f}\n\n"
    
    send_message(chat_id, msg)


def handle_help(chat_id):
    """Handle /help command"""
    handle_start(chat_id)


def process_message(update):
    """Process incoming message"""
    if "message" not in update:
        return
    
    message = update["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip().lower()
    
    print(f"Received: {text} from {chat_id}")
    
    if text == "/start":
        handle_start(chat_id)
    elif text == "/status":
        handle_status(chat_id)
    elif text == "/analysis":
        handle_analysis(chat_id)
    elif text == "/signals":
        handle_signals(chat_id)
    elif text == "/trades":
        handle_trades(chat_id)
    elif text == "/help":
        handle_help(chat_id)
    else:
        send_message(chat_id, 
            "🤔 Unknown command. Use /help to see available commands."
        )


def main():
    """Main bot loop"""
    print("🤖 Telegram bot starting...")
    print(f"Token: {TELEGRAM_TOKEN[:20]}...")
    
    # Get bot info
    try:
        response = requests.get(f"{TELEGRAM_API}/getMe", timeout=10)
        bot_info = response.json()
        if bot_info.get("ok"):
            username = bot_info["result"]["username"]
            print(f"✅ Connected as @{username}")
        else:
            print(f"❌ Failed to connect: {bot_info}")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    print("Listening for messages...")
    
    offset = None
    while True:
        updates = get_updates(offset)
        
        for update in updates:
            offset = update["update_id"] + 1
            process_message(update)


if __name__ == "__main__":
    main()

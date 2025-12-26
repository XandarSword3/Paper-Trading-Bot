"""
V4 Strategy Dashboard - 1H High-Frequency
Entry=8 | Exit=16 | Trail=3.5x ATR | ~1.33 trades/day
"""
import streamlit as st
import json
import requests
from pathlib import Path
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="V4 Strategy", page_icon="⚡", layout="wide")

# === CONFIGURATION ===
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"
STATE_FILE = Path("bot_state_v4.json")
TRADES_FILE = Path("trades_v4.json")

# V4 Parameters
ENTRY_LEN = 8
EXIT_LEN = 16
TRAIL_MULT = 3.5


@st.cache_data(ttl=60)
def get_price():
    """Fetch current BTC price from Kraken"""
    try:
        response = requests.get(f"{KRAKEN_API}/Ticker", params={"pair": PAIR}, timeout=10)
        data = response.json()
        result_key = list(data["result"].keys())[0]
        return float(data["result"][result_key]["c"][0])
    except:
        return None


@st.cache_data(ttl=180)
def get_candles(limit=200):
    """Fetch 1H candles from Kraken"""
    try:
        response = requests.get(f"{KRAKEN_API}/OHLC", 
                               params={"pair": PAIR, "interval": 60}, timeout=30)
        data = response.json()
        result_key = [k for k in data["result"].keys() if k != "last"][0]
        return data["result"][result_key]
    except:
        return []


def load_state():
    """Load V4 bot state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"strategy": "V4", "equity": 1000.0, "position_size": 0.0, "position_units": [], "trade_count": 0}


def load_trades():
    """Load V4 trade history"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


# === MAIN PAGE ===
st.title("⚡ V4 High-Frequency Dashboard")
st.caption("1-Hour Donchian | Entry=8 | Exit=16 | Trail=3.5x ATR | ~1 trade/day")

# Performance comparison badge
st.success("🏆 V4 beats V1 by +717% in backtests (1572% vs 855%)")

# Load data
state = load_state()
trades = load_trades()
price = get_price()

# Metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("BTC Price", f"${price:,.2f}" if price else "N/A")

with col2:
    equity = state.get('equity', 1000)
    initial = state.get('initial_capital', 1000)
    pnl_pct = ((equity / initial) - 1) * 100 if initial else 0
    st.metric("Equity", f"${equity:,.2f}", f"{pnl_pct:+.1f}%")

with col3:
    pos = state.get('position_size', 0)
    st.metric("Position", f"{pos:.5f} BTC" if pos > 0 else "FLAT")

with col4:
    units = len(state.get('position_units', []))
    st.metric("Units", f"{units}/4")

with col5:
    st.metric("Total Trades", state.get('trade_count', 0))

st.divider()

# Two columns: Chart and Position
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Price Chart (1H)")
    
    candles = get_candles()
    if candles:
        # Prepare data - last 100 candles
        times = [datetime.fromtimestamp(c[0]) for c in candles[-100:]]
        opens = [float(c[1]) for c in candles[-100:]]
        highs = [float(c[2]) for c in candles[-100:]]
        lows = [float(c[3]) for c in candles[-100:]]
        closes = [float(c[4]) for c in candles[-100:]]
        
        # Calculate Donchian channels
        entry_highs = []
        exit_lows = []
        for i in range(len(closes)):
            start = max(0, i - ENTRY_LEN)
            entry_highs.append(max(highs[start:i]) if i > 0 else highs[0])
            start = max(0, i - EXIT_LEN)
            exit_lows.append(min(lows[start:i]) if i > 0 else lows[0])
        
        # Create chart
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Candlestick(
            x=times, open=opens, high=highs, low=lows, close=closes,
            name="BTC/USD"
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=entry_highs, mode='lines', name=f"Entry ({ENTRY_LEN})",
            line=dict(color='lime', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=exit_lows, mode='lines', name=f"Exit ({EXIT_LEN})",
            line=dict(color='red', width=2, dash='dot')
        ))
        
        fig.update_layout(
            height=400, xaxis_rangeslider_visible=False,
            template="plotly_dark", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📦 Current Position")
    
    units_list = state.get('position_units', [])
    if units_list:
        total_pnl = 0
        for i, unit in enumerate(units_list):
            with st.expander(f"Unit {i+1}", expanded=True):
                entry = unit.get('entry_price', 0)
                stop = unit.get('trailing_stop', 0)
                size = unit.get('size', 0)
                
                st.write(f"**Entry:** ${entry:,.2f}")
                st.write(f"**Size:** {size:.5f} BTC")
                st.write(f"**Stop:** ${stop:,.2f}")
                
                if price:
                    unit_pnl = (price - entry) * size
                    total_pnl += unit_pnl
                    color = "green" if unit_pnl >= 0 else "red"
                    st.markdown(f"**P&L:** :{color}[${unit_pnl:+,.2f}]")
        
        if price:
            st.divider()
            st.metric("Total Open P&L", f"${total_pnl:+,.2f}")
    else:
        st.info("No open position")
        st.caption("Waiting for 8-hour high breakout")

# Trade History
st.divider()
st.subheader("📜 Trade History")

if trades:
    # Prepare trade data for display
    display_trades = []
    for t in reversed(trades[-50:]):
        display_trades.append({
            "Time": t.get('time', '')[:19],
            "Type": t.get('type', ''),
            "Reason": t.get('reason', ''),
            "Price": f"${t.get('price', 0):,.2f}",
            "Size": f"{t.get('size', 0):.5f}",
            "P&L": f"${t.get('pnl', 0):+,.2f}" if t.get('pnl') else "-",
            "Equity": f"${t.get('equity', 0):,.2f}"
        })
    
    st.dataframe(display_trades, use_container_width=True, hide_index=True)
else:
    st.info("No trades yet - V4 bot is waiting for signals")

# Equity Curve
if trades:
    st.divider()
    st.subheader("📈 Equity Curve")
    
    equity_data = [t.get('equity', 1000) for t in trades]
    times_data = list(range(len(equity_data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_data, y=equity_data, mode='lines+markers',
        name='Equity', line=dict(color='cyan', width=2)
    ))
    fig.update_layout(height=300, template="plotly_dark", 
                     xaxis_title="Trade #", yaxis_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)

# Strategy Stats
st.divider()
st.subheader("📊 V4 Strategy Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Strategy Parameters:**
    - Timeframe: 1H
    - Entry: 8-bar high breakout
    - Exit: 16-bar low
    - Trail: 3.5x ATR
    - Risk: 1% per trade
    """)

with col2:
    st.markdown("""
    **Backtest Results:**
    - Return: **1572%** 🏆
    - Max Drawdown: -30%
    - Win Rate: 34.5%
    - Trades/Day: ~1.33
    """)

with col3:
    last_run = state.get('last_run', 'Never')
    st.markdown(f"""
    **Bot Status:**
    - Last Run: {last_run[:19] if last_run else 'Never'}
    - Schedule: Every 1 hour
    - Mode: Simulation
    """)

# V1 vs V4 Comparison
st.divider()
st.subheader("⚔️ V1 vs V4 Comparison")

comparison_data = {
    "Metric": ["Timeframe", "Entry Period", "Exit Period", "Trail Mult", "Backtest Return", "Trades/Day", "Win Rate"],
    "V1 (Turtle)": ["4H", "40 bars", "16 bars", "4.0x ATR", "855%", "~0.14", "35.4%"],
    "V4 (Fast)": ["1H", "8 bars", "16 bars", "3.5x ATR", "1572%", "~1.33", "34.5%"]
}

st.dataframe(comparison_data, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption("V4 High-Frequency Strategy | Simulation Mode | Data: Kraken API")

"""
V1 Strategy Dashboard - 4H Turtle-Donchian
Entry=40 | Exit=16 | Trail=4.0x ATR | ~0.14 trades/day
"""
import streamlit as st
import json
import requests
from pathlib import Path
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="V1 Strategy", page_icon="🐢", layout="wide")

# === CONFIGURATION ===
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"
STATE_FILE = Path("bot_state.json")
TRADES_FILE = Path("trades.json")
WALK_FORWARD_FILE = Path("walk_forward_results_v1.json")


def load_walk_forward_summary():
    """Load the real out-of-sample walk-forward summary if present. This is
    the only performance number that should be shown as 'validated' — it was
    never fit on the data it's scored on. Returns None if the file is
    missing so callers can show an explicit 'not yet validated' state rather
    than silently falling back to a stale hardcoded figure."""
    if not WALK_FORWARD_FILE.exists():
        return None
    try:
        with open(WALK_FORWARD_FILE) as f:
            return json.load(f)
    except Exception:
        return None

# V1 Parameters
ENTRY_LEN = 40
EXIT_LEN = 16
TRAIL_MULT = 4.0


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


@st.cache_data(ttl=300)
def get_candles(limit=200):
    """Fetch 4H candles from Kraken"""
    try:
        response = requests.get(f"{KRAKEN_API}/OHLC", 
                               params={"pair": PAIR, "interval": 240}, timeout=30)
        data = response.json()
        result_key = [k for k in data["result"].keys() if k != "last"][0]
        return data["result"][result_key]
    except:
        return []


def load_state():
    """Load V1 bot state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"equity": 1000.0, "position_size": 0.0, "position_units": [], "trade_count": 0}


def load_trades():
    """Load V1 trade history"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


# === MAIN PAGE ===
st.title("🐢 V1 Strategy Dashboard")
st.caption("4-Hour Turtle-Donchian | Entry=40 | Exit=16 | Trail=4.0x ATR")

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
    st.subheader("📊 Price Chart (4H)")
    
    candles = get_candles()
    if candles:
        # Prepare data
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
            line=dict(color='green', width=1, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=exit_lows, mode='lines', name=f"Exit ({EXIT_LEN})",
            line=dict(color='red', width=1, dash='dot')
        ))
        
        fig.update_layout(
            height=400, xaxis_rangeslider_visible=False,
            template="plotly_dark", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📦 Current Position")
    
    units = state.get('position_units', [])
    if units:
        for i, unit in enumerate(units):
            with st.expander(f"Unit {i+1}", expanded=True):
                entry = unit.get('entry_price', 0)
                stop = unit.get('trailing_stop', 0)
                size = unit.get('size', 0)
                
                st.write(f"**Entry:** ${entry:,.2f}")
                st.write(f"**Size:** {size:.5f} BTC")
                st.write(f"**Stop:** ${stop:,.2f}")
                
                if price:
                    unit_pnl = (price - entry) * size
                    st.write(f"**P&L:** ${unit_pnl:+,.2f}")
    else:
        st.info("No open position")
        st.caption("Waiting for breakout signal above Donchian high")

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
    st.info("No trades yet - bot is waiting for signals")

# Equity Curve
if trades:
    st.divider()
    st.subheader("📈 Equity Curve")
    
    equity_data = [t.get('equity', 1000) for t in trades]
    times_data = list(range(len(equity_data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_data, y=equity_data, mode='lines+markers',
        name='Equity', line=dict(color='green', width=2)
    ))
    fig.update_layout(height=300, template="plotly_dark", 
                     xaxis_title="Trade #", yaxis_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)

# Strategy Stats
st.divider()
st.subheader("📊 V1 Strategy Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Strategy Parameters:**
    - Timeframe: 4H
    - Entry: 40-bar high breakout
    - Exit: 16-bar low
    - Trail: 4.0x ATR
    - Risk: 1% per trade
    """)

with col2:
    wf = load_walk_forward_summary()
    if wf and wf.get("oos_sharpe") is not None:
        span_start = str(wf.get("oos_coverage_start", ""))[:10]
        span_end = str(wf.get("oos_coverage_end", ""))[:10]
        st.markdown(f"""
        **Walk-Forward OOS Results** ({span_start} → {span_end}):
        - Total return: {wf.get('oos_total_return_pct', 0):+.1f}%
        - CAGR: {wf.get('oos_cagr_pct', 0):+.1f}%
        - Max Drawdown: {wf.get('oos_max_drawdown_pct', 0):.1f}%
        - Sharpe: {wf.get('oos_sharpe', 0):.2f}  |  Calmar: {wf.get('oos_calmar_ratio', 0):.2f}
        - Win Rate: {wf.get('oos_win_rate_pct', 0):.1f}%  ({wf.get('total_oos_trades', 0)} OOS trades, {wf.get('num_folds', 0)} folds)
        """)
        st.caption(
            "These are stitched out-of-sample walk-forward numbers — parameters "
            "for each fold were chosen only on data before that fold's test "
            "window. This is the number to trust, not an in-sample backtest."
        )
    else:
        st.warning(
            "No walk-forward validation file found (walk_forward_results_v1.json). "
            "This strategy has NOT been out-of-sample validated — do not treat any "
            "in-sample backtest number as expected live performance."
        )

with col3:
    last_run = state.get('last_run', 'Never')
    st.markdown(f"""
    **Bot Status:**
    - Last Run: {last_run[:19] if last_run else 'Never'}
    - Schedule: Every 4 hours
    - Mode: Simulation
    """)

# Footer
st.divider()
st.caption("V1 Turtle-Donchian Strategy | Simulation Mode | Data: Kraken API")

"""
🚀 BTC Trading Bot - Premium Dashboard
Dual Strategy: V1 Turtle (4H) + V4 Fast (1H)
Deploy free on Streamlit Cloud: https://share.streamlit.io
"""
import streamlit as st
import json
import requests
from datetime import datetime, timezone
import pandas as pd
import time

# === PAGE CONFIG ===
st.set_page_config(
    page_title="BTC Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("## 📊 Navigation")
st.sidebar.markdown("""
- **Home** - Combined Overview
- **📊 Candle Analysis** - Interactive chart
- **🐢 V1 Strategy** - Turtle (4H)
- **⚡ V4 Strategy** - Fast (1H)
- **🎯 Liquidation Hunter** - Cascades
""")

st.sidebar.divider()

# === GITHUB CONFIG ===
GITHUB_USER = "XandarSword3"
GITHUB_REPO = "Paper-Trading-Bot"
GITHUB_BRANCH = "master"

STATE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/bot_state.json"
TRADES_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/trades.json"
STATE_V4_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/bot_state_v4.json"
TRADES_V4_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/trades_v4.json"
KRAKEN_PRICE_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Dark theme enhancements */
    .main { background-color: #0e1117; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Positive/Negative colors */
    .profit { color: #00ff88 !important; font-weight: bold; }
    .loss { color: #ff4444 !important; font-weight: bold; }
    .neutral { color: #888888 !important; }
    
    /* Headers */
    h1 { 
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    
    /* Status indicator */
    .status-live {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    /* Cards */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Trade row styling */
    .trade-entry { border-left: 3px solid #00ff88; padding-left: 10px; }
    .trade-exit { border-left: 3px solid #ff4444; padding-left: 10px; }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Better scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)


# === DATA LOADING ===
@st.cache_data(ttl=30)
def load_state():
    """Load V1 bot state from GitHub"""
    try:
        response = requests.get(STATE_URL + f"?t={int(time.time())}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to load state: {e}")
    return None


@st.cache_data(ttl=30)
def load_trades():
    """Load V1 trade history from GitHub"""
    try:
        response = requests.get(TRADES_URL + f"?t={int(time.time())}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=30)
def load_state_v4():
    """Load V4 bot state from GitHub"""
    try:
        response = requests.get(STATE_V4_URL + f"?t={int(time.time())}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=30)
def load_trades_v4():
    """Load V4 trade history from GitHub"""
    try:
        response = requests.get(TRADES_V4_URL + f"?t={int(time.time())}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=10)
def get_live_price():
    """Get live BTC price from Kraken"""
    try:
        response = requests.get(KRAKEN_PRICE_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            result_key = [k for k in data["result"].keys() if k != "last"][0]
            return float(data["result"][result_key]["c"][0])
    except:
        pass
    return None


def calculate_stats(trades, initial_capital):
    """Calculate comprehensive trading statistics"""
    if not trades:
        return {}
    
    exits = [t for t in trades if t.get('type') == 'EXIT']
    entries = [t for t in trades if t.get('type') in ('ENTRY', 'Breakout', 'Pyramid')]
    
    if not exits:
        return {'total_trades': len(entries), 'closed_trades': 0}
    
    pnls = [t.get('pnl', 0) for t in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    total_pnl = sum(pnls)
    
    # Calculate max drawdown from equity curve
    equities = [initial_capital]
    for t in trades:
        if 'equity' in t:
            equities.append(t['equity'])
    
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'total_trades': len(entries),
        'closed_trades': len(exits),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'max_drawdown': max_dd,
        'wins': len(wins),
        'losses': len(losses)
    }


def format_time_ago(iso_time):
    """Convert ISO time to 'X mins ago' format"""
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return iso_time


# === MAIN APP ===
def main():
    # Header
    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.markdown("# 🚀 BTC Trading Bot")
        st.caption("Dual Strategy: 🐢 V1 Turtle (4H) + ⚡ V4 Fast (1H)")
    with col_status:
        st.markdown("""
            <div style='text-align: right; padding-top: 20px;'>
                <span class='status-live'></span>
                <span style='color: #00ff88; font-weight: bold;'>LIVE</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Load data for both strategies
    state = load_state()
    trades = load_trades()
    state_v4 = load_state_v4()
    trades_v4 = load_trades_v4()
    live_price = get_live_price()
    
    # === COMBINED STRATEGY OVERVIEW ===
    st.markdown("---")
    st.subheader("📊 Strategy Overview")
    
    col_v1, col_v4, col_combined = st.columns(3)
    
    # V1 Stats
    with col_v1:
        st.markdown("### 🐢 V1 Turtle")
        if state:
            v1_equity = state.get('equity', 1000)
            v1_initial = state.get('initial_capital', 1000)
            v1_return = ((v1_equity / v1_initial) - 1) * 100
            v1_pos = state.get('position_size', 0)
            st.metric("Equity", f"${v1_equity:,.2f}", f"{v1_return:+.1f}%")
            st.caption(f"Position: {'LONG' if v1_pos > 0 else 'FLAT'}")
            st.caption(f"Trades: {state.get('trade_count', 0)}")
        else:
            st.info("V1 not yet started")
    
    # V4 Stats
    with col_v4:
        st.markdown("### ⚡ V4 Fast")
        if state_v4:
            v4_equity = state_v4.get('equity', 1000)
            v4_initial = state_v4.get('initial_capital', 1000)
            v4_return = ((v4_equity / v4_initial) - 1) * 100
            v4_pos = state_v4.get('position_size', 0)
            st.metric("Equity", f"${v4_equity:,.2f}", f"{v4_return:+.1f}%")
            st.caption(f"Position: {'LONG' if v4_pos > 0 else 'FLAT'}")
            st.caption(f"Trades: {state_v4.get('trade_count', 0)}")
        else:
            st.info("V4 not yet started")
    
    # Combined Stats
    with col_combined:
        st.markdown("### 💼 Combined")
        v1_eq = state.get('equity', 1000) if state else 1000
        v4_eq = state_v4.get('equity', 1000) if state_v4 else 1000
        combined_equity = v1_eq + v4_eq
        combined_initial = 2000
        combined_return = ((combined_equity / combined_initial) - 1) * 100
        st.metric("Total Equity", f"${combined_equity:,.2f}", f"{combined_return:+.1f}%")
        v1_trades = state.get('trade_count', 0) if state else 0
        v4_trades = state_v4.get('trade_count', 0) if state_v4 else 0
        st.caption(f"Total Trades: {v1_trades + v4_trades}")
        st.caption(f"BTC: ${live_price:,.2f}" if live_price else "")
    
    if not state:
        st.error("⚠️ Could not connect to V1 bot. Check GitHub configuration.")
        st.code(f"STATE_URL = {STATE_URL}")
        return
    
    # === TOP METRICS (V1 focus for backward compatibility) ===
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate returns
    initial = state.get('initial_capital', 1000)
    equity = state.get('equity', initial)
    returns_pct = ((equity / initial) - 1) * 100
    
    with col1:
        st.metric(
            "💰 Equity",
            f"${equity:,.2f}",
            f"{returns_pct:+.2f}%",
            delta_color="normal"
        )
    
    with col2:
        if live_price:
            st.metric(
                "₿ BTC Price",
                f"${live_price:,.2f}",
                "Live from Kraken"
            )
        else:
            st.metric("₿ BTC Price", "Loading...", "")
    
    with col3:
        position = state.get('position_size', 0)
        units = len(state.get('position_units', []))
        if position > 0:
            pos_value = position * (live_price or 0)
            st.metric(
                "📊 Position",
                f"{position:.5f} BTC",
                f"${pos_value:,.0f} | {units}/4 units"
            )
        else:
            st.metric("📊 Position", "No Position", "Waiting for signal")
    
    with col4:
        st.metric(
            "🔄 Total Trades",
            state.get('trade_count', 0),
            "All time"
        )
    
    with col5:
        last_run = state.get('last_run', '')
        if last_run:
            time_ago = format_time_ago(last_run)
            st.metric("⏰ Last Run", time_ago, "Every 4 hours")
        else:
            st.metric("⏰ Last Run", "Never", "")
    
    # === MAIN CONTENT ===
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📜 Trade History", "📊 Analytics", "⚙️ Strategy", "🎯 Liquidation Hunter"])
    
    # === TAB 1: OVERVIEW ===
    with tab1:
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader("📈 Equity Curve")
            
            if trades:
                # Build equity curve
                equity_data = [{"time": datetime.now(timezone.utc) - pd.Timedelta(days=30), "equity": initial}]
                for trade in trades:
                    if 'equity' in trade and 'time' in trade:
                        equity_data.append({
                            "time": trade['time'],
                            "equity": trade['equity']
                        })
                
                # Add current equity
                equity_data.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "equity": equity
                })
                
                if len(equity_data) > 1:
                    df = pd.DataFrame(equity_data)
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    df = df.dropna(subset=['time']).sort_values('time')
                    
                    if not df.empty and len(df) > 1:
                        # Add color based on trend
                        st.line_chart(
                            df.set_index('time')['equity'],
                            use_container_width=True
                        )
                    else:
                        st.info("📊 Equity chart will populate after first trade")
                else:
                    st.info("📊 Waiting for first trade to show equity curve")
            else:
                # Show placeholder chart
                st.info("📊 No trades yet - chart will appear after first signal")
            
            # Unrealized P&L
            if state.get('position_size', 0) > 0 and live_price:
                st.subheader("💹 Unrealized P&L")
                units = state.get('position_units', [])
                total_cost = sum(u.get('entry_price', 0) * u.get('size', 0) for u in units)
                current_value = state['position_size'] * live_price
                unrealized = current_value - total_cost
                unrealized_pct = (unrealized / total_cost * 100) if total_cost > 0 else 0
                
                if unrealized >= 0:
                    st.success(f"**+${unrealized:,.2f}** ({unrealized_pct:+.2f}%)")
                else:
                    st.error(f"**${unrealized:,.2f}** ({unrealized_pct:+.2f}%)")
        
        with right_col:
            st.subheader("🎯 Current Position")
            
            if state.get('position_size', 0) > 0:
                position_units = state.get('position_units', [])
                
                for i, unit in enumerate(position_units):
                    entry_price = unit.get('entry_price', 0)
                    size = unit.get('size', 0)
                    stop = unit.get('trailing_stop', 0)
                    
                    # Calculate unit P&L
                    if live_price:
                        unit_pnl = (live_price - entry_price) * size
                        unit_pnl_pct = ((live_price / entry_price) - 1) * 100
                        pnl_color = "profit" if unit_pnl >= 0 else "loss"
                    else:
                        unit_pnl = 0
                        unit_pnl_pct = 0
                        pnl_color = "neutral"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class='info-card'>
                            <strong>Unit #{i+1}</strong><br>
                            📍 Entry: <strong>${entry_price:,.2f}</strong><br>
                            📦 Size: <strong>{size:.5f} BTC</strong><br>
                            🛑 Stop: <strong>${stop:,.2f}</strong><br>
                            💰 P&L: <span class='{pnl_color}'>${unit_pnl:+,.2f} ({unit_pnl_pct:+.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("""
                    🔍 **No Active Position**
                    
                    Waiting for price to break above the 40-period high 
                    to trigger entry signal.
                """)
            
            # Next scheduled run
            st.subheader("⏰ Schedule")
            now = datetime.now(timezone.utc)
            run_hours = [0, 4, 8, 12, 16, 20]
            current_hour = now.hour
            
            next_run = None
            for h in run_hours:
                if h > current_hour:
                    next_run = h
                    break
            if next_run is None:
                next_run = run_hours[0]
            
            hours_until = (next_run - current_hour) % 24
            if hours_until == 0:
                hours_until = 4
            
            st.markdown(f"""
                **Next run in:** {hours_until} hours  
                **Schedule:** Every 4 hours at :05  
                **Times:** 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
            """)
    
    # === TAB 2: TRADE HISTORY ===
    with tab2:
        st.subheader("📜 Complete Trade History")
        
        if trades:
            # Process trades for display
            display_data = []
            for t in reversed(trades):  # Most recent first
                row = {
                    'Time': t.get('time', '')[:16].replace('T', ' '),
                    'Type': t.get('type', ''),
                    'Reason': t.get('reason', ''),
                    'Price': f"${t.get('price', 0):,.2f}",
                    'Size': f"{t.get('size', 0):.5f}",
                }
                
                if 'pnl' in t and t.get('type') == 'EXIT':
                    pnl = t.get('pnl', 0)
                    pnl_pct = t.get('pnl_pct', 0)
                    row['P&L'] = f"${pnl:+,.2f}"
                    row['P&L %'] = f"{pnl_pct:+.2f}%"
                else:
                    row['P&L'] = '-'
                    row['P&L %'] = '-'
                
                row['Equity'] = f"${t.get('equity', 0):,.2f}"
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            
            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Type": st.column_config.TextColumn(width="small"),
                    "P&L": st.column_config.TextColumn(width="medium"),
                }
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History (CSV)",
                csv,
                "trade_history.csv",
                "text/csv"
            )
        else:
            st.info("📭 No trades executed yet. Waiting for first signal...")
    
    # === TAB 3: ANALYTICS ===
    with tab3:
        st.subheader("📊 Performance Analytics")
        
        stats = calculate_stats(trades, initial)
        
        if stats.get('closed_trades', 0) > 0:
            # Stats grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_rate = stats.get('win_rate', 0)
                color = "#00ff88" if win_rate >= 50 else "#ff4444"
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h2 style='color: {color}; margin: 0;'>{win_rate:.1f}%</h2>
                        <p style='color: #888; margin: 0;'>Win Rate</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pf = stats.get('profit_factor', 0)
                pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
                color = "#00ff88" if pf >= 1 else "#ff4444"
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h2 style='color: {color}; margin: 0;'>{pf_display}</h2>
                        <p style='color: #888; margin: 0;'>Profit Factor</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_pnl = stats.get('total_pnl', 0)
                color = "#00ff88" if total_pnl >= 0 else "#ff4444"
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h2 style='color: {color}; margin: 0;'>${total_pnl:+,.0f}</h2>
                        <p style='color: #888; margin: 0;'>Total P&L</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                max_dd = stats.get('max_drawdown', 0)
                st.markdown(f"""
                    <div class='info-card' style='text-align: center;'>
                        <h2 style='color: #ff4444; margin: 0;'>-{max_dd:.1f}%</h2>
                        <p style='color: #888; margin: 0;'>Max Drawdown</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Wins & Losses")
                st.write(f"✅ Winning trades: **{stats.get('wins', 0)}**")
                st.write(f"❌ Losing trades: **{stats.get('losses', 0)}**")
                st.write(f"💵 Avg win: **${stats.get('avg_win', 0):,.2f}**")
                st.write(f"💸 Avg loss: **${stats.get('avg_loss', 0):,.2f}**")
            
            with col2:
                st.markdown("### 📊 Trade Summary")
                st.write(f"🔄 Total entries: **{stats.get('total_trades', 0)}**")
                st.write(f"✅ Closed trades: **{stats.get('closed_trades', 0)}**")
                st.write(f"📈 Returns: **{returns_pct:+.2f}%**")
                st.write(f"💰 Current equity: **${equity:,.2f}**")
        else:
            st.info("📊 Analytics will appear after completing first trade cycle (entry + exit)")
    
    # === TAB 4: STRATEGY ===
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 V1 Turtle-Donchian Strategy")
            
            st.markdown("""
            #### Entry Rules
            - **Breakout:** Enter when price breaks above 40-period high
            - **Pyramiding:** Add up to 4 units, spaced 1 ATR apart
            - **Position sizing:** Risk 1% of equity per unit
            
            #### Exit Rules
            - **Donchian Exit:** Close when price breaks below 16-period low
            - **Trailing Stop:** 4x ATR below highest price since entry
            
            #### Risk Management
            - **Max risk per trade:** 1% of equity
            - **Max units:** 4 (pyramiding)
            - **Position type:** Long only
            """)
        
        with col2:
            st.subheader("📊 Current Indicators")
            
            # These would come from the bot's last calculation
            # For now, show placeholder
            st.markdown("""
            <div class='info-card'>
                <strong>Strategy Parameters</strong><br>
                📏 Entry Length: 40 periods<br>
                📏 Exit Length: 16 periods<br>
                📏 ATR Length: 20 periods<br>
                📏 Trail Multiplier: 4.0x ATR<br>
                📏 Risk per Trade: 1%<br>
                📏 Max Pyramids: 4 units
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🔗 Links")
            st.markdown(f"""
            - [📁 GitHub Repository](https://github.com/{GITHUB_USER}/{GITHUB_REPO})
            - [📊 Bot State (JSON)]({STATE_URL})
            - [📜 Trade History (JSON)]({TRADES_URL})
            """)
    
    # === TAB 5: LIQUIDATION HUNTER ===
    with tab5:
        st.subheader("🎯 Liquidation Cascade Hunter")
        st.markdown("""
        **Strategy Overview:**
        This is a highly defensive mean-reversion algorithm that hunts for intense Liquidation Cascades on Bybit Perpetuals.
        It uses Orderbook analytics and ATR volatility checks to snipe entries using Limit Orders after extreme capitulations.
        It runs entirely independently from the Turtle-Donchian trend-following systems.
        
        **Key Defenses:**
        - 🛑 Intraday Drawdown limit (5%)
        - 📅 Macro Event Blackouts (FOMC, CPI skips)
        - 📉 Real-time Bid/Ask Spread & Open Interest validation
        """)
        
        st.markdown("### 📊 Live Engine Metrics")
        try:
            import os
            # Attempt to read local LH logs
            lh_path = os.path.join("Trading Bot", "liquidation-hunter", "logs", "paper_readiness.json")
            if os.path.exists(lh_path):
                with open(lh_path, "r", encoding="utf-8") as f:
                    lh_stats = json.load(f)
                st.json(lh_stats)
            else:
                st.info("Liquidation Hunter metrics will appear here once the engine generates `paper_readiness.json`.")
        except Exception as e:
            st.error(f"Could not load Liquidation Hunter stats: {e}")
    
    # === SIDEBAR ===
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=80)
        st.markdown("## BTC Trading Bot")
        st.caption("V1 Turtle-Donchian Strategy")
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("### 💼 Portfolio")
        st.write(f"**Equity:** ${equity:,.2f}")
        st.write(f"**Returns:** {returns_pct:+.2f}%")
        st.write(f"**Trades:** {state.get('trade_count', 0)}")
        
        if live_price:
            st.markdown("---")
            st.markdown("### 📈 Market")
            st.write(f"**BTC Price:** ${live_price:,.2f}")
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.caption("Data refreshes every 30 seconds")
    
    # Auto-refresh script
    if auto_refresh:
        st.markdown("""
            <script>
                setTimeout(function(){
                    window.location.reload();
                }, 30000);
            </script>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

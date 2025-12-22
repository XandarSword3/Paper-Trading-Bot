"""
Streamlit Dashboard for BTC Trading Bot
Displays real-time equity, trades, and signals
Deploy free on Streamlit Cloud: https://share.streamlit.io
"""
import streamlit as st
import json
import requests
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="BTC Trading Bot",
    page_icon="📈",
    layout="wide"
)

# GitHub raw file URLs
GITHUB_USER = "XandarSword3"
GITHUB_REPO = "Paper-Trading-Bot"
GITHUB_BRANCH = "master"

STATE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/bot_state.json"
TRADES_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/trades.json"


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_state():
    """Load bot state from GitHub"""
    try:
        response = requests.get(STATE_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to load state: {e}")
    return None


@st.cache_data(ttl=60)
def load_trades():
    """Load trade history from GitHub"""
    try:
        response = requests.get(TRADES_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to load trades: {e}")
    return []


def main():
    st.title("BTC Trading Bot Dashboard")
    st.caption("V1 Turtle-Donchian Strategy | Runs every 4 hours via GitHub Actions")
    
    # Load data
    state = load_state()
    trades = load_trades()
    
    if not state:
        st.warning("Could not load bot state. Make sure GitHub URLs are configured correctly.")
        st.code(f"STATE_URL = {STATE_URL}")
        return
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Equity",
            f"${state['equity']:,.2f}",
            f"{((state['equity'] / state['initial_capital']) - 1) * 100:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Position",
            f"{state['position_size']:.5f} BTC",
            f"{len(state['position_units'])} units"
        )
    
    with col3:
        st.metric("Total Trades", state['trade_count'])
    
    with col4:
        last_run = state.get('last_run', 'Never')
        if last_run and last_run != 'Never':
            try:
                dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                last_run = dt.strftime('%Y-%m-%d %H:%M UTC')
            except:
                pass
        st.metric("Last Run", last_run)
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("Equity Curve")
        
        if trades:
            # Build equity curve from trades
            equity_data = [{"time": None, "equity": state['initial_capital']}]
            for trade in trades:
                if 'equity' in trade:
                    equity_data.append({
                        "time": trade.get('time', ''),
                        "equity": trade['equity']
                    })
            
            if len(equity_data) > 1:
                df = pd.DataFrame(equity_data)
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.dropna(subset=['time'])
                
                if not df.empty:
                    st.line_chart(df.set_index('time')['equity'])
                else:
                    st.info("Equity chart will appear after first trade")
            else:
                st.info("No trades yet - equity chart will appear after first trade")
        else:
            st.info("No trades yet - waiting for first signal")
        
        st.subheader("Trade History")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Format columns
            if 'time' in trades_df.columns:
                trades_df['time'] = pd.to_datetime(trades_df['time']).dt.strftime('%Y-%m-%d %H:%M')
            if 'price' in trades_df.columns:
                trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
            if 'pnl' in trades_df.columns:
                trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "-")
            if 'pnl_pct' in trades_df.columns:
                trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
            if 'quantity' in trades_df.columns:
                trades_df['quantity'] = trades_df['quantity'].apply(lambda x: f"{x:.5f}")
            
            # Show most recent first
            st.dataframe(
                trades_df[['time', 'type', 'reason', 'price', 'quantity', 'pnl', 'pnl_pct']].iloc[::-1],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trades executed yet")
    
    with right_col:
        st.subheader("Current Position")
        
        if state['position_size'] > 0:
            st.success(f"LONG: {state['position_size']:.5f} BTC")
            
            for i, unit in enumerate(state['position_units']):
                with st.expander(f"Unit #{i+1}", expanded=True):
                    st.write(f"**Entry:** ${unit['entry_price']:,.2f}")
                    st.write(f"**Qty:** {unit['quantity']:.5f} BTC")
                    st.write(f"**Stop:** ${unit['trailing_stop']:,.2f}")
                    if 'entry_time' in unit:
                        st.write(f"**Time:** {unit['entry_time'][:16]}")
        else:
            st.info("No open position - waiting for entry signal")
        
        st.subheader("Strategy Info")
        
        st.markdown("""
        **V1 Turtle-Donchian**
        - Entry: 40-period high breakout
        - Exit: 16-period low break
        - Trail: 4.0 x ATR
        - Risk: 1% per trade
        - Max units: 4 (pyramiding)
        - Long only
        
        **Schedule**
        - Runs every 4 hours
        - Checks at: 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
        """)
        
        st.subheader("Quick Stats")
        
        if trades:
            entries = [t for t in trades if t['type'] in ('ENTRY', 'PYRAMID')]
            exits = [t for t in trades if t['type'] == 'EXIT']
            
            wins = [t for t in exits if t.get('pnl', 0) > 0]
            losses = [t for t in exits if t.get('pnl', 0) < 0]
            
            win_rate = len(wins) / len(exits) * 100 if exits else 0
            total_pnl = sum(t.get('pnl', 0) for t in exits)
            
            st.write(f"**Entry trades:** {len(entries)}")
            st.write(f"**Closed trades:** {len(exits)}")
            st.write(f"**Win rate:** {win_rate:.1f}%")
            st.write(f"**Total P&L:** ${total_pnl:+,.2f}")
        else:
            st.write("No trades yet")
    
    # Footer
    st.divider()
    st.caption("Dashboard auto-refreshes every 60 seconds. Last page load: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Auto-refresh button
    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

"""
Liquidation Hunter Dashboard - 1m Cascade Reversion
Dedicated page for tracking Bybit Liquidation anomalies.
"""
import streamlit as st
import json
import os
from pathlib import Path

# Page config
st.set_page_config(page_title="Liquidation Hunter", page_icon="🎯", layout="wide")

# === CUSTOM CSS (Glassmorphism + Animations) ===
st.markdown("""
<style>
    /* Dark theme enhancements */
    .main { background-color: #0e1117; }
    
    /* Metric cards with Modern Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(13, 17, 23, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 136, 0.15);
    }
    
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
</style>
""", unsafe_allow_html=True)

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("# 🎯 Liquidation Hunter")
    st.caption("Mean-Reversion capitulation sniper | 1m Timeframe | Live Orderbook validation")
with col_status:
    st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <span class='status-live'></span>
            <span style='color: #00ff88; font-weight: bold;'>ENGINE ACTIVE</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

def load_metrics():
    """Load the JSON metrics from the Liquidation Hunter logs director"""
    # Assuming the user runs Streamlit from 'BTC Strategy' root directory
    filepath = Path("Trading Bot") / "liquidation-hunter" / "logs" / "paper_readiness.json"
    
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error parsing metrics: {e}")
            return None
    return None

data = load_metrics()

if data is None:
    st.warning("⚠️ Could not locate `paper_readiness.json`. The Liquidation Hunter engine might not have generated it yet. Please run the engine in paper mode first.")
else:
    # --- Metrics Section ---
    st.subheader("📊 Live Readiness Metrics")
    
    paper = data.get("paper_metrics", {})
    backtest = data.get("backtest_metrics", {})
    gates = data.get("gates", {})
    ready = data.get("ready_for_live", False)
    
    if ready:
        st.success("✅ Engine is formally **READY FOR LIVE DEPLOYMENT** according to statistical gates.")
    else:
        st.info("🕒 Engine is still undergoing paper validation bounds testing.")
        
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Trades (Paper)", paper.get("total_trades", 0))
    with cols[1]:
        st.metric("Win Rate (Paper)", f"{paper.get('win_rate', 0)*100:.1f}%")
    with cols[2]:
        st.metric("Sharpe (Paper)", f"{paper.get('sharpe', 0):.2f}")
    with cols[3]:
        st.metric("Avg R:R", f"{paper.get('avg_rr', 0):.2f}")
        
    st.markdown("---")
    
    # --- Advanced Breakdown ---
    st.subheader("🕵️ Deviation & Gate Checks")
    st.markdown("This compares the Live Paper results against the Historical Backtest baseline to ensure the alpha isn't degrading.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### 🧮 Baselines (Backtest)")
        st.write(f"**Baseline Win Rate:** {backtest.get('win_rate', 0)*100:.1f}%")
        st.write(f"**Baseline Sharpe:** {backtest.get('sharpe', 0):.2f}")
        st.write(f"**Baseline R:R:** {backtest.get('avg_rr', 0):.2f}")
        
    with col_g2:
        st.markdown("### 🚦 Statistical Gates")
        def render_gate(name, val):
            return "✅ Passed" if val else "❌ Failing/Waiting"
            
        st.write(f"**Minimum Trades Achieved:** {render_gate('min_trades', gates.get('min_trades'))}")
        st.write(f"**Sharpe Degradation OK:** {render_gate('min_sharpe', gates.get('min_sharpe'))}")
        st.write(f"**Win Rate Tolerance OK:** {render_gate('win_rate_tolerance', gates.get('win_rate_tolerance'))}")
        st.write(f"**Max Degradation Guard:** {render_gate('max_degradation', gates.get('max_degradation'))}")

st.markdown("---")
st.subheader("🔍 Local Debug Tooling")
st.write("To regenerate these metrics natively, start your engine using the integrated bridge:")
st.code("python liquidation_hunter_bridge.py --mode paper --iterations 100", language="bash")

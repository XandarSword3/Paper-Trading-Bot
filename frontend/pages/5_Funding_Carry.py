"""
Funding-Rate Carry — Forward Paper Track
Pre-gate diagnostic tracker (not a live/paper trading bot — see
research/bots/funding_carry_paper_bot.py docstring). Displays the
real, forward-only settlement data it has collected since going live.
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone

st.set_page_config(page_title="Funding Carry", page_icon="🔁", layout="wide")

GITHUB_USER = "XandarSword3"
GITHUB_REPO = "Paper-Trading-Bot"
GITHUB_BRANCH = "master"

STATE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data/funding_carry_paper_state.json"
TRADES_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data/funding_carry_paper_trades.json"


@st.cache_data(ttl=300)
def load_json(url: str):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


st.markdown("# 🔁 Funding-Rate Carry — Forward Paper Track")
st.caption(
    "Static long-spot / short-perp hedge. No signal, no capital at risk — "
    "this is real forward OKX settlement data, not a backtest. Started "
    "2026-07-25; every point here accumulated after that date."
)
st.markdown("---")

state = load_json(STATE_URL)
trades = load_json(TRADES_URL)

if state is None:
    st.warning(
        "No forward data yet — the first scheduled run (00:00 / 08:00 / 16:00 UTC) "
        "hasn't landed, or the workflow hasn't fired since this page last redeployed."
    )
    st.stop()

real_capital = state["real_capital"]
equity = state["equity"]
total_return_pct = (equity - real_capital) / real_capital * 100
started_at = datetime.fromisoformat(state["started_at"].replace("Z", "+00:00"))
days_running = (datetime.now(timezone.utc) - started_at).total_seconds() / 86400

col1, col2, col3, col4 = st.columns(4)
col1.metric("Equity", f"${equity:,.2f}", f"{total_return_pct:+.3f}%")
col2.metric("Real capital deployed", f"${real_capital:,.0f}", "2x notional, unlevered")
col3.metric("Settlements collected", state["num_settlements"])
col4.metric("Days running", f"{days_running:.1f}")

if state.get("last_settlement_applied"):
    st.caption(f"Last settlement applied: {state['last_settlement_applied']}")

st.markdown("---")

if trades:
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.subheader("Forward equity curve")
    st.line_chart(df.set_index("timestamp")["equity_after"])

    st.subheader("Settlement log")
    st.dataframe(
        df.sort_values("timestamp", ascending=False)[
            ["timestamp", "funding_rate", "pnl", "equity_after"]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No settlements recorded yet.")

st.markdown("---")
st.caption(
    "This page reads committed state directly from GitHub, so it lags the "
    "live repo by a few minutes (cache) plus however long the last scheduled "
    "Action took to commit. Days-running is not a validation period — treat "
    "any number here as provisional until there's enough history for it to "
    "mean something statistically."
)

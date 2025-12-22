"""
Interactive Strategy Analysis Chart
Click on any candle to see strategy fit percentage
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# Strategy parameters
ENTRY_LEN = 40
EXIT_LEN = 16
ATR_LEN = 20
TRAIL_MULT = 4.0

# Kraken API
KRAKEN_API = "https://api.kraken.com/0/public"
PAIR = "XBTUSD"


@st.cache_data(ttl=60)
def get_candles(interval=240, count=200):
    """Fetch candles from Kraken"""
    try:
        since = int((datetime.now(timezone.utc) - timedelta(hours=interval/60 * count)).timestamp())
        response = requests.get(
            f"{KRAKEN_API}/OHLC",
            params={"pair": PAIR, "interval": interval, "since": since},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("error"):
            return pd.DataFrame()
        
        result_key = list(data["result"].keys())[0]
        ohlc = data["result"][result_key]
        
        df = pd.DataFrame(ohlc, columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df.tail(count)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    """Calculate strategy indicators"""
    df = df.copy()
    
    # Donchian Channels
    df['entry_high'] = df['high'].shift(1).rolling(ENTRY_LEN).max()
    df['exit_low'] = df['low'].shift(1).rolling(EXIT_LEN).min()
    
    # ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['atr'] = df['tr'].rolling(ATR_LEN).mean()
    
    # Breakout signals
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['breakout'] = df['prev_high'] > df['entry_high']
    df['exit_signal'] = df['prev_low'] < df['exit_low']
    
    return df


def analyze_candle(row, df, idx):
    """
    Analyze a single candle and return strategy fit percentage (0-100%)
    
    Scoring Components:
    1. Distance to breakout (0-40 pts) - How close is price to entry?
    2. Bullish candle strength (0-20 pts) - Is this a strong bullish candle?
    3. Close position (0-20 pts) - Did price close near the high?
    4. Trend strength (0-20 pts) - Is the overall trend up?
    """
    score = 0
    breakdown = {}
    
    # Component 1: Distance to Entry (0-40 points)
    if pd.notna(row['entry_high']) and row['entry_high'] > 0:
        pct_to_entry = (row['close'] - row['entry_high']) / row['entry_high'] * 100
        
        if pct_to_entry > 0:  # Above breakout = perfect for entry
            dist_score = 40
            breakdown['distance'] = f"✅ Above breakout (+{pct_to_entry:.1f}%) → 40/40"
        elif pct_to_entry > -2:  # Within 2% of breakout
            dist_score = int(35 + pct_to_entry * 2.5)
            breakdown['distance'] = f"🟡 Near breakout ({pct_to_entry:.1f}%) → {dist_score}/40"
        elif pct_to_entry > -5:  # Within 5%
            dist_score = int(20 + pct_to_entry * 3)
            breakdown['distance'] = f"🟠 Approaching ({pct_to_entry:.1f}%) → {dist_score}/40"
        else:
            dist_score = max(0, int(5 + pct_to_entry))
            breakdown['distance'] = f"⚪ Far from breakout ({pct_to_entry:.1f}%) → {dist_score}/40"
        
        score += dist_score
    else:
        breakdown['distance'] = "❌ No entry level calculated"
    
    # Component 2: Bullish Candle (0-20 points)
    body = row['close'] - row['open']
    range_size = row['high'] - row['low'] if row['high'] != row['low'] else 1
    body_ratio = body / range_size
    
    if body_ratio > 0.6:  # Strong bullish (>60% body)
        bull_score = 20
        breakdown['bullish'] = f"✅ Strong bullish candle ({body_ratio*100:.0f}% body) → 20/20"
    elif body_ratio > 0.3:  # Moderate bullish
        bull_score = int(10 + body_ratio * 16)
        breakdown['bullish'] = f"🟡 Moderate bullish ({body_ratio*100:.0f}% body) → {bull_score}/20"
    elif body_ratio > 0:  # Weak bullish
        bull_score = int(body_ratio * 33)
        breakdown['bullish'] = f"🟠 Weak bullish ({body_ratio*100:.0f}% body) → {bull_score}/20"
    else:  # Bearish
        bull_score = 0
        breakdown['bullish'] = f"❌ Bearish candle ({body_ratio*100:.0f}%) → 0/20"
    
    score += bull_score
    
    # Component 3: Close Position (0-20 points)
    if range_size > 0:
        close_pct = (row['close'] - row['low']) / range_size
        
        if close_pct > 0.8:  # Closed in top 20%
            close_score = 20
            breakdown['close_pos'] = f"✅ Closed near high (top {(1-close_pct)*100:.0f}%) → 20/20"
        elif close_pct > 0.5:  # Closed in top half
            close_score = int(close_pct * 20)
            breakdown['close_pos'] = f"🟡 Closed upper half → {close_score}/20"
        else:  # Closed in lower half
            close_score = int(close_pct * 10)
            breakdown['close_pos'] = f"🟠 Closed lower half → {close_score}/20"
        
        score += close_score
    else:
        breakdown['close_pos'] = "⚪ Doji candle → 10/20"
        score += 10
    
    # Component 4: Trend Strength (0-20 points)
    if idx >= 10:
        lookback = df.iloc[idx-10:idx]
        if len(lookback) > 0:
            trend_start = lookback['close'].iloc[0]
            trend_end = row['close']
            trend_pct = (trend_end - trend_start) / trend_start * 100
            
            if trend_pct > 5:
                trend_score = 20
                breakdown['trend'] = f"✅ Strong uptrend (+{trend_pct:.1f}%) → 20/20"
            elif trend_pct > 0:
                trend_score = int(10 + trend_pct * 2)
                breakdown['trend'] = f"🟡 Uptrend (+{trend_pct:.1f}%) → {trend_score}/20"
            elif trend_pct > -5:
                trend_score = int(5 + trend_pct)
                breakdown['trend'] = f"🟠 Sideways ({trend_pct:+.1f}%) → {trend_score}/20"
            else:
                trend_score = 0
                breakdown['trend'] = f"❌ Downtrend ({trend_pct:.1f}%) → 0/20"
            
            score += trend_score
        else:
            breakdown['trend'] = "⚪ Not enough data"
    else:
        breakdown['trend'] = "⚪ Not enough history"
    
    # Classification
    if score >= 80:
        rating = "🔥 EXCELLENT"
        color = "#00ff00"
    elif score >= 60:
        rating = "✅ GOOD"
        color = "#90EE90"
    elif score >= 40:
        rating = "🟡 MODERATE"
        color = "#FFD700"
    elif score >= 20:
        rating = "🟠 WEAK"
        color = "#FFA500"
    else:
        rating = "❌ POOR"
        color = "#FF6B6B"
    
    return {
        'score': score,
        'rating': rating,
        'color': color,
        'breakdown': breakdown,
        'is_breakout': row.get('breakout', False),
        'is_exit': row.get('exit_signal', False)
    }


def create_chart(df):
    """Create interactive candlestick chart with signals"""
    
    # Calculate fit scores for all candles
    fit_scores = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        analysis = analyze_candle(row, df, idx)
        fit_scores.append(analysis['score'])
    
    df['fit_score'] = fit_scores
    
    # Create figure
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BTC/USD',
        increasing=dict(line=dict(color='#26A69A'), fillcolor='#26A69A'),
        decreasing=dict(line=dict(color='#EF5350'), fillcolor='#EF5350'),
        hoverinfo='text',
        text=[
            f"Time: {t.strftime('%Y-%m-%d %H:%M')}<br>"
            f"O: ${o:,.0f}<br>"
            f"H: ${h:,.0f}<br>"
            f"L: ${l:,.0f}<br>"
            f"C: ${c:,.0f}<br>"
            f"<b>Strategy Fit: {s}%</b>"
            for t, o, h, l, c, s in zip(
                df['time'], df['open'], df['high'], df['low'], df['close'], df['fit_score']
            )
        ]
    ))
    
    # Entry channel (40-period high)
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['entry_high'],
        mode='lines',
        name=f'Entry High ({ENTRY_LEN})',
        line=dict(color='#00FF00', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Exit channel (16-period low)
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['exit_low'],
        mode='lines',
        name=f'Exit Low ({EXIT_LEN})',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Mark breakouts
    breakouts = df[df['breakout'] == True]
    if len(breakouts) > 0:
        fig.add_trace(go.Scatter(
            x=breakouts['time'],
            y=breakouts['high'] * 1.005,
            mode='markers',
            name='Breakout Signal',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='#00FF00',
                line=dict(width=1, color='white')
            )
        ))
    
    # Mark exit signals
    exits = df[df['exit_signal'] == True]
    if len(exits) > 0:
        fig.add_trace(go.Scatter(
            x=exits['time'],
            y=exits['low'] * 0.995,
            mode='markers',
            name='Exit Signal',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='#FF6B6B',
                line=dict(width=1, color='white')
            )
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='🔍 Interactive Strategy Analysis - Click Any Candle',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor='#1E2329',
            title='',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#1E2329',
            title='Price ($)',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        height=600,
        hovermode='x unified'
    )
    
    return fig, df


def main():
    st.set_page_config(
        page_title="Strategy Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1f2c, #252b3a);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #2d3547;
        margin: 5px;
    }
    .score-excellent { color: #00ff00; font-size: 48px; font-weight: bold; }
    .score-good { color: #90EE90; font-size: 48px; font-weight: bold; }
    .score-moderate { color: #FFD700; font-size: 48px; font-weight: bold; }
    .score-weak { color: #FFA500; font-size: 48px; font-weight: bold; }
    .score-poor { color: #FF6B6B; font-size: 48px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Interactive Strategy Analysis")
    st.caption("Click on any candle to see detailed strategy fit analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        timeframe = st.selectbox(
            "Timeframe",
            options=[60, 240, 1440],
            format_func=lambda x: {60: "1 Hour", 240: "4 Hour", 1440: "1 Day"}[x],
            index=1
        )
        candle_count = st.slider("Candles to show", 50, 200, 100)
        
        st.divider()
        st.subheader("📋 Strategy Rules")
        st.markdown(f"""
        - **Entry**: High > {ENTRY_LEN}-period high
        - **Exit**: Low < {EXIT_LEN}-period low
        - **ATR**: {ATR_LEN}-period average
        - **Trail Stop**: {TRAIL_MULT}x ATR
        """)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
    
    # Fetch data
    df = get_candles(interval=timeframe, count=candle_count)
    
    if df.empty:
        st.error("Failed to load data")
        return
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create and display chart
    fig, df = create_chart(df)
    
    # Use plotly_chart with click events
    st.plotly_chart(fig, use_container_width=True)
    
    # Candle selector
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔎 Analyze Specific Candle")
        
        # Select candle by index
        valid_indices = list(range(max(0, ENTRY_LEN), len(df)))
        if valid_indices:
            selected_idx = st.slider(
                "Select candle",
                min_value=min(valid_indices),
                max_value=max(valid_indices),
                value=max(valid_indices),
                format="%d"
            )
            
            # Get selected candle
            row = df.iloc[selected_idx]
            analysis = analyze_candle(row, df, selected_idx)
            
            # Display analysis
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <h3>📅 {row['time'].strftime('%Y-%m-%d %H:%M')}</h3>
                <div class='score-{"excellent" if analysis["score"]>=80 else "good" if analysis["score"]>=60 else "moderate" if analysis["score"]>=40 else "weak" if analysis["score"]>=20 else "poor"}'>
                    {analysis['score']}%
                </div>
                <p style='font-size:18px;'>{analysis['rating']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Price info
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Open | ${row['open']:,.2f} |
            | High | ${row['high']:,.2f} |
            | Low | ${row['low']:,.2f} |
            | Close | ${row['close']:,.2f} |
            | Entry Level | ${row['entry_high']:,.2f} |
            | Exit Level | ${row['exit_low']:,.2f} |
            """)
            
            # Breakdown
            st.subheader("📊 Score Breakdown")
            for component, detail in analysis['breakdown'].items():
                st.markdown(f"- {detail}")
            
            # Signal status
            if analysis['is_breakout']:
                st.success("🚀 BREAKOUT SIGNAL - Entry conditions met!")
            elif analysis['is_exit']:
                st.error("🛑 EXIT SIGNAL - Exit conditions triggered!")
            else:
                st.info("⏳ No active signal - Monitoring...")
    
    # Stats summary
    st.divider()
    st.subheader("📈 Recent Signal Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Recent breakouts
    recent_breakouts = df[df['breakout'] == True].tail(5)
    with col1:
        st.metric("Breakouts (visible)", len(df[df['breakout'] == True]))
    
    # Recent exits
    with col2:
        st.metric("Exit Signals (visible)", len(df[df['exit_signal'] == True]))
    
    # Average fit score
    valid_scores = df['fit_score'].dropna()
    with col3:
        st.metric("Avg Fit Score", f"{valid_scores.mean():.0f}%" if len(valid_scores) > 0 else "N/A")
    
    # Current fit
    with col4:
        current_fit = df['fit_score'].iloc[-1] if len(df) > 0 else 0
        st.metric("Current Candle Fit", f"{current_fit}%")


if __name__ == "__main__":
    main()

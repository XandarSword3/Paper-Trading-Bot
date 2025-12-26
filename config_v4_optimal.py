"""
V4 OPTIMAL CONFIGURATION
High-Frequency Turtle-Donchian Strategy

Result: 1572% return vs V1's 855% (+717%)
Trades: ~1.33/day (vs V1's 0.14/day)
"""

# === V4 BEST PARAMETERS ===
V4_PARAMS = {
    "timeframe": "1H",           # 1-hour candles
    "entry_len": 8,              # 8-hour breakout (vs V1's 40 on 4H)
    "exit_len": 16,              # 16-hour exit channel
    "atr_len": 14,               # ATR period
    "trail_mult": 3.5,           # Trailing stop = 3.5x ATR
    "risk_pct": 1.0,             # 1% risk per trade
    "max_units": 4,              # Max pyramid units
    "long_only": True,           # Only long positions
}

# === PERFORMANCE COMPARISON ===
"""
Metric              V1 (4H)         V4 (1H)
----------------------------------------------
Total Return:       855%            1572%
Max Drawdown:       ~-55%           -65%
Trades/Day:         0.14            1.33
Win Rate:           ~40%            44%
Profit Factor:      ~1.8            1.27
Timeframe:          4H              1H
Entry Lookback:     40 bars         8 bars
                    (160 hours)     (8 hours)
"""

# === KEY DIFFERENCES FROM V1 ===
"""
1. TIMEFRAME: 1H vs 4H (4x more data points)
2. ENTRY: 8-hour breakout vs 160-hour (V1's 40 * 4H)
   - Much faster to catch breakouts
   - More trades but smaller individual moves
   
3. EXIT: 16-hour channel vs 64-hour (V1's 16 * 4H)
   - Faster exits preserve profits
   
4. TRAILING STOP: 3.5x ATR vs 4.0x
   - Slightly tighter stops for faster timeframe
   
5. TRADE FREQUENCY: 10x more trades
   - V1: ~52 trades/year
   - V4: ~485 trades/year
"""

# === USAGE ===
"""
To backtest with these parameters:

from strategy_v4_fast import fast_backtest, load_data
import pandas as pd

df = load_data()
high = df['high'].values
low = df['low'].values
close = df['close'].values

# Pre-compute indicators
entry_high = pd.Series(high).rolling(8).max().shift(1).values
exit_low = pd.Series(low).rolling(16).min().shift(1).values
tr = ... # Calculate ATR
atr = pd.Series(tr).rolling(14).mean().values

result = fast_backtest(
    high, low, close, entry_high, exit_low, atr,
    entry_len=8, trail_mult=3.5, risk_pct=1.0
)
print(f"Return: {result['return']:.1f}%")
"""

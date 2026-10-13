"""
Technical Indicator Suite for Quantitative Research.
Indicators:
1. Bollinger Bands (20, 2)
2. RSI (length 6)
3. StochRSI (14, 14, 3, 3)
4. MACD (12, 26, 9)
5. Exponential Moving Averages: 20, 50, 80, 200
6. ADX (14) with +DI / -DI
7. Volume MA (20)
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1. Bollinger Bands (20, 2.0)
    # ddof=0 (population standard deviation) is required to reproduce Binance's own
    # published bands: at the validated bar (SOLUSDT 3m, 2025-05-01 12:27 UTC) ddof=0
    # is within 0.0035 of UP/MB/DN, while ddof=1 is 0.0107 off on UP and 0.0077 off on
    # DN. See quant_research/indicator_validation.py and REPORT section 17.
    df["bb_basis"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["bb_upper"] = df["bb_basis"] + 2.0 * bb_std
    df["bb_lower"] = df["bb_basis"] - 2.0 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_basis"]

    # 2. RSI (length 6)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain_6 = gain.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    avg_loss_6 = loss.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    rs_6 = avg_gain_6 / avg_loss_6.replace(0, np.nan)
    df["rsi6"] = (100.0 - (100.0 / (1.0 + rs_6))).fillna(50.0)

    # 3. StochRSI (14, 14, 3, 3)
    avg_gain_14 = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss_14 = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
    rsi14 = (100.0 - (100.0 / (1.0 + rs_14))).fillna(50.0)
    rsi14_min = rsi14.rolling(14).min()
    rsi14_max = rsi14.rolling(14).max()
    rsi_range = (rsi14_max - rsi14_min).replace(0, np.nan)
    stoch = (rsi14 - rsi14_min) / rsi_range
    df["stoch_k"] = (stoch.rolling(3).mean() * 100.0).fillna(50.0)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(50.0)

    # 4. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # 5. EMAs: 20, 50, 80, 200
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema80"] = close.ewm(span=80, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # 6. ADX (14)
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    df["atr14"] = atr14

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()

    df["plus_di"] = (100.0 * (plus_dm_s / atr14.replace(0, np.nan))).fillna(0.0)
    df["minus_di"] = (100.0 * (minus_dm_s / atr14.replace(0, np.nan))).fillna(0.0)
    di_sum = (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    dx = (100.0 * (df["plus_di"] - df["minus_di"]).abs() / di_sum).fillna(0.0)
    df["adx14"] = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    # 7. Volume MA (20)
    df["vol_ma20"] = volume.rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    return df

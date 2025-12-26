"""
V3 FAST - Vectorized High-Frequency Turtle Strategy
Optimized for speed using numpy/pandas vectorization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from datetime import datetime
import os
import time
import requests
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings('ignore')

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def fetch_1h_data() -> pd.DataFrame:
    """Fetch 1H data with caching"""
    cache_file = os.path.join(DATA_DIR, "BTCUSDT_1h.parquet")
    
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if len(df) > 50000:
            print(f"Loaded cached data: {len(df)} candles")
            return df
    
    print("Fetching from Binance...")
    start_ts = int(datetime(2017, 8, 1).timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    with tqdm(desc="Downloading", total=80) as pbar:
        while current_ts < end_ts:
            try:
                response = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "1h",
                        "startTime": current_ts,
                        "endTime": end_ts,
                        "limit": 1000
                    },
                    timeout=10
                )
                data = response.json()
                if not data:
                    break
                all_data.extend(data)
                current_ts = data[-1][0] + 1
                pbar.update(1)
                time.sleep(0.1)
            except:
                time.sleep(1)
    
    df = pd.DataFrame(all_data)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume",
                  "close_time", "quote_volume", "trades", "taker_buy_base",
                  "taker_buy_quote", "ignore"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_parquet(cache_file)
    return df


def fast_backtest(
    df: pd.DataFrame,
    entry_len: int,
    exit_len: int,
    atr_len: int,
    trail_mult: float,
    trend_len: int,
    risk_pct: float,
    max_units: int = 4,
    initial_capital: float = 10000
) -> Dict:
    """
    Fast vectorized backtest
    Returns performance metrics
    """
    n = len(df)
    
    # Pre-calculate indicators
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Donchian channels
    entry_high = pd.Series(high).shift(1).rolling(entry_len).max().values
    exit_low = pd.Series(low).shift(1).rolling(exit_len).min().values
    
    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).rolling(atr_len).mean().values
    
    # Trend filter
    trend_ma = pd.Series(close).rolling(trend_len).mean().values
    trend_up = close > trend_ma
    
    # Warmup
    warmup = max(entry_len, trend_len, atr_len) + 10
    
    # State
    equity = initial_capital
    position = 0.0
    units = []  # List of (entry_price, size, stop)
    
    num_trades = 0
    wins = 0
    total_profit = 0.0
    total_loss = 0.0
    peak_equity = equity
    max_dd = 0.0
    
    for i in range(warmup, n):
        price = close[i]
        prev_high = high[i-1]
        prev_low = low[i-1]
        curr_atr = atr[i]
        
        if np.isnan(entry_high[i]) or np.isnan(curr_atr) or curr_atr <= 0:
            continue
        
        # === EXITS ===
        if position > 0:
            exit_triggered = False
            
            # Donchian exit
            if prev_low < exit_low[i]:
                exit_triggered = True
            
            # Trailing stop
            if not exit_triggered:
                for unit in units:
                    if price <= unit[2]:  # stop price
                        exit_triggered = True
                        break
            
            if exit_triggered:
                total_cost = sum(u[0] * u[1] for u in units)
                exit_value = price * position
                pnl = exit_value - total_cost
                equity += pnl
                
                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    total_loss += abs(pnl)
                
                num_trades += 1
                position = 0.0
                units = []
            else:
                # Update trailing stops
                new_units = []
                for entry_price, size, stop in units:
                    new_stop = max(stop, price - trail_mult * curr_atr)
                    new_units.append((entry_price, size, new_stop))
                units = new_units
        
        # === ENTRIES ===
        if len(units) < max_units and trend_up[i]:
            # Breakout signal
            if prev_high > entry_high[i]:
                # Pyramid spacing check
                can_enter = True
                if units:
                    last_entry = units[-1][0]
                    if price < last_entry + curr_atr:
                        can_enter = False
                
                if can_enter:
                    risk = equity * (risk_pct / 100)
                    stop_dist = trail_mult * curr_atr
                    unit_size = risk / stop_dist
                    
                    # Limit size
                    max_size = (equity * 0.25) / price
                    unit_size = min(unit_size, max_size)
                    
                    if unit_size * price > 10:
                        stop = price - stop_dist
                        units.append((price, unit_size, stop))
                        position += unit_size
        
        # Track drawdown
        total_value = equity + position * price if position > 0 else equity
        if total_value > peak_equity:
            peak_equity = total_value
        dd = (total_value - peak_equity) / peak_equity * 100
        if dd < max_dd:
            max_dd = dd
    
    # Close remaining position
    if position > 0:
        final_price = close[-1]
        total_cost = sum(u[0] * u[1] for u in units)
        pnl = final_price * position - total_cost
        equity += pnl
        if pnl > 0:
            wins += 1
            total_profit += pnl
        else:
            total_loss += abs(pnl)
        num_trades += 1
    
    total_return = (equity / initial_capital - 1) * 100
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 999
    
    days = (n - warmup) / 24  # hours to days
    trades_per_day = num_trades / days if days > 0 else 0
    
    return {
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': num_trades,
        'trades_per_day': trades_per_day,
        'final_equity': equity
    }


def optimize_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Grid search optimization"""
    
    # Parameter grid
    entry_lens = [12, 16, 20, 24, 30, 36, 48]
    exit_lens = [4, 6, 8, 10, 12]
    trail_mults = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    trend_lens = [50, 100, 150, 200]
    risk_pcts = [0.5, 1.0, 1.5, 2.0]
    
    combinations = list(itertools.product(
        entry_lens, exit_lens, trail_mults, trend_lens, risk_pcts
    ))
    
    print(f"\nOptimizing V3 ({len(combinations)} combinations)...")
    
    results = []
    
    for entry_len, exit_len, trail_mult, trend_len, risk_pct in tqdm(combinations):
        result = fast_backtest(
            df, entry_len, exit_len, 14, trail_mult, trend_len, risk_pct
        )
        
        # Score: Return - 0.5 * |MaxDD| (penalize drawdown)
        score = result['return'] - 0.5 * abs(result['max_dd'])
        
        results.append({
            'entry_len': entry_len,
            'exit_len': exit_len,
            'trail_mult': trail_mult,
            'trend_len': trend_len,
            'risk_pct': risk_pct,
            'return': result['return'],
            'max_dd': result['max_dd'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'trades': result['trades'],
            'trades_per_day': result['trades_per_day'],
            'score': score
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'v3_optimization.csv'), index=False)
    
    return results_df


def main():
    print("=" * 70)
    print("V3 FAST - ENHANCED TURTLE STRATEGY")
    print("Goal: Beat V1's 855% return with higher frequency")
    print("=" * 70)
    
    # Load data
    df = fetch_1h_data()
    print(f"\nData: {len(df)} hourly candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Optimize
    results = optimize_v3(df)
    
    print("\n" + "=" * 70)
    print("TOP 20 RESULTS")
    print("=" * 70)
    print(results.head(20).to_string())
    
    # Best result
    best = results.iloc[0]
    
    print("\n" + "=" * 70)
    print("BEST V3 CONFIGURATION")
    print("=" * 70)
    print(f"  Entry Length: {int(best['entry_len'])} hours")
    print(f"  Exit Length:  {int(best['exit_len'])} hours")
    print(f"  Trail Mult:   {best['trail_mult']}x ATR")
    print(f"  Trend Filter: {int(best['trend_len'])} hours MA")
    print(f"  Risk/Trade:   {best['risk_pct']}%")
    
    print("\n  PERFORMANCE:")
    print(f"    Total Return:  {best['return']:.1f}%")
    print(f"    Max Drawdown:  {best['max_dd']:.1f}%")
    print(f"    Win Rate:      {best['win_rate']:.1f}%")
    print(f"    Profit Factor: {best['profit_factor']:.2f}")
    print(f"    Total Trades:  {int(best['trades'])}")
    print(f"    Trades/Day:    {best['trades_per_day']:.2f}")
    
    # Compare to V1
    v1_return = 855
    print("\n" + "=" * 70)
    print("COMPARISON TO V1")
    print("=" * 70)
    print(f"  V1 Return: {v1_return}%")
    print(f"  V3 Return: {best['return']:.1f}%")
    
    if best['return'] > v1_return:
        print(f"\n  ✅ V3 BEATS V1 by {best['return'] - v1_return:.1f}%!")
    else:
        gap = v1_return - best['return']
        print(f"\n  Gap to V1: {gap:.1f}%")
        print("  → Need to iterate further")


if __name__ == "__main__":
    main()

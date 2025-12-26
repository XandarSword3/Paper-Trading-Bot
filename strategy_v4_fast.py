"""
V4 ULTRA-FAST - Fully Vectorized Turtle Strategy
Uses pandas rolling for speed, then iterates only for trading logic
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
import requests
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load cached 1H data"""
    cache_file = os.path.join(DATA_DIR, "BTCUSDT_1h.parquet")
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    raise FileNotFoundError("Run strategy_v4.py first to download data")


def fast_backtest(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_high: np.ndarray,
    exit_low: np.ndarray,
    atr: np.ndarray,
    entry_len: int,
    trail_mult: float,
    risk_pct: float,
    max_units: int = 4,
    initial_capital: float = 10000
) -> dict:
    """Ultra-fast backtest using pre-computed indicators"""
    
    n = len(close)
    warmup = entry_len + 50
    
    equity = initial_capital
    position = 0.0
    units = []  # [(entry_price, size, stop)]
    
    num_entries = 0
    wins = 0
    total_profit = 0.0
    total_loss = 0.0
    peak_equity = equity
    max_dd = 0.0
    
    for i in range(warmup, n):
        price = close[i]
        
        if np.isnan(entry_high[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        curr_atr = atr[i]
        
        # === EXITS ===
        if position > 0:
            exit_triggered = False
            
            # Donchian exit
            if low[i] < exit_low[i]:
                exit_triggered = True
            
            # Trailing stop
            if not exit_triggered:
                for u in units:
                    if price <= u[2]:
                        exit_triggered = True
                        break
            
            if exit_triggered:
                total_cost = sum(u[0] * u[1] for u in units)
                pnl = price * position - total_cost
                equity += pnl
                
                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    total_loss += abs(pnl)
                
                position = 0.0
                units = []
            else:
                # Update stops
                units = [(ep, sz, max(st, price - trail_mult * curr_atr)) for ep, sz, st in units]
        
        # === ENTRIES ===
        if len(units) < max_units:
            if high[i] > entry_high[i]:
                can_enter = True
                if units and price < units[-1][0] + curr_atr:
                    can_enter = False
                
                if can_enter:
                    risk = equity * (risk_pct / 100)
                    stop_dist = trail_mult * curr_atr
                    unit_size = min(risk / stop_dist, (equity * 0.25) / price)
                    
                    if unit_size * price > 10:
                        units.append((price, unit_size, price - stop_dist))
                        position += unit_size
                        num_entries += 1
        
        # Drawdown
        val = equity + position * price if position > 0 else equity
        if val > peak_equity:
            peak_equity = val
        dd = (val - peak_equity) / peak_equity * 100
        if dd < max_dd:
            max_dd = dd
    
    # Close remaining
    if position > 0:
        pnl = close[-1] * position - sum(u[0] * u[1] for u in units)
        equity += pnl
        if pnl > 0:
            wins += 1
            total_profit += pnl
        else:
            total_loss += abs(pnl)
    
    days = (n - warmup) / 24
    
    return {
        'return': (equity / initial_capital - 1) * 100,
        'max_dd': max_dd,
        'win_rate': (wins / max(1, num_entries // max_units)) * 100,
        'profit_factor': total_profit / max(0.01, total_loss),
        'trades': num_entries,
        'trades_per_day': num_entries / max(1, days),
        'equity': equity
    }


def optimize(df: pd.DataFrame):
    """Fast grid search"""
    
    print("Pre-computing indicators...")
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Pre-compute ATR once
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr_14 = pd.Series(tr).rolling(14).mean().values
    
    # Parameter grid
    entry_lens = [8, 12, 16, 20, 24, 30, 40, 48, 60, 80]
    exit_lens = [4, 6, 8, 10, 12, 16, 20]
    trail_mults = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    risk_pcts = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    combos = list(itertools.product(entry_lens, exit_lens, trail_mults, risk_pcts))
    print(f"Testing {len(combos)} combinations...")
    
    results = []
    
    for entry_len, exit_len, trail_mult, risk_pct in tqdm(combos):
        # Compute entry/exit levels
        entry_high = pd.Series(high).rolling(entry_len).max().shift(1).values
        exit_low = pd.Series(low).rolling(exit_len).min().shift(1).values
        
        result = fast_backtest(
            high, low, close, entry_high, exit_low, atr_14,
            entry_len, trail_mult, risk_pct
        )
        
        score = result['return'] - 0.5 * abs(result['max_dd'])
        
        results.append({
            'entry': entry_len,
            'exit': exit_len,
            'trail': trail_mult,
            'risk': risk_pct,
            'return': result['return'],
            'dd': result['max_dd'],
            'win_rate': result['win_rate'],
            'pf': result['profit_factor'],
            'trades': result['trades'],
            'tpd': result['trades_per_day'],
            'score': score
        })
    
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'v4_fast_results.csv'), index=False)
    
    return results_df


def main():
    print("=" * 70)
    print("V4 ULTRA-FAST OPTIMIZATION")
    print("=" * 70)
    
    df = load_data()
    print(f"Data: {len(df)} candles, {df.index[0]} to {df.index[-1]}")
    
    results = optimize(df)
    
    print("\n" + "=" * 70)
    print("TOP 20 CONFIGURATIONS")
    print("=" * 70)
    print(results.head(20).to_string())
    
    best = results.iloc[0]
    
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Entry: {int(best['entry'])}h | Exit: {int(best['exit'])}h | Trail: {best['trail']}x | Risk: {best['risk']}%")
    print(f"\n  Return: {best['return']:.1f}%")
    print(f"  Max DD: {best['dd']:.1f}%")
    print(f"  Trades: {int(best['trades'])} ({best['tpd']:.2f}/day)")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['pf']:.2f}")
    
    # Compare to V1
    v1 = 855
    print("\n" + "=" * 70)
    if best['return'] > v1:
        print(f"✅ BEATS V1 ({v1}%) by {best['return']-v1:.1f}%!")
    else:
        print(f"Gap to V1: {v1 - best['return']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

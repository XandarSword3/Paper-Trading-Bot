"""
V4 TURTLE STRATEGY - Fixed and Enhanced
Targeting ~1 trade/day with > 855% return

Key fixes from V3:
- Fixed signal detection logic
- Optional trend filter
- Better debugging
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


def backtest_v4(
    df: pd.DataFrame,
    entry_len: int,
    exit_len: int,
    atr_len: int,
    trail_mult: float,
    risk_pct: float,
    max_units: int = 4,
    use_trend_filter: bool = False,
    trend_len: int = 100,
    initial_capital: float = 10000,
    debug: bool = False
) -> dict:
    """
    Backtest V4 strategy
    """
    n = len(df)
    
    # Arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Indicators
    entry_high = np.full(n, np.nan)
    exit_low = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    trend_ma = np.full(n, np.nan)
    
    # Calculate rolling indicators
    for i in range(max(entry_len, exit_len, atr_len, trend_len) + 1, n):
        # Entry: max of previous entry_len highs (not including current)
        entry_high[i] = np.max(high[i-entry_len:i])
        
        # Exit: min of previous exit_len lows
        exit_low[i] = np.min(low[i-exit_len:i])
        
        # ATR
        trs = []
        for j in range(i-atr_len, i):
            tr = max(
                high[j] - low[j],
                abs(high[j] - close[j-1]),
                abs(low[j] - close[j-1])
            )
            trs.append(tr)
        atr[i] = np.mean(trs)
        
        # Trend MA
        trend_ma[i] = np.mean(close[i-trend_len:i])
    
    # Warmup
    warmup = max(entry_len, exit_len, atr_len, trend_len) + 10
    
    # State
    equity = initial_capital
    position = 0.0
    units = []  # [(entry_price, size, stop), ...]
    
    num_entries = 0
    num_exits = 0
    wins = 0
    total_profit = 0.0
    total_loss = 0.0
    peak_equity = equity
    max_dd = 0.0
    
    trades_log = []
    
    for i in range(warmup, n):
        price = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        if np.isnan(entry_high[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        curr_atr = atr[i]
        
        # === CHECK EXITS ===
        if position > 0:
            exit_triggered = False
            exit_reason = ""
            
            # Donchian exit: current low breaks exit_low
            if curr_low < exit_low[i]:
                exit_triggered = True
                exit_reason = "Donchian"
            
            # Trailing stop
            if not exit_triggered:
                for unit in units:
                    if price <= unit[2]:  # stop
                        exit_triggered = True
                        exit_reason = "Stop"
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
                
                num_exits += 1
                
                if debug and num_exits <= 20:
                    trades_log.append(f"EXIT {exit_reason}: price={price:.2f}, pnl={pnl:.2f}, equity={equity:.2f}")
                
                position = 0.0
                units = []
            else:
                # Update trailing stops
                new_units = []
                for entry_price, size, stop in units:
                    new_stop = max(stop, price - trail_mult * curr_atr)
                    new_units.append((entry_price, size, new_stop))
                units = new_units
        
        # === CHECK ENTRIES ===
        if len(units) < max_units:
            # Trend filter (optional)
            trend_ok = True
            if use_trend_filter:
                trend_ok = price > trend_ma[i]
            
            # Breakout: current high breaks entry_high
            if curr_high > entry_high[i] and trend_ok:
                # Pyramid check
                can_enter = True
                if units:
                    last_entry = units[-1][0]
                    if price < last_entry + curr_atr:
                        can_enter = False
                
                if can_enter:
                    # Position sizing
                    risk = equity * (risk_pct / 100)
                    stop_dist = trail_mult * curr_atr
                    unit_size = risk / stop_dist
                    
                    # Limit
                    max_size = (equity * 0.25) / price
                    unit_size = min(unit_size, max_size)
                    
                    if unit_size * price > 10:
                        stop = price - stop_dist
                        units.append((price, unit_size, stop))
                        position += unit_size
                        num_entries += 1
                        
                        if debug and num_entries <= 20:
                            trades_log.append(f"ENTRY: price={price:.2f}, size={unit_size:.6f}, stop={stop:.2f}")
        
        # Drawdown tracking
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
        num_exits += 1
    
    # Metrics
    total_return = (equity / initial_capital - 1) * 100
    win_rate = (wins / num_exits * 100) if num_exits > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 999
    
    days = (n - warmup) / 24
    trades_per_day = num_entries / days if days > 0 else 0
    
    if debug:
        print(f"\nDEBUG: {num_entries} entries, {num_exits} exits")
        for log in trades_log[:10]:
            print(f"  {log}")
    
    return {
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': num_entries,
        'trades_per_day': trades_per_day,
        'final_equity': equity
    }


def optimize_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive optimization"""
    
    # Wide parameter ranges
    entry_lens = [8, 12, 16, 20, 24, 30, 40, 48, 60]
    exit_lens = [4, 6, 8, 10, 12, 16]
    trail_mults = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    risk_pcts = [0.5, 1.0, 1.5, 2.0]
    use_trend = [False, True]
    trend_lens = [50, 100, 150]
    
    # Build combinations
    combinations = []
    for entry_len, exit_len, trail_mult, risk_pct, trend, trend_len in itertools.product(
        entry_lens, exit_lens, trail_mults, risk_pcts, use_trend, trend_lens
    ):
        if not trend:
            if trend_len != 50:  # Avoid duplicates when not using trend
                continue
        combinations.append((entry_len, exit_len, trail_mult, risk_pct, trend, trend_len))
    
    print(f"\nOptimizing V4 ({len(combinations)} combinations)...")
    
    results = []
    
    for entry_len, exit_len, trail_mult, risk_pct, use_trend, trend_len in tqdm(combinations):
        result = backtest_v4(
            df,
            entry_len=entry_len,
            exit_len=exit_len,
            atr_len=14,
            trail_mult=trail_mult,
            risk_pct=risk_pct,
            use_trend_filter=use_trend,
            trend_len=trend_len
        )
        
        # Score: prioritize return, penalize drawdown
        score = result['return'] - 0.5 * abs(result['max_dd'])
        
        results.append({
            'entry_len': entry_len,
            'exit_len': exit_len,
            'trail_mult': trail_mult,
            'risk_pct': risk_pct,
            'use_trend': use_trend,
            'trend_len': trend_len if use_trend else 0,
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
    results_df.to_csv(os.path.join(RESULTS_DIR, 'v4_optimization.csv'), index=False)
    
    return results_df


def main():
    print("=" * 70)
    print("V4 TURTLE STRATEGY - FIXED")
    print("Goal: Beat V1's 855% return with ~1 trade/day")
    print("=" * 70)
    
    # Load data
    df = fetch_1h_data()
    print(f"\nData: {len(df)} hourly candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Quick sanity check
    print("\n--- Sanity Check (no filters, aggressive params) ---")
    test = backtest_v4(
        df,
        entry_len=20,
        exit_len=10,
        atr_len=14,
        trail_mult=3.0,
        risk_pct=1.0,
        use_trend_filter=False,
        debug=True
    )
    print(f"Return: {test['return']:.1f}%, Trades: {test['trades']}, DD: {test['max_dd']:.1f}%")
    
    # Full optimization
    results = optimize_v4(df)
    
    print("\n" + "=" * 70)
    print("TOP 20 RESULTS")
    print("=" * 70)
    print(results.head(20).to_string())
    
    # Best result
    best = results.iloc[0]
    
    print("\n" + "=" * 70)
    print("BEST V4 CONFIGURATION")
    print("=" * 70)
    print(f"  Entry Length: {int(best['entry_len'])} hours")
    print(f"  Exit Length:  {int(best['exit_len'])} hours")
    print(f"  Trail Mult:   {best['trail_mult']}x ATR")
    print(f"  Risk/Trade:   {best['risk_pct']}%")
    print(f"  Trend Filter: {'Yes, ' + str(int(best['trend_len'])) + 'h MA' if best['use_trend'] else 'No'}")
    
    print("\n  PERFORMANCE:")
    print(f"    Total Return:  {best['return']:.1f}%")
    print(f"    Max Drawdown:  {best['max_dd']:.1f}%")
    print(f"    Win Rate:      {best['win_rate']:.1f}%")
    print(f"    Profit Factor: {best['profit_factor']:.2f}")
    print(f"    Total Trades:  {int(best['trades'])}")
    print(f"    Trades/Day:    {best['trades_per_day']:.2f}")
    
    # V1 comparison
    v1_return = 855
    print("\n" + "=" * 70)
    print("COMPARISON TO V1")
    print("=" * 70)
    print(f"  V1 Return: {v1_return}%")
    print(f"  V4 Return: {best['return']:.1f}%")
    
    if best['return'] > v1_return:
        print(f"\n  ✅ V4 BEATS V1 by {best['return'] - v1_return:.1f}%!")
    else:
        gap = v1_return - best['return']
        print(f"\n  Gap to V1: {gap:.1f}%")
        print("  → Trying more aggressive parameters...")


if __name__ == "__main__":
    main()

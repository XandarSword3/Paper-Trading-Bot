"""
V3 Enhanced High-Frequency Turtle Strategy
Goal: Beat V1's 855% return with ~1 trade/day frequency

Key improvements over V2:
1. Trend filter (only trade in direction of higher timeframe trend)
2. Volatility filter (skip low-volatility chop)
3. Better stop management
4. Smarter pyramiding
5. Wider parameter search
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import os
import time
import requests
from tqdm import tqdm
import itertools

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# DATA FETCHER
# ============================================================================

def fetch_1h_data(start_date: str = "2017-08-01", end_date: str = "2025-12-26") -> pd.DataFrame:
    """Fetch 1H data from Binance with caching"""
    cache_file = os.path.join(DATA_DIR, "BTCUSDT_1h.parquet")
    
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if len(df) > 1000:  # Valid cache
            print(f"Loaded cached data: {len(df)} candles")
            return df
    
    print(f"Fetching 1H data from Binance...")
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    with tqdm(desc="Downloading", total=74) as pbar:
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
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_parquet(cache_file)
    print(f"Saved {len(df)} candles")
    
    return df


# ============================================================================
# V3 STRATEGY PARAMETERS
# ============================================================================

@dataclass
class V3Params:
    """V3 Strategy Parameters"""
    # Core Donchian
    entry_len: int = 20          # Breakout lookback
    exit_len: int = 10           # Exit lookback
    atr_len: int = 14            # ATR period
    
    # Trailing stop
    trail_mult: float = 3.0      # ATR multiplier for stop
    
    # Risk management
    risk_percent: float = 1.0    # Risk per trade
    max_units: int = 4           # Max pyramid units
    
    # FILTERS (new in V3)
    trend_len: int = 100         # Higher TF trend filter (100 hours = ~4 days)
    volatility_filter: bool = True
    vol_min_atr: float = 0.5     # Min ATR% to take trades
    vol_max_atr: float = 8.0     # Max ATR% (avoid extreme volatility)
    
    # Trade direction
    long_only: bool = True       # Only long trades (BTC bias)


# ============================================================================
# V3 BACKTEST ENGINE
# ============================================================================

class V3Backtester:
    """
    V3 Enhanced Backtester with:
    - Trend filter
    - Volatility filter
    - Better stop management
    """
    
    def __init__(self, params: V3Params):
        self.params = params
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = df.copy()
        
        # Donchian Channels (shifted by 1 for no lookahead)
        df['entry_high'] = df['high'].shift(1).rolling(self.params.entry_len).max()
        df['exit_low'] = df['low'].shift(1).rolling(self.params.exit_len).min()
        
        # ATR
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        df['atr'] = df['tr'].rolling(self.params.atr_len).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as % of price
        
        # Trend filter: price above/below long-term MA
        df['trend_ma'] = df['close'].rolling(self.params.trend_len).mean()
        df['trend_up'] = df['close'] > df['trend_ma']
        df['trend_dn'] = df['close'] < df['trend_ma']
        
        # Momentum: rate of change
        df['roc'] = df['close'].pct_change(20) * 100
        
        return df
    
    def run(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Run backtest"""
        df = self.calculate_indicators(df)
        
        # Warmup period
        warmup = max(self.params.entry_len, self.params.trend_len, self.params.atr_len) + 10
        
        if len(df) <= warmup:
            return {'total_return': 0, 'trades': 0, 'max_dd': -100}
        
        # State
        equity = initial_capital
        position = 0.0
        units = []  # List of {entry_price, size, stop}
        trades = []
        peak_equity = equity
        max_dd = 0
        
        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = row['close']
            
            # Skip if no indicators
            if pd.isna(row['entry_high']) or pd.isna(row['atr']):
                continue
            
            atr = row['atr']
            atr_pct = row['atr_pct']
            
            # === VOLATILITY FILTER ===
            if self.params.volatility_filter:
                if atr_pct < self.params.vol_min_atr or atr_pct > self.params.vol_max_atr:
                    # Still check exits but skip entries
                    pass
            
            # === CHECK EXITS (always check) ===
            if position > 0:
                exit_triggered = False
                exit_reason = ""
                
                # Donchian exit
                if prev['low'] < row['exit_low']:
                    exit_triggered = True
                    exit_reason = "Donchian Exit"
                
                # Trailing stop check
                if not exit_triggered:
                    for unit in units:
                        if price <= unit['stop']:
                            exit_triggered = True
                            exit_reason = "Trailing Stop"
                            break
                
                if exit_triggered:
                    # Close all units
                    total_cost = sum(u['entry_price'] * u['size'] for u in units)
                    exit_value = price * position
                    pnl = exit_value - total_cost
                    equity += pnl
                    
                    trades.append({
                        'type': 'EXIT',
                        'time': row.name,
                        'price': price,
                        'pnl': pnl,
                        'reason': exit_reason,
                        'equity': equity
                    })
                    
                    position = 0.0
                    units = []
                
                # Update trailing stops (move up, never down)
                else:
                    for unit in units:
                        new_stop = price - self.params.trail_mult * atr
                        if new_stop > unit['stop']:
                            unit['stop'] = new_stop
            
            # === CHECK ENTRIES ===
            if len(units) < self.params.max_units:
                # Volatility filter for entries
                vol_ok = True
                if self.params.volatility_filter:
                    vol_ok = self.params.vol_min_atr <= atr_pct <= self.params.vol_max_atr
                
                # Trend filter for entries
                trend_ok = True
                if self.params.long_only:
                    trend_ok = row['trend_up']
                
                # Breakout signal
                if prev['high'] > row['entry_high'] and vol_ok and trend_ok:
                    # Check pyramid spacing
                    can_enter = True
                    if units:
                        last_entry = units[-1]['entry_price']
                        if price < last_entry + atr:  # Must be 1 ATR above last entry
                            can_enter = False
                    
                    if can_enter:
                        # Position sizing based on risk
                        risk = equity * (self.params.risk_percent / 100)
                        stop_dist = self.params.trail_mult * atr
                        unit_size = risk / stop_dist
                        
                        # Limit size
                        max_size = (equity * 0.25) / price  # Max 25% per unit
                        unit_size = min(unit_size, max_size)
                        
                        if unit_size * price > 10:  # Min $10 trade
                            stop = price - stop_dist
                            
                            units.append({
                                'entry_price': price,
                                'size': unit_size,
                                'stop': stop
                            })
                            position += unit_size
                            
                            trades.append({
                                'type': 'ENTRY',
                                'time': row.name,
                                'price': price,
                                'size': unit_size,
                                'units': len(units),
                                'equity': equity
                            })
            
            # Track drawdown
            total_value = equity + position * price if position > 0 else equity
            if total_value > peak_equity:
                peak_equity = total_value
            dd = (total_value - peak_equity) / peak_equity * 100
            if dd < max_dd:
                max_dd = dd
        
        # Close any remaining position
        if position > 0:
            final_price = df.iloc[-1]['close']
            total_cost = sum(u['entry_price'] * u['size'] for u in units)
            equity += final_price * position - total_cost
        
        # Calculate stats
        entry_trades = [t for t in trades if t['type'] == 'ENTRY']
        exit_trades = [t for t in trades if t['type'] == 'EXIT']
        
        wins = len([t for t in exit_trades if t.get('pnl', 0) > 0])
        total_exits = len(exit_trades)
        win_rate = (wins / total_exits * 100) if total_exits > 0 else 0
        
        total_profit = sum(t['pnl'] for t in exit_trades if t.get('pnl', 0) > 0)
        total_loss = abs(sum(t['pnl'] for t in exit_trades if t.get('pnl', 0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 999
        
        total_return = (equity / initial_capital - 1) * 100
        
        days = (df.index[-1] - df.index[warmup]).days
        trades_per_day = len(entry_trades) / days if days > 0 else 0
        
        return {
            'total_return': total_return,
            'final_equity': equity,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(entry_trades),
            'trades_per_day': trades_per_day,
            'trades': trades
        }


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_v3(df: pd.DataFrame, initial_capital: float = 10000) -> Tuple[V3Params, Dict]:
    """
    Grid search for best V3 parameters
    """
    
    # Parameter ranges
    entry_lens = [16, 20, 24, 30, 36]
    exit_lens = [6, 8, 10, 12]
    trail_mults = [2.0, 2.5, 3.0, 3.5, 4.0]
    trend_lens = [50, 100, 150, 200]
    risk_pcts = [0.5, 1.0, 1.5]
    
    combinations = list(itertools.product(
        entry_lens, exit_lens, trail_mults, trend_lens, risk_pcts
    ))
    
    print(f"\nOptimizing V3 ({len(combinations)} combinations)...")
    
    best_result = None
    best_params = None
    best_score = -float('inf')
    
    results = []
    
    for entry_len, exit_len, trail_mult, trend_len, risk_pct in tqdm(combinations):
        params = V3Params(
            entry_len=entry_len,
            exit_len=exit_len,
            trail_mult=trail_mult,
            trend_len=trend_len,
            risk_percent=risk_pct,
            volatility_filter=True,
            long_only=True
        )
        
        bt = V3Backtester(params)
        result = bt.run(df, initial_capital)
        
        # Score: prioritize return while penalizing drawdown
        # Score = Return - 0.5 * |MaxDD|
        score = result['total_return'] - 0.5 * abs(result['max_dd'])
        
        results.append({
            'entry_len': entry_len,
            'exit_len': exit_len,
            'trail_mult': trail_mult,
            'trend_len': trend_len,
            'risk_pct': risk_pct,
            'return': result['total_return'],
            'max_dd': result['max_dd'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'trades': result['total_trades'],
            'trades_per_day': result['trades_per_day'],
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'v3_optimization.csv'), index=False)
    
    return best_params, best_result, results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V3 ENHANCED TURTLE STRATEGY")
    print("Goal: Beat V1's 855% return with ~1 trade/day")
    print("=" * 70)
    
    # Fetch data
    df = fetch_1h_data()
    print(f"\nData: {len(df)} hourly candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Optimize
    best_params, best_result, results_df = optimize_v3(df)
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE - TOP 10 RESULTS")
    print("=" * 70)
    print(results_df.head(10).to_string())
    
    print("\n" + "=" * 70)
    print("BEST V3 PARAMETERS")
    print("=" * 70)
    print(f"  Entry Length: {best_params.entry_len} hours")
    print(f"  Exit Length:  {best_params.exit_len} hours")
    print(f"  Trail Mult:   {best_params.trail_mult}x ATR")
    print(f"  Trend Filter: {best_params.trend_len} hours MA")
    print(f"  Risk/Trade:   {best_params.risk_percent}%")
    
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    print(f"  Total Return:  {best_result['total_return']:.1f}%")
    print(f"  Max Drawdown:  {best_result['max_dd']:.1f}%")
    print(f"  Win Rate:      {best_result['win_rate']:.1f}%")
    print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
    print(f"  Total Trades:  {best_result['total_trades']}")
    print(f"  Trades/Day:    {best_result['trades_per_day']:.2f}")
    
    # Compare to V1
    print("\n" + "=" * 70)
    print("COMPARISON TO V1")
    print("=" * 70)
    v1_return = 855
    print(f"  V1 Return: {v1_return}%")
    print(f"  V3 Return: {best_result['total_return']:.1f}%")
    
    if best_result['total_return'] > v1_return:
        print(f"\n  ✅ V3 BEATS V1 by {best_result['total_return'] - v1_return:.1f}%!")
    else:
        print(f"\n  ❌ V3 underperforms by {v1_return - best_result['total_return']:.1f}%")
        print("  → Need further iteration")
    
    return best_params, best_result


if __name__ == "__main__":
    main()

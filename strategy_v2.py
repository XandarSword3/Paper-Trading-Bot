"""
V2 High-Frequency Turtle-Donchian Strategy
Same core logic as V1, but tuned for 1H timeframe and ~1 trade/day

Run this to:
1. Fetch 1H data
2. Run optimization to find best parameters
3. Backtest with optimized params
4. Compare to V1 results
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
from datetime import datetime
import os
import time
import requests
from tqdm import tqdm

from config_v2 import (
    StrategyParamsV2, BacktestConfigV2, OptimizationRangesV2,
    DEFAULT_PARAMS_V2, DEFAULT_BACKTEST_V2, DATA_DIR, RESULTS_DIR
)


# ============================================================================
# DATA FETCHER (1H specific)
# ============================================================================

class DataFetcherV2:
    """Fetches 1H historical data from Binance"""
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.cache_file = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
    
    def fetch(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """Fetch 1H data, using cache if available"""
        
        if use_cache and os.path.exists(self.cache_file):
            df = pd.read_parquet(self.cache_file)
            print(f"Loaded cached data: {len(df)} candles")
            
            # Check if we need to update
            last_date = df.index[-1].strftime("%Y-%m-%d")
            if last_date >= end_date:
                return df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                print(f"Cache ends at {last_date}, fetching updates...")
                start_date = last_date
        
        df = self._fetch_from_binance(start_date, end_date)
        
        # Save to cache
        df.to_parquet(self.cache_file)
        print(f"Saved {len(df)} candles to cache")
        
        return df
    
    def _fetch_from_binance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download from Binance API"""
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        limit = 1000
        
        interval_ms = 3600 * 1000  # 1 hour
        total_batches = max(1, (end_ts - start_ts) // (limit * interval_ms) + 1)
        
        print(f"Fetching {self.symbol} 1H data from {start_date} to {end_date}...")
        
        with tqdm(total=total_batches, desc="Downloading 1H data") as pbar:
            while current_ts < end_ts:
                params = {
                    "symbol": self.symbol,
                    "interval": "1h",
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": limit
                }
                
                try:
                    response = requests.get(self.BASE_URL, params=params, timeout=10)
                    response.raise_for_status()
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
                    continue
        
        if not all_data:
            raise ValueError("No data fetched")
        
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
        return df


# ============================================================================
# STRATEGY V2 (Same logic, different defaults)
# ============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    exit_reason: Optional[str]
    unit_number: int


class TurtleStrategyV2:
    """
    V2 High-Frequency Turtle Strategy
    Same core logic as V1, tuned for 1H timeframe
    """
    
    def __init__(self, params: StrategyParamsV2 = None):
        self.params = params or DEFAULT_PARAMS_V2
        self.trades: List[TradeRecord] = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        df = df.copy()
        
        # Donchian channels
        df['upper_entry'] = df['high'].shift(1).rolling(self.params.entry_len).max()
        df['lower_entry'] = df['low'].shift(1).rolling(self.params.entry_len).min()
        df['upper_exit'] = df['high'].shift(1).rolling(self.params.exit_len).max()
        df['lower_exit'] = df['low'].shift(1).rolling(self.params.exit_len).min()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(self.params.atr_len).mean()
        
        # 200 EMA for regime filter
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        return df
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        verbose: bool = False
    ) -> dict:
        """Run backtest and return results"""
        
        df = self.calculate_indicators(df)
        
        # State
        equity = initial_capital
        position_size = 0.0
        avg_entry = 0.0
        units = 0
        last_add_price = 0.0
        highest = 0.0
        
        self.trades = []
        current_units = []
        equity_curve = []
        
        warmup = max(self.params.entry_len, self.params.atr_len, 200) + 1
        
        for i in range(len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            
            if i < warmup:
                equity_curve.append(equity)
                continue
            
            close = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            upper_entry = row['upper_entry']
            lower_exit = row['lower_exit']
            ema200 = row['ema200']
            
            if pd.isna(atr) or pd.isna(upper_entry):
                equity_curve.append(equity + position_size * close)
                continue
            
            # === EXIT LOGIC ===
            if position_size > 0:
                highest = max(highest, high)
                trail_stop = highest - self.params.trail_mult * atr
                
                # Trailing stop
                if low <= trail_stop:
                    exit_price = trail_stop * (1 - 0.0013)  # costs
                    pnl = position_size * (exit_price - avg_entry)
                    equity += pnl
                    
                    for unit in current_units:
                        unit['exit_time'] = ts
                        unit['exit_price'] = exit_price
                        unit['pnl'] = unit['qty'] * (exit_price - unit['entry'])
                        unit['exit_reason'] = 'Trail Stop'
                        self.trades.append(TradeRecord(
                            entry_time=unit['entry_time'],
                            exit_time=ts,
                            direction='long',
                            entry_price=unit['entry'],
                            exit_price=exit_price,
                            quantity=unit['qty'],
                            pnl=unit['pnl'],
                            exit_reason='Trail Stop',
                            unit_number=unit['unit_num']
                        ))
                    
                    position_size = 0
                    current_units = []
                    units = 0
                    highest = 0
                
                # Donchian exit
                elif low <= lower_exit:
                    exit_price = lower_exit * (1 - 0.0013)
                    pnl = position_size * (exit_price - avg_entry)
                    equity += pnl
                    
                    for unit in current_units:
                        self.trades.append(TradeRecord(
                            entry_time=unit['entry_time'],
                            exit_time=ts,
                            direction='long',
                            entry_price=unit['entry'],
                            exit_price=exit_price,
                            quantity=unit['qty'],
                            pnl=unit['qty'] * (exit_price - unit['entry']),
                            exit_reason='Donchian Exit',
                            unit_number=unit['unit_num']
                        ))
                    
                    position_size = 0
                    current_units = []
                    units = 0
                    highest = 0
            
            # === ENTRY LOGIC ===
            if units < self.params.max_units:
                # Regime filter
                if self.params.use_regime_filter and close < ema200:
                    equity_curve.append(equity + position_size * close)
                    continue
                
                # Entry signal
                if high > upper_entry:
                    can_add = False
                    
                    if units == 0:
                        can_add = True
                    elif close >= last_add_price + self.params.pyramid_spacing_n * atr:
                        can_add = True
                    
                    if can_add:
                        # Position sizing
                        risk_usd = equity * (self.params.risk_percent / 100)
                        stop_dist = self.params.size_stop_mult * atr
                        unit_size = risk_usd / stop_dist if stop_dist > 0 else 0
                        unit_size = max(0.001, round(unit_size, 4))
                        
                        entry_price = upper_entry * (1 + 0.0013)  # costs
                        
                        # Update position
                        total_cost = avg_entry * position_size + entry_price * unit_size
                        position_size += unit_size
                        avg_entry = total_cost / position_size if position_size > 0 else 0
                        units += 1
                        last_add_price = entry_price
                        highest = high
                        
                        current_units.append({
                            'entry_time': ts,
                            'entry': entry_price,
                            'qty': unit_size,
                            'unit_num': units
                        })
            
            equity_curve.append(equity + position_size * close)
        
        # Close any remaining position
        if position_size > 0:
            final_price = df.iloc[-1]['close']
            pnl = position_size * (final_price - avg_entry)
            equity += pnl
            equity_curve[-1] = equity
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=df.index)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity - initial_capital) / initial_capital * 100
        
        # Drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        max_dd = drawdown.min()
        
        # Trade stats
        trade_pnls = [t.pnl for t in self.trades if t.pnl is not None]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        
        win_rate = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # Sharpe
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(8760)  # 1H = 8760/year
        else:
            sharpe = 0
        
        # Trades per day
        days = (df.index[-1] - df.index[warmup]).days
        trades_per_day = len(self.trades) / days if days > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'trades_per_day': trades_per_day,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity,
            'equity_curve': equity_series,
            'trades': self.trades
        }


# ============================================================================
# OPTIMIZER
# ============================================================================

def optimize_v2(df: pd.DataFrame, initial_capital: float = 100_000.0) -> dict:
    """
    Grid search optimization for V2 parameters
    Target: ~1 trade per day with good risk-adjusted returns
    """
    
    ranges = OptimizationRangesV2()
    
    entry_lens = range(ranges.entry_len[0], ranges.entry_len[1]+1, ranges.entry_len[2])
    exit_lens = range(ranges.exit_len[0], ranges.exit_len[1]+1, ranges.exit_len[2])
    trail_mults = np.arange(ranges.trail_mult[0], ranges.trail_mult[1]+0.1, ranges.trail_mult[2])
    
    results = []
    total = len(list(entry_lens)) * len(list(exit_lens)) * len(list(trail_mults))
    
    print(f"\nOptimizing V2 parameters ({total} combinations)...")
    
    for entry_len in range(ranges.entry_len[0], ranges.entry_len[1]+1, ranges.entry_len[2]):
        for exit_len in range(ranges.exit_len[0], ranges.exit_len[1]+1, ranges.exit_len[2]):
            if exit_len >= entry_len:
                continue
            
            for trail_mult in np.arange(ranges.trail_mult[0], ranges.trail_mult[1]+0.1, ranges.trail_mult[2]):
                params = StrategyParamsV2(
                    entry_len=entry_len,
                    exit_len=exit_len,
                    trail_mult=trail_mult
                )
                
                strategy = TurtleStrategyV2(params)
                result = strategy.run_backtest(df, initial_capital)
                
                # Score: balance return, drawdown, and trade frequency
                # Target ~1 trade/day = 0.5-1.5 trades/day is ideal
                freq_score = 1.0 if 0.5 <= result['trades_per_day'] <= 1.5 else 0.5
                
                score = (
                    result['total_return'] / 100 * 0.3 +  # Return weight
                    (100 + result['max_drawdown']) / 100 * 0.3 +  # DD weight (less negative = better)
                    result['sharpe'] * 0.2 +  # Sharpe weight
                    freq_score * 0.2  # Frequency weight
                )
                
                results.append({
                    'entry_len': entry_len,
                    'exit_len': exit_len,
                    'trail_mult': trail_mult,
                    'return': result['total_return'],
                    'max_dd': result['max_drawdown'],
                    'sharpe': result['sharpe'],
                    'trades': result['total_trades'],
                    'trades_per_day': result['trades_per_day'],
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor'],
                    'score': score
                })
    
    # Sort by score
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)
    
    best = results_df.iloc[0]
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest V2 Parameters:")
    print(f"  Entry Length: {int(best['entry_len'])} hours")
    print(f"  Exit Length:  {int(best['exit_len'])} hours")
    print(f"  Trail Mult:   {best['trail_mult']:.1f}x ATR")
    print(f"\nPerformance:")
    print(f"  Total Return:  {best['return']:.1f}%")
    print(f"  Max Drawdown:  {best['max_dd']:.1f}%")
    print(f"  Sharpe Ratio:  {best['sharpe']:.2f}")
    print(f"  Win Rate:      {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total Trades:  {int(best['trades'])}")
    print(f"  Trades/Day:    {best['trades_per_day']:.2f}")
    
    return {
        'best_params': {
            'entry_len': int(best['entry_len']),
            'exit_len': int(best['exit_len']),
            'trail_mult': best['trail_mult']
        },
        'best_result': best.to_dict(),
        'all_results': results_df
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("V2 HIGH-FREQUENCY TURTLE STRATEGY")
    print("Target: ~1 trade per day (vs V1's ~1 trade per week)")
    print("="*70)
    
    # Fetch 1H data
    fetcher = DataFetcherV2()
    config = DEFAULT_BACKTEST_V2
    
    df = fetcher.fetch(config.start_date, config.end_date)
    print(f"\nData: {len(df)} hourly candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Run optimization
    opt_result = optimize_v2(df, config.initial_capital)
    
    # Run final backtest with best params
    best_params = StrategyParamsV2(**opt_result['best_params'])
    strategy = TurtleStrategyV2(best_params)
    result = strategy.run_backtest(df, config.initial_capital)
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, "v2_optimization_results.csv")
    opt_result['all_results'].to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Compare to V1 target
    print("\n" + "="*70)
    print("COMPARISON TO V1")
    print("="*70)
    print(f"V1 (4H, Entry=40): ~52 trades/year, ~0.14 trades/day")
    print(f"V2 (1H, Entry={best_params.entry_len}): ~{result['trades_per_day']*365:.0f} trades/year, {result['trades_per_day']:.2f} trades/day")
    print(f"\nFrequency multiplier: {result['trades_per_day'] / 0.14:.1f}x")
    
    return result


if __name__ == "__main__":
    main()

"""
Enhanced Strategy V3 - Selective Filtering
Only apply filters during historically bad periods, otherwise trade aggressively
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, PLOTS_DIR
from data_fetcher import download_btc_data

os.makedirs(PLOTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 1_000.0


@dataclass
class V3Params:
    """V3 strategy - selective protection only in severe bear markets"""
    # Original Donchian params
    entry_len: int = 40
    exit_len: int = 16
    atr_len: int = 20
    trail_mult: float = 4.0
    size_stop_mult: float = 2.0
    risk_percent: float = 1.0
    max_units: int = 4
    pyramid_spacing_n: float = 1.5
    
    # V3: Selective filters - only active in severe conditions
    severe_bear_sma_slope: float = -0.02  # 200 SMA falling >2% over lookback
    slope_lookback: int = 30              # Days to check slope
    min_price_vs_sma: float = 0.90        # Price must be at least 90% of 200 SMA
    
    # Reduce size instead of stopping trades
    bear_size_reduction: float = 0.5      # Use 50% size in mild bear
    severe_bear_size: float = 0.0         # 0 = don't trade, or 0.25 = 25% size
    
    # Execution
    long_only: bool = True
    commission_pct: float = 0.08
    slippage_pct: float = 0.05
    lot_step: float = 0.001


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: str
    quantity: float
    stop_loss: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    market_mode: Optional[str] = None


class V3Strategy:
    """V3 Strategy - Only pause in SEVERE bear markets"""
    
    def __init__(self, params: V3Params):
        self.params = params
        self.trades: List[Trade] = []
        self.position = 0
        self.active_trades: List[Trade] = []
        self.equity = 0
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators with selective bear market detection"""
        df = df.copy()
        
        # Original Donchian indicators
        df['donchian_high'] = df['high'].rolling(self.params.entry_len).max()
        df['donchian_low'] = df['low'].rolling(self.params.entry_len).min()
        df['exit_low'] = df['low'].rolling(self.params.exit_len).min()
        df['exit_high'] = df['high'].rolling(self.params.exit_len).max()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.params.atr_len).mean()
        
        # 200 SMA
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # SMA slope (percentage change)
        df['sma_slope'] = (df['sma_200'] - df['sma_200'].shift(self.params.slope_lookback)) / df['sma_200'].shift(self.params.slope_lookback)
        
        # Price vs SMA ratio
        df['price_vs_sma'] = df['close'] / df['sma_200']
        
        # Market mode classification
        df['market_mode'] = 'Normal'
        
        # Mild bear: Below SMA but not severely
        df.loc[df['price_vs_sma'] < 1.0, 'market_mode'] = 'Mild Bear'
        
        # Severe bear: Below SMA AND SMA falling
        severe_bear = (df['price_vs_sma'] < self.params.min_price_vs_sma) & (df['sma_slope'] < self.params.severe_bear_sma_slope)
        df.loc[severe_bear, 'market_mode'] = 'Severe Bear'
        
        # Bull: Above SMA
        df.loc[df['price_vs_sma'] > 1.05, 'market_mode'] = 'Bull'
        
        return df
    
    def get_size_multiplier(self, market_mode: str) -> float:
        """Get position size multiplier based on market mode"""
        if market_mode == 'Severe Bear':
            return self.params.severe_bear_size
        elif market_mode == 'Mild Bear':
            return self.params.bear_size_reduction
        else:
            return 1.0
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 1000.0, verbose: bool = False) -> pd.DataFrame:
        """Run V3 backtest"""
        df = self.calculate_indicators(df)
        
        self.equity = initial_capital
        self.trades = []
        self.active_trades = []
        self.position = 0
        
        equity_curve = []
        warmup = 210
        
        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            current_price = row['close']
            atr = row['atr']
            market_mode = row['market_mode']
            
            if pd.isna(atr) or atr <= 0:
                equity_curve.append({'timestamp': row.name, 'equity': self.equity, 'mode': market_mode})
                continue
            
            # Check exits first
            trades_to_close = []
            for trade in self.active_trades:
                should_exit = False
                exit_reason = None
                
                # Trailing stop
                trail_stop = current_price - self.params.trail_mult * atr
                if trade.direction == 'long':
                    trade.stop_loss = max(trade.stop_loss, trail_stop)
                    if row['low'] <= trade.stop_loss:
                        should_exit = True
                        exit_reason = "Trailing Stop"
                    elif row['low'] <= prev_row['exit_low']:
                        should_exit = True
                        exit_reason = "Donchian Exit"
                
                if should_exit:
                    exit_price = min(current_price, trade.stop_loss) if exit_reason == "Trailing Stop" else current_price
                    exit_price *= (1 - self.params.slippage_pct / 100)
                    
                    pnl = (exit_price - trade.entry_price) * trade.quantity
                    pnl -= trade.entry_price * trade.quantity * self.params.commission_pct / 100
                    pnl -= exit_price * trade.quantity * self.params.commission_pct / 100
                    
                    trade.exit_time = row.name
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.exit_reason = exit_reason
                    
                    self.equity += pnl
                    self.trades.append(trade)
                    trades_to_close.append(trade)
            
            for t in trades_to_close:
                self.active_trades.remove(t)
            
            self.position = sum(t.quantity for t in self.active_trades)
            
            # Get size multiplier for current market mode
            size_mult = self.get_size_multiplier(market_mode)
            
            # Check entries
            if size_mult > 0 and self.position == 0:
                # Entry signal
                if row['high'] >= prev_row['donchian_high']:
                    risk_per_trade = self.equity * self.params.risk_percent / 100
                    stop_distance = self.params.size_stop_mult * atr
                    
                    if stop_distance > 0:
                        base_size = risk_per_trade / stop_distance
                        quantity = base_size * size_mult
                        quantity = max(self.params.lot_step, round(quantity / self.params.lot_step) * self.params.lot_step)
                        
                        entry_price = prev_row['donchian_high'] * (1 + self.params.slippage_pct / 100)
                        stop_loss = entry_price - stop_distance
                        
                        if quantity * entry_price <= self.equity * 0.95:
                            trade = Trade(
                                entry_time=row.name,
                                entry_price=entry_price,
                                direction='long',
                                quantity=quantity,
                                stop_loss=stop_loss,
                                market_mode=market_mode
                            )
                            self.active_trades.append(trade)
            
            # Pyramiding
            elif size_mult > 0 and len(self.active_trades) < self.params.max_units:
                if row['high'] >= prev_row['donchian_high']:
                    last_entry = self.active_trades[-1].entry_price if self.active_trades else 0
                    pyramid_threshold = last_entry + self.params.pyramid_spacing_n * atr
                    
                    if current_price >= pyramid_threshold:
                        risk_per_trade = self.equity * self.params.risk_percent / 100
                        stop_distance = self.params.size_stop_mult * atr
                        
                        if stop_distance > 0:
                            base_size = risk_per_trade / stop_distance
                            quantity = base_size * size_mult
                            quantity = max(self.params.lot_step, round(quantity / self.params.lot_step) * self.params.lot_step)
                            
                            entry_price = current_price * (1 + self.params.slippage_pct / 100)
                            stop_loss = entry_price - stop_distance
                            
                            total_position_value = sum(t.quantity * t.entry_price for t in self.active_trades)
                            if (total_position_value + quantity * entry_price) <= self.equity * 0.95:
                                trade = Trade(
                                    entry_time=row.name,
                                    entry_price=entry_price,
                                    direction='long',
                                    quantity=quantity,
                                    stop_loss=stop_loss,
                                    market_mode=market_mode
                                )
                                self.active_trades.append(trade)
            
            unrealized_pnl = sum((current_price - t.entry_price) * t.quantity for t in self.active_trades)
            total_equity = self.equity + unrealized_pnl
            equity_curve.append({'timestamp': row.name, 'equity': total_equity, 'mode': market_mode})
        
        # Close remaining positions
        if self.active_trades:
            final_price = df.iloc[-1]['close']
            for trade in self.active_trades:
                pnl = (final_price - trade.entry_price) * trade.quantity
                pnl -= trade.entry_price * trade.quantity * self.params.commission_pct / 100
                pnl -= final_price * trade.quantity * self.params.commission_pct / 100
                
                trade.exit_time = df.index[-1]
                trade.exit_price = final_price
                trade.pnl = pnl
                trade.exit_reason = "End of Backtest"
                self.equity += pnl
                self.trades.append(trade)
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        return equity_df


def optimize_v3(df: pd.DataFrame, initial_capital: float = 1000.0):
    """Test different V3 parameter combinations"""
    
    print("=" * 80)
    print("V3 OPTIMIZATION - Finding Best Selective Filter Settings")
    print("=" * 80)
    
    # Parameter grid
    severe_bear_sizes = [0.0, 0.25, 0.5]  # 0 = pause, 0.25 = 25% size, 0.5 = 50% size
    bear_size_reductions = [0.5, 0.75, 1.0]  # Mild bear adjustments
    min_price_vs_sma = [0.85, 0.90, 0.95]  # Threshold for severe bear
    
    results = []
    
    # Also run original for comparison
    from strategy import TurtleDonchianStrategy
    from config import StrategyParams
    
    print("\nRunning original strategy for baseline...")
    orig_params = StrategyParams(
        entry_len=40, exit_len=16, atr_len=20, trail_mult=4.0,
        size_stop_mult=2.0, risk_percent=1.0, max_units=4,
        pyramid_spacing_n=1.5, long_only=True, use_regime_filter=True,
        lot_step=0.001, commission_pct=0.08, slippage_pct=0.05
    )
    orig_strategy = TurtleDonchianStrategy(orig_params)
    orig_results = orig_strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
    orig_equity = orig_results['equity']
    orig_trades = [t for t in orig_strategy.trades]
    
    orig_max_dd = ((orig_equity.cummax() - orig_equity) / orig_equity.cummax()).max() * 100
    orig_return = (orig_equity.iloc[-1] / initial_capital - 1) * 100
    
    results.append({
        'name': 'Original V1',
        'severe_bear_size': 1.0,
        'bear_reduction': 1.0,
        'min_price_sma': 0,
        'final_equity': orig_equity.iloc[-1],
        'return': orig_return,
        'max_dd': orig_max_dd,
        'trades': len(orig_trades),
        'sharpe': 0,
        'return_dd_ratio': orig_return / orig_max_dd if orig_max_dd > 0 else 0,
    })
    
    print(f"  Original: Return={orig_return:.1f}%, MaxDD={orig_max_dd:.1f}%")
    
    # Test V3 combinations
    print("\nTesting V3 parameter combinations...")
    total_combos = len(severe_bear_sizes) * len(bear_size_reductions) * len(min_price_vs_sma)
    combo_num = 0
    
    for sbs in severe_bear_sizes:
        for bsr in bear_size_reductions:
            for mps in min_price_vs_sma:
                combo_num += 1
                
                params = V3Params(
                    severe_bear_size=sbs,
                    bear_size_reduction=bsr,
                    min_price_vs_sma=mps,
                )
                
                strategy = V3Strategy(params)
                equity_df = strategy.run_backtest(df, initial_capital=initial_capital)
                
                final_equity = equity_df['equity'].iloc[-1]
                returns = equity_df['equity'].pct_change().dropna()
                max_dd = ((equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()).max() * 100
                total_return = (final_equity / initial_capital - 1) * 100
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6) if returns.std() > 0 else 0
                
                results.append({
                    'name': f'V3_sbs{sbs}_bsr{bsr}_mps{mps}',
                    'severe_bear_size': sbs,
                    'bear_reduction': bsr,
                    'min_price_sma': mps,
                    'final_equity': final_equity,
                    'return': total_return,
                    'max_dd': max_dd,
                    'trades': len(strategy.trades),
                    'sharpe': sharpe,
                    'return_dd_ratio': total_return / max_dd if max_dd > 0 else 0,
                })
                
                print(f"  [{combo_num}/{total_combos}] SevBear={sbs}, MildBear={bsr}, MinPrice={mps}: " +
                      f"Return={total_return:.1f}%, MaxDD={max_dd:.1f}%, Trades={len(strategy.trades)}")
    
    # Find best configurations
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("TOP 5 BY RETURN")
    print("=" * 80)
    top_return = results_df.nlargest(5, 'return')
    print(top_return[['name', 'return', 'max_dd', 'trades', 'return_dd_ratio']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP 5 BY RISK-ADJUSTED RETURN (Return/MaxDD)")
    print("=" * 80)
    top_risk_adj = results_df.nlargest(5, 'return_dd_ratio')
    print(top_risk_adj[['name', 'return', 'max_dd', 'trades', 'return_dd_ratio']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP 5 BY LOWEST MAX DRAWDOWN")
    print("=" * 80)
    top_dd = results_df.nsmallest(5, 'max_dd')
    print(top_dd[['name', 'return', 'max_dd', 'trades', 'return_dd_ratio']].to_string(index=False))
    
    # Find the sweet spot - best return with acceptable drawdown
    acceptable_dd = results_df[results_df['max_dd'] < 35]  # Max 35% drawdown
    if len(acceptable_dd) > 0:
        best_in_dd_range = acceptable_dd.nlargest(1, 'return').iloc[0]
        print(f"\n🏆 BEST WITH <35% DRAWDOWN: {best_in_dd_range['name']}")
        print(f"   Return: {best_in_dd_range['return']:.1f}%")
        print(f"   Max DD: {best_in_dd_range['max_dd']:.1f}%")
        print(f"   Trades: {best_in_dd_range['trades']}")
    
    return results_df


def run_best_v3_with_yearly(df: pd.DataFrame, initial_capital: float = 1000.0):
    """Run the best V3 configuration and show yearly comparison"""
    
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: ORIGINAL vs BEST V3")
    print("=" * 80)
    
    # Original
    from strategy import TurtleDonchianStrategy
    from config import StrategyParams
    
    orig_params = StrategyParams(
        entry_len=40, exit_len=16, atr_len=20, trail_mult=4.0,
        size_stop_mult=2.0, risk_percent=1.0, max_units=4,
        pyramid_spacing_n=1.5, long_only=True, use_regime_filter=True,
        lot_step=0.001, commission_pct=0.08, slippage_pct=0.05
    )
    orig_strategy = TurtleDonchianStrategy(orig_params)
    orig_results = orig_strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
    orig_equity = orig_results['equity']
    orig_trades_df = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0
    } for t in orig_strategy.trades])
    
    # Best V3 - Original without any filters (since that performed best)
    # But let's try the severe bear pause only version
    v3_params = V3Params(
        severe_bear_size=0.0,     # Pause in severe bear
        bear_size_reduction=1.0,  # Normal size in mild bear
        min_price_vs_sma=0.85,    # Only trigger if 15% below SMA
    )
    v3_strategy = V3Strategy(v3_params)
    v3_equity = v3_strategy.run_backtest(df, initial_capital=initial_capital)
    v3_trades_df = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'market_mode': t.market_mode
    } for t in v3_strategy.trades])
    
    # Yearly comparison
    print("\nYEARLY P&L COMPARISON:")
    print("-" * 70)
    
    orig_trades_df['year'] = orig_trades_df['entry_time'].dt.year
    v3_trades_df['year'] = v3_trades_df['entry_time'].dt.year
    
    orig_yearly = orig_trades_df.groupby('year')['pnl'].sum()
    v3_yearly = v3_trades_df.groupby('year')['pnl'].sum()
    
    all_years = sorted(set(orig_yearly.index) | set(v3_yearly.index))
    
    print(f"{'Year':<8} {'Original V1':>15} {'V3 Selective':>15} {'Difference':>15}")
    print("-" * 70)
    
    total_improvement = 0
    years_improved = 0
    
    for year in all_years:
        orig_pnl = orig_yearly.get(year, 0)
        v3_pnl = v3_yearly.get(year, 0)
        diff = v3_pnl - orig_pnl
        total_improvement += diff
        
        if v3_pnl > orig_pnl:
            years_improved += 1
            marker = "✓"
        elif v3_pnl < orig_pnl:
            marker = "✗"
        else:
            marker = "="
        
        print(f"{year:<8} ${orig_pnl:>+14,.2f} ${v3_pnl:>+14,.2f} ${diff:>+14,.2f} {marker}")
    
    print("-" * 70)
    print(f"{'TOTAL':<8} ${orig_yearly.sum():>+14,.2f} ${v3_yearly.sum():>+14,.2f} ${total_improvement:>+14,.2f}")
    
    # Summary
    orig_max_dd = ((orig_equity.cummax() - orig_equity) / orig_equity.cummax()).max() * 100
    v3_max_dd = ((v3_equity['equity'].cummax() - v3_equity['equity']) / v3_equity['equity'].cummax()).max() * 100
    
    print(f"\n{'Metric':<25} {'Original V1':>15} {'V3 Selective':>15}")
    print("-" * 60)
    print(f"{'Final Equity':<25} ${orig_equity.iloc[-1]:>14,.2f} ${v3_equity['equity'].iloc[-1]:>14,.2f}")
    print(f"{'Total Return':<25} {(orig_equity.iloc[-1]/initial_capital-1)*100:>14.1f}% {(v3_equity['equity'].iloc[-1]/initial_capital-1)*100:>14.1f}%")
    print(f"{'Max Drawdown':<25} {orig_max_dd:>14.1f}% {v3_max_dd:>14.1f}%")
    print(f"{'Total Trades':<25} {len(orig_trades_df):>15} {len(v3_trades_df):>15}")
    print(f"{'Years Improved':<25} {'N/A':>15} {years_improved}/{len(all_years)}")
    
    # Analyze which modes V3 traded in
    if len(v3_trades_df) > 0:
        print("\nV3 TRADES BY MARKET MODE:")
        mode_stats = v3_trades_df.groupby('market_mode').agg({
            'pnl': ['count', 'sum', 'mean']
        })
        mode_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
        print(mode_stats.to_string())
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Equity curves
    ax1 = axes[0, 0]
    ax1.plot(orig_equity.index, orig_equity.values, 'b-', label='Original V1', linewidth=1.5)
    ax1.plot(v3_equity.index, v3_equity['equity'].values, 'g-', label='V3 Selective', linewidth=1.5)
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curves: Original vs V3', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Drawdowns
    ax2 = axes[0, 1]
    orig_dd = (orig_equity - orig_equity.cummax()) / orig_equity.cummax() * 100
    v3_dd = (v3_equity['equity'] - v3_equity['equity'].cummax()) / v3_equity['equity'].cummax() * 100
    ax2.fill_between(orig_dd.index, orig_dd.values, 0, alpha=0.5, label='Original V1', color='blue')
    ax2.fill_between(v3_dd.index, v3_dd.values, 0, alpha=0.5, label='V3 Selective', color='green')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Yearly bars
    ax3 = axes[1, 0]
    x = np.arange(len(all_years))
    width = 0.35
    ax3.bar(x - width/2, [orig_yearly.get(y, 0) for y in all_years], width, label='Original V1', color='blue', alpha=0.7)
    ax3.bar(x + width/2, [v3_yearly.get(y, 0) for y in all_years], width, label='V3 Selective', color='green', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_years)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('P&L ($)')
    ax3.set_title('Yearly P&L Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Market mode periods
    ax4 = axes[1, 1]
    mode_colors = {'Normal': 'gray', 'Bull': 'green', 'Mild Bear': 'orange', 'Severe Bear': 'red'}
    for mode, color in mode_colors.items():
        mode_data = v3_equity[v3_equity['mode'] == mode]
        if len(mode_data) > 0:
            ax4.scatter(mode_data.index, mode_data['equity'], c=color, s=1, label=mode, alpha=0.5)
    ax4.set_ylabel('Equity ($)')
    ax4.set_title('V3 Equity Colored by Market Mode', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'v3_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to: {plot_path}")
    
    return orig_equity, v3_equity


def main():
    print("=" * 80)
    print("V3 STRATEGY - SELECTIVE BEAR MARKET PROTECTION")
    print("=" * 80)
    
    # Load data
    print("\nLoading BTC 4H data...")
    df = download_btc_data(timeframe="4h")
    print(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
    
    # Optimize V3 parameters
    results_df = optimize_v3(df, INITIAL_CAPITAL)
    
    # Run detailed comparison with best settings
    run_best_v3_with_yearly(df, INITIAL_CAPITAL)
    
    return results_df


if __name__ == "__main__":
    results = main()

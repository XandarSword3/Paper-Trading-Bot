"""
Enhanced Strategy V2 - With Bear Market Protection & Regime Filters
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
class EnhancedParams:
    """Enhanced strategy parameters with regime filters"""
    # Original Donchian params
    entry_len: int = 40
    exit_len: int = 16
    atr_len: int = 20
    trail_mult: float = 4.0
    size_stop_mult: float = 2.0
    risk_percent: float = 1.0
    max_units: int = 4
    pyramid_spacing_n: float = 1.5
    
    # NEW: Regime filters
    use_trend_filter: bool = True          # Only trade when above 200 SMA
    trend_sma_len: int = 200               # SMA period for trend filter
    
    use_momentum_filter: bool = True       # Only trade when RSI > threshold
    rsi_len: int = 14
    rsi_threshold: float = 45.0            # RSI must be above this for longs
    
    use_volatility_scaling: bool = True    # Scale size based on volatility
    vol_lookback: int = 50                 # Period for vol calculation
    vol_target: float = 0.02               # Target daily volatility (2%)
    
    use_drawdown_reduction: bool = True    # Reduce size after drawdowns
    dd_threshold: float = 0.15             # Reduce at 15% drawdown
    dd_size_mult: float = 0.5              # Reduce to 50% size
    
    use_bear_market_pause: bool = True     # Pause in bear markets
    bear_sma_slope_len: int = 20           # Check if 200 SMA is falling over this period
    
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
    regime: Optional[str] = None


class EnhancedStrategy:
    """Enhanced Donchian strategy with regime filters"""
    
    def __init__(self, params: EnhancedParams):
        self.params = params
        self.trades: List[Trade] = []
        self.position = 0
        self.active_trades: List[Trade] = []
        self.equity = 0
        self.peak_equity = 0
        self.current_drawdown = 0
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators including regime filters"""
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
        
        # NEW: Trend filter - 200 SMA
        df['sma_200'] = df['close'].rolling(self.params.trend_sma_len).mean()
        df['above_sma'] = df['close'] > df['sma_200']
        
        # NEW: SMA slope (is trend rising or falling)
        df['sma_slope'] = df['sma_200'].diff(self.params.bear_sma_slope_len)
        df['sma_rising'] = df['sma_slope'] > 0
        
        # NEW: RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.params.rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params.rsi_len).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # NEW: Volatility for scaling
        df['daily_returns'] = df['close'].pct_change()
        df['volatility'] = df['daily_returns'].rolling(self.params.vol_lookback).std()
        
        # NEW: Market regime classification
        df['regime'] = 'Neutral'
        df.loc[df['above_sma'] & df['sma_rising'], 'regime'] = 'Bull'
        df.loc[~df['above_sma'] & ~df['sma_rising'], 'regime'] = 'Bear'
        df.loc[df['above_sma'] & ~df['sma_rising'], 'regime'] = 'Bull Weakening'
        df.loc[~df['above_sma'] & df['sma_rising'], 'regime'] = 'Bear Recovering'
        
        return df
    
    def get_position_size_multiplier(self, df_row, current_equity: float) -> float:
        """Calculate position size multiplier based on filters"""
        multiplier = 1.0
        
        # Volatility scaling
        if self.params.use_volatility_scaling:
            current_vol = df_row['volatility']
            if not pd.isna(current_vol) and current_vol > 0:
                vol_ratio = self.params.vol_target / current_vol
                vol_mult = min(max(vol_ratio, 0.5), 2.0)  # Clamp between 0.5x and 2x
                multiplier *= vol_mult
        
        # Drawdown reduction
        if self.params.use_drawdown_reduction:
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            if self.current_drawdown > self.params.dd_threshold:
                multiplier *= self.params.dd_size_mult
        
        return multiplier
    
    def should_trade(self, df_row) -> tuple:
        """Check if we should trade based on regime filters"""
        can_trade = True
        reasons = []
        
        # Trend filter
        if self.params.use_trend_filter:
            if not df_row['above_sma']:
                can_trade = False
                reasons.append("Below 200 SMA")
        
        # Momentum filter
        if self.params.use_momentum_filter:
            if df_row['rsi'] < self.params.rsi_threshold:
                can_trade = False
                reasons.append(f"RSI {df_row['rsi']:.1f} < {self.params.rsi_threshold}")
        
        # Bear market pause
        if self.params.use_bear_market_pause:
            if df_row['regime'] == 'Bear':
                can_trade = False
                reasons.append("Bear market regime")
        
        return can_trade, reasons
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 1000.0, verbose: bool = False) -> pd.DataFrame:
        """Run enhanced backtest"""
        df = self.calculate_indicators(df)
        
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.active_trades = []
        self.position = 0
        
        equity_curve = []
        signals_log = []
        
        warmup = max(self.params.entry_len, self.params.trend_sma_len, self.params.vol_lookback) + 10
        
        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            current_price = row['close']
            atr = row['atr']
            
            if pd.isna(atr) or atr <= 0:
                equity_curve.append({'timestamp': row.name, 'equity': self.equity})
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
            
            # Check entries
            can_trade, skip_reasons = self.should_trade(row)
            
            if can_trade and self.position == 0:
                # Entry signal
                if row['high'] >= prev_row['donchian_high']:
                    # Calculate position size
                    risk_per_trade = self.equity * self.params.risk_percent / 100
                    stop_distance = self.params.size_stop_mult * atr
                    
                    if stop_distance > 0:
                        base_size = risk_per_trade / stop_distance
                        size_mult = self.get_position_size_multiplier(row, self.equity)
                        quantity = base_size * size_mult
                        quantity = max(self.params.lot_step, round(quantity / self.params.lot_step) * self.params.lot_step)
                        
                        entry_price = prev_row['donchian_high'] * (1 + self.params.slippage_pct / 100)
                        stop_loss = entry_price - stop_distance
                        
                        if quantity * entry_price <= self.equity * 0.95:  # Max 95% of equity
                            trade = Trade(
                                entry_time=row.name,
                                entry_price=entry_price,
                                direction='long',
                                quantity=quantity,
                                stop_loss=stop_loss,
                                regime=row['regime']
                            )
                            self.active_trades.append(trade)
                            
                            if verbose:
                                print(f"ENTRY: {row.name} @ ${entry_price:.2f}, Qty: {quantity:.4f}, Regime: {row['regime']}")
            
            # Pyramiding (add to winners)
            elif can_trade and len(self.active_trades) < self.params.max_units:
                if row['high'] >= prev_row['donchian_high']:
                    last_entry = self.active_trades[-1].entry_price if self.active_trades else 0
                    pyramid_threshold = last_entry + self.params.pyramid_spacing_n * atr
                    
                    if current_price >= pyramid_threshold:
                        risk_per_trade = self.equity * self.params.risk_percent / 100
                        stop_distance = self.params.size_stop_mult * atr
                        
                        if stop_distance > 0:
                            base_size = risk_per_trade / stop_distance
                            size_mult = self.get_position_size_multiplier(row, self.equity)
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
                                    regime=row['regime']
                                )
                                self.active_trades.append(trade)
            
            # Track equity
            unrealized_pnl = sum((current_price - t.entry_price) * t.quantity for t in self.active_trades)
            total_equity = self.equity + unrealized_pnl
            equity_curve.append({'timestamp': row.name, 'equity': total_equity})
        
        # Close any remaining positions
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


def compare_strategies(df: pd.DataFrame, initial_capital: float = 1000.0):
    """Compare original vs enhanced strategy"""
    
    print("=" * 80)
    print("STRATEGY COMPARISON: ORIGINAL vs ENHANCED V2")
    print("=" * 80)
    
    # Import original strategy
    from strategy import TurtleDonchianStrategy
    from config import StrategyParams
    
    # Original strategy
    print("\nRunning ORIGINAL strategy...")
    orig_params = StrategyParams(
        entry_len=40, exit_len=16, atr_len=20, trail_mult=4.0,
        size_stop_mult=2.0, risk_percent=1.0, max_units=4,
        pyramid_spacing_n=1.5, long_only=True, use_regime_filter=True,
        lot_step=0.001, commission_pct=0.08, slippage_pct=0.05
    )
    orig_strategy = TurtleDonchianStrategy(orig_params)
    orig_results = orig_strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
    orig_equity = orig_results['equity']
    orig_trades = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'direction': t.direction
    } for t in orig_strategy.trades])
    
    # Enhanced strategy - V2a: Trend filter only
    print("Running ENHANCED V2a (Trend Filter)...")
    v2a_params = EnhancedParams(
        use_trend_filter=True,
        use_momentum_filter=False,
        use_volatility_scaling=False,
        use_drawdown_reduction=False,
        use_bear_market_pause=False,
    )
    v2a_strategy = EnhancedStrategy(v2a_params)
    v2a_equity = v2a_strategy.run_backtest(df, initial_capital=initial_capital)
    v2a_trades = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'regime': t.regime
    } for t in v2a_strategy.trades])
    
    # Enhanced strategy - V2b: Trend + Bear pause
    print("Running ENHANCED V2b (Trend + Bear Pause)...")
    v2b_params = EnhancedParams(
        use_trend_filter=True,
        use_momentum_filter=False,
        use_volatility_scaling=False,
        use_drawdown_reduction=False,
        use_bear_market_pause=True,
    )
    v2b_strategy = EnhancedStrategy(v2b_params)
    v2b_equity = v2b_strategy.run_backtest(df, initial_capital=initial_capital)
    v2b_trades = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'regime': t.regime
    } for t in v2b_strategy.trades])
    
    # Enhanced strategy - V2c: All filters
    print("Running ENHANCED V2c (All Filters)...")
    v2c_params = EnhancedParams(
        use_trend_filter=True,
        use_momentum_filter=True,
        use_volatility_scaling=True,
        use_drawdown_reduction=True,
        use_bear_market_pause=True,
    )
    v2c_strategy = EnhancedStrategy(v2c_params)
    v2c_equity = v2c_strategy.run_backtest(df, initial_capital=initial_capital)
    v2c_trades = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'regime': t.regime
    } for t in v2c_strategy.trades])
    
    # Enhanced V2d: Trend + RSI + Bear pause (no vol scaling/DD reduction)
    print("Running ENHANCED V2d (Trend + RSI + Bear Pause)...")
    v2d_params = EnhancedParams(
        use_trend_filter=True,
        use_momentum_filter=True,
        rsi_threshold=40.0,  # Slightly lower threshold
        use_volatility_scaling=False,
        use_drawdown_reduction=False,
        use_bear_market_pause=True,
    )
    v2d_strategy = EnhancedStrategy(v2d_params)
    v2d_equity = v2d_strategy.run_backtest(df, initial_capital=initial_capital)
    v2d_trades = pd.DataFrame([{
        'entry_time': t.entry_time, 'pnl': t.pnl or 0, 'regime': t.regime
    } for t in v2d_strategy.trades])
    
    # Calculate metrics
    def calc_metrics(equity, trades_df, name):
        final = equity['equity'].iloc[-1]
        returns = equity['equity'].pct_change().dropna()
        
        # Max drawdown
        running_max = equity['equity'].cummax()
        drawdown = (equity['equity'] - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # Sharpe
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6) if returns.std() > 0 else 0
        
        # Win rate
        if len(trades_df) > 0:
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = wins / len(trades_df) * 100
            total_pnl = trades_df['pnl'].sum()
        else:
            win_rate = 0
            total_pnl = 0
        
        return {
            'name': name,
            'final_equity': final,
            'total_return': (final / initial_capital - 1) * 100,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
        }
    
    results = [
        calc_metrics(orig_results[['equity']], orig_trades, 'Original V1'),
        calc_metrics(v2a_equity, v2a_trades, 'V2a: Trend Filter'),
        calc_metrics(v2b_equity, v2b_trades, 'V2b: Trend+Bear Pause'),
        calc_metrics(v2c_equity, v2c_trades, 'V2c: All Filters'),
        calc_metrics(v2d_equity, v2d_trades, 'V2d: Trend+RSI+Bear'),
    ]
    
    # Print comparison
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)
    
    print(f"\n{'Strategy':<25} {'Final Equity':>14} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Trades':>8} {'Win Rate':>10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['name']:<25} ${r['final_equity']:>13,.2f} {r['total_return']:>9.1f}% {r['max_drawdown']:>9.1f}% {r['sharpe']:>8.2f} {r['trades']:>8} {r['win_rate']:>9.1f}%")
    
    # Find best strategy
    best = max(results, key=lambda x: x['total_return'] / abs(x['max_drawdown']) if x['max_drawdown'] != 0 else 0)
    print(f"\n🏆 Best Risk-Adjusted: {best['name']}")
    
    # Yearly comparison for the best enhanced
    print("\n" + "=" * 80)
    print("YEARLY COMPARISON: ORIGINAL vs BEST ENHANCED")
    print("=" * 80)
    
    # Find which enhanced was best
    best_enhanced_equity = None
    best_enhanced_trades = None
    if best['name'] == 'V2a: Trend Filter':
        best_enhanced_equity = v2a_equity
        best_enhanced_trades = v2a_trades
    elif best['name'] == 'V2b: Trend+Bear Pause':
        best_enhanced_equity = v2b_equity
        best_enhanced_trades = v2b_trades
    elif best['name'] == 'V2c: All Filters':
        best_enhanced_equity = v2c_equity
        best_enhanced_trades = v2c_trades
    elif best['name'] == 'V2d: Trend+RSI+Bear':
        best_enhanced_equity = v2d_equity
        best_enhanced_trades = v2d_trades
    
    if best_enhanced_trades is not None and len(best_enhanced_trades) > 0 and len(orig_trades) > 0:
        orig_trades['year'] = orig_trades['entry_time'].dt.year
        best_enhanced_trades['year'] = best_enhanced_trades['entry_time'].dt.year
        
        orig_yearly = orig_trades.groupby('year')['pnl'].sum()
        enhanced_yearly = best_enhanced_trades.groupby('year')['pnl'].sum()
        
        all_years = sorted(set(orig_yearly.index) | set(enhanced_yearly.index))
        
        print(f"\n{'Year':<8} {'Original':>15} {'Enhanced':>15} {'Difference':>15}")
        print("-" * 60)
        
        for year in all_years:
            orig_pnl = orig_yearly.get(year, 0)
            enh_pnl = enhanced_yearly.get(year, 0)
            diff = enh_pnl - orig_pnl
            
            marker = "✓" if enh_pnl > orig_pnl else "✗" if enh_pnl < orig_pnl else "="
            print(f"{year:<8} ${orig_pnl:>+14,.2f} ${enh_pnl:>+14,.2f} ${diff:>+14,.2f} {marker}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Equity curves
    ax1 = axes[0, 0]
    ax1.plot(orig_equity.index, orig_equity.values, 'b-', label='Original V1', linewidth=1.5)
    ax1.plot(v2b_equity.index, v2b_equity['equity'].values, 'g-', label='V2b: Trend+Bear', linewidth=1.5, alpha=0.8)
    ax1.plot(v2d_equity.index, v2d_equity['equity'].values, 'r-', label='V2d: Trend+RSI+Bear', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curves Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Drawdown comparison
    ax2 = axes[0, 1]
    orig_dd = (orig_equity - orig_equity.cummax()) / orig_equity.cummax() * 100
    v2b_dd = (v2b_equity['equity'] - v2b_equity['equity'].cummax()) / v2b_equity['equity'].cummax() * 100
    v2d_dd = (v2d_equity['equity'] - v2d_equity['equity'].cummax()) / v2d_equity['equity'].cummax() * 100
    
    ax2.fill_between(orig_dd.index, orig_dd.values, 0, alpha=0.4, label='Original V1', color='blue')
    ax2.fill_between(v2b_dd.index, v2b_dd.values, 0, alpha=0.4, label='V2b', color='green')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Yearly returns bar chart
    ax3 = axes[1, 0]
    if len(orig_trades) > 0:
        orig_trades['year'] = orig_trades['entry_time'].dt.year
        yearly_orig = orig_trades.groupby('year')['pnl'].sum()
        
        x = np.arange(len(yearly_orig))
        width = 0.35
        
        ax3.bar(x - width/2, yearly_orig.values, width, label='Original V1', color='blue', alpha=0.7)
        
        if best_enhanced_trades is not None and len(best_enhanced_trades) > 0:
            yearly_enh = best_enhanced_trades.groupby('year')['pnl'].sum().reindex(yearly_orig.index).fillna(0)
            ax3.bar(x + width/2, yearly_enh.values, width, label=best['name'], color='green', alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(yearly_orig.index)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('P&L ($)')
        ax3.set_title('Yearly P&L Comparison', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Summary metrics bar chart
    ax4 = axes[1, 1]
    metrics_names = ['Return/DD', 'Sharpe', 'Win Rate/10']
    orig_metrics = [results[0]['total_return'] / abs(results[0]['max_drawdown']), 
                    results[0]['sharpe'], results[0]['win_rate'] / 10]
    
    # Find V2b metrics
    v2b_result = [r for r in results if r['name'] == 'V2b: Trend+Bear Pause'][0]
    v2b_metrics = [v2b_result['total_return'] / abs(v2b_result['max_drawdown']) if v2b_result['max_drawdown'] != 0 else 0,
                   v2b_result['sharpe'], v2b_result['win_rate'] / 10]
    
    x = np.arange(len(metrics_names))
    ax4.bar(x - 0.2, orig_metrics, 0.4, label='Original V1', color='blue', alpha=0.7)
    ax4.bar(x + 0.2, v2b_metrics, 0.4, label='V2b', color='green', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names)
    ax4.set_title('Risk-Adjusted Metrics Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'strategy_comparison_v2.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comparison plots saved to: {plot_path}")
    
    return results, {
        'original': (orig_equity, orig_trades),
        'v2a': (v2a_equity, v2a_trades),
        'v2b': (v2b_equity, v2b_trades),
        'v2c': (v2c_equity, v2c_trades),
        'v2d': (v2d_equity, v2d_trades),
    }


def main():
    print("=" * 80)
    print("ENHANCED STRATEGY V2 - TESTING IMPROVEMENTS")
    print("=" * 80)
    
    # Load data
    print("\nLoading BTC 4H data...")
    df = download_btc_data(timeframe="4h")
    print(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
    
    # Run comparison
    results, strategies = compare_strategies(df, INITIAL_CAPITAL)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    orig = results[0]
    best_enh = max(results[1:], key=lambda x: x['total_return'])
    best_risk_adj = max(results[1:], key=lambda x: x['total_return'] / abs(x['max_drawdown']) if x['max_drawdown'] != 0 else 0)
    
    print(f"""
    Original V1:
      - Return: {orig['total_return']:.1f}%
      - Max Drawdown: {orig['max_drawdown']:.1f}%
      - Sharpe: {orig['sharpe']:.2f}
      - Trades: {orig['trades']}
    
    Best Return ({best_enh['name']}):
      - Return: {best_enh['total_return']:.1f}%
      - Max Drawdown: {best_enh['max_drawdown']:.1f}%
      - Sharpe: {best_enh['sharpe']:.2f}
      - Trades: {best_enh['trades']}
    
    Best Risk-Adjusted ({best_risk_adj['name']}):
      - Return: {best_risk_adj['total_return']:.1f}%
      - Max Drawdown: {best_risk_adj['max_drawdown']:.1f}%
      - Sharpe: {best_risk_adj['sharpe']:.2f}
      - Return/MaxDD: {best_risk_adj['total_return']/abs(best_risk_adj['max_drawdown']):.2f}
    """)
    
    return results, strategies


if __name__ == "__main__":
    results, strategies = main()

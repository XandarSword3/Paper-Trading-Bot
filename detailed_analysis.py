"""
Detailed Trading Analysis
Generates comprehensive statistics matching TradingView format
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StrategyParams, RESULTS_DIR, PLOTS_DIR
from data_fetcher import download_btc_data
from strategy import TurtleDonchianStrategy


# Optimized parameters from robustness testing
OPTIMIZED_PARAMS = StrategyParams(
    entry_len=40,
    exit_len=16,
    atr_len=20,
    trail_mult=4.0,
    size_stop_mult=2.0,
    risk_percent=1.0,
    max_units=4,
    pyramid_spacing_n=1.5,
    long_only=True,
    use_regime_filter=True,
    lot_step=0.001,
    commission_pct=0.08,
    slippage_pct=0.05,
)

INITIAL_CAPITAL = 1_000_000.0  # Match TradingView


def calculate_detailed_stats(strategy, results, initial_capital):
    """Calculate comprehensive trading statistics"""
    
    trades = strategy.trades
    equity = results['equity']
    
    if not trades:
        return {}
    
    # Build trades DataFrame
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'quantity': t.quantity,
        'pnl': t.pnl or 0,
        'exit_reason': t.exit_reason,
        'unit_number': t.unit_number
    } for t in trades])
    
    # Calculate trade returns as percentage
    trades_df['trade_value'] = trades_df['quantity'] * trades_df['entry_price']
    trades_df['pnl_pct'] = (trades_df['pnl'] / trades_df['trade_value']) * 100
    
    # Calculate holding periods (in bars, assuming 4H) - BEFORE filtering
    trades_df['bars_held'] = (
        (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / (4 * 3600)
    ).astype(int)
    
    # Separate winning and losing trades - AFTER adding bars_held
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]
    
    # Commission calculation (already included in PnL, but estimate separately)
    total_volume = (trades_df['quantity'] * trades_df['entry_price']).sum() * 2  # entry + exit
    commission_paid = total_volume * (OPTIMIZED_PARAMS.commission_pct / 100)
    
    # Equity curve analysis
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    
    # Run-up analysis (when equity is making new highs)
    is_runup = equity >= rolling_max
    
    # Drawdown periods
    in_dd = drawdown < 0
    
    # Calculate drawdown durations
    dd_durations = []
    current_dd_start = None
    
    for i, (idx, dd_val) in enumerate(drawdown.items()):
        if dd_val < 0 and current_dd_start is None:
            current_dd_start = idx
        elif dd_val >= 0 and current_dd_start is not None:
            duration = (idx - current_dd_start).total_seconds() / (24 * 3600)
            dd_durations.append(duration)
            current_dd_start = None
    
    # Buy and hold calculation
    start_price = results['close'].iloc[0]
    end_price = results['close'].iloc[-1]
    buy_hold_return = (end_price / start_price - 1) * initial_capital
    buy_hold_pct = (end_price / start_price - 1) * 100
    
    # Sharpe & Sortino (annualized, 4H bars = 6 per day = 2190 per year)
    returns = equity.pct_change().dropna()
    bars_per_year = 365 * 6
    
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0
    
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino = (returns.mean() / negative_returns.std()) * np.sqrt(bars_per_year)
    else:
        sortino = 0
    
    # Profit factor
    gross_profit = winning['pnl'].sum() if len(winning) > 0 else 0
    gross_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max contracts held (approximate based on pyramiding)
    max_position_value = (trades_df['quantity'] * trades_df['entry_price']).max()
    
    stats = {
        # Capital & Returns
        'initial_capital': initial_capital,
        'final_equity': equity.iloc[-1],
        'net_profit': equity.iloc[-1] - initial_capital,
        'net_profit_pct': (equity.iloc[-1] / initial_capital - 1) * 100,
        'gross_profit': gross_profit,
        'gross_profit_pct': (gross_profit / initial_capital) * 100,
        'gross_loss': gross_loss,
        'gross_loss_pct': (gross_loss / initial_capital) * 100,
        'commission_paid': commission_paid,
        
        # Buy & Hold
        'buy_hold_return': buy_hold_return,
        'buy_hold_pct': buy_hold_pct,
        
        # Drawdown
        'max_drawdown': abs(drawdown.min()) * 100,
        'max_drawdown_value': abs(drawdown.min() * rolling_max[drawdown.idxmin()]),
        'avg_drawdown': abs(drawdown[drawdown < 0].mean()) * 100 if (drawdown < 0).any() else 0,
        'avg_dd_duration': np.mean(dd_durations) if dd_durations else 0,
        'max_dd_duration': max(dd_durations) if dd_durations else 0,
        
        # Run-up
        'max_runup': equity.max() - initial_capital,
        'max_runup_pct': (equity.max() / initial_capital - 1) * 100,
        
        # Trade Statistics
        'total_trades': len(trades_df),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'percent_profitable': len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        
        # P&L Statistics
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_pnl_pct': trades_df['pnl_pct'].mean(),
        'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
        'avg_win_pct': winning['pnl_pct'].mean() if len(winning) > 0 else 0,
        'avg_loss': abs(losing['pnl'].mean()) if len(losing) > 0 else 0,
        'avg_loss_pct': abs(losing['pnl_pct'].mean()) if len(losing) > 0 else 0,
        'ratio_avg_win_loss': (winning['pnl'].mean() / abs(losing['pnl'].mean())) if len(losing) > 0 and len(winning) > 0 else 0,
        
        'largest_win': winning['pnl'].max() if len(winning) > 0 else 0,
        'largest_win_pct': winning['pnl_pct'].max() if len(winning) > 0 else 0,
        'largest_loss': abs(losing['pnl'].min()) if len(losing) > 0 else 0,
        'largest_loss_pct': abs(losing['pnl_pct'].min()) if len(losing) > 0 else 0,
        
        # Bars in trades
        'avg_bars_all': trades_df['bars_held'].mean(),
        'avg_bars_winners': winning['bars_held'].mean() if len(winning) > 0 else 0,
        'avg_bars_losers': losing['bars_held'].mean() if len(losing) > 0 else 0,
        
        # Risk metrics
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'profit_factor': profit_factor,
        
        # Position info
        'max_position_value': max_position_value,
    }
    
    return stats, trades_df


def print_tradingview_format(stats, trades_df):
    """Print statistics in TradingView-like format"""
    
    print("\n" + "=" * 80)
    print("DETAILED TRADING ANALYSIS - OPTIMIZED STRATEGY")
    print("Parameters: Entry=40, Exit=16, Trail=4.0, Risk=1.0%, PyramidN=1.5")
    print("=" * 80)
    
    print("\n" + "-" * 60)
    print("CAPITAL & RETURNS")
    print("-" * 60)
    print(f"{'Initial Capital':<35} {stats['initial_capital']:>20,.2f} USDT")
    print(f"{'Final Equity':<35} {stats['final_equity']:>20,.2f} USDT")
    print(f"{'Net Profit':<35} {stats['net_profit']:>+20,.2f} USDT ({stats['net_profit_pct']:+.2f}%)")
    print(f"{'Gross Profit':<35} {stats['gross_profit']:>20,.2f} USDT ({stats['gross_profit_pct']:.2f}%)")
    print(f"{'Gross Loss':<35} {stats['gross_loss']:>20,.2f} USDT ({stats['gross_loss_pct']:.2f}%)")
    print(f"{'Commission Paid (estimated)':<35} {stats['commission_paid']:>20,.2f} USDT")
    print(f"{'Buy & Hold Return':<35} {stats['buy_hold_return']:>+20,.2f} USDT ({stats['buy_hold_pct']:+.2f}%)")
    
    print("\n" + "-" * 60)
    print("DRAWDOWN ANALYSIS")
    print("-" * 60)
    print(f"{'Max Equity Drawdown':<35} {stats['max_drawdown_value']:>20,.2f} USDT ({stats['max_drawdown']:.2f}%)")
    print(f"{'Avg Equity Drawdown':<35} {stats['avg_drawdown']:>20.2f}%")
    print(f"{'Avg Drawdown Duration':<35} {stats['avg_dd_duration']:>20.0f} days")
    print(f"{'Max Drawdown Duration':<35} {stats['max_dd_duration']:>20.0f} days")
    
    print("\n" + "-" * 60)
    print("RUN-UP ANALYSIS")
    print("-" * 60)
    print(f"{'Max Equity Run-up':<35} {stats['max_runup']:>20,.2f} USDT ({stats['max_runup_pct']:.2f}%)")
    
    print("\n" + "-" * 60)
    print("TRADE STATISTICS")
    print("-" * 60)
    print(f"{'Total Trades':<35} {stats['total_trades']:>20}")
    print(f"{'Winning Trades':<35} {stats['winning_trades']:>20}")
    print(f"{'Losing Trades':<35} {stats['losing_trades']:>20}")
    print(f"{'Percent Profitable':<35} {stats['percent_profitable']:>20.2f}%")
    
    print("\n" + "-" * 60)
    print("PROFIT/LOSS BREAKDOWN")
    print("-" * 60)
    print(f"{'Avg P&L per Trade':<35} {stats['avg_pnl']:>+20,.2f} USDT ({stats['avg_pnl_pct']:+.2f}%)")
    print(f"{'Avg Winning Trade':<35} {stats['avg_win']:>20,.2f} USDT ({stats['avg_win_pct']:.2f}%)")
    print(f"{'Avg Losing Trade':<35} {stats['avg_loss']:>20,.2f} USDT ({stats['avg_loss_pct']:.2f}%)")
    print(f"{'Ratio Avg Win / Avg Loss':<35} {stats['ratio_avg_win_loss']:>20.3f}")
    print(f"{'Largest Winning Trade':<35} {stats['largest_win']:>20,.2f} USDT ({stats['largest_win_pct']:.2f}%)")
    print(f"{'Largest Losing Trade':<35} {stats['largest_loss']:>20,.2f} USDT ({stats['largest_loss_pct']:.2f}%)")
    
    print("\n" + "-" * 60)
    print("TRADE DURATION (in 4H bars)")
    print("-" * 60)
    print(f"{'Avg # Bars in All Trades':<35} {stats['avg_bars_all']:>20.0f}")
    print(f"{'Avg # Bars in Winning Trades':<35} {stats['avg_bars_winners']:>20.0f}")
    print(f"{'Avg # Bars in Losing Trades':<35} {stats['avg_bars_losers']:>20.0f}")
    
    print("\n" + "-" * 60)
    print("RISK METRICS")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<35} {stats['sharpe_ratio']:>20.3f}")
    print(f"{'Sortino Ratio':<35} {stats['sortino_ratio']:>20.3f}")
    print(f"{'Profit Factor':<35} {stats['profit_factor']:>20.2f}")
    
    # P&L Distribution
    print("\n" + "-" * 60)
    print("P&L DISTRIBUTION")
    print("-" * 60)
    
    pnl = trades_df['pnl']
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(pnl, p)
        print(f"{'  ' + str(p) + 'th Percentile':<35} {val:>+20,.2f} USDT")
    
    # Trade breakdown by exit reason
    print("\n" + "-" * 60)
    print("TRADES BY EXIT REASON")
    print("-" * 60)
    exit_breakdown = trades_df.groupby('exit_reason').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    exit_breakdown.columns = ['Count', 'Total P&L', 'Avg P&L']
    print(exit_breakdown.to_string())
    
    # Trade breakdown by direction
    print("\n" + "-" * 60)
    print("TRADES BY DIRECTION")
    print("-" * 60)
    dir_breakdown = trades_df.groupby('direction').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    dir_breakdown.columns = ['Count', 'Total P&L', 'Avg P&L']
    print(dir_breakdown.to_string())
    
    # Pyramid level analysis
    print("\n" + "-" * 60)
    print("TRADES BY PYRAMID LEVEL")
    print("-" * 60)
    pyramid_breakdown = trades_df.groupby('unit_number').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    pyramid_breakdown.columns = ['Count', 'Total P&L', 'Avg P&L']
    print(pyramid_breakdown.to_string())
    
    # Yearly breakdown
    print("\n" + "-" * 60)
    print("YEARLY PERFORMANCE")
    print("-" * 60)
    trades_df['year'] = trades_df['exit_time'].dt.year
    yearly = trades_df.groupby('year').agg({
        'pnl': ['count', 'sum', 'mean'],
    }).round(2)
    yearly.columns = ['Trades', 'Total P&L', 'Avg P&L']
    
    winning_by_year = trades_df[trades_df['pnl'] > 0].groupby('year').size()
    yearly['Win Rate'] = (winning_by_year / yearly['Trades'] * 100).round(1)
    yearly['Win Rate'] = yearly['Win Rate'].fillna(0).astype(str) + '%'
    
    print(yearly.to_string())
    
    return stats


def plot_pnl_distribution(trades_df, save_path=None):
    """Plot P&L distribution histogram"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trade P&L Distribution Analysis', fontsize=14)
        
        # P&L histogram
        ax1 = axes[0, 0]
        pnl = trades_df['pnl']
        ax1.hist(pnl, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.axvline(pnl.mean(), color='green', linestyle='-', label=f'Mean: ${pnl.mean():,.0f}')
        ax1.set_xlabel('P&L (USDT)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('P&L Distribution')
        ax1.legend()
        
        # Cumulative P&L
        ax2 = axes[0, 1]
        cumulative = pnl.cumsum()
        colors = ['green' if x >= 0 else 'red' for x in cumulative]
        ax2.plot(range(len(cumulative)), cumulative, linewidth=1)
        ax2.fill_between(range(len(cumulative)), cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.3)
        ax2.fill_between(range(len(cumulative)), cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.3)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L (USDT)')
        ax2.set_title('Cumulative P&L')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Win/Loss by year
        ax3 = axes[1, 0]
        trades_df['year'] = trades_df['exit_time'].dt.year
        yearly_pnl = trades_df.groupby('year')['pnl'].sum()
        colors = ['green' if x > 0 else 'red' for x in yearly_pnl]
        ax3.bar(yearly_pnl.index, yearly_pnl.values, color=colors, alpha=0.7)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Total P&L (USDT)')
        ax3.set_title('Yearly P&L')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Trade return percentage distribution
        ax4 = axes[1, 1]
        pnl_pct = trades_df['pnl_pct']
        ax4.hist(pnl_pct, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Trade Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Trade Return % Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nP&L distribution chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Could not generate plots: {e}")


def main():
    print("=" * 80)
    print("LOADING DATA AND RUNNING OPTIMIZED STRATEGY")
    print("=" * 80)
    
    # Load data
    print("\nLoading BTC 4H data...")
    df = download_btc_data(timeframe="4h")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total candles: {len(df)}")
    
    # Run strategy
    print("\nRunning backtest with optimized parameters...")
    strategy = TurtleDonchianStrategy(OPTIMIZED_PARAMS)
    results = strategy.run_backtest(df, initial_capital=INITIAL_CAPITAL, verbose=False)
    
    # Calculate detailed stats
    print("Calculating detailed statistics...")
    stats, trades_df = calculate_detailed_stats(strategy, results, INITIAL_CAPITAL)
    
    # Print TradingView-style report
    print_tradingview_format(stats, trades_df)
    
    # Generate plots
    plot_pnl_distribution(trades_df, os.path.join(PLOTS_DIR, "pnl_distribution.png"))
    
    # Save trades to CSV
    trades_df.to_csv(os.path.join(RESULTS_DIR, "all_trades.csv"), index=False)
    print(f"\nAll trades saved to {RESULTS_DIR}/all_trades.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

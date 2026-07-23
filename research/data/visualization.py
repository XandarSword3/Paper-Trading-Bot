"""
Visualization utilities for backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
import os

from config import PLOTS_DIR


def plot_equity_curve(
    equity: pd.Series,
    btc_prices: pd.Series = None,
    title: str = "Strategy Equity Curve",
    save_path: str = None
):
    """Plot equity curve with optional BTC comparison"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Main equity plot
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, label='Strategy Equity', linewidth=1.5, color='blue')
    
    if btc_prices is not None:
        # Normalize BTC to same starting value
        btc_normalized = btc_prices / btc_prices.iloc[0] * equity.iloc[0]
        ax1.plot(btc_prices.index, btc_normalized.values, 
                label='BTC Buy & Hold', linewidth=1, color='orange', alpha=0.7)
    
    ax1.set_ylabel('Equity ($)')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Drawdown plot
    ax2 = axes[1]
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Equity curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trade_analysis(
    trades: list,
    title: str = "Trade Analysis",
    save_path: str = None
):
    """Plot trade statistics and distributions"""
    
    if not trades:
        print("No trades to analyze")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'pnl': t.pnl or 0,
        'quantity': t.quantity,
        'unit_number': t.unit_number
    } for t in trades])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    # P&L Distribution
    ax1 = axes[0, 0]
    pnl_values = df['pnl'].values
    colors = ['green' if p > 0 else 'red' for p in pnl_values]
    ax1.hist(pnl_values, bins=50, color='blue', edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--')
    ax1.axvline(np.mean(pnl_values), color='green', linestyle='-', label=f'Mean: ${np.mean(pnl_values):,.0f}')
    ax1.set_xlabel('P&L ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('P&L Distribution')
    ax1.legend()
    
    # Cumulative P&L
    ax2 = axes[0, 1]
    cumulative_pnl = df['pnl'].cumsum()
    ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=1.5)
    ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=(cumulative_pnl >= 0), color='green', alpha=0.3)
    ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=(cumulative_pnl < 0), color='red', alpha=0.3)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.set_title('Cumulative P&L Over Trades')
    ax2.grid(True, alpha=0.3)
    
    # Win/Loss by Direction
    ax3 = axes[1, 0]
    direction_pnl = df.groupby('direction')['pnl'].agg(['sum', 'count', 'mean'])
    colors = ['green' if d == 'long' else 'red' for d in direction_pnl.index]
    bars = ax3.bar(direction_pnl.index, direction_pnl['sum'], color=colors, alpha=0.7)
    ax3.set_xlabel('Direction')
    ax3.set_ylabel('Total P&L ($)')
    ax3.set_title('P&L by Direction')
    for bar, count in zip(bars, direction_pnl['count']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={int(count)}', ha='center', va='bottom')
    
    # P&L by Unit Number (pyramiding)
    ax4 = axes[1, 1]
    unit_pnl = df.groupby('unit_number')['pnl'].agg(['sum', 'count'])
    ax4.bar(unit_pnl.index, unit_pnl['sum'], color='blue', alpha=0.7)
    ax4.set_xlabel('Unit Number (Pyramid Level)')
    ax4.set_ylabel('Total P&L ($)')
    ax4.set_title('P&L by Pyramid Level')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trade analysis saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_regime_comparison(
    regime_df: pd.DataFrame,
    title: str = "Regime Performance Comparison",
    save_path: str = None
):
    """Plot regime-by-regime performance comparison"""
    
    if regime_df.empty:
        print("No regime data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    # Returns comparison
    ax1 = axes[0, 0]
    x = range(len(regime_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], regime_df['btc_return_pct'], width, 
            label='BTC', color='orange', alpha=0.7)
    ax1.bar([i + width/2 for i in x], regime_df['strategy_return_pct'], width,
            label='Strategy', color='blue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_df['name'], rotation=45, ha='right')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Returns by Regime')
    ax1.legend()
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Max Drawdown by regime
    ax2 = axes[0, 1]
    colors = ['green' if t == 'bull' else 'red' if t == 'bear' else 'gray' 
              for t in regime_df['type']]
    ax2.bar(regime_df['name'], regime_df['max_drawdown_pct'], color=colors, alpha=0.7)
    ax2.set_xticklabels(regime_df['name'], rotation=45, ha='right')
    ax2.set_ylabel('Max Drawdown (%)')
    ax2.set_title('Max Drawdown by Regime')
    
    # Win rate by regime
    ax3 = axes[1, 0]
    ax3.bar(regime_df['name'], regime_df['win_rate'], color='blue', alpha=0.7)
    ax3.set_xticklabels(regime_df['name'], rotation=45, ha='right')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate by Regime')
    ax3.axhline(50, color='red', linestyle='--', label='50%')
    ax3.legend()
    
    # Number of trades by regime
    ax4 = axes[1, 1]
    ax4.bar(regime_df['name'], regime_df['num_trades'], color='purple', alpha=0.7)
    ax4.set_xticklabels(regime_df['name'], rotation=45, ha='right')
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trade Count by Regime')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Regime comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(
    results_df: pd.DataFrame,
    strategy,
    regime_df: pd.DataFrame = None
):
    """Generate all visualization plots"""
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Equity curve
    if 'equity' in results_df.columns:
        plot_equity_curve(
            results_df['equity'],
            results_df['close'],
            save_path=os.path.join(PLOTS_DIR, "equity_curve.png")
        )
    
    # Trade analysis
    if hasattr(strategy, 'trades') and strategy.trades:
        plot_trade_analysis(
            strategy.trades,
            save_path=os.path.join(PLOTS_DIR, "trade_analysis.png")
        )
    
    # Regime comparison
    if regime_df is not None and not regime_df.empty:
        plot_regime_comparison(
            regime_df,
            save_path=os.path.join(PLOTS_DIR, "regime_comparison.png")
        )
    
    print(f"\nAll plots saved to {PLOTS_DIR}/")

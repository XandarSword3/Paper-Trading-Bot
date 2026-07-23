"""
Deep Strategy Analysis - Identify Weak Periods and Potential Improvements
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import StrategyParams, RESULTS_DIR, PLOTS_DIR
from data_fetcher import download_btc_data
from strategy import TurtleDonchianStrategy

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 1_000.0


def run_strategy_with_details(df, params, initial_capital):
    """Run strategy and return detailed results"""
    strategy = TurtleDonchianStrategy(params)
    results = strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
    
    trades_list = []
    for t in strategy.trades:
        trades_list.append({
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl or 0,
            'exit_reason': t.exit_reason,
        })
    
    trades_df = pd.DataFrame(trades_list)
    return results, trades_df, strategy


def analyze_drawdowns(equity):
    """Analyze drawdown periods in detail"""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    
    # Find drawdown periods
    in_drawdown = drawdown < -5  # Significant drawdown threshold
    
    drawdown_periods = []
    start_idx = None
    
    for i, (idx, is_dd) in enumerate(in_drawdown.items()):
        if is_dd and start_idx is None:
            start_idx = idx
        elif not is_dd and start_idx is not None:
            # End of drawdown
            period_dd = drawdown[start_idx:idx]
            max_dd = period_dd.min()
            max_dd_date = period_dd.idxmin()
            
            drawdown_periods.append({
                'start': start_idx,
                'end': idx,
                'recovery_date': idx,
                'max_drawdown': max_dd,
                'max_dd_date': max_dd_date,
                'duration_days': (idx - start_idx).days,
                'equity_at_start': equity[start_idx],
                'equity_at_bottom': equity[max_dd_date],
            })
            start_idx = None
    
    # Handle ongoing drawdown
    if start_idx is not None:
        period_dd = drawdown[start_idx:]
        max_dd = period_dd.min()
        max_dd_date = period_dd.idxmin()
        drawdown_periods.append({
            'start': start_idx,
            'end': equity.index[-1],
            'recovery_date': None,
            'max_drawdown': max_dd,
            'max_dd_date': max_dd_date,
            'duration_days': (equity.index[-1] - start_idx).days,
            'equity_at_start': equity[start_idx],
            'equity_at_bottom': equity[max_dd_date],
        })
    
    return pd.DataFrame(drawdown_periods), drawdown


def analyze_stalling_periods(equity, threshold_days=60):
    """Find periods where equity stalls (no new highs)"""
    running_max = equity.cummax()
    
    stalling_periods = []
    last_high_date = equity.index[0]
    last_high_value = equity.iloc[0]
    stall_start = None
    
    for idx, (date, eq) in enumerate(equity.items()):
        current_max = running_max[date]
        
        if eq >= current_max and eq > last_high_value:
            # New high
            if stall_start is not None:
                stall_duration = (date - stall_start).days
                if stall_duration >= threshold_days:
                    stalling_periods.append({
                        'start': stall_start,
                        'end': date,
                        'duration_days': stall_duration,
                        'equity_at_start': equity[stall_start],
                        'equity_at_end': eq,
                        'min_equity': equity[stall_start:date].min(),
                        'max_drawdown_pct': (equity[stall_start:date].min() / equity[stall_start] - 1) * 100,
                    })
                stall_start = None
            last_high_date = date
            last_high_value = eq
        elif stall_start is None:
            stall_start = last_high_date
    
    # Handle ongoing stall
    if stall_start is not None:
        stall_duration = (equity.index[-1] - stall_start).days
        if stall_duration >= threshold_days:
            stalling_periods.append({
                'start': stall_start,
                'end': equity.index[-1],
                'duration_days': stall_duration,
                'equity_at_start': equity[stall_start],
                'equity_at_end': equity.iloc[-1],
                'min_equity': equity[stall_start:].min(),
                'max_drawdown_pct': (equity[stall_start:].min() / equity[stall_start] - 1) * 100,
            })
    
    return pd.DataFrame(stalling_periods)


def analyze_market_regimes(df, equity, trades_df):
    """Analyze performance in different market regimes"""
    
    # Calculate market indicators
    df = df.copy()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['volatility'] = df['close'].pct_change().rolling(50).std() * np.sqrt(252 * 6)  # Annualized vol for 4H
    df['returns_30d'] = df['close'].pct_change(30 * 6)  # 30 day returns (6 candles per day)
    
    # Define regimes
    df['regime'] = 'Neutral'
    df.loc[(df['sma_50'] > df['sma_200']) & (df['returns_30d'] > 0.1), 'regime'] = 'Strong Bull'
    df.loc[(df['sma_50'] > df['sma_200']) & (df['returns_30d'] <= 0.1) & (df['returns_30d'] > 0), 'regime'] = 'Mild Bull'
    df.loc[(df['sma_50'] < df['sma_200']) & (df['returns_30d'] < -0.1), 'regime'] = 'Strong Bear'
    df.loc[(df['sma_50'] < df['sma_200']) & (df['returns_30d'] >= -0.1) & (df['returns_30d'] < 0), 'regime'] = 'Mild Bear'
    df.loc[df['volatility'] > df['volatility'].quantile(0.8), 'regime'] = 'High Volatility'
    
    # Analyze trades by regime
    if len(trades_df) > 0:
        trades_df = trades_df.copy()
        trades_df['entry_regime'] = trades_df['entry_time'].apply(
            lambda x: df.loc[df.index.asof(x), 'regime'] if x >= df.index[0] else 'Unknown'
        )
        
        regime_stats = trades_df.groupby('entry_regime').agg({
            'pnl': ['count', 'sum', 'mean'],
        }).round(2)
        regime_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
        
        # Win rate by regime
        for regime in trades_df['entry_regime'].unique():
            regime_trades = trades_df[trades_df['entry_regime'] == regime]
            wins = len(regime_trades[regime_trades['pnl'] > 0])
            regime_stats.loc[regime, 'win_rate'] = wins / len(regime_trades) * 100 if len(regime_trades) > 0 else 0
        
        return regime_stats, df
    
    return None, df


def analyze_by_year_month(trades_df, equity):
    """Analyze performance by year and month"""
    
    # Monthly equity returns
    monthly_equity = equity.resample('ME').last()
    monthly_returns = monthly_equity.pct_change() * 100
    
    # Trades by year
    if len(trades_df) > 0:
        trades_df = trades_df.copy()
        trades_df['year'] = trades_df['entry_time'].dt.year
        trades_df['month'] = trades_df['entry_time'].dt.month
        
        yearly_stats = trades_df.groupby('year').agg({
            'pnl': ['count', 'sum', 'mean'],
        }).round(2)
        yearly_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
        
        # Win rate by year
        for year in trades_df['year'].unique():
            year_trades = trades_df[trades_df['year'] == year]
            wins = len(year_trades[year_trades['pnl'] > 0])
            yearly_stats.loc[year, 'win_rate'] = wins / len(year_trades) * 100 if len(year_trades) > 0 else 0
        
        return yearly_stats, monthly_returns
    
    return None, monthly_returns


def create_analysis_plots(df, equity, trades_df, drawdown, stalling_periods, drawdown_periods):
    """Create comprehensive analysis plots"""
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Equity curve with drawdown overlay
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(equity.index, equity.values, 'b-', linewidth=1.5, label='Equity')
    ax1.fill_between(equity.index, equity.values, alpha=0.3)
    ax1.set_ylabel('Equity ($)', color='blue')
    ax1.set_title('Equity Curve with Stalling Periods Highlighted', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Highlight stalling periods
    for _, period in stalling_periods.iterrows():
        ax1.axvspan(period['start'], period['end'], alpha=0.3, color='red', 
                    label='Stalling' if _ == 0 else '')
    
    ax1_dd = ax1.twinx()
    ax1_dd.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax1_dd.set_ylabel('Drawdown (%)', color='red')
    ax1_dd.set_ylim(-50, 5)
    
    # 2. BTC price with trade entries/exits
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(df.index, df['close'], 'gray', alpha=0.7, linewidth=0.8, label='BTC Price')
    ax2.set_ylabel('BTC Price ($)')
    ax2.set_title('BTC Price with Trade Entries', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot trade entries
    if len(trades_df) > 0:
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        ax2.scatter(winning['entry_time'], winning['entry_price'], 
                   c='green', marker='^', s=20, alpha=0.6, label=f'Wins ({len(winning)})')
        ax2.scatter(losing['entry_time'], losing['entry_price'], 
                   c='red', marker='v', s=20, alpha=0.6, label=f'Losses ({len(losing)})')
    ax2.legend(loc='upper left')
    
    # 3. Monthly returns heatmap-style bar chart
    ax3 = fig.add_subplot(4, 2, 3)
    monthly_equity = equity.resample('ME').last()
    monthly_returns = monthly_equity.pct_change() * 100
    
    colors = ['green' if r > 0 else 'red' for r in monthly_returns.values]
    ax3.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.7, width=20)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Monthly Return (%)')
    ax3.set_title('Monthly Returns', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Rolling Sharpe Ratio
    ax4 = fig.add_subplot(4, 2, 4)
    daily_returns = equity.pct_change()
    rolling_sharpe = (daily_returns.rolling(252).mean() / daily_returns.rolling(252).std()) * np.sqrt(252)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 'purple', linewidth=1)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.axhline(y=1, color='green', linestyle='--', linewidth=1, label='Sharpe = 1')
    ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                     where=rolling_sharpe.values > 0, alpha=0.3, color='green')
    ax4.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                     where=rolling_sharpe.values < 0, alpha=0.3, color='red')
    ax4.set_ylabel('Rolling Sharpe (1Y)')
    ax4.set_title('Rolling 1-Year Sharpe Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Trade P&L distribution
    ax5 = fig.add_subplot(4, 2, 5)
    if len(trades_df) > 0:
        ax5.hist(trades_df['pnl'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=2, 
                   label=f"Mean: ${trades_df['pnl'].mean():.2f}")
    ax5.set_xlabel('P&L ($)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Consecutive wins/losses
    ax6 = fig.add_subplot(4, 2, 6)
    if len(trades_df) > 0:
        trades_df_sorted = trades_df.sort_values('entry_time')
        is_win = (trades_df_sorted['pnl'] > 0).astype(int)
        
        # Calculate streaks
        streak_changes = is_win.diff().fillna(1).abs().cumsum()
        streaks = is_win.groupby(streak_changes).cumcount() + 1
        
        win_streaks = streaks[is_win == 1]
        loss_streaks = streaks[is_win == 0]
        
        ax6.hist(win_streaks, bins=range(1, 15), alpha=0.7, color='green', label='Win Streaks')
        ax6.hist(loss_streaks, bins=range(1, 15), alpha=0.7, color='red', label='Loss Streaks')
    ax6.set_xlabel('Streak Length')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Consecutive Win/Loss Streaks', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Yearly P&L
    ax7 = fig.add_subplot(4, 2, 7)
    if len(trades_df) > 0:
        yearly_pnl = trades_df.groupby(trades_df['entry_time'].dt.year)['pnl'].sum()
        colors = ['green' if p > 0 else 'red' for p in yearly_pnl.values]
        bars = ax7.bar(yearly_pnl.index, yearly_pnl.values, color=colors, alpha=0.7, edgecolor='black')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, yearly_pnl.values):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=8)
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Total P&L ($)')
    ax7.set_title('Yearly P&L', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Drawdown duration histogram
    ax8 = fig.add_subplot(4, 2, 8)
    if len(drawdown_periods) > 0:
        ax8.bar(range(len(drawdown_periods)), drawdown_periods['duration_days'], 
               color='red', alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Drawdown Event #')
        ax8.set_ylabel('Duration (days)')
        
        # Add max DD labels
        for i, (_, row) in enumerate(drawdown_periods.iterrows()):
            ax8.text(i, row['duration_days'] + 5, f"{row['max_drawdown']:.1f}%", 
                    ha='center', fontsize=8)
    ax8.set_title('Drawdown Durations (>5%)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'strategy_deep_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Analysis plots saved to: {plot_path}")
    return plot_path


def main():
    print("=" * 80)
    print("DEEP STRATEGY ANALYSIS - Finding Weak Points")
    print("=" * 80)
    
    # Load data
    print("\nLoading BTC 4H data...")
    df = download_btc_data(timeframe="4h")
    print(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
    
    # Run strategy with optimized parameters
    params = StrategyParams(
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
    
    print("\nRunning strategy...")
    results, trades_df, strategy = run_strategy_with_details(df, params, INITIAL_CAPITAL)
    equity = results['equity']
    
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:        ${equity.iloc[-1]:,.2f}")
    print(f"Total Return:        {(equity.iloc[-1]/INITIAL_CAPITAL - 1)*100:,.2f}%")
    print(f"Total Trades:        {len(trades_df)}")
    
    # Analyze drawdowns
    print(f"\n{'='*60}")
    print("DRAWDOWN ANALYSIS")
    print(f"{'='*60}")
    
    drawdown_periods, drawdown = analyze_drawdowns(equity)
    
    if len(drawdown_periods) > 0:
        print(f"\nSignificant Drawdowns (>5%):")
        print("-" * 80)
        for i, row in drawdown_periods.iterrows():
            print(f"  #{i+1}: {row['start'].strftime('%Y-%m-%d')} to {row['end'].strftime('%Y-%m-%d')}")
            print(f"      Duration: {row['duration_days']} days | Max DD: {row['max_drawdown']:.1f}%")
            print(f"      Equity: ${row['equity_at_start']:,.2f} → ${row['equity_at_bottom']:,.2f}")
            print()
    
    # Analyze stalling periods
    print(f"\n{'='*60}")
    print("STALLING PERIODS (No new highs for 60+ days)")
    print(f"{'='*60}")
    
    stalling_periods = analyze_stalling_periods(equity, threshold_days=60)
    
    if len(stalling_periods) > 0:
        print(f"\nFound {len(stalling_periods)} stalling periods:")
        print("-" * 80)
        total_stall_days = 0
        for i, row in stalling_periods.iterrows():
            total_stall_days += row['duration_days']
            print(f"  #{i+1}: {row['start'].strftime('%Y-%m-%d')} to {row['end'].strftime('%Y-%m-%d')}")
            print(f"      Duration: {row['duration_days']} days | Max Drawdown: {row['max_drawdown_pct']:.1f}%")
            print(f"      Equity: ${row['equity_at_start']:,.2f} → ${row['equity_at_end']:,.2f}")
            print()
        
        total_days = (equity.index[-1] - equity.index[0]).days
        print(f"Total time in stalling periods: {total_stall_days} days ({total_stall_days/total_days*100:.1f}% of backtest)")
    
    # Analyze by regime
    print(f"\n{'='*60}")
    print("PERFORMANCE BY MARKET REGIME")
    print(f"{'='*60}")
    
    regime_stats, df_with_regime = analyze_market_regimes(df, equity, trades_df)
    if regime_stats is not None:
        print("\n")
        print(regime_stats.to_string())
    
    # Analyze by year
    print(f"\n{'='*60}")
    print("YEARLY PERFORMANCE")
    print(f"{'='*60}")
    
    yearly_stats, monthly_returns = analyze_by_year_month(trades_df, equity)
    if yearly_stats is not None:
        print("\n")
        print(yearly_stats.to_string())
        
        # Identify worst years
        worst_years = yearly_stats[yearly_stats['total_pnl'] < 0].sort_values('total_pnl')
        if len(worst_years) > 0:
            print(f"\n⚠️  LOSING YEARS:")
            for year, row in worst_years.iterrows():
                print(f"   {year}: ${row['total_pnl']:+,.2f} ({row['trades']:.0f} trades, {row['win_rate']:.1f}% win rate)")
    
    # Analyze trades during stalling periods
    print(f"\n{'='*60}")
    print("TRADES DURING STALLING PERIODS")
    print(f"{'='*60}")
    
    if len(stalling_periods) > 0 and len(trades_df) > 0:
        stall_trades = []
        for _, period in stalling_periods.iterrows():
            period_trades = trades_df[
                (trades_df['entry_time'] >= period['start']) & 
                (trades_df['entry_time'] <= period['end'])
            ]
            stall_trades.append(period_trades)
        
        all_stall_trades = pd.concat(stall_trades) if stall_trades else pd.DataFrame()
        
        if len(all_stall_trades) > 0:
            wins = len(all_stall_trades[all_stall_trades['pnl'] > 0])
            losses = len(all_stall_trades[all_stall_trades['pnl'] <= 0])
            
            print(f"\nTrades during stalling periods: {len(all_stall_trades)}")
            print(f"Win Rate: {wins/len(all_stall_trades)*100:.1f}%")
            print(f"Total P&L: ${all_stall_trades['pnl'].sum():+,.2f}")
            print(f"Avg P&L: ${all_stall_trades['pnl'].mean():+,.2f}")
            
            # Exit reasons during stalls
            print(f"\nExit Reasons during stalls:")
            print(all_stall_trades['exit_reason'].value_counts().to_string())
    
    # Create plots
    print(f"\n{'='*60}")
    print("GENERATING ANALYSIS PLOTS...")
    print(f"{'='*60}")
    
    plot_path = create_analysis_plots(df, equity, trades_df, drawdown, stalling_periods, drawdown_periods)
    
    # Key insights and recommendations
    print(f"\n{'='*80}")
    print("KEY INSIGHTS & POTENTIAL IMPROVEMENTS")
    print(f"{'='*80}")
    
    insights = []
    
    # Check for bear market issues
    if regime_stats is not None and 'Strong Bear' in regime_stats.index:
        bear_stats = regime_stats.loc['Strong Bear']
        if bear_stats['total_pnl'] < 0:
            insights.append(f"⚠️  BEAR MARKET LOSSES: ${bear_stats['total_pnl']:,.2f} lost in strong bear markets")
            insights.append("   → Consider: Tighter stops or pause trading in bear regimes")
    
    # Check for high volatility issues
    if regime_stats is not None and 'High Volatility' in regime_stats.index:
        hv_stats = regime_stats.loc['High Volatility']
        if hv_stats['win_rate'] < 40:
            insights.append(f"⚠️  HIGH VOLATILITY STRUGGLES: {hv_stats['win_rate']:.1f}% win rate")
            insights.append("   → Consider: Reduce position size or widen stops in high vol")
    
    # Check for prolonged drawdowns
    if len(drawdown_periods) > 0:
        max_dd_duration = drawdown_periods['duration_days'].max()
        max_dd = drawdown_periods['max_drawdown'].min()
        if max_dd_duration > 200:
            insights.append(f"⚠️  LONG DRAWDOWN: {max_dd_duration} days with {max_dd:.1f}% max drawdown")
            insights.append("   → Consider: Add trend filter or reduce exposure in choppy markets")
    
    # Check for losing years
    if yearly_stats is not None:
        losing_years = yearly_stats[yearly_stats['total_pnl'] < 0]
        if len(losing_years) > 0:
            for year in losing_years.index:
                insights.append(f"⚠️  LOSING YEAR {year}: ${losing_years.loc[year, 'total_pnl']:+,.2f}")
    
    # Print insights
    for insight in insights:
        print(insight)
    
    if not insights:
        print("✓ No major issues identified - strategy performs consistently!")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print(f"{'='*80}")
    
    print("""
    Based on the analysis, consider testing these enhancements:
    
    1. REGIME FILTER: Pause or reduce size when 200 SMA is falling
    2. VOLATILITY ADJUSTMENT: Scale position size inversely with ATR
    3. DRAWDOWN LIMIT: Reduce size by 50% after 20% drawdown
    4. MOMENTUM CONFIRMATION: Only enter when RSI > 50 for longs
    5. TIME FILTER: Avoid entries in historically weak months
    """)
    
    # Save detailed results
    results_path = os.path.join(RESULTS_DIR, 'deep_analysis_results.csv')
    if len(trades_df) > 0:
        trades_df.to_csv(results_path, index=False)
        print(f"\nTrade details saved to: {results_path}")
    
    return {
        'equity': equity,
        'trades': trades_df,
        'drawdown_periods': drawdown_periods,
        'stalling_periods': stalling_periods,
        'regime_stats': regime_stats,
        'yearly_stats': yearly_stats,
    }


if __name__ == "__main__":
    analysis = main()

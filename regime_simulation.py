"""
Historical Regime Analysis + 20-Year Forward Simulation
Based on V1 Strategy Performance
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, PLOTS_DIR
from data_fetcher import download_btc_data

os.makedirs(PLOTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 1_000.0


def identify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Identify market regimes from historical data"""
    
    df = df.copy()
    
    # Calculate indicators for regime detection
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # Returns over different periods
    df['return_30d'] = df['close'].pct_change(30 * 6)  # 30 days in 4H candles
    df['return_90d'] = df['close'].pct_change(90 * 6)
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(50).std() * np.sqrt(252 * 6)
    
    # Drawdown from ATH
    df['ath'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['ath']) / df['ath']
    
    # Define regimes
    df['regime'] = 'Unknown'
    
    # 1. STRONG BULL: Above 200 SMA, strong positive returns, making new highs
    strong_bull = (
        (df['close'] > df['sma_200']) & 
        (df['return_30d'] > 0.15) & 
        (df['drawdown'] > -0.10)
    )
    df.loc[strong_bull, 'regime'] = 'Strong Bull'
    
    # 2. MILD BULL: Above 200 SMA, positive but moderate returns
    mild_bull = (
        (df['close'] > df['sma_200']) & 
        (df['return_30d'] > 0) & 
        (df['return_30d'] <= 0.15) &
        (df['regime'] == 'Unknown')
    )
    df.loc[mild_bull, 'regime'] = 'Mild Bull'
    
    # 3. STRONG BEAR: Below 200 SMA, significant negative returns
    strong_bear = (
        (df['close'] < df['sma_200']) & 
        (df['return_30d'] < -0.15) &
        (df['drawdown'] < -0.30)
    )
    df.loc[strong_bear, 'regime'] = 'Strong Bear'
    
    # 4. MILD BEAR: Below 200 SMA, negative but moderate returns
    mild_bear = (
        (df['close'] < df['sma_200']) & 
        (df['return_30d'] < 0) & 
        (df['return_30d'] >= -0.15) &
        (df['regime'] == 'Unknown')
    )
    df.loc[mild_bear, 'regime'] = 'Mild Bear'
    
    # 5. RECOVERY: Below 200 SMA but rising, positive returns
    recovery = (
        (df['close'] < df['sma_200']) & 
        (df['return_30d'] > 0.05) &
        (df['regime'] == 'Unknown')
    )
    df.loc[recovery, 'regime'] = 'Recovery'
    
    # 6. CONSOLIDATION/CHOP: Near 200 SMA, low returns, range-bound
    consolidation = (
        (abs(df['close'] / df['sma_200'] - 1) < 0.10) &
        (abs(df['return_30d']) < 0.10) &
        (df['regime'] == 'Unknown')
    )
    df.loc[consolidation, 'regime'] = 'Consolidation'
    
    # 7. EUPHORIA: Extreme gains, usually at cycle tops
    euphoria = (
        (df['return_30d'] > 0.40) &
        (df['close'] > df['sma_200'] * 1.5)
    )
    df.loc[euphoria, 'regime'] = 'Euphoria'
    
    # 8. CAPITULATION: Extreme losses, usually at cycle bottoms
    capitulation = (
        (df['return_30d'] < -0.30) &
        (df['drawdown'] < -0.50)
    )
    df.loc[capitulation, 'regime'] = 'Capitulation'
    
    # Fill remaining as Neutral
    df.loc[df['regime'] == 'Unknown', 'regime'] = 'Neutral'
    
    return df


def analyze_regime_statistics(df: pd.DataFrame):
    """Analyze regime durations and transitions"""
    
    print("=" * 80)
    print("HISTORICAL REGIME ANALYSIS (2017-2025)")
    print("=" * 80)
    
    # Regime distribution
    regime_counts = df['regime'].value_counts()
    total_candles = len(df)
    
    print("\n📊 REGIME DISTRIBUTION:")
    print("-" * 60)
    for regime, count in regime_counts.items():
        pct = count / total_candles * 100
        days = count / 6  # 4H candles to days
        print(f"  {regime:<20} {count:>6} candles ({pct:>5.1f}%) ≈ {days:>6.0f} days")
    
    # Calculate regime durations
    print("\n📊 REGIME DURATIONS:")
    print("-" * 60)
    
    # Find regime periods
    df['regime_change'] = df['regime'] != df['regime'].shift(1)
    df['regime_id'] = df['regime_change'].cumsum()
    
    regime_durations = df.groupby(['regime_id', 'regime']).size().reset_index(name='duration')
    regime_durations['duration_days'] = regime_durations['duration'] / 6
    
    duration_stats = regime_durations.groupby('regime')['duration_days'].agg(['mean', 'std', 'min', 'max', 'count'])
    
    print(f"\n{'Regime':<20} {'Avg Days':>10} {'Std':>10} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 70)
    for regime, row in duration_stats.iterrows():
        print(f"{regime:<20} {row['mean']:>10.1f} {row['std']:>10.1f} {row['min']:>8.1f} {row['max']:>8.1f} {int(row['count']):>8}")
    
    # Calculate transition probabilities (Markov chain)
    print("\n📊 REGIME TRANSITION PROBABILITIES:")
    print("-" * 60)
    
    transitions = defaultdict(lambda: defaultdict(int))
    prev_regime = None
    
    for regime in df['regime']:
        if prev_regime is not None:
            transitions[prev_regime][regime] += 1
        prev_regime = regime
    
    # Convert to probabilities
    transition_probs = {}
    for from_regime, to_regimes in transitions.items():
        total = sum(to_regimes.values())
        transition_probs[from_regime] = {to: count/total for to, count in to_regimes.items()}
    
    regimes_list = ['Strong Bull', 'Mild Bull', 'Euphoria', 'Consolidation', 'Neutral', 
                    'Recovery', 'Mild Bear', 'Strong Bear', 'Capitulation']
    
    # Print transition matrix
    print("\nTransition Matrix (row = from, col = to):")
    print(f"{'From \\ To':<15}", end='')
    for r in regimes_list:
        if r in transitions:
            print(f"{r[:8]:>10}", end='')
    print()
    
    for from_r in regimes_list:
        if from_r in transition_probs:
            print(f"{from_r[:14]:<15}", end='')
            for to_r in regimes_list:
                if to_r in transitions:
                    prob = transition_probs[from_r].get(to_r, 0)
                    print(f"{prob*100:>9.1f}%", end='')
            print()
    
    return transition_probs, duration_stats, regime_durations


def get_v1_performance_by_regime(df: pd.DataFrame) -> dict:
    """Get V1 strategy performance statistics by regime"""
    
    # Run V1 strategy to get trade data
    from strategy import TurtleDonchianStrategy
    from config import StrategyParams
    
    params = StrategyParams(
        entry_len=40, exit_len=16, atr_len=20, trail_mult=4.0,
        size_stop_mult=2.0, risk_percent=1.0, max_units=4,
        pyramid_spacing_n=1.5, long_only=True, use_regime_filter=True,
        lot_step=0.001, commission_pct=0.08, slippage_pct=0.05
    )
    
    strategy = TurtleDonchianStrategy(params)
    results = strategy.run_backtest(df, initial_capital=INITIAL_CAPITAL, verbose=False)
    
    # Get regime for each trade
    trades_data = []
    for t in strategy.trades:
        # Find regime at entry
        if t.entry_time in df.index:
            regime = df.loc[t.entry_time, 'regime']
        else:
            # Find closest
            idx = df.index.get_indexer([t.entry_time], method='nearest')[0]
            regime = df.iloc[idx]['regime']
        
        trades_data.append({
            'entry_time': t.entry_time,
            'pnl': t.pnl or 0,
            'pnl_pct': ((t.exit_price - t.entry_price) / t.entry_price * 100) if t.exit_price and t.entry_price else 0,
            'regime': regime,
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Performance by regime
    print("\n📊 V1 STRATEGY PERFORMANCE BY REGIME:")
    print("-" * 70)
    
    regime_perf = {}
    
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        if len(regime_trades) > 0:
            wins = len(regime_trades[regime_trades['pnl'] > 0])
            regime_perf[regime] = {
                'trades': len(regime_trades),
                'win_rate': wins / len(regime_trades),
                'avg_pnl_pct': regime_trades['pnl_pct'].mean(),
                'total_pnl': regime_trades['pnl'].sum(),
                'std_pnl_pct': regime_trades['pnl_pct'].std(),
            }
    
    print(f"\n{'Regime':<20} {'Trades':>8} {'Win Rate':>10} {'Avg P&L%':>10} {'Total P&L':>12}")
    print("-" * 70)
    for regime, stats in sorted(regime_perf.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        print(f"{regime:<20} {stats['trades']:>8} {stats['win_rate']*100:>9.1f}% {stats['avg_pnl_pct']:>+9.2f}% ${stats['total_pnl']:>+11,.2f}")
    
    return regime_perf


def simulate_20_years(transition_probs: dict, duration_stats: pd.DataFrame, 
                      regime_perf: dict, initial_capital: float, 
                      num_simulations: int = 1000):
    """Simulate 20 years of trading based on regime transitions"""
    
    print("\n" + "=" * 80)
    print("20-YEAR FORWARD SIMULATION")
    print("=" * 80)
    
    np.random.seed(42)
    
    years = 20
    days_per_year = 365
    total_days = years * days_per_year
    
    # Average trades per day in each regime (from historical data)
    # Roughly 442 trades over ~3000 days = 0.15 trades/day on average
    trades_per_day_base = 0.15
    
    # Regime-specific trade frequency multipliers
    regime_trade_freq = {
        'Strong Bull': 1.5,
        'Mild Bull': 1.2,
        'Euphoria': 2.0,
        'Consolidation': 0.8,
        'Neutral': 1.0,
        'Recovery': 1.3,
        'Mild Bear': 0.7,
        'Strong Bear': 0.5,
        'Capitulation': 0.3,
    }
    
    all_simulations = []
    final_values = []
    
    # Default regime performance for regimes we haven't seen
    default_perf = {
        'win_rate': 0.45,
        'avg_pnl_pct': 2.0,
        'std_pnl_pct': 5.0,
    }
    
    regimes_list = list(transition_probs.keys())
    
    for sim in range(num_simulations):
        # Start with random regime based on historical distribution
        current_regime = np.random.choice(regimes_list)
        
        equity = initial_capital
        equity_curve = [equity]
        regime_history = [current_regime]
        
        days_in_regime = 0
        
        # Get average duration for current regime
        if current_regime in duration_stats.index:
            avg_duration = duration_stats.loc[current_regime, 'mean']
            std_duration = duration_stats.loc[current_regime, 'std']
            if pd.isna(std_duration):
                std_duration = avg_duration * 0.5
        else:
            avg_duration = 30
            std_duration = 15
        
        regime_duration = max(5, np.random.normal(avg_duration, std_duration))
        
        for day in range(total_days):
            days_in_regime += 1
            
            # Check for regime transition
            if days_in_regime >= regime_duration:
                # Transition to new regime based on probabilities
                if current_regime in transition_probs:
                    to_regimes = list(transition_probs[current_regime].keys())
                    probs = list(transition_probs[current_regime].values())
                    current_regime = np.random.choice(to_regimes, p=probs)
                else:
                    current_regime = np.random.choice(regimes_list)
                
                days_in_regime = 0
                
                # New duration
                if current_regime in duration_stats.index:
                    avg_duration = duration_stats.loc[current_regime, 'mean']
                    std_duration = duration_stats.loc[current_regime, 'std']
                    if pd.isna(std_duration):
                        std_duration = avg_duration * 0.5
                else:
                    avg_duration = 30
                    std_duration = 15
                
                regime_duration = max(5, np.random.normal(avg_duration, std_duration))
            
            # Simulate trades for this day
            trade_freq = trades_per_day_base * regime_trade_freq.get(current_regime, 1.0)
            num_trades = np.random.poisson(trade_freq)
            
            for _ in range(num_trades):
                # Get regime performance
                if current_regime in regime_perf:
                    perf = regime_perf[current_regime]
                else:
                    perf = default_perf
                
                # Simulate trade outcome
                win_rate = perf.get('win_rate', 0.45)
                avg_pnl = perf.get('avg_pnl_pct', 2.0)
                std_pnl = perf.get('std_pnl_pct', 5.0)
                
                if np.random.random() < win_rate:
                    # Winning trade
                    pnl_pct = abs(np.random.normal(avg_pnl, std_pnl))
                else:
                    # Losing trade
                    pnl_pct = -abs(np.random.normal(avg_pnl * 0.6, std_pnl * 0.5))
                
                # Apply P&L (1% risk per trade, so P&L is relative to position size)
                risk_amount = equity * 0.01
                pnl = risk_amount * (pnl_pct / 2.0)  # Normalized to risk
                equity += pnl
                
                # Floor at 10% of initial capital (can't go negative)
                equity = max(equity, initial_capital * 0.1)
            
            equity_curve.append(equity)
            regime_history.append(current_regime)
        
        all_simulations.append(equity_curve)
        final_values.append(equity)
    
    # Convert to numpy for analysis
    all_simulations = np.array(all_simulations)
    final_values = np.array(final_values)
    
    # Statistics
    print(f"\nSimulated {num_simulations} paths over {years} years")
    print("-" * 60)
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = np.percentile(final_values, percentiles)
    
    print(f"\n{'Percentile':<15} {'Final Value':>15} {'Total Return':>15}")
    print("-" * 50)
    for p, v in zip(percentiles, pct_values):
        ret = (v / initial_capital - 1) * 100
        print(f"{p}th{'':<11} ${v:>14,.2f} {ret:>+14.1f}%")
    
    print(f"\n{'Mean Final Value:':<30} ${final_values.mean():>14,.2f}")
    print(f"{'Median Final Value:':<30} ${np.median(final_values):>14,.2f}")
    print(f"{'Std Dev:':<30} ${final_values.std():>14,.2f}")
    print(f"{'Min:':<30} ${final_values.min():>14,.2f}")
    print(f"{'Max:':<30} ${final_values.max():>14,.2f}")
    
    # Probability of outcomes
    print("\n📊 PROBABILITY OF OUTCOMES:")
    print("-" * 50)
    
    thresholds = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    for thresh in thresholds:
        prob = (final_values >= thresh).mean() * 100
        print(f"  P(Portfolio ≥ ${thresh:>10,}) = {prob:>6.1f}%")
    
    # Probability of loss
    loss_prob = (final_values < initial_capital).mean() * 100
    print(f"\n  P(Loss after 20 years) = {loss_prob:.1f}%")
    
    return all_simulations, final_values


def create_simulation_plots(all_simulations: np.ndarray, final_values: np.ndarray, 
                            initial_capital: float, df: pd.DataFrame):
    """Create visualization plots"""
    
    years = 20
    days = all_simulations.shape[1]
    time_axis = np.linspace(0, years, days)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Simulation fan chart
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Plot percentile bands
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    colors = ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#fc9272', '#fcbba1', '#fee5d9']
    
    pct_curves = np.percentile(all_simulations, percentiles, axis=0)
    
    # Fill between percentile bands
    ax1.fill_between(time_axis, pct_curves[0], pct_curves[6], alpha=0.3, color='blue', label='5-95th pct')
    ax1.fill_between(time_axis, pct_curves[1], pct_curves[5], alpha=0.4, color='blue', label='10-90th pct')
    ax1.fill_between(time_axis, pct_curves[2], pct_curves[4], alpha=0.5, color='blue', label='25-75th pct')
    ax1.plot(time_axis, pct_curves[3], 'b-', linewidth=2, label='Median')
    
    # Plot a few random paths
    for i in np.random.choice(len(all_simulations), 20, replace=False):
        ax1.plot(time_axis, all_simulations[i], 'gray', alpha=0.2, linewidth=0.5)
    
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('20-Year Portfolio Projection (1,000 Simulations)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # 2. Final value distribution
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Use log scale for better visualization
    log_values = np.log10(final_values)
    ax2.hist(log_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
    
    # Add percentile lines
    for p, label in [(50, 'Median'), (5, '5th pct'), (95, '95th pct')]:
        val = np.percentile(final_values, p)
        ax2.axvline(x=np.log10(val), color='red' if p == 50 else 'orange', 
                   linestyle='--', linewidth=2, label=f'{label}: ${val:,.0f}')
    
    ax2.set_xlabel('Final Portfolio Value (log scale)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
    
    # Custom x-axis labels
    log_ticks = [3, 4, 5, 6, 7]  # $1K, $10K, $100K, $1M, $10M
    ax2.set_xticks(log_ticks)
    ax2.set_xticklabels(['$1K', '$10K', '$100K', '$1M', '$10M'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Historical regime timeline
    ax3 = fig.add_subplot(2, 2, 3)
    
    regime_colors = {
        'Strong Bull': 'darkgreen',
        'Mild Bull': 'lightgreen',
        'Euphoria': 'gold',
        'Consolidation': 'gray',
        'Neutral': 'lightgray',
        'Recovery': 'cyan',
        'Mild Bear': 'lightsalmon',
        'Strong Bear': 'red',
        'Capitulation': 'darkred',
    }
    
    # Plot BTC price with regime coloring
    ax3.plot(df.index, df['close'], 'black', linewidth=0.5, alpha=0.5)
    
    for regime, color in regime_colors.items():
        mask = df['regime'] == regime
        if mask.any():
            ax3.scatter(df.index[mask], df['close'][mask], c=color, s=1, alpha=0.5, label=regime)
    
    ax3.set_ylabel('BTC Price ($)')
    ax3.set_title('Historical BTC Price Colored by Regime', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability over time
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate probability of reaching thresholds over time
    thresholds = [5000, 10000, 50000, 100000]
    colors = ['green', 'blue', 'purple', 'red']
    
    for thresh, color in zip(thresholds, colors):
        probs = (all_simulations >= thresh).mean(axis=0) * 100
        ax4.plot(time_axis, probs, color=color, linewidth=2, label=f'≥${thresh:,}')
    
    ax4.set_xlabel('Years')
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Probability of Reaching Portfolio Thresholds Over Time', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, '20_year_simulation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Simulation plots saved to: {plot_path}")
    
    # Additional plot: Regime distribution and performance
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regime time distribution
    ax1 = axes2[0]
    regime_counts = df['regime'].value_counts()
    colors_list = [regime_colors.get(r, 'gray') for r in regime_counts.index]
    ax1.pie(regime_counts.values, labels=regime_counts.index, colors=colors_list, 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Historical Time Spent in Each Regime', fontsize=12, fontweight='bold')
    
    # Transition heatmap placeholder (simplified)
    ax2 = axes2[1]
    ax2.text(0.5, 0.5, 'Regime Transition\nProbabilities\n(See console output)', 
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    ax2.set_title('Markov Transition Matrix', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, 'regime_analysis.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Regime analysis saved to: {plot_path2}")
    
    return plot_path


def main():
    print("=" * 80)
    print("20-YEAR FORWARD SIMULATION BASED ON HISTORICAL REGIMES")
    print("=" * 80)
    
    # Load data
    print("\nLoading BTC 4H data...")
    df = download_btc_data(timeframe="4h")
    print(f"Data: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
    
    # Identify regimes
    print("\nIdentifying market regimes...")
    df = identify_regimes(df)
    
    # Analyze regime statistics
    transition_probs, duration_stats, regime_durations = analyze_regime_statistics(df)
    
    # Get V1 performance by regime
    regime_perf = get_v1_performance_by_regime(df)
    
    # Run 20-year simulation
    all_simulations, final_values = simulate_20_years(
        transition_probs, duration_stats, regime_perf, 
        INITIAL_CAPITAL, num_simulations=1000
    )
    
    # Create plots
    create_simulation_plots(all_simulations, final_values, INITIAL_CAPITAL, df)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: 20-YEAR PROJECTION FROM $1,000")
    print("=" * 80)
    
    median_final = np.median(final_values)
    p5 = np.percentile(final_values, 5)
    p95 = np.percentile(final_values, 95)
    
    print(f"""
    Starting Capital:          ${INITIAL_CAPITAL:,.2f}
    
    After 20 Years:
      - Pessimistic (5th pct):  ${p5:,.2f} ({(p5/INITIAL_CAPITAL-1)*100:+.0f}%)
      - Expected (Median):      ${median_final:,.2f} ({(median_final/INITIAL_CAPITAL-1)*100:+.0f}%)
      - Optimistic (95th pct):  ${p95:,.2f} ({(p95/INITIAL_CAPITAL-1)*100:+.0f}%)
    
    Key Probabilities:
      - 90% chance of portfolio > ${np.percentile(final_values, 10):,.0f}
      - 50% chance of portfolio > ${median_final:,.0f}
      - 10% chance of portfolio > ${np.percentile(final_values, 90):,.0f}
    """)
    
    return df, all_simulations, final_values


if __name__ == "__main__":
    df, sims, finals = main()

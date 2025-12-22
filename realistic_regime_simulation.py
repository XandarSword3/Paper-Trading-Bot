"""
Realistic 20-Year Regime Simulation with Costs
Uses same methodology as regime_simulation.py but adds realistic trading costs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, PLOTS_DIR
from data_fetcher import download_btc_data
from regime_simulation import identify_regimes, analyze_regime_statistics, get_v1_performance_by_regime

os.makedirs(PLOTS_DIR, exist_ok=True)

# Realistic costs for automated trading
EXCHANGE_FEE_PER_TRADE = 0.0015    # 0.15% round trip
SLIPPAGE_PER_TRADE = 0.0005        # 0.05% round trip (limit orders)
TOTAL_COST_PER_TRADE = EXCHANGE_FEE_PER_TRADE + SLIPPAGE_PER_TRADE  # 0.20%
ANNUAL_ERROR_RATE = 0.01           # 1% annual from execution issues


def simulate_20_years_with_costs(transition_probs: dict, duration_stats: pd.DataFrame, 
                                  regime_perf: dict, initial_capital: float, 
                                  apply_costs: bool = True,
                                  num_simulations: int = 1000):
    """
    Simulate 20 years - SAME as regime_simulation.py but with optional costs
    """
    
    label = "WITH COSTS" if apply_costs else "NO COSTS (Optimistic)"
    print(f"\n{'=' * 80}")
    print(f"20-YEAR SIMULATION - {label}")
    print(f"{'=' * 80}")
    
    if apply_costs:
        print(f"\nApplying realistic costs:")
        print(f"  - Exchange fees: {EXCHANGE_FEE_PER_TRADE*100:.2f}% per trade")
        print(f"  - Slippage: {SLIPPAGE_PER_TRADE*100:.2f}% per trade")
        print(f"  - Annual errors: {ANNUAL_ERROR_RATE*100:.0f}%")
        print(f"  - Total per-trade cost: {TOTAL_COST_PER_TRADE*100:.2f}%\n")
    
    np.random.seed(42)
    
    years = 20
    days_per_year = 365
    total_days = years * days_per_year
    
    trades_per_day_base = 0.15
    
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
    
    default_perf = {
        'win_rate': 0.45,
        'avg_pnl_pct': 2.0,
        'std_pnl_pct': 5.0,
    }
    
    regimes_list = list(transition_probs.keys())
    
    for sim in range(num_simulations):
        current_regime = np.random.choice(regimes_list)
        
        equity = initial_capital
        equity_curve = [equity]
        
        days_in_regime = 0
        
        if current_regime in duration_stats.index:
            avg_duration = duration_stats.loc[current_regime, 'mean']
            std_duration = duration_stats.loc[current_regime, 'std']
            if pd.isna(std_duration):
                std_duration = avg_duration * 0.5
        else:
            avg_duration = 30
            std_duration = 15
        
        regime_duration = max(1, int(np.random.normal(avg_duration, std_duration)))
        
        year_start_equity = equity
        
        for day in range(total_days):
            # Check for regime transition
            days_in_regime += 1
            if days_in_regime >= regime_duration:
                next_regime_probs = transition_probs.get(current_regime, {})
                if next_regime_probs:
                    next_regimes = list(next_regime_probs.keys())
                    probs = list(next_regime_probs.values())
                    probs = np.array(probs) / sum(probs)
                    current_regime = np.random.choice(next_regimes, p=probs)
                
                if current_regime in duration_stats.index:
                    avg_duration = duration_stats.loc[current_regime, 'mean']
                    std_duration = duration_stats.loc[current_regime, 'std']
                    if pd.isna(std_duration):
                        std_duration = avg_duration * 0.5
                else:
                    avg_duration = 30
                    std_duration = 15
                
                regime_duration = max(1, int(np.random.normal(avg_duration, std_duration)))
                days_in_regime = 0
            
            # Simulate trades for this day
            trade_freq = trades_per_day_base * regime_trade_freq.get(current_regime, 1.0)
            num_trades = np.random.poisson(trade_freq)
            
            for _ in range(num_trades):
                if current_regime in regime_perf:
                    perf = regime_perf[current_regime]
                else:
                    perf = default_perf
                
                win_rate = perf.get('win_rate', 0.45)
                avg_pnl = perf.get('avg_pnl_pct', 2.0)
                std_pnl = perf.get('std_pnl_pct', 5.0)
                
                if np.random.random() < win_rate:
                    pnl_pct = abs(np.random.normal(avg_pnl, std_pnl))
                else:
                    pnl_pct = -abs(np.random.normal(avg_pnl * 0.6, std_pnl * 0.5))
                
                # Apply costs if realistic mode
                if apply_costs:
                    pnl_pct = pnl_pct - (TOTAL_COST_PER_TRADE * 100)  # Convert to percentage
                
                # Apply P&L - SAME AS ORIGINAL
                risk_amount = equity * 0.01
                pnl = risk_amount * (pnl_pct / 2.0)  # Normalized to risk
                equity += pnl
                equity = max(equity, 10)
            
            # Apply annual error rate
            if apply_costs and day > 0 and day % 365 == 0:
                year_profit = equity - year_start_equity
                if year_profit > 0:
                    equity -= year_profit * ANNUAL_ERROR_RATE
                year_start_equity = equity
            
            equity_curve.append(equity)
        
        all_simulations.append(equity_curve)
        final_values.append(equity)
    
    all_simulations = np.array(all_simulations)
    final_values = np.array(final_values)
    
    # Print results
    print(f"\nSimulated {num_simulations} paths over {years} years")
    print("-" * 60)
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n{'Percentile':<15} {'Final Value':>18} {'Total Return':>15}")
    print("-" * 50)
    
    for p in percentiles:
        val = np.percentile(final_values, p)
        ret = ((val - initial_capital) / initial_capital) * 100
        label = "th" if p not in [5, 25] else ("th" if p == 25 else "th")
        print(f"{p}{label:<12} ${val:>17,.2f}   {ret:>+13.1f}%")
    
    print(f"\n{'Mean':<15} ${np.mean(final_values):>17,.2f}")
    print(f"{'Median':<15} ${np.median(final_values):>17,.2f}")
    print(f"{'Std Dev':<15} ${np.std(final_values):>17,.2f}")
    
    thresholds = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    print(f"\n📊 Probability of Outcomes:")
    print("-" * 50)
    for thresh in thresholds:
        prob = (final_values >= thresh).mean() * 100
        print(f"  P(Portfolio >= ${thresh:>10,}) = {prob:>6.1f}%")
    
    loss_prob = (final_values < initial_capital).mean() * 100
    print(f"\n  P(Loss after {years} years) = {loss_prob:.1f}%")
    
    return all_simulations, final_values


def create_comparison_plot(opt_sims, opt_vals, real_sims, real_vals, initial_capital):
    """Create side-by-side comparison"""
    
    fig = plt.figure(figsize=(18, 10))
    
    years_axis = np.linspace(0, 20, opt_sims.shape[1])
    
    # Optimistic
    ax1 = plt.subplot(2, 2, 1)
    pcts = [5, 25, 50, 75, 95]
    opt_curves = np.percentile(opt_sims, pcts, axis=0)
    
    ax1.fill_between(years_axis, opt_curves[0], opt_curves[4], alpha=0.2, color='green', label='5-95th')
    ax1.fill_between(years_axis, opt_curves[1], opt_curves[3], alpha=0.4, color='green', label='25-75th')
    ax1.plot(years_axis, opt_curves[2], 'g-', linewidth=3, label='Median')
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('OPTIMISTIC (No Costs)', fontsize=14, fontweight='bold', color='darkgreen')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Realistic
    ax2 = plt.subplot(2, 2, 2)
    real_curves = np.percentile(real_sims, pcts, axis=0)
    
    ax2.fill_between(years_axis, real_curves[0], real_curves[4], alpha=0.2, color='red', label='5-95th')
    ax2.fill_between(years_axis, real_curves[1], real_curves[3], alpha=0.4, color='red', label='25-75th')
    ax2.plot(years_axis, real_curves[2], 'r-', linewidth=3, label='Median')
    ax2.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.set_title('REALISTIC (With Costs)', fontsize=14, fontweight='bold', color='darkred')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Distribution
    ax3 = plt.subplot(2, 2, 3)
    opt_clip = np.clip(opt_vals, 0, np.percentile(opt_vals, 99))
    real_clip = np.clip(real_vals, 0, np.percentile(real_vals, 99))
    
    ax3.hist(opt_clip, bins=50, alpha=0.5, color='green', label='Optimistic', density=True)
    ax3.hist(real_clip, bins=50, alpha=0.5, color='red', label='Realistic', density=True)
    ax3.axvline(x=np.median(opt_vals), color='green', linewidth=2, label=f'Opt Median: ${np.median(opt_vals):,.0f}')
    ax3.axvline(x=np.median(real_vals), color='red', linewidth=2, label=f'Real Median: ${np.median(real_vals):,.0f}')
    ax3.set_xlabel('Final Value ($)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Distribution After 20 Years', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    diff_pct = ((np.median(real_vals) - np.median(opt_vals)) / np.median(opt_vals)) * 100
    
    summary = f"""
    COMPARISON SUMMARY (Starting: ${initial_capital:,.0f})
    
    ═══════════════════════════════════════════════════════════
                        OPTIMISTIC         REALISTIC      
    ═══════════════════════════════════════════════════════════
    Median Outcome      ${np.median(opt_vals):>12,.0f}     ${np.median(real_vals):>12,.0f}
                                                ({diff_pct:+.1f}%)
    
    5th Percentile      ${np.percentile(opt_vals, 5):>12,.0f}     ${np.percentile(real_vals, 5):>12,.0f}
    
    95th Percentile     ${np.percentile(opt_vals, 95):>12,.0f}     ${np.percentile(real_vals, 95):>12,.0f}
    
    P(>= $100K)         {(opt_vals >= 100000).mean()*100:>12.1f}%    {(real_vals >= 100000).mean()*100:>12.1f}%
    
    P(>= $1M)           {(opt_vals >= 1000000).mean()*100:>12.1f}%    {(real_vals >= 1000000).mean()*100:>12.1f}%
    
    P(Loss)             {(opt_vals < initial_capital).mean()*100:>12.1f}%    {(real_vals < initial_capital).mean()*100:>12.1f}%
    ═══════════════════════════════════════════════════════════
    
    Realistic Costs Applied:
    • Exchange fees: {EXCHANGE_FEE_PER_TRADE*100:.2f}% per trade
    • Slippage: {SLIPPAGE_PER_TRADE*100:.2f}% per trade
    • Annual errors: {ANNUAL_ERROR_RATE*100:.0f}% of gains
    • No taxes (Lebanon / reinvesting)
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('20-Year Simulation: Optimistic vs Realistic\n(Based on Historical Regime Analysis)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, 'realistic_vs_optimistic_final.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("REALISTIC 20-YEAR SIMULATION COMPARISON")
    print("="*80)
    
    # Load and analyze data
    df = download_btc_data()
    df = identify_regimes(df)
    transition_probs, duration_stats, _ = analyze_regime_statistics(df)
    regime_perf = get_v1_performance_by_regime(df)
    
    initial_capital = 1000.0
    
    # Run optimistic
    opt_sims, opt_vals = simulate_20_years_with_costs(
        transition_probs, duration_stats, regime_perf,
        initial_capital, apply_costs=False, num_simulations=1000
    )
    
    # Run realistic
    real_sims, real_vals = simulate_20_years_with_costs(
        transition_probs, duration_stats, regime_perf,
        initial_capital, apply_costs=True, num_simulations=1000
    )
    
    # Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Optimistic':>18} {'Realistic':>18} {'Difference':>15}")
    print("-"*80)
    
    metrics = [
        ("Median Outcome", np.median(opt_vals), np.median(real_vals)),
        ("5th Percentile", np.percentile(opt_vals, 5), np.percentile(real_vals, 5)),
        ("95th Percentile", np.percentile(opt_vals, 95), np.percentile(real_vals, 95)),
        ("P(>= $100K)", (opt_vals >= 100000).mean()*100, (real_vals >= 100000).mean()*100),
        ("P(>= $1M)", (opt_vals >= 1000000).mean()*100, (real_vals >= 1000000).mean()*100),
    ]
    
    for name, opt, real in metrics:
        if "P(" in name:
            diff = real - opt
            print(f"{name:<25} {opt:>17.1f}% {real:>17.1f}% {diff:>+14.1f}%")
        else:
            diff_pct = ((real - opt) / opt) * 100
            print(f"{name:<25} ${opt:>16,.0f} ${real:>16,.0f} {diff_pct:>+14.1f}%")
    
    create_comparison_plot(opt_sims, opt_vals, real_sims, real_vals, initial_capital)
    
    print("\n" + "="*80)
    print("⚠️  READ CRITICAL_WARNINGS.md FOR FULL RISK DISCLOSURE")
    print("="*80)

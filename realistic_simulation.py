"""
REALISTIC 20-Year Simulation - Properly Calibrated
Based on ACTUAL backtest performance with realistic cost adjustments
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, PLOTS_DIR, StrategyParams
from strategy import TurtleDonchianStrategy
from data_fetcher import download_btc_data

os.makedirs(PLOTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 1000.0

# =============================================================================
# REALISTIC COST PARAMETERS
# =============================================================================

# Per-trade costs (applied to each trade)
EXCHANGE_FEE_ROUND_TRIP = 0.0015    # 0.075% x 2 (entry + exit) = 0.15%
SLIPPAGE_ROUND_TRIP = 0.0005        # 0.025% x 2 = 0.05% (limit orders, liquid market)
TOTAL_TRADE_COST = EXCHANGE_FEE_ROUND_TRIP + SLIPPAGE_ROUND_TRIP  # 0.20% per trade

# Annual adjustments
TAX_RATE = 0.0                      # NO TAX (Lebanon / reinvesting)
ANNUAL_ERROR_RATE = 0.01            # 1% of annual returns lost to errors/missed trades (automated)


def get_actual_trade_returns():
    """Get actual trade-by-trade returns from V1 backtest"""
    
    print("Loading actual trade data from V1 backtest...")
    
    # Load data
    df = download_btc_data()
    
    # Run V1 strategy
    params = StrategyParams(
        entry_len=40, exit_len=16, atr_len=20, trail_mult=4.0,
        size_stop_mult=2.0, risk_percent=1.0, max_units=4,
        pyramid_spacing_n=1.5, long_only=True, use_regime_filter=True,
        lot_step=0.001, commission_pct=0.0, slippage_pct=0.0  # No costs in backtest
    )
    
    strategy = TurtleDonchianStrategy(params)
    strategy.run_backtest(df, initial_capital=INITIAL_CAPITAL, verbose=False)
    
    # Extract trade returns
    trade_returns = []
    for t in strategy.trades:
        if t.exit_price and t.entry_price:
            # Return as percentage of position
            ret_pct = (t.exit_price - t.entry_price) / t.entry_price
            trade_returns.append(ret_pct)
    
    trade_returns = np.array(trade_returns)
    
    print(f"\nActual V1 Backtest Statistics:")
    print(f"  Total trades: {len(trade_returns)}")
    print(f"  Win rate: {(trade_returns > 0).mean()*100:.1f}%")
    print(f"  Mean return per trade: {trade_returns.mean()*100:.2f}%")
    print(f"  Median return per trade: {np.median(trade_returns)*100:.2f}%")
    print(f"  Std dev: {trade_returns.std()*100:.2f}%")
    print(f"  Best trade: {trade_returns.max()*100:.2f}%")
    print(f"  Worst trade: {trade_returns.min()*100:.2f}%")
    
    # Calculate years of data
    years = (df.index[-1] - df.index[0]).days / 365.25
    trades_per_year = len(trade_returns) / years
    print(f"  Trades per year: {trades_per_year:.1f}")
    
    return trade_returns, trades_per_year


def simulate_20_years(trade_returns: np.ndarray, trades_per_year: float,
                      initial_capital: float, apply_costs: bool,
                      num_simulations: int = 500):
    """
    Monte Carlo simulation using bootstrap resampling of actual trades
    Calibrated to match actual backtest performance
    """
    
    years = 20
    trades_per_sim = int(trades_per_year * years)
    
    # Adjust returns for costs if realistic mode
    if apply_costs:
        # Subtract per-trade costs from each trade return
        adjusted_returns = trade_returns - TOTAL_TRADE_COST
    else:
        adjusted_returns = trade_returns.copy()
    
    # Calculate effective risk multiplier to match backtest
    # Backtest: 855% return over 8.3 years = 442 trades
    # Average capital growth per trade = (9.55)^(1/442) - 1 = 0.51% per trade
    # With 2.07% avg return and 1% risk: expected_growth = 0.01 * 2.07 = 2.07% per trade
    # Actual was less due to variable sizing - use effective_risk = 0.25 (0.25 x avg_return)
    effective_risk_mult = 0.25  # Calibrated to match 855% over 8.3 years
    
    results = []
    all_equity_curves = []
    
    np.random.seed(42)
    
    for sim in range(num_simulations):
        # Bootstrap sample trades
        sampled_indices = np.random.choice(len(adjusted_returns), trades_per_sim, replace=True)
        sampled_returns = adjusted_returns[sampled_indices]
        
        # Calculate equity curve
        capital = initial_capital
        equity_curve = [capital]
        
        for year in range(years):
            year_start = capital
            
            # Get trades for this year
            start_idx = int(year * trades_per_year)
            end_idx = int((year + 1) * trades_per_year)
            year_trades = sampled_returns[start_idx:end_idx]
            
            # Apply each trade with calibrated risk
            for trade_ret in year_trades:
                # P&L proportional to trade return and risk multiplier
                pnl = capital * effective_risk_mult * trade_ret
                capital += pnl
                capital = max(capital, 10)  # Floor at $10
            
            # Apply annual costs if realistic
            if apply_costs:
                # Apply error rate (missed trades, bad execution)
                year_gain = capital - year_start
                if year_gain > 0:
                    capital -= year_gain * ANNUAL_ERROR_RATE
                
                # Apply taxes on profits
                year_profit = capital - year_start
                if year_profit > 0:
                    tax = year_profit * TAX_RATE
                    capital -= tax
            
            equity_curve.append(capital)
        
        results.append(capital)
        all_equity_curves.append(equity_curve)
    
    return np.array(results), np.array(all_equity_curves)


def run_comparison():
    """Run both optimistic and realistic simulations and compare"""
    
    print("=" * 80)
    print("20-YEAR SIMULATION: OPTIMISTIC vs REALISTIC")
    print("=" * 80)
    
    # Get actual trade data
    trade_returns, trades_per_year = get_actual_trade_returns()
    
    # Run optimistic simulation (no costs)
    print("\n" + "=" * 60)
    print("OPTIMISTIC SIMULATION (Perfect Execution, No Costs)")
    print("=" * 60)
    
    opt_results, opt_curves = simulate_20_years(
        trade_returns, trades_per_year, INITIAL_CAPITAL,
        apply_costs=False, num_simulations=500
    )
    
    print_results("OPTIMISTIC", opt_results, INITIAL_CAPITAL)
    
    # Run realistic simulation (with costs)
    print("\n" + "=" * 60)
    print("REALISTIC SIMULATION (Fees, Slippage, Taxes, Errors)")
    print("=" * 60)
    print(f"\nCost assumptions:")
    print(f"  - Exchange fees (round trip): {EXCHANGE_FEE_ROUND_TRIP*100:.2f}%")
    print(f"  - Slippage (round trip): {SLIPPAGE_ROUND_TRIP*100:.2f}%")
    print(f"  - Total per-trade cost: {TOTAL_TRADE_COST*100:.2f}%")
    print(f"  - Annual error rate: {ANNUAL_ERROR_RATE*100:.1f}%")
    print(f"  - Tax rate on profits: {TAX_RATE*100:.0f}%")
    
    real_results, real_curves = simulate_20_years(
        trade_returns, trades_per_year, INITIAL_CAPITAL,
        apply_costs=True, num_simulations=500
    )
    
    print_results("REALISTIC", real_results, INITIAL_CAPITAL)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Optimistic':>18} {'Realistic':>18} {'Difference':>15}")
    print("-" * 80)
    
    comparisons = [
        ("5th Percentile", np.percentile(opt_results, 5), np.percentile(real_results, 5)),
        ("25th Percentile", np.percentile(opt_results, 25), np.percentile(real_results, 25)),
        ("Median (50th)", np.percentile(opt_results, 50), np.percentile(real_results, 50)),
        ("75th Percentile", np.percentile(opt_results, 75), np.percentile(real_results, 75)),
        ("95th Percentile", np.percentile(opt_results, 95), np.percentile(real_results, 95)),
    ]
    
    for name, opt, real in comparisons:
        diff_pct = (real - opt) / opt * 100
        print(f"{name:<25} ${opt:>16,.0f} ${real:>16,.0f} {diff_pct:>+14.1f}%")
    
    print("-" * 80)
    
    # Probability comparisons
    thresholds = [10000, 50000, 100000, 500000, 1000000]
    print(f"\n{'Probability of Reaching':<25} {'Optimistic':>18} {'Realistic':>18}")
    print("-" * 65)
    for thresh in thresholds:
        opt_prob = (opt_results >= thresh).mean() * 100
        real_prob = (real_results >= thresh).mean() * 100
        label = f">= ${thresh:,}"
        print(f"{label:<25} {opt_prob:>17.1f}% {real_prob:>17.1f}%")
    
    opt_loss = (opt_results < INITIAL_CAPITAL).mean() * 100
    real_loss = (real_results < INITIAL_CAPITAL).mean() * 100
    print(f"\n{'Probability of LOSS':<25} {opt_loss:>17.1f}% {real_loss:>17.1f}%")
    
    # Create plots
    create_comparison_plots(opt_results, opt_curves, real_results, real_curves)
    
    # Print warnings
    print("\n" + "=" * 80)
    print("CRITICAL WARNINGS")
    print("=" * 80)
    print("""
    1. These simulations assume BTC behaves similarly to 2017-2025
    2. Future volatility/trends may be completely different
    3. Black swan events (exchange hacks, regulations) NOT modeled
    4. Psychological errors likely WORSE than 5% assumed
    5. Tax treatment varies by jurisdiction
    6. Past performance does NOT guarantee future results
    
    READ CRITICAL_WARNINGS.md for full risk disclosure!
    """)
    
    return opt_results, real_results


def print_results(label: str, results: np.ndarray, initial: float):
    """Print simulation results"""
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    
    print(f"\n{label} Results (from ${initial:,.0f}):")
    print("-" * 50)
    print(f"{'Percentile':<15} {'Final Value':>15} {'Return':>15}")
    print("-" * 50)
    
    for p in percentiles:
        val = np.percentile(results, p)
        ret = (val - initial) / initial * 100
        print(f"{p}th {'':<10} ${val:>14,.0f} {ret:>+14.0f}%")
    
    print("-" * 50)
    print(f"{'Mean':<15} ${np.mean(results):>14,.0f} {(np.mean(results)/initial - 1)*100:>+14.0f}%")
    print(f"{'Std Dev':<15} ${np.std(results):>14,.0f}")
    print(f"{'Min':<15} ${np.min(results):>14,.0f}")
    print(f"{'Max':<15} ${np.max(results):>14,.0f}")
    
    loss_prob = (results < initial).mean() * 100
    print(f"\nProbability of loss: {loss_prob:.1f}%")


def create_comparison_plots(opt_results, opt_curves, real_results, real_curves):
    """Create comparison visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    
    years = np.arange(21)
    
    # 1. Optimistic fan chart
    ax1 = plt.subplot(2, 2, 1)
    pcts = [5, 25, 50, 75, 95]
    opt_pct_curves = np.percentile(opt_curves, pcts, axis=0)
    
    ax1.fill_between(years, opt_pct_curves[0], opt_pct_curves[4], alpha=0.2, color='green', label='5-95th')
    ax1.fill_between(years, opt_pct_curves[1], opt_pct_curves[3], alpha=0.4, color='green', label='25-75th')
    ax1.plot(years, opt_pct_curves[2], 'g-', linewidth=2, label='Median')
    ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('OPTIMISTIC (No Costs)', fontsize=14, fontweight='bold', color='darkgreen')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(100, np.percentile(opt_curves, 99).max() * 1.5)
    
    # 2. Realistic fan chart
    ax2 = plt.subplot(2, 2, 2)
    real_pct_curves = np.percentile(real_curves, pcts, axis=0)
    
    ax2.fill_between(years, real_pct_curves[0], real_pct_curves[4], alpha=0.2, color='red', label='5-95th')
    ax2.fill_between(years, real_pct_curves[1], real_pct_curves[3], alpha=0.4, color='red', label='25-75th')
    ax2.plot(years, real_pct_curves[2], 'r-', linewidth=2, label='Median')
    ax2.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('REALISTIC (With Costs + Taxes)', fontsize=14, fontweight='bold', color='darkred')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(100, np.percentile(opt_curves, 99).max() * 1.5)
    
    # 3. Distribution comparison
    ax3 = plt.subplot(2, 2, 3)
    
    # Clip for visualization
    opt_clip = np.clip(opt_results, 0, np.percentile(opt_results, 95))
    real_clip = np.clip(real_results, 0, np.percentile(real_results, 95))
    
    ax3.hist(opt_clip, bins=40, alpha=0.5, color='green', label='Optimistic', density=True)
    ax3.hist(real_clip, bins=40, alpha=0.5, color='red', label='Realistic', density=True)
    ax3.axvline(x=INITIAL_CAPITAL, color='black', linestyle='--', label='Break Even')
    ax3.axvline(x=np.median(opt_results), color='green', linestyle='-', linewidth=2, alpha=0.7)
    ax3.axvline(x=np.median(real_results), color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Final Portfolio Value ($)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Outcomes After 20 Years', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    SIMULATION SUMMARY (20 Years, Starting with ${INITIAL_CAPITAL:,.0f})
    
    ===================================================================
                          OPTIMISTIC        REALISTIC
    ===================================================================
    Median Outcome        ${np.median(opt_results):>12,.0f}    ${np.median(real_results):>12,.0f}
    
    5th Percentile        ${np.percentile(opt_results, 5):>12,.0f}    ${np.percentile(real_results, 5):>12,.0f}
    (Worst 5%)
    
    95th Percentile       ${np.percentile(opt_results, 95):>12,.0f}    ${np.percentile(real_results, 95):>12,.0f}
    (Best 5%)
    
    Probability of Loss   {(opt_results < INITIAL_CAPITAL).mean()*100:>12.1f}%   {(real_results < INITIAL_CAPITAL).mean()*100:>12.1f}%
    
    P(> $100K)            {(opt_results >= 100000).mean()*100:>12.1f}%   {(real_results >= 100000).mean()*100:>12.1f}%
    
    P(> $1M)              {(opt_results >= 1000000).mean()*100:>12.1f}%   {(real_results >= 1000000).mean()*100:>12.1f}%
    ===================================================================
    
    Cost Assumptions (Realistic):
    - Trading fees: {EXCHANGE_FEE_ROUND_TRIP*100:.2f}% per trade
    - Slippage: {SLIPPAGE_ROUND_TRIP*100:.2f}% per trade  
    - Execution errors: {ANNUAL_ERROR_RATE*100:.0f}% of profits/year
    - Taxes: {TAX_RATE*100:.0f}% of annual profits
    
    WARNING: DOES NOT INCLUDE: Black swans, exchange failures,
        regulatory bans, psychological errors, or regime changes
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('20-Year Forward Simulation: Optimistic vs Realistic\n(Based on 2017-2025 BTC Data)',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, 'realistic_vs_optimistic_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    run_comparison()

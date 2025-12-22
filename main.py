"""
Main Runner - Executes all phases of the backtesting framework
Turtle-Inspired Donchian Strategy Analysis Pipeline
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_PARAMS, DEFAULT_BACKTEST, RESULTS_DIR, PLOTS_DIR,
    StrategyParams
)


def print_header():
    """Print program header"""
    print("\n" + "=" * 80)
    print("TURTLE-INSPIRED DONCHIAN STRATEGY BACKTESTER")
    print("Complete Analysis Pipeline")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def phase_1_setup():
    """Phase 1: Download data and run initial backtest"""
    print("\n" + "=" * 80)
    print("PHASE 1: BUILD LOCAL BACKTEST ENVIRONMENT")
    print("=" * 80)
    
    from data_fetcher import download_btc_data
    from strategy import TurtleDonchianStrategy
    
    # Download data
    print("\n[1.1] Downloading BTC historical data...")
    df = download_btc_data(
        timeframe="4h",
        start_date="2017-01-01",
        force_refresh=False
    )
    
    print(f"\nData loaded: {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['low'].min():,.0f} to ${df['high'].max():,.0f}")
    
    # Run initial backtest
    print("\n[1.2] Running initial backtest with default parameters...")
    print(f"\nDefault Parameters:")
    print(f"  Entry Length: {DEFAULT_PARAMS.entry_len}")
    print(f"  Exit Length: {DEFAULT_PARAMS.exit_len}")
    print(f"  ATR Length: {DEFAULT_PARAMS.atr_len}")
    print(f"  Trail Multiplier: {DEFAULT_PARAMS.trail_mult}")
    print(f"  Risk %: {DEFAULT_PARAMS.risk_percent}")
    print(f"  Max Units: {DEFAULT_PARAMS.max_units}")
    print(f"  Commission: {DEFAULT_PARAMS.commission_pct}%")
    print(f"  Slippage: {DEFAULT_PARAMS.slippage_pct}%")
    
    strategy = TurtleDonchianStrategy(DEFAULT_PARAMS)
    results = strategy.run_backtest(
        df,
        initial_capital=DEFAULT_BACKTEST.initial_capital,
        verbose=False
    )
    
    # Print results
    trade_stats = strategy.get_trade_stats()
    equity_stats = strategy.get_equity_stats(DEFAULT_BACKTEST.initial_capital)
    
    print("\n" + "-" * 40)
    print("INITIAL BACKTEST RESULTS")
    print("-" * 40)
    
    print(f"\nEquity Statistics:")
    print(f"  Initial Capital:  ${DEFAULT_BACKTEST.initial_capital:,.0f}")
    print(f"  Final Equity:     ${equity_stats.get('final_equity', 0):,.0f}")
    print(f"  Total Return:     {equity_stats.get('total_return_pct', 0):+.1f}%")
    print(f"  CAGR:             {equity_stats.get('cagr_pct', 0):.1f}%")
    print(f"  Max Drawdown:     {equity_stats.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Sharpe Ratio:     {equity_stats.get('sharpe_ratio', 0):.2f}")
    print(f"  Calmar Ratio:     {equity_stats.get('calmar_ratio', 0):.2f}")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:     {trade_stats.get('total_trades', 0)}")
    print(f"  Win Rate:         {trade_stats.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:    {trade_stats.get('profit_factor', 0):.2f}")
    print(f"  Avg Win:          ${trade_stats.get('avg_win', 0):,.0f}")
    print(f"  Avg Loss:         ${trade_stats.get('avg_loss', 0):,.0f}")
    
    # Save results
    results.to_csv(os.path.join(RESULTS_DIR, "backtest_results.csv"))
    print(f"\nResults saved to {RESULTS_DIR}/backtest_results.csv")
    
    return df, strategy, results


def phase_2_regime_analysis(df):
    """Phase 2: Regime Decomposition"""
    from regime_analysis import RegimeAnalyzer
    
    analyzer = RegimeAnalyzer(DEFAULT_PARAMS)
    analyzer.run_full_analysis(df)
    
    print(analyzer.get_summary_report())
    
    # Save results
    regime_df = analyzer.to_dataframe()
    regime_df.to_csv(os.path.join(RESULTS_DIR, "regime_analysis.csv"), index=False)
    
    return analyzer, regime_df


def phase_3_robustness_test(df):
    """Phase 3: Robustness Testing"""
    from robustness_test import RobustnessTester
    
    tester = RobustnessTester()
    results_df = tester.run_full_test(df)
    
    print(tester.get_summary_report())
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"), index=False)
    
    # Generate heatmaps
    try:
        tester.plot_heatmaps(os.path.join(PLOTS_DIR, "robustness_heatmaps.png"))
    except Exception as e:
        print(f"Could not generate heatmaps: {e}")
    
    return tester, results_df


def phase_4_survivability(df):
    """Phase 4: Capital Survivability Analysis"""
    from survivability import SurvivabilityAnalyzer
    
    analyzer = SurvivabilityAnalyzer(DEFAULT_PARAMS)
    metrics = analyzer.analyze(df)
    
    print(analyzer.get_summary_report())
    print(analyzer.get_psychological_assessment())
    
    # Save data
    if analyzer.drawdown_curve is not None:
        dd_df = pd.DataFrame({
            'equity': analyzer.equity_curve,
            'drawdown_pct': analyzer.drawdown_curve * 100
        })
        dd_df.to_csv(os.path.join(RESULTS_DIR, "survivability_data.csv"))
    
    return analyzer, metrics


def phase_5_monte_carlo(df):
    """Phase 5: Monte Carlo Simulation"""
    from monte_carlo import MonteCarloSimulator
    
    simulator = MonteCarloSimulator()
    results = simulator.run_simulation(df, n_simulations=1000)
    
    print(simulator.get_summary_report())
    
    # Generate plots
    try:
        simulator.plot_distribution(os.path.join(PLOTS_DIR, "monte_carlo_distribution.png"))
    except Exception as e:
        print(f"Could not generate Monte Carlo plots: {e}")
    
    # Save results
    import pandas as pd
    pd.DataFrame({
        'final_equity': results.equity_distribution
    }).to_csv(os.path.join(RESULTS_DIR, "monte_carlo_equities.csv"), index=False)
    
    return simulator, results


def phase_6_7_forward_test():
    """Phase 6-7: Forward Testing Framework"""
    from forward_test import ForwardTestTracker, generate_deployment_checklist
    
    print("\n" + "=" * 80)
    print("PHASE 6-7: FORWARD TESTING & DEPLOYMENT")
    print("=" * 80)
    
    print("\nForward testing framework is ready.")
    print("Run 'python forward_test.py' to start paper trading.")
    
    print(generate_deployment_checklist())
    
    return True


def generate_final_report(
    strategy,
    regime_analyzer,
    robustness_tester,
    survivability_analyzer,
    monte_carlo_simulator
):
    """Generate comprehensive final report"""
    
    report = []
    report.append("\n" + "=" * 80)
    report.append("FINAL COMPREHENSIVE REPORT")
    report.append("Turtle-Inspired Donchian Strategy Analysis")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Strategy summary
    equity_stats = strategy.get_equity_stats(DEFAULT_BACKTEST.initial_capital)
    trade_stats = strategy.get_trade_stats()
    
    report.append("\n" + "-" * 40)
    report.append("STRATEGY PERFORMANCE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Return: {equity_stats.get('total_return_pct', 0):+.1f}%")
    report.append(f"CAGR: {equity_stats.get('cagr_pct', 0):.1f}%")
    report.append(f"Sharpe Ratio: {equity_stats.get('sharpe_ratio', 0):.2f}")
    report.append(f"Max Drawdown: {equity_stats.get('max_drawdown_pct', 0):.1f}%")
    report.append(f"Win Rate: {trade_stats.get('win_rate', 0):.1f}%")
    
    # Verdict
    report.append("\n" + "-" * 40)
    report.append("OVERALL ASSESSMENT")
    report.append("-" * 40)
    
    # Check all phases
    mc_results = monte_carlo_simulator.results if monte_carlo_simulator else None
    surv_metrics = survivability_analyzer.metrics if survivability_analyzer else None
    
    issues = []
    strengths = []
    
    # Check Monte Carlo results
    if mc_results:
        if mc_results.prob_ruin > 10:
            issues.append(f"High probability of ruin ({mc_results.prob_ruin:.1f}%)")
        else:
            strengths.append(f"Low probability of ruin ({mc_results.prob_ruin:.1f}%)")
        
        if mc_results.median_cagr > 10:
            strengths.append(f"Strong median CAGR ({mc_results.median_cagr:.1f}%)")
        elif mc_results.median_cagr < 0:
            issues.append(f"Negative median CAGR ({mc_results.median_cagr:.1f}%)")
    
    # Check survivability
    if surv_metrics:
        if not surv_metrics.survives_stress:
            issues.append("Fails stress test")
        else:
            strengths.append("Passes stress test")
    
    report.append("\nStrengths:")
    for s in strengths:
        report.append(f"  ✓ {s}")
    
    if issues:
        report.append("\nConcerns:")
        for i in issues:
            report.append(f"  ⚠ {i}")
    
    # Final recommendation
    report.append("\n" + "-" * 40)
    report.append("RECOMMENDATION")
    report.append("-" * 40)
    
    if len(issues) == 0:
        report.append("✓ STRATEGY IS DEPLOYABLE")
        report.append("")
        report.append("Proceed to paper trading for 3-6 months before live deployment.")
    elif len(issues) <= 2:
        report.append("⚠ STRATEGY IS MARGINAL")
        report.append("")
        report.append("Consider parameter adjustments before deployment:")
        report.append("  - Reduce risk_percent to lower drawdowns")
        report.append("  - Reduce max_units to limit pyramiding risk")
    else:
        report.append("✗ STRATEGY NEEDS WORK")
        report.append("")
        report.append("Significant issues must be addressed before deployment.")
    
    final_report = "\n".join(report)
    
    # Save report
    with open(os.path.join(RESULTS_DIR, "final_report.txt"), 'w') as f:
        f.write(final_report)
    
    return final_report


def main():
    """Main execution function"""
    import pandas as pd
    
    print_header()
    
    start_time = time.time()
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Phase 1: Setup and initial backtest
    df, strategy, results = phase_1_setup()
    
    # Phase 2: Regime analysis
    regime_analyzer, regime_df = phase_2_regime_analysis(df)
    
    # Phase 3: Robustness testing
    robustness_tester, robustness_df = phase_3_robustness_test(df)
    
    # Phase 4: Survivability analysis
    survivability_analyzer, surv_metrics = phase_4_survivability(df)
    
    # Phase 5: Monte Carlo simulation
    monte_carlo_sim, mc_results = phase_5_monte_carlo(df)
    
    # Phase 6-7: Forward testing framework
    phase_6_7_forward_test()
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    try:
        from visualization import generate_all_plots
        generate_all_plots(results, strategy, regime_df)
    except Exception as e:
        print(f"Could not generate some plots: {e}")
    
    # Final report
    final_report = generate_final_report(
        strategy,
        regime_analyzer,
        robustness_tester,
        survivability_analyzer,
        monte_carlo_sim
    )
    print(final_report)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("\nNext steps:")
    print("1. Review the final_report.txt")
    print("2. Examine the plots in the plots/ directory")
    print("3. If satisfied, run forward_test.py to begin paper trading")
    print("=" * 80)


if __name__ == "__main__":
    main()

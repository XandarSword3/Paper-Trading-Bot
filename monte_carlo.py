"""
Phase 5: Monte Carlo & Path Risk Analysis
Randomizes trade outcomes to assess probability of ruin
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import warnings

from config import StrategyParams, DEFAULT_PARAMS, MonteCarloConfig, DEFAULT_MONTE_CARLO
from strategy import TurtleDonchianStrategy

warnings.filterwarnings('ignore')


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results"""
    n_simulations: int
    median_final_equity: float
    mean_final_equity: float
    std_final_equity: float
    
    # Percentile outcomes
    p5_equity: float    # 5th percentile (bad)
    p25_equity: float   # 25th percentile
    p50_equity: float   # Median
    p75_equity: float   # 75th percentile
    p95_equity: float   # 95th percentile (good)
    
    # CAGR distribution
    median_cagr: float
    p5_cagr: float
    p95_cagr: float
    
    # Drawdown distribution
    median_max_dd: float
    p95_max_dd: float  # 95th percentile worst drawdown
    
    # Risk metrics
    prob_50pct_drawdown: float  # Probability of 50%+ drawdown
    prob_ruin: float            # Probability of losing >80%
    prob_profitable: float      # Probability of making money
    prob_double: float          # Probability of doubling capital
    
    # Distribution of final equities
    equity_distribution: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo simulation for path risk analysis.
    
    Randomizes:
    - Trade order (sequence matters for compounding)
    - Execution noise
    - Slippage variation
    
    Outputs:
    - Probability of ruin
    - Probability of 50% drawdown
    - Median vs tail outcomes
    """
    
    def __init__(
        self,
        params: StrategyParams = None,
        mc_config: MonteCarloConfig = None
    ):
        self.params = params or DEFAULT_PARAMS
        self.mc_config = mc_config or DEFAULT_MONTE_CARLO
        self.results: MonteCarloResults = None
        self.all_equity_curves: List[np.ndarray] = []
    
    def extract_trade_returns(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0
    ) -> List[float]:
        """Extract individual trade returns from backtest"""
        
        strategy = TurtleDonchianStrategy(self.params)
        strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
        
        trades = strategy.trades
        
        # Calculate return per trade (as fraction of equity at entry)
        trade_returns = []
        
        for trade in trades:
            if trade.pnl is not None and trade.entry_price > 0:
                # Approximate equity at entry
                trade_value = trade.quantity * trade.entry_price
                if trade_value > 0:
                    pct_return = trade.pnl / trade_value
                    trade_returns.append(pct_return)
        
        return trade_returns
    
    def simulate_path(
        self,
        trade_returns: List[float],
        initial_capital: float,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        Simulate a single equity path with randomized trade order.
        
        Returns:
            equity_curve: Array of equity values
            final_equity: Final equity value
            max_drawdown: Maximum drawdown as fraction
        """
        
        # Shuffle trade order
        shuffled = np.random.permutation(trade_returns)
        
        # Add execution noise if enabled
        if add_noise:
            noise = np.random.normal(
                0,
                self.mc_config.execution_noise_std,
                len(shuffled)
            )
            slippage_var = np.random.normal(
                0,
                self.mc_config.slippage_std,
                len(shuffled)
            )
            shuffled = shuffled + noise - np.abs(slippage_var)
        
        # Build equity curve
        equity = initial_capital
        equity_curve = [equity]
        peak = equity
        max_dd = 0.0
        
        for ret in shuffled:
            # Apply return (compounding)
            equity = equity * (1 + ret)
            equity = max(0, equity)  # Can't go negative
            equity_curve.append(equity)
            
            # Track drawdown
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return np.array(equity_curve), equity, max_dd
    
    def run_simulation(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        n_simulations: int = None
    ) -> MonteCarloResults:
        """Run full Monte Carlo simulation"""
        
        n_sims = n_simulations or self.mc_config.n_simulations
        
        print("\n" + "=" * 80)
        print("PHASE 5: MONTE CARLO SIMULATION")
        print("=" * 80)
        print(f"\nRunning {n_sims} simulations...")
        
        # Set random seed for reproducibility
        np.random.seed(self.mc_config.random_seed)
        
        # Extract trade returns from actual backtest
        print("Extracting trade returns from historical backtest...")
        trade_returns = self.extract_trade_returns(df, initial_capital)
        
        if len(trade_returns) < 10:
            print("WARNING: Too few trades for reliable Monte Carlo simulation")
        
        print(f"Found {len(trade_returns)} trades to simulate")
        print(f"Average trade return: {np.mean(trade_returns)*100:.2f}%")
        print(f"Trade return std: {np.std(trade_returns)*100:.2f}%")
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        self.all_equity_curves = []
        
        for _ in tqdm(range(n_sims), desc="Simulating"):
            eq_curve, final_eq, max_dd = self.simulate_path(
                trade_returns, initial_capital, add_noise=True
            )
            final_equities.append(final_eq)
            max_drawdowns.append(max_dd)
            self.all_equity_curves.append(eq_curve)
        
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate CAGRs (assuming ~8 years of history)
        years = len(df) / (365 * 6)  # 4H bars
        cagrs = (final_equities / initial_capital) ** (1 / years) - 1
        
        # Calculate probabilities
        prob_ruin = np.mean(final_equities < initial_capital * 0.2)
        prob_50dd = np.mean(max_drawdowns >= 0.5)
        prob_profit = np.mean(final_equities > initial_capital)
        prob_double = np.mean(final_equities >= initial_capital * 2)
        
        self.results = MonteCarloResults(
            n_simulations=n_sims,
            median_final_equity=np.median(final_equities),
            mean_final_equity=np.mean(final_equities),
            std_final_equity=np.std(final_equities),
            
            p5_equity=np.percentile(final_equities, 5),
            p25_equity=np.percentile(final_equities, 25),
            p50_equity=np.percentile(final_equities, 50),
            p75_equity=np.percentile(final_equities, 75),
            p95_equity=np.percentile(final_equities, 95),
            
            median_cagr=np.median(cagrs) * 100,
            p5_cagr=np.percentile(cagrs, 5) * 100,
            p95_cagr=np.percentile(cagrs, 95) * 100,
            
            median_max_dd=np.median(max_drawdowns) * 100,
            p95_max_dd=np.percentile(max_drawdowns, 95) * 100,
            
            prob_50pct_drawdown=prob_50dd * 100,
            prob_ruin=prob_ruin * 100,
            prob_profitable=prob_profit * 100,
            prob_double=prob_double * 100,
            
            equity_distribution=final_equities
        )
        
        return self.results
    
    def get_summary_report(self) -> str:
        """Generate Monte Carlo analysis report"""
        
        if self.results is None:
            return "No Monte Carlo results available."
        
        r = self.results
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("MONTE CARLO SIMULATION REPORT")
        report.append("=" * 80)
        
        report.append(f"\nSimulations run: {r.n_simulations}")
        
        report.append("\n" + "-" * 40)
        report.append("FINAL EQUITY DISTRIBUTION")
        report.append("-" * 40)
        
        report.append(f"5th Percentile (Bad):   ${r.p5_equity:,.0f}")
        report.append(f"25th Percentile:        ${r.p25_equity:,.0f}")
        report.append(f"Median (50th):          ${r.p50_equity:,.0f}")
        report.append(f"75th Percentile:        ${r.p75_equity:,.0f}")
        report.append(f"95th Percentile (Good): ${r.p95_equity:,.0f}")
        
        report.append("\n" + "-" * 40)
        report.append("CAGR DISTRIBUTION")
        report.append("-" * 40)
        
        report.append(f"5th Percentile CAGR:  {r.p5_cagr:+.1f}%")
        report.append(f"Median CAGR:          {r.median_cagr:+.1f}%")
        report.append(f"95th Percentile CAGR: {r.p95_cagr:+.1f}%")
        
        report.append("\n" + "-" * 40)
        report.append("DRAWDOWN DISTRIBUTION")
        report.append("-" * 40)
        
        report.append(f"Median Max Drawdown:  {r.median_max_dd:.1f}%")
        report.append(f"95th Pctl Max DD:     {r.p95_max_dd:.1f}%")
        
        report.append("\n" + "-" * 40)
        report.append("PROBABILITY METRICS")
        report.append("-" * 40)
        
        report.append(f"Probability of 50%+ Drawdown: {r.prob_50pct_drawdown:.1f}%")
        report.append(f"Probability of Ruin (>80%):   {r.prob_ruin:.1f}%")
        report.append(f"Probability of Profit:        {r.prob_profitable:.1f}%")
        report.append(f"Probability of Doubling:      {r.prob_double:.1f}%")
        
        report.append("\n" + "-" * 40)
        report.append("DEPLOYMENT DECISION")
        report.append("-" * 40)
        
        # Decision logic
        deployable = (
            r.median_cagr > 0 and
            r.prob_ruin < 10 and
            r.prob_profitable > 60 and
            r.p95_max_dd < 80
        )
        
        if deployable:
            report.append("✓ STRATEGY IS DEPLOYABLE")
            report.append("")
            report.append("Conditions met:")
            report.append(f"  • Median outcome is positive ({r.median_cagr:+.1f}% CAGR)")
            report.append(f"  • Probability of ruin is low ({r.prob_ruin:.1f}%)")
            report.append(f"  • Left tail does not destroy you ({r.p95_max_dd:.1f}% worst DD)")
            report.append("")
            report.append("However, prepare for:")
            report.append(f"  • 5% chance of only ${r.p5_equity:,.0f} final equity")
            report.append(f"  • {r.prob_50pct_drawdown:.0f}% chance of 50%+ drawdown")
        else:
            report.append("✗ STRATEGY HAS SIGNIFICANT TAIL RISK")
            report.append("")
            issues = []
            if r.median_cagr <= 0:
                issues.append(f"  • Median outcome is negative ({r.median_cagr:+.1f}%)")
            if r.prob_ruin >= 10:
                issues.append(f"  • High probability of ruin ({r.prob_ruin:.1f}%)")
            if r.prob_profitable <= 60:
                issues.append(f"  • Low probability of profit ({r.prob_profitable:.1f}%)")
            if r.p95_max_dd >= 80:
                issues.append(f"  • Extreme tail drawdowns ({r.p95_max_dd:.1f}%)")
            
            report.append("Issues:")
            report.extend(issues)
            report.append("")
            report.append("Recommendations:")
            report.append("  1. Reduce position sizing (risk_percent)")
            report.append("  2. Reduce pyramiding (max_units)")
            report.append("  3. Consider the strategy non-deployable")
        
        return "\n".join(report)
    
    def plot_distribution(self, save_path: str = None):
        """Plot Monte Carlo results distribution"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib required for plotting")
            return
        
        if self.results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=14)
        
        # Equity distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(self.results.equity_distribution, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(self.results.p5_equity, color='red', linestyle='--', label='5th pctl')
        ax1.axvline(self.results.p50_equity, color='green', linestyle='-', label='Median')
        ax1.axvline(self.results.p95_equity, color='blue', linestyle='--', label='95th pctl')
        ax1.set_xlabel('Final Equity ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Final Equity Distribution')
        ax1.legend()
        
        # Sample equity paths
        ax2 = axes[0, 1]
        n_paths = min(100, len(self.all_equity_curves))
        for i in range(n_paths):
            alpha = 0.1 if i < n_paths - 5 else 0.5
            ax2.plot(self.all_equity_curves[i], alpha=alpha, linewidth=0.5)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Equity ($)')
        ax2.set_title(f'Sample Equity Paths (n={n_paths})')
        
        # CAGR distribution
        ax3 = axes[1, 0]
        cagrs = (self.results.equity_distribution / 100_000) ** (1/8) - 1  # Approximate 8 years
        ax3.hist(cagrs * 100, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='-', linewidth=2)
        ax3.set_xlabel('CAGR (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('CAGR Distribution')
        
        # Cumulative probability
        ax4 = axes[1, 1]
        sorted_eq = np.sort(self.results.equity_distribution)
        cumprob = np.arange(1, len(sorted_eq) + 1) / len(sorted_eq)
        ax4.plot(sorted_eq, cumprob)
        ax4.axhline(0.5, color='green', linestyle='--', label='Median')
        ax4.axvline(100_000, color='red', linestyle='-', label='Initial Capital')
        ax4.set_xlabel('Final Equity ($)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Monte Carlo plots saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    import os
    from data_fetcher import download_btc_data
    from config import RESULTS_DIR, PLOTS_DIR
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")
    
    simulator = MonteCarloSimulator()
    results = simulator.run_simulation(df, n_simulations=1000)
    
    print(simulator.get_summary_report())
    
    # Save plots
    simulator.plot_distribution(os.path.join(PLOTS_DIR, "monte_carlo_distribution.png"))
    
    # Save equity distribution
    pd.DataFrame({
        'final_equity': results.equity_distribution
    }).to_csv(os.path.join(RESULTS_DIR, "monte_carlo_equities.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/monte_carlo_equities.csv")

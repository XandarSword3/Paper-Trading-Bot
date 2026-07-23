"""
Phase 5: Monte Carlo & Path Risk Analysis
Randomizes trade outcomes to assess probability of ruin

SEQUENCE-RISK ONLY — READ THIS BEFORE TRUSTING WHAT THIS FILE TELLS YOU.
MonteCarloSimulator below reshuffles the order of the SAME in-sample trades
a backtest already produced. That's a real and useful question — "given
these trades happen, in what order, how bad can the ride get?" — but it is
NOT a test of whether the edge is real: every shuffled path is built from
trades that were selected on the same data their own performance is judged
on. A curve-fit strategy with zero real edge can still produce a Monte Carlo
report that looks fine, because reshuffling can't detect that the trades
themselves were cherry-picked.

For the "is the edge real" question, this file also provides
OOSBlockBootstrap: block-bootstrap resampling of the walk-forward
out-of-sample RETURNS from Phase 2 (walk_forward.py), not in-sample trades.
Those returns were never touched by parameter selection, so resampling them
tests forward reliability instead of just sequence risk. See
edge_validation_test.py for the Phase 3 driver script that runs both this
and the deflated Sharpe ratio (deflated_sharpe.py) together.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import warnings

from config import (
    StrategyParams, DEFAULT_PARAMS, MonteCarloConfig, DEFAULT_MONTE_CARLO,
    BlockBootstrapConfig, DEFAULT_BLOCK_BOOTSTRAP,
)
from strategy import TurtleDonchianStrategy
from walk_forward import _infer_bars_per_year

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


def format_monte_carlo_report(results: "MonteCarloResults", title: str, caveat: str = "") -> str:
    """Shared pretty-printer for MonteCarloResults, used by both the
    sequence-risk simulator and the OOS block-bootstrap below so the two
    report formats stay directly comparable side by side."""

    r = results
    report = []
    report.append("\n" + "=" * 80)
    report.append(title)
    report.append("=" * 80)
    if caveat:
        report.append(f"\n{caveat}")

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

    deployable = (
        r.median_cagr > 0 and
        r.prob_ruin < 10 and
        r.prob_profitable > 60 and
        r.p95_max_dd < 80
    )

    if deployable:
        report.append("[OK] conditions favorable in this test")
        report.append("")
        report.append("Conditions met:")
        report.append(f"  - Median outcome is positive ({r.median_cagr:+.1f}% CAGR)")
        report.append(f"  - Probability of ruin is low ({r.prob_ruin:.1f}%)")
        report.append(f"  - Left tail does not destroy you ({r.p95_max_dd:.1f}% worst DD)")
        report.append("")
        report.append("However, prepare for:")
        report.append(f"  - 5% chance of only ${r.p5_equity:,.0f} final equity")
        report.append(f"  - {r.prob_50pct_drawdown:.0f}% chance of 50%+ drawdown")
    else:
        report.append("[FAIL] significant tail risk in this test")
        report.append("")
        issues = []
        if r.median_cagr <= 0:
            issues.append(f"  - Median outcome is negative ({r.median_cagr:+.1f}%)")
        if r.prob_ruin >= 10:
            issues.append(f"  - High probability of ruin ({r.prob_ruin:.1f}%)")
        if r.prob_profitable <= 60:
            issues.append(f"  - Low probability of profit ({r.prob_profitable:.1f}%)")
        if r.p95_max_dd >= 80:
            issues.append(f"  - Extreme tail drawdowns ({r.p95_max_dd:.1f}%)")

        report.append("Issues:")
        report.extend(issues)
        report.append("")
        report.append("Recommendations:")
        report.append("  1. Reduce position sizing (risk_percent)")
        report.append("  2. Reduce pyramiding (max_units)")
        report.append("  3. Consider the strategy non-deployable")

    return "\n".join(report)


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

        return format_monte_carlo_report(
            self.results,
            title="MONTE CARLO SIMULATION REPORT — SEQUENCE-RISK ONLY",
            caveat=(
                "This reshuffles the ORDER of the same in-sample trades a backtest already "
                "produced. It tests sequence/ruin risk given those trades happen — it says "
                "NOTHING about whether the underlying edge is real. For that, see "
                "OOSBlockBootstrap in this file / edge_validation_test.py."
            ),
        )
    
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


class OOSBlockBootstrap:
    """
    Block-bootstrap resampling of walk-forward OUT-OF-SAMPLE returns —
    Phase 3 of VALIDATION_REMEDIATION_PLAN.md.

    Unlike MonteCarloSimulator (which reshuffles in-sample trades and
    therefore can't detect curve-fitting), this resamples the per-bar return
    series from walk_forward.py's stitched OOS equity curve — bars that were
    never inside any fold's optimization window. Resampling in contiguous
    BLOCKS (rather than shuffling bars independently) preserves the
    short-run autocorrelation and volatility clustering real returns have;
    an iid shuffle would understate how bad a real bad stretch can look.

    This answers "how much would the OOS result plausibly vary in expectation
    if we lived through a different sample of similar market conditions" —
    a genuine forward-reliability question, not just a sequence-of-known-
    trades question.
    """

    def __init__(self, config: BlockBootstrapConfig = None):
        self.config = config or DEFAULT_BLOCK_BOOTSTRAP
        self.results: MonteCarloResults = None

    @staticmethod
    def load_oos_returns(equity_csv_path: str) -> pd.Series:
        """Load the per-bar return series from a walk_forward_test.py
        walk_forward_oos_equity.csv artifact."""
        equity_df = pd.read_csv(equity_csv_path, index_col="timestamp", parse_dates=True)
        returns = equity_df["equity"].pct_change().dropna()
        return returns

    def _simulate_path(self, returns: np.ndarray, n_periods: int) -> np.ndarray:
        """One moving-block-bootstrap resample: repeatedly splice in a
        random contiguous block from the real OOS return series until the
        resampled path reaches n_periods, then compound it into an equity
        multiplier path (starting at 1.0)."""
        block_size = max(1, min(self.config.block_size_bars, len(returns)))
        n_blocks_needed = int(np.ceil(n_periods / block_size))
        max_start = len(returns) - block_size

        pieces = []
        for _ in range(n_blocks_needed):
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            pieces.append(returns[start:start + block_size])
        resampled = np.concatenate(pieces)[:n_periods]
        return np.cumprod(1 + resampled)

    def run(
        self,
        oos_returns: pd.Series,
        initial_capital: float = 100_000.0,
        n_simulations: int = None,
    ) -> MonteCarloResults:
        n_sims = n_simulations or self.config.n_simulations
        returns = oos_returns.dropna().to_numpy()
        n_periods = len(returns)

        if n_periods < self.config.block_size_bars * 3:
            print(f"WARNING: only {n_periods} OOS bars available for a block size of "
                  f"{self.config.block_size_bars} — bootstrap distribution will be narrow/unreliable. "
                  f"Consider a longer walk-forward run or a smaller block_size_bars.")

        print("\n" + "=" * 80)
        print("PHASE 3: OOS BLOCK-BOOTSTRAP (out-of-sample returns, not in-sample trades)")
        print("=" * 80)
        print(f"\nOOS bars available: {n_periods}  |  block size: {self.config.block_size_bars} bars  "
              f"|  simulations: {n_sims}")

        np.random.seed(self.config.random_seed)

        bars_per_year = _infer_bars_per_year(oos_returns.dropna().index)

        final_equities = []
        max_drawdowns = []

        for _ in tqdm(range(n_sims), desc="Bootstrapping OOS returns"):
            path = self._simulate_path(returns, n_periods)
            equity_curve = initial_capital * path
            final_equities.append(equity_curve[-1])
            running_max = np.maximum.accumulate(equity_curve)
            dd = (running_max - equity_curve) / running_max
            max_drawdowns.append(dd.max())

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        years = n_periods / bars_per_year if bars_per_year > 0 else 1.0
        cagrs = (final_equities / initial_capital) ** (1 / years) - 1 if years > 0 else np.zeros_like(final_equities)

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

            prob_50pct_drawdown=np.mean(max_drawdowns >= 0.5) * 100,
            prob_ruin=np.mean(final_equities < initial_capital * 0.2) * 100,
            prob_profitable=np.mean(final_equities > initial_capital) * 100,
            prob_double=np.mean(final_equities >= initial_capital * 2) * 100,

            equity_distribution=final_equities,
        )
        return self.results

    def get_summary_report(self) -> str:
        if self.results is None:
            return "No OOS bootstrap results available."
        return format_monte_carlo_report(
            self.results,
            title="OOS BLOCK-BOOTSTRAP REPORT — forward reliability, not sequence risk",
            caveat=(
                "Built entirely from walk-forward OUT-OF-SAMPLE returns (walk_forward.py), "
                "resampled in contiguous blocks. None of these bars were seen during the "
                "parameter selection that produced them."
            ),
        )


if __name__ == "__main__":
    import os
    from data_fetcher import download_btc_data
    from data_splits import get_development, describe_split
    from config import RESULTS_DIR, PLOTS_DIR
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")

    # Phase 1 of the remediation plan: never touch the true holdout here.
    # NOTE: as-is, this still reshuffles in-sample trades — that's Phase 3's
    # "label it accurately as sequence-risk only" fix, not this one.
    print(f"\n{describe_split()}")
    df = get_development(df)
    print(f"Restricted to development window (in-sample + validation): "
          f"{len(df)} candles, {df.index[0]} -> {df.index[-1]}")
    
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

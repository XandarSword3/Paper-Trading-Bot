"""
Phase 3: Robustness Testing
Tests parameter ranges to find stable plateaus of profitability
NO OPTIMIZATION - only testing for robustness
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
from tqdm import tqdm

from config import (
    StrategyParams, DEFAULT_PARAMS, DEFAULT_ROBUSTNESS,
    RobustnessRanges, RESULTS_DIR
)
from strategy import TurtleDonchianStrategy

warnings.filterwarnings('ignore')


@dataclass
class ParameterResult:
    """Result of a single parameter combination test"""
    entry_len: int
    exit_len: int
    trail_mult: float
    risk_percent: float
    pyramid_spacing: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    win_rate: float
    profit_factor: float
    cagr_pct: float
    calmar_ratio: float


class RobustnessTester:
    """
    Tests parameter ranges for strategy robustness.
    
    NOT for optimization - looking for:
    - Broad plateaus of profitability
    - Smooth degradation, not cliffs
    - Parameter insensitivity
    """
    
    def __init__(self, ranges: RobustnessRanges = None):
        self.ranges = ranges or DEFAULT_ROBUSTNESS
        self.results: List[ParameterResult] = []
        self.results_df: pd.DataFrame = None
    
    def generate_parameter_grid(self) -> List[dict]:
        """Generate all parameter combinations to test"""
        
        entry_lens = list(range(
            self.ranges.entry_len[0],
            self.ranges.entry_len[1] + 1,
            self.ranges.entry_len[2]
        ))
        
        exit_lens = list(range(
            self.ranges.exit_len[0],
            self.ranges.exit_len[1] + 1,
            self.ranges.exit_len[2]
        ))
        
        trail_mults = np.arange(
            self.ranges.trail_mult[0],
            self.ranges.trail_mult[1] + 0.01,
            self.ranges.trail_mult[2]
        ).tolist()
        
        risk_pcts = np.arange(
            self.ranges.risk_percent[0],
            self.ranges.risk_percent[1] + 0.01,
            self.ranges.risk_percent[2]
        ).tolist()
        
        pyramid_spacings = np.arange(
            self.ranges.pyramid_spacing_n[0],
            self.ranges.pyramid_spacing_n[1] + 0.01,
            self.ranges.pyramid_spacing_n[2]
        ).tolist()
        
        # Generate all combinations
        combinations = list(product(
            entry_lens, exit_lens, trail_mults, risk_pcts, pyramid_spacings
        ))
        
        # Filter invalid combinations (exit_len should be < entry_len)
        valid_combinations = [
            {
                'entry_len': c[0],
                'exit_len': c[1],
                'trail_mult': c[2],
                'risk_percent': c[3],
                'pyramid_spacing_n': c[4]
            }
            for c in combinations
            if c[1] < c[0]  # exit_len < entry_len
        ]
        
        return valid_combinations
    
    def test_parameters(
        self,
        df: pd.DataFrame,
        params_dict: dict,
        initial_capital: float = 100_000.0
    ) -> ParameterResult:
        """Test a single parameter combination"""
        
        # Create params with overrides
        params = StrategyParams(
            entry_len=params_dict['entry_len'],
            exit_len=params_dict['exit_len'],
            trail_mult=params_dict['trail_mult'],
            risk_percent=params_dict['risk_percent'],
            pyramid_spacing_n=params_dict['pyramid_spacing_n'],
            # Keep other defaults
            atr_len=DEFAULT_PARAMS.atr_len,
            size_stop_mult=DEFAULT_PARAMS.size_stop_mult,
            max_units=DEFAULT_PARAMS.max_units,
            long_only=DEFAULT_PARAMS.long_only,
            use_regime_filter=DEFAULT_PARAMS.use_regime_filter,
            lot_step=DEFAULT_PARAMS.lot_step,
            commission_pct=DEFAULT_PARAMS.commission_pct,
            slippage_pct=DEFAULT_PARAMS.slippage_pct,
        )
        
        strategy = TurtleDonchianStrategy(params)
        results = strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
        
        trade_stats = strategy.get_trade_stats()
        equity_stats = strategy.get_equity_stats(initial_capital)
        
        return ParameterResult(
            entry_len=params_dict['entry_len'],
            exit_len=params_dict['exit_len'],
            trail_mult=params_dict['trail_mult'],
            risk_percent=params_dict['risk_percent'],
            pyramid_spacing=params_dict['pyramid_spacing_n'],
            total_return_pct=equity_stats.get('total_return_pct', 0),
            max_drawdown_pct=abs(equity_stats.get('max_drawdown_pct', 0)),
            sharpe_ratio=equity_stats.get('sharpe_ratio', 0),
            num_trades=trade_stats.get('total_trades', 0),
            win_rate=trade_stats.get('win_rate', 0),
            profit_factor=trade_stats.get('profit_factor', 0),
            cagr_pct=equity_stats.get('cagr_pct', 0),
            calmar_ratio=equity_stats.get('calmar_ratio', 0)
        )
    
    def run_full_test(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0
    ) -> pd.DataFrame:
        """Run robustness test across all parameter combinations"""
        
        print("\n" + "=" * 80)
        print("PHASE 3: ROBUSTNESS TESTING")
        print("=" * 80)
        print("\nNOT optimization - testing for stability and robustness")
        
        param_grid = self.generate_parameter_grid()
        print(f"\nTesting {len(param_grid)} parameter combinations...")
        
        self.results = []
        
        for params in tqdm(param_grid, desc="Testing parameters"):
            try:
                result = self.test_parameters(df, params, initial_capital)
                self.results.append(result)
            except Exception as e:
                # Skip failed combinations
                continue
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame([
            {
                'entry_len': r.entry_len,
                'exit_len': r.exit_len,
                'trail_mult': r.trail_mult,
                'risk_pct': r.risk_percent,
                'pyramid_spacing': r.pyramid_spacing,
                'return_pct': r.total_return_pct,
                'max_dd_pct': r.max_drawdown_pct,
                'sharpe': r.sharpe_ratio,
                'trades': r.num_trades,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'cagr_pct': r.cagr_pct,
                'calmar': r.calmar_ratio
            }
            for r in self.results
        ])
        
        return self.results_df
    
    def analyze_robustness(self) -> Dict:
        """Analyze robustness of results"""
        
        if self.results_df is None or len(self.results_df) == 0:
            return {}
        
        df = self.results_df
        
        analysis = {}
        
        # Profitable combinations
        profitable = df[df['return_pct'] > 0]
        analysis['pct_profitable'] = len(profitable) / len(df) * 100
        
        # Strongly profitable (>100% return)
        strong = df[df['return_pct'] > 100]
        analysis['pct_strong'] = len(strong) / len(df) * 100
        
        # Parameter sensitivity analysis
        for param in ['entry_len', 'exit_len', 'trail_mult', 'risk_pct', 'pyramid_spacing']:
            param_groups = df.groupby(param)['return_pct'].agg(['mean', 'std', 'min', 'max'])
            
            # Check for cliffs (large drops in performance)
            returns_by_param = df.groupby(param)['return_pct'].mean()
            diffs = returns_by_param.diff().abs()
            max_cliff = diffs.max() if len(diffs) > 0 else 0
            
            analysis[f'{param}_mean_return'] = param_groups['mean'].mean()
            analysis[f'{param}_std_return'] = param_groups['mean'].std()
            analysis[f'{param}_max_cliff'] = max_cliff
            
            # Is it a plateau? (low variance in means)
            is_plateau = param_groups['mean'].std() < param_groups['mean'].mean() * 0.5
            analysis[f'{param}_is_plateau'] = is_plateau
        
        # Overall robustness score
        num_plateaus = sum(1 for k, v in analysis.items() if k.endswith('_is_plateau') and v)
        analysis['robustness_score'] = (
            analysis['pct_profitable'] * 0.3 +
            analysis['pct_strong'] * 0.3 +
            num_plateaus * 10
        )
        
        return analysis
    
    def get_summary_report(self) -> str:
        """Generate robustness test summary report"""
        
        if self.results_df is None or len(self.results_df) == 0:
            return "No robustness test results available."
        
        df = self.results_df
        analysis = self.analyze_robustness()
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("ROBUSTNESS TEST SUMMARY REPORT")
        report.append("=" * 80)
        
        report.append(f"\nTotal combinations tested: {len(df)}")
        report.append(f"Profitable combinations: {analysis['pct_profitable']:.1f}%")
        report.append(f"Strongly profitable (>100%): {analysis['pct_strong']:.1f}%")
        
        report.append("\n" + "-" * 80)
        report.append("PARAMETER SENSITIVITY ANALYSIS")
        report.append("-" * 80)
        
        params = ['entry_len', 'exit_len', 'trail_mult', 'risk_pct', 'pyramid_spacing']
        for param in params:
            is_plateau = analysis.get(f'{param}_is_plateau', False)
            mean_ret = analysis.get(f'{param}_mean_return', 0)
            max_cliff = analysis.get(f'{param}_max_cliff', 0)
            
            status = "✓ PLATEAU" if is_plateau else "⚠ SENSITIVE"
            report.append(f"\n{param}:")
            report.append(f"  Mean return: {mean_ret:.1f}%")
            report.append(f"  Max cliff: {max_cliff:.1f}%")
            report.append(f"  Status: {status}")
        
        report.append("\n" + "-" * 80)
        report.append("TOP 10 PARAMETER COMBINATIONS (by Sharpe Ratio)")
        report.append("-" * 80)
        
        top_10 = df.nlargest(10, 'sharpe')
        report.append("\n{:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8}".format(
            "Entry", "Exit", "Trail", "Risk%", "PyramN", "Return%", "MaxDD%", "Sharpe"
        ))
        
        for _, row in top_10.iterrows():
            report.append("{:>8d} {:>8d} {:>8.1f} {:>8.2f} {:>8.2f} {:>+10.1f} {:>8.1f} {:>8.2f}".format(
                int(row['entry_len']), int(row['exit_len']), row['trail_mult'],
                row['risk_pct'], row['pyramid_spacing'], row['return_pct'],
                row['max_dd_pct'], row['sharpe']
            ))
        
        report.append("\n" + "-" * 80)
        report.append("ROBUSTNESS VERDICT")
        report.append("-" * 80)
        
        score = analysis.get('robustness_score', 0)
        if score >= 50:
            report.append(f"✓ ROBUST (Score: {score:.1f})")
            report.append("  Strategy shows broad plateaus of profitability")
        elif score >= 30:
            report.append(f"⚠ MARGINAL (Score: {score:.1f})")
            report.append("  Strategy works but is sensitive to parameters")
        else:
            report.append(f"✗ FRAGILE (Score: {score:.1f})")
            report.append("  Strategy only works in narrow parameter bands - CAUTION")
        
        return "\n".join(report)
    
    def plot_heatmaps(self, save_path: str = None):
        """Generate parameter sensitivity heatmaps"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Matplotlib/Seaborn required for plotting")
            return
        
        if self.results_df is None:
            return
        
        df = self.results_df
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Parameter Sensitivity Heatmaps', fontsize=14)
        
        # Entry vs Exit length
        pivot1 = df.pivot_table(
            values='return_pct',
            index='entry_len',
            columns='exit_len',
            aggfunc='mean'
        )
        sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[0, 0])
        axes[0, 0].set_title('Return% by Entry/Exit Length')
        
        # Trail mult vs Risk %
        pivot2 = df.pivot_table(
            values='return_pct',
            index='trail_mult',
            columns='risk_pct',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Return% by Trail Mult/Risk%')
        
        # Entry vs Pyramid spacing
        pivot3 = df.pivot_table(
            values='sharpe',
            index='entry_len',
            columns='pyramid_spacing',
            aggfunc='mean'
        )
        sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Sharpe by Entry Length/Pyramid Spacing')
        
        # Max DD by parameters
        pivot4 = df.pivot_table(
            values='max_dd_pct',
            index='entry_len',
            columns='risk_pct',
            aggfunc='mean'
        )
        sns.heatmap(pivot4, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=axes[1, 1])
        axes[1, 1].set_title('Max Drawdown% by Entry Length/Risk%')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmaps saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    import os
    from data_fetcher import download_btc_data
    from config import RESULTS_DIR, PLOTS_DIR
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")
    
    tester = RobustnessTester()
    results_df = tester.run_full_test(df)
    
    print(tester.get_summary_report())
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/robustness_results.csv")
    
    # Generate plots
    tester.plot_heatmaps(os.path.join(PLOTS_DIR, "robustness_heatmaps.png"))

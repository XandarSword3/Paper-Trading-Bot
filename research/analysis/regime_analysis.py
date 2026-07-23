"""
Phase 2: Regime Decomposition Analysis
Analyzes strategy performance across different BTC market regimes
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from config import DEFAULT_REGIMES, StrategyParams, DEFAULT_PARAMS
from strategy import TurtleDonchianStrategy


@dataclass
class RegimeStats:
    """Statistics for a single regime"""
    name: str
    regime_type: str
    start_date: str
    end_date: str
    btc_return_pct: float
    strategy_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    profit_factor: float
    time_to_recovery_days: float
    sharpe_ratio: float
    survived: bool  # Did not go to zero


class RegimeAnalyzer:
    """
    Analyzes strategy performance across different market regimes.
    
    The key pattern we're looking for:
    - Loses or flat in chop
    - Survives bear
    - Explodes in bull
    - Recovers within 1-2 regimes
    """
    
    def __init__(self, params: StrategyParams = None):
        self.params = params or DEFAULT_PARAMS
        self.regimes = DEFAULT_REGIMES.regimes
        self.results: List[RegimeStats] = []
    
    def analyze_regime(
        self,
        df: pd.DataFrame,
        regime: dict,
        initial_capital: float = 100_000.0
    ) -> RegimeStats:
        """Analyze strategy performance for a single regime"""
        
        # Filter data for regime period
        start = pd.Timestamp(regime['start'])
        end = pd.Timestamp(regime['end'])
        regime_df = df[(df.index >= start) & (df.index <= end)].copy()
        
        if len(regime_df) < 50:  # Need minimum data
            return RegimeStats(
                name=regime['name'],
                regime_type=regime['type'],
                start_date=regime['start'],
                end_date=regime['end'],
                btc_return_pct=0.0,
                strategy_return_pct=0.0,
                max_drawdown_pct=0.0,
                num_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                time_to_recovery_days=0.0,
                sharpe_ratio=0.0,
                survived=True
            )
        
        # Calculate BTC buy-and-hold return
        btc_return = (regime_df['close'].iloc[-1] / regime_df['close'].iloc[0] - 1) * 100
        
        # Run strategy backtest
        strategy = TurtleDonchianStrategy(self.params)
        results = strategy.run_backtest(regime_df, initial_capital=initial_capital, verbose=False)
        
        # Get stats
        trade_stats = strategy.get_trade_stats()
        equity_stats = strategy.get_equity_stats(initial_capital)
        
        # Determine if strategy survived
        final_equity = equity_stats.get('final_equity', initial_capital)
        survived = final_equity > initial_capital * 0.1  # Survived if > 10% left
        
        return RegimeStats(
            name=regime['name'],
            regime_type=regime['type'],
            start_date=regime['start'],
            end_date=regime['end'],
            btc_return_pct=btc_return,
            strategy_return_pct=equity_stats.get('total_return_pct', 0),
            max_drawdown_pct=abs(equity_stats.get('max_drawdown_pct', 0)),
            num_trades=trade_stats.get('total_trades', 0),
            win_rate=trade_stats.get('win_rate', 0),
            profit_factor=trade_stats.get('profit_factor', 0),
            time_to_recovery_days=equity_stats.get('avg_recovery_days', 0),
            sharpe_ratio=equity_stats.get('sharpe_ratio', 0),
            survived=survived
        )
    
    def run_full_analysis(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0
    ) -> List[RegimeStats]:
        """Run analysis for all defined regimes"""
        
        self.results = []
        
        print("\n" + "=" * 80)
        print("PHASE 2: REGIME DECOMPOSITION ANALYSIS")
        print("=" * 80)
        
        for regime in self.regimes:
            print(f"\nAnalyzing {regime['name']} ({regime['type']})...")
            stats = self.analyze_regime(df, regime, initial_capital)
            self.results.append(stats)
            
            print(f"  BTC Return: {stats.btc_return_pct:+.1f}%")
            print(f"  Strategy Return: {stats.strategy_return_pct:+.1f}%")
            print(f"  Max Drawdown: {stats.max_drawdown_pct:.1f}%")
            print(f"  Trades: {stats.num_trades}, Win Rate: {stats.win_rate:.1f}%")
            print(f"  Survived: {'✓' if stats.survived else '✗'}")
        
        return self.results
    
    def get_summary_report(self) -> str:
        """Generate a summary report of regime analysis"""
        
        if not self.results:
            return "No regime analysis results available."
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("REGIME ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        
        # Summary table
        report.append("\n{:<25} {:>8} {:>12} {:>12} {:>10} {:>8}".format(
            "Regime", "Type", "BTC Ret%", "Strat Ret%", "Max DD%", "Survive"
        ))
        report.append("-" * 80)
        
        for r in self.results:
            report.append("{:<25} {:>8} {:>+12.1f} {:>+12.1f} {:>10.1f} {:>8}".format(
                r.name, r.regime_type, r.btc_return_pct, r.strategy_return_pct,
                r.max_drawdown_pct, "✓" if r.survived else "✗"
            ))
        
        # Pattern analysis
        report.append("\n" + "-" * 80)
        report.append("PATTERN ANALYSIS (Expected: Flat/Loss in chop, Survive bear, Explode in bull)")
        report.append("-" * 80)
        
        bull_regimes = [r for r in self.results if r.regime_type == 'bull']
        bear_regimes = [r for r in self.results if r.regime_type == 'bear']
        chop_regimes = [r for r in self.results if r.regime_type == 'chop']
        
        if bull_regimes:
            avg_bull = np.mean([r.strategy_return_pct for r in bull_regimes])
            report.append(f"Bull Market Performance: {avg_bull:+.1f}% avg return")
            if avg_bull > 50:
                report.append("  ✓ PASS - Captures upside in bull markets")
            else:
                report.append("  ⚠ WARNING - Underperforming in bull markets")
        
        if bear_regimes:
            avg_bear = np.mean([r.strategy_return_pct for r in bear_regimes])
            all_survived = all(r.survived for r in bear_regimes)
            report.append(f"Bear Market Performance: {avg_bear:+.1f}% avg return")
            if all_survived and avg_bear > -30:
                report.append("  ✓ PASS - Survives bear markets")
            else:
                report.append("  ✗ FAIL - Does not survive bear markets well")
        
        if chop_regimes:
            avg_chop = np.mean([r.strategy_return_pct for r in chop_regimes])
            report.append(f"Chop/Sideways Performance: {avg_chop:+.1f}% avg return")
            if avg_chop > -20:
                report.append("  ✓ PASS - Acceptable losses in chop")
            else:
                report.append("  ⚠ WARNING - Large losses in choppy markets")
        
        # Overall viability
        report.append("\n" + "-" * 80)
        report.append("VIABILITY ASSESSMENT")
        report.append("-" * 80)
        
        all_survived = all(r.survived for r in self.results)
        max_dd = max(r.max_drawdown_pct for r in self.results)
        total_return = sum(r.strategy_return_pct for r in self.results)
        
        if all_survived and max_dd < 60 and total_return > 0:
            report.append("✓ VIABLE - Strategy survived all regimes with acceptable drawdowns")
        elif all_survived:
            report.append("⚠ MARGINAL - Survived but performance needs review")
        else:
            report.append("✗ NOT VIABLE - Strategy died in at least one regime")
        
        return "\n".join(report)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for further analysis"""
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'name': r.name,
                'type': r.regime_type,
                'start': r.start_date,
                'end': r.end_date,
                'btc_return_pct': r.btc_return_pct,
                'strategy_return_pct': r.strategy_return_pct,
                'max_drawdown_pct': r.max_drawdown_pct,
                'num_trades': r.num_trades,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'recovery_days': r.time_to_recovery_days,
                'sharpe': r.sharpe_ratio,
                'survived': r.survived
            }
            for r in self.results
        ])


if __name__ == "__main__":
    from data_fetcher import download_btc_data
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")
    
    analyzer = RegimeAnalyzer()
    analyzer.run_full_analysis(df)
    
    print(analyzer.get_summary_report())
    
    # Save results
    results_df = analyzer.to_dataframe()
    results_df.to_csv("results/regime_analysis.csv", index=False)
    print("\nResults saved to results/regime_analysis.csv")

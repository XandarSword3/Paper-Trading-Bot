"""
Phase 4: Capital Survivability Analysis
Determines if you get rich or go broke
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import StrategyParams, DEFAULT_PARAMS
from strategy import TurtleDonchianStrategy


@dataclass
class SurvivabilityMetrics:
    """Capital survivability metrics"""
    worst_drawdown_pct: float
    longest_drawdown_days: float
    worst_losing_streak: int
    worst_losing_streak_pnl: float
    max_consecutive_losses: int
    worst_month_return_pct: float
    worst_quarter_return_pct: float
    worst_year_return_pct: float
    equity_volatility_pct: float
    time_underwater_pct: float
    ulcer_index: float
    pain_index: float
    
    # Stress-tested versions (50% haircut, 1.5x drawdown)
    stress_test_equity: float
    stress_test_drawdown: float
    survives_stress: bool


class SurvivabilityAnalyzer:
    """
    Analyzes capital survivability under various stress scenarios.
    
    Key metrics:
    - Worst historical drawdown
    - Longest drawdown duration
    - Worst losing streak
    - Worst equity volatility window
    
    Then applies:
    - 50% haircut to performance
    - 1.5× increase in drawdown
    """
    
    def __init__(self, params: StrategyParams = None):
        self.params = params or DEFAULT_PARAMS
        self.metrics: SurvivabilityMetrics = None
        self.equity_curve: pd.Series = None
        self.drawdown_curve: pd.Series = None
    
    def analyze(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0
    ) -> SurvivabilityMetrics:
        """Run full survivability analysis"""
        
        print("\n" + "=" * 80)
        print("PHASE 4: CAPITAL SURVIVABILITY ANALYSIS")
        print("=" * 80)
        
        # Run backtest
        strategy = TurtleDonchianStrategy(self.params)
        results = strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
        
        self.equity_curve = results['equity']
        
        # Calculate drawdown curve
        rolling_max = self.equity_curve.expanding().max()
        self.drawdown_curve = (self.equity_curve - rolling_max) / rolling_max
        
        # === Worst Drawdown ===
        worst_drawdown_pct = abs(self.drawdown_curve.min()) * 100
        print(f"\nWorst Historical Drawdown: {worst_drawdown_pct:.1f}%")
        
        # === Longest Drawdown Duration ===
        longest_dd_days = self._calculate_longest_drawdown()
        print(f"Longest Drawdown Duration: {longest_dd_days:.0f} days")
        
        # === Worst Losing Streak ===
        trades = strategy.trades
        losing_streak, streak_pnl = self._calculate_worst_losing_streak(trades)
        print(f"Worst Losing Streak: {losing_streak} trades (${streak_pnl:,.0f} loss)")
        
        # === Period Returns ===
        worst_month = self._calculate_worst_period_return('ME')  # Month end
        worst_quarter = self._calculate_worst_period_return('QE')  # Quarter end
        worst_year = self._calculate_worst_period_return('YE')  # Year end
        
        print(f"Worst Month: {worst_month:.1f}%")
        print(f"Worst Quarter: {worst_quarter:.1f}%")
        print(f"Worst Year: {worst_year:.1f}%")
        
        # === Equity Volatility ===
        daily_returns = self.equity_curve.resample('D').last().pct_change().dropna()
        equity_vol = daily_returns.std() * np.sqrt(365) * 100
        print(f"Annualized Equity Volatility: {equity_vol:.1f}%")
        
        # === Time Underwater ===
        time_underwater = (self.drawdown_curve < 0).sum() / len(self.drawdown_curve) * 100
        print(f"Time Underwater: {time_underwater:.1f}%")
        
        # === Ulcer Index (pain of drawdowns) ===
        ulcer_index = np.sqrt((self.drawdown_curve ** 2).mean()) * 100
        print(f"Ulcer Index: {ulcer_index:.2f}")
        
        # === Pain Index (average drawdown) ===
        pain_index = abs(self.drawdown_curve.mean()) * 100
        print(f"Pain Index: {pain_index:.2f}%")
        
        # === STRESS TEST ===
        print("\n" + "-" * 40)
        print("STRESS TEST (50% haircut, 1.5x drawdown)")
        print("-" * 40)
        
        final_equity = self.equity_curve.iloc[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # 50% haircut to performance
        stress_return = total_return * 0.5
        stress_equity = initial_capital * (1 + stress_return / 100)
        
        # 1.5x drawdown
        stress_drawdown = worst_drawdown_pct * 1.5
        
        print(f"Stress-Tested Return: {stress_return:+.1f}% (was {total_return:+.1f}%)")
        print(f"Stress-Tested Drawdown: {stress_drawdown:.1f}% (was {worst_drawdown_pct:.1f}%)")
        
        # Survival check
        survives = stress_equity > initial_capital * 0.5 and stress_drawdown < 80
        
        self.metrics = SurvivabilityMetrics(
            worst_drawdown_pct=worst_drawdown_pct,
            longest_drawdown_days=longest_dd_days,
            worst_losing_streak=losing_streak,
            worst_losing_streak_pnl=streak_pnl,
            max_consecutive_losses=losing_streak,
            worst_month_return_pct=worst_month,
            worst_quarter_return_pct=worst_quarter,
            worst_year_return_pct=worst_year,
            equity_volatility_pct=equity_vol,
            time_underwater_pct=time_underwater,
            ulcer_index=ulcer_index,
            pain_index=pain_index,
            stress_test_equity=stress_equity,
            stress_test_drawdown=stress_drawdown,
            survives_stress=survives
        )
        
        return self.metrics
    
    def _calculate_longest_drawdown(self) -> float:
        """Calculate longest drawdown period in days"""
        in_drawdown = self.drawdown_curve < 0
        
        longest = 0
        current = 0
        
        for val in in_drawdown:
            if val:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        
        # Convert to days (assuming 4H bars = 6 per day)
        bars_per_day = 6
        return longest / bars_per_day
    
    def _calculate_worst_losing_streak(self, trades) -> Tuple[int, float]:
        """Calculate worst consecutive losing trades"""
        if not trades:
            return 0, 0.0
        
        current_streak = 0
        worst_streak = 0
        current_pnl = 0.0
        worst_pnl = 0.0
        
        for trade in trades:
            if trade.pnl and trade.pnl < 0:
                current_streak += 1
                current_pnl += trade.pnl
                if current_streak > worst_streak:
                    worst_streak = current_streak
                    worst_pnl = current_pnl
            else:
                current_streak = 0
                current_pnl = 0.0
        
        return worst_streak, worst_pnl
    
    def _calculate_worst_period_return(self, freq: str) -> float:
        """Calculate worst return for given period frequency"""
        try:
            period_equity = self.equity_curve.resample(freq).last().dropna()
            period_returns = period_equity.pct_change().dropna() * 100
            return period_returns.min() if len(period_returns) > 0 else 0.0
        except:
            return 0.0
    
    def get_summary_report(self) -> str:
        """Generate survivability analysis report"""
        
        if self.metrics is None:
            return "No survivability analysis results available."
        
        m = self.metrics
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("CAPITAL SURVIVABILITY REPORT")
        report.append("=" * 80)
        
        report.append("\n" + "-" * 40)
        report.append("HISTORICAL WORST CASE METRICS")
        report.append("-" * 40)
        
        report.append(f"Maximum Drawdown:        {m.worst_drawdown_pct:.1f}%")
        report.append(f"Longest Drawdown:        {m.longest_drawdown_days:.0f} days")
        report.append(f"Worst Losing Streak:     {m.worst_losing_streak} trades")
        report.append(f"Streak P&L Loss:         ${m.worst_losing_streak_pnl:,.0f}")
        report.append(f"Worst Month:             {m.worst_month_return_pct:+.1f}%")
        report.append(f"Worst Quarter:           {m.worst_quarter_return_pct:+.1f}%")
        report.append(f"Worst Year:              {m.worst_year_return_pct:+.1f}%")
        
        report.append("\n" + "-" * 40)
        report.append("PAIN METRICS")
        report.append("-" * 40)
        
        report.append(f"Equity Volatility (Ann): {m.equity_volatility_pct:.1f}%")
        report.append(f"Time Underwater:         {m.time_underwater_pct:.1f}%")
        report.append(f"Ulcer Index:             {m.ulcer_index:.2f}")
        report.append(f"Pain Index:              {m.pain_index:.2f}%")
        
        report.append("\n" + "-" * 40)
        report.append("STRESS TEST RESULTS")
        report.append("-" * 40)
        
        report.append(f"Stress-Tested Equity:    ${m.stress_test_equity:,.0f}")
        report.append(f"Stress-Tested Drawdown:  {m.stress_test_drawdown:.1f}%")
        
        report.append("\n" + "-" * 40)
        report.append("SURVIVABILITY VERDICT")
        report.append("-" * 40)
        
        if m.survives_stress:
            report.append("✓ SURVIVES STRESS TEST")
            report.append("")
            report.append("Recommendations:")
            
            if m.worst_drawdown_pct > 40:
                report.append("  ⚠ Consider reducing risk% to lower drawdowns")
            if m.longest_drawdown_days > 180:
                report.append("  ⚠ Prepare for multi-month drawdown periods")
            if m.time_underwater_pct > 50:
                report.append("  ⚠ Expect to be in drawdown more than half the time")
            
            report.append("\nYou can proceed with deployment, but:")
            report.append("  1. Use only capital you can afford to lose")
            report.append("  2. Expect drawdowns 50% worse than historical")
            report.append("  3. Accept performance may be 50% lower")
        else:
            report.append("✗ FAILS STRESS TEST")
            report.append("")
            report.append("Required adjustments:")
            report.append("  1. Reduce risk_percent significantly")
            report.append("  2. Reduce max_units (pyramiding)")
            report.append("  3. Increase trail_mult for tighter stops")
            report.append("  4. Accept lower returns for survivability")
        
        return "\n".join(report)
    
    def get_psychological_assessment(self) -> str:
        """Assess if strategy is psychologically tradeable"""
        
        if self.metrics is None:
            return "No metrics available."
        
        m = self.metrics
        
        assessment = []
        assessment.append("\n" + "=" * 80)
        assessment.append("PSYCHOLOGICAL ASSESSMENT")
        assessment.append("=" * 80)
        assessment.append("\nCan you actually trade this strategy?")
        assessment.append("")
        
        issues = []
        
        if m.worst_drawdown_pct > 50:
            issues.append(f"• You will see 50%+ of your account disappear ({m.worst_drawdown_pct:.0f}%)")
        
        if m.longest_drawdown_days > 365:
            issues.append(f"• You may wait {m.longest_drawdown_days/365:.1f} years to recover from a drawdown")
        
        if m.worst_losing_streak > 5:
            issues.append(f"• You will experience {m.worst_losing_streak}+ consecutive losses")
        
        if m.time_underwater_pct > 60:
            issues.append(f"• You will be losing {m.time_underwater_pct:.0f}% of the time")
        
        if m.worst_month_return_pct < -15:
            issues.append(f"• Single months can lose {abs(m.worst_month_return_pct):.0f}%+")
        
        if issues:
            assessment.append("You must be prepared for:")
            assessment.extend(issues)
            assessment.append("")
            assessment.append("If ANY of these would cause you to abandon the strategy,")
            assessment.append("DO NOT TRADE IT with real money.")
        else:
            assessment.append("✓ Strategy has manageable psychological demands")
        
        return "\n".join(assessment)


if __name__ == "__main__":
    import os
    from data_fetcher import download_btc_data
    from config import RESULTS_DIR
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")
    
    analyzer = SurvivabilityAnalyzer()
    metrics = analyzer.analyze(df)
    
    print(analyzer.get_summary_report())
    print(analyzer.get_psychological_assessment())
    
    # Save drawdown curve
    if analyzer.drawdown_curve is not None:
        dd_df = pd.DataFrame({
            'equity': analyzer.equity_curve,
            'drawdown_pct': analyzer.drawdown_curve * 100
        })
        dd_df.to_csv(os.path.join(RESULTS_DIR, "survivability_data.csv"))
        print(f"\nData saved to {RESULTS_DIR}/survivability_data.csv")

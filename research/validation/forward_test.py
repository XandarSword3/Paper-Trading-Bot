"""
Phase 6-7: Forward Testing Framework & Deployment Guide
Paper trading setup and real money deployment guidelines
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
import os

from config import StrategyParams, DEFAULT_PARAMS, DATA_DIR, RESULTS_DIR
from strategy import TurtleDonchianStrategy
from data_fetcher import BinanceDataFetcher


class ForwardTestTracker:
    """
    Forward testing tracker for paper trading.
    
    Phase 6 Requirements:
    - Paper trade exactly for 3-6 months
    - Same rules as backtest
    - Same position sizing
    - No overrides
    """
    
    def __init__(self, params: StrategyParams = None, initial_capital: float = 100_000.0):
        self.params = params or DEFAULT_PARAMS
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Tracking
        self.trades = []
        self.daily_equity = []
        self.signals = []
        self.state_file = os.path.join(RESULTS_DIR, "forward_test_state.json")
        
        # Position state
        self.position_size = 0.0
        self.position_direction = None
        self.entry_price = 0.0
        self.units_count = 0
        self.last_add_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float('inf')
    
    def save_state(self):
        """Save current state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'current_capital': self.current_capital,
            'position_size': self.position_size,
            'position_direction': self.position_direction,
            'entry_price': self.entry_price,
            'units_count': self.units_count,
            'last_add_price': self.last_add_price,
            'highest_since_entry': self.highest_since_entry,
            'lowest_since_entry': self.lowest_since_entry,
            'trades': self.trades,
            'daily_equity': self.daily_equity,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"State saved to {self.state_file}")
    
    def load_state(self) -> bool:
        """Load state from file if exists"""
        if not os.path.exists(self.state_file):
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.current_capital = state['current_capital']
            self.position_size = state['position_size']
            self.position_direction = state['position_direction']
            self.entry_price = state['entry_price']
            self.units_count = state['units_count']
            self.last_add_price = state['last_add_price']
            self.highest_since_entry = state['highest_since_entry']
            self.lowest_since_entry = state['lowest_since_entry']
            self.trades = state['trades']
            self.daily_equity = state['daily_equity']
            
            print(f"State loaded from {self.state_file}")
            print(f"Current capital: ${self.current_capital:,.2f}")
            print(f"Position: {self.position_size:.4f} BTC ({self.position_direction})")
            
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def generate_current_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals for current market state"""
        
        strategy = TurtleDonchianStrategy(self.params)
        df = strategy.calculate_indicators(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'timestamp': str(df.index[-1]),
            'close': latest['close'],
            'upper_entry': latest['upper_entry'],
            'lower_entry': latest['lower_entry'],
            'upper_exit': latest['upper_exit'],
            'lower_exit': latest['lower_exit'],
            'atr': latest['atr'],
            'ema200': latest['ema200'],
            
            'long_entry': latest['close'] > latest['upper_entry'],
            'short_entry': (
                not self.params.long_only and
                latest['close'] < latest['lower_entry'] and
                (not self.params.use_regime_filter or latest['close'] < latest['ema200'])
            ),
            'long_exit': latest['close'] < latest['lower_exit'],
            'short_exit': latest['close'] > latest['upper_exit'],
            
            'suggested_position_size': strategy.calculate_unit_size(
                self.current_capital, latest['atr'], latest['close']
            ),
        }
        
        # Add trailing stop levels
        if self.position_size > 0:
            self.highest_since_entry = max(self.highest_since_entry, latest['high'])
            signals['trail_stop'] = self.highest_since_entry - self.params.trail_mult * latest['atr']
        elif self.position_size < 0:
            self.lowest_since_entry = min(self.lowest_since_entry, latest['low'])
            signals['trail_stop'] = self.lowest_since_entry + self.params.trail_mult * latest['atr']
        else:
            signals['trail_stop'] = None
        
        # Add pyramiding levels
        if self.position_size > 0 and self.units_count < self.params.max_units:
            signals['pyramid_trigger'] = self.last_add_price + (self.params.pyramid_spacing_n * latest['atr'])
        elif self.position_size < 0 and self.units_count < self.params.max_units:
            signals['pyramid_trigger'] = self.last_add_price - (self.params.pyramid_spacing_n * latest['atr'])
        else:
            signals['pyramid_trigger'] = None
        
        return signals
    
    def record_trade(
        self,
        action: str,
        price: float,
        quantity: float,
        timestamp: str = None
    ):
        """Record a trade execution"""
        timestamp = timestamp or datetime.now().isoformat()
        
        trade = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'quantity': quantity,
            'capital_before': self.current_capital,
        }
        
        # Update position
        if action == 'BUY':
            if self.position_size <= 0:
                # New long or closing short
                if self.position_size < 0:
                    # Close short
                    pnl = abs(self.position_size) * (self.entry_price - price)
                    self.current_capital += pnl
                    trade['pnl'] = pnl
                
                self.position_size = quantity
                self.position_direction = 'long'
                self.entry_price = price
                self.units_count = 1
                self.last_add_price = price
                self.highest_since_entry = price
            else:
                # Adding to long
                total_cost = self.entry_price * self.position_size + price * quantity
                self.position_size += quantity
                self.entry_price = total_cost / self.position_size
                self.units_count += 1
                self.last_add_price = price
        
        elif action == 'SELL':
            if self.position_size >= 0:
                # New short or closing long
                if self.position_size > 0:
                    # Close long
                    pnl = self.position_size * (price - self.entry_price)
                    self.current_capital += pnl
                    trade['pnl'] = pnl
                
                if not self.params.long_only:
                    self.position_size = -quantity
                    self.position_direction = 'short'
                    self.entry_price = price
                    self.units_count = 1
                    self.last_add_price = price
                    self.lowest_since_entry = price
                else:
                    self.position_size = 0
                    self.position_direction = None
            else:
                # Adding to short
                total_cost = self.entry_price * abs(self.position_size) + price * quantity
                self.position_size -= quantity
                self.entry_price = total_cost / abs(self.position_size)
                self.units_count += 1
                self.last_add_price = price
        
        elif action == 'CLOSE':
            if self.position_size > 0:
                pnl = self.position_size * (price - self.entry_price)
            else:
                pnl = abs(self.position_size) * (self.entry_price - price)
            
            self.current_capital += pnl
            trade['pnl'] = pnl
            
            self.position_size = 0
            self.position_direction = None
            self.units_count = 0
            self.highest_since_entry = 0
            self.lowest_since_entry = float('inf')
        
        trade['capital_after'] = self.current_capital
        self.trades.append(trade)
        self.save_state()
        
        return trade
    
    def record_daily_equity(self, current_price: float):
        """Record daily equity snapshot"""
        
        # Calculate unrealized PnL
        if self.position_size > 0:
            unrealized = self.position_size * (current_price - self.entry_price)
        elif self.position_size < 0:
            unrealized = abs(self.position_size) * (self.entry_price - current_price)
        else:
            unrealized = 0
        
        total_equity = self.current_capital + unrealized
        
        self.daily_equity.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'capital': self.current_capital,
            'unrealized': unrealized,
            'total_equity': total_equity,
            'position': self.position_size,
            'price': current_price
        })
        
        self.save_state()
    
    def get_performance_summary(self) -> str:
        """Get forward test performance summary"""
        
        if not self.daily_equity:
            return "No forward test data available."
        
        df = pd.DataFrame(self.daily_equity)
        
        report = []
        report.append("\n" + "=" * 60)
        report.append("FORWARD TEST PERFORMANCE SUMMARY")
        report.append("=" * 60)
        
        start_equity = self.initial_capital
        current_equity = df['total_equity'].iloc[-1]
        total_return = (current_equity / start_equity - 1) * 100
        
        report.append(f"\nStart Date: {df['date'].iloc[0]}")
        report.append(f"Current Date: {df['date'].iloc[-1]}")
        report.append(f"Days: {len(df)}")
        
        report.append(f"\nInitial Capital: ${start_equity:,.2f}")
        report.append(f"Current Equity: ${current_equity:,.2f}")
        report.append(f"Total Return: {total_return:+.2f}%")
        
        # Drawdown
        rolling_max = pd.Series(df['total_equity']).expanding().max()
        drawdown = (df['total_equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        report.append(f"Max Drawdown: {max_dd:.1f}%")
        
        # Trade stats
        if self.trades:
            trades_with_pnl = [t for t in self.trades if 'pnl' in t]
            if trades_with_pnl:
                total_pnl = sum(t['pnl'] for t in trades_with_pnl)
                wins = [t for t in trades_with_pnl if t['pnl'] > 0]
                losses = [t for t in trades_with_pnl if t['pnl'] < 0]
                
                report.append(f"\nTotal Trades: {len(trades_with_pnl)}")
                report.append(f"Win Rate: {len(wins)/len(trades_with_pnl)*100:.1f}%")
                report.append(f"Total P&L: ${total_pnl:,.2f}")
        
        return "\n".join(report)


def generate_deployment_checklist() -> str:
    """Generate deployment checklist for Phase 7"""
    
    checklist = """
================================================================================
PHASE 7: REAL MONEY DEPLOYMENT CHECKLIST
================================================================================

PRE-DEPLOYMENT REQUIREMENTS:
----------------------------
□ Completed 3-6 months of paper trading
□ Paper trading results within 20% of backtest expectations
□ Survived at least one significant drawdown in paper trading
□ Followed rules mechanically without overrides
□ Comfortable with expected drawdowns psychologically

CAPITAL ALLOCATION:
-------------------
□ Starting with 10-20% of intended final capital
□ Using only money you can afford to lose completely
□ No borrowed money or money needed for living expenses
□ Separate trading account from main savings

EXCHANGE SETUP:
---------------
□ Account on reliable exchange (Binance, Bybit, etc.)
□ API keys generated with trading permissions only (no withdrawal)
□ IP whitelist configured for API access
□ 2FA enabled on account
□ Withdrawal address whitelist enabled

RISK CONTROLS:
--------------
□ Maximum position size limits set
□ Daily loss limits configured
□ Emergency stop-loss orders in place
□ Plan for exchange downtime/issues

EXECUTION PLAN:
---------------
□ Signal generation automated or scheduled manual checks
□ Trade logging system in place
□ Daily equity recording
□ Weekly performance review scheduled

SCALING RULES:
--------------
□ Will NOT scale up during winning streaks
□ Will scale up only after surviving full drawdown cycle
□ Maximum 2x increase per scaling step
□ 3-month minimum between scaling decisions

PSYCHOLOGICAL PREPARATION:
--------------------------
□ Accept you will have losing months
□ Accept drawdowns will feel worse in real money
□ Commit to following rules for minimum 1 year
□ Have plan for handling urge to override system

================================================================================
HARD TRUTHS TO ACCEPT BEFORE DEPLOYING
================================================================================

1. Backtest results are OPTIMISTIC. Real results will be worse.

2. The first 6 months will likely be mediocre or losing.

3. You will want to quit. Have a plan for this.

4. "Improving" the strategy during live trading is usually destroying it.

5. The money you make (if any) comes from:
   - Staying in during boredom
   - Not quitting after underperformance
   - Not "improving" it to death
   - BTC happening to trend again

6. If you cannot follow the rules mechanically, YOU are the problem,
   not the system.

================================================================================
DEPLOYMENT TIMELINE
================================================================================

Week 1-2:    Deploy with minimum size (0.001 BTC per unit)
             Focus on execution, not returns

Week 3-4:    Increase to 10% of target size if execution is clean

Month 2-3:   Maintain 10-20% size
             First drawdown likely to occur

Month 4-6:   IF survived drawdown and still following rules,
             consider increasing to 30-40% of target

Month 7-12:  Gradual scaling only after proven performance
             Never more than 2x increase at once

Year 2+:     Full target size only after surviving complete
             bull/bear cycle

================================================================================
"""
    
    return checklist


if __name__ == "__main__":
    from data_fetcher import download_btc_data
    
    print("=" * 60)
    print("FORWARD TESTING FRAMEWORK")
    print("=" * 60)
    
    # Initialize tracker
    tracker = ForwardTestTracker(initial_capital=100_000.0)
    
    # Try to load existing state
    if not tracker.load_state():
        print("\nNo existing forward test state. Starting fresh.")
    
    # Get current market data and signals
    print("\nFetching current market data...")
    df = download_btc_data(timeframe="4h")
    
    signals = tracker.generate_current_signals(df)
    
    print("\n" + "-" * 40)
    print("CURRENT MARKET SIGNALS")
    print("-" * 40)
    print(f"Timestamp: {signals['timestamp']}")
    print(f"Price: ${signals['close']:,.2f}")
    print(f"ATR: ${signals['atr']:,.2f}")
    print(f"Upper Entry: ${signals['upper_entry']:,.2f}")
    print(f"Lower Entry: ${signals['lower_entry']:,.2f}")
    print(f"Lower Exit: ${signals['lower_exit']:,.2f}")
    print(f"Upper Exit: ${signals['upper_exit']:,.2f}")
    print(f"EMA 200: ${signals['ema200']:,.2f}")
    
    print("\n" + "-" * 40)
    print("SIGNAL STATUS")
    print("-" * 40)
    print(f"Long Entry Signal: {'✓ ACTIVE' if signals['long_entry'] else '✗ No'}")
    print(f"Short Entry Signal: {'✓ ACTIVE' if signals['short_entry'] else '✗ No'}")
    print(f"Long Exit Signal: {'✓ ACTIVE' if signals['long_exit'] else '✗ No'}")
    print(f"Short Exit Signal: {'✓ ACTIVE' if signals['short_exit'] else '✗ No'}")
    
    if signals['trail_stop']:
        print(f"Trailing Stop: ${signals['trail_stop']:,.2f}")
    if signals['pyramid_trigger']:
        print(f"Pyramid Trigger: ${signals['pyramid_trigger']:,.2f}")
    
    print(f"\nSuggested Position Size: {signals['suggested_position_size']:.4f} BTC")
    
    print(tracker.get_performance_summary())
    
    print(generate_deployment_checklist())

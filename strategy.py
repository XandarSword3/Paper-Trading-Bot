"""
Turtle-Inspired Donchian Strategy Implementation
Exact replication of the TradingView Pine Script logic
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
from config import StrategyParams, DEFAULT_PARAMS


@dataclass
class TradeRecord:
    """Record of a single trade"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    exit_reason: Optional[str]
    unit_number: int  # For pyramiding tracking


class TurtleDonchianStrategy:
    """
    Turtle-Inspired Donchian Channel Strategy
    
    Replicates the exact logic from the TradingView Pine Script:
    - Donchian channel breakout entries
    - ATR-based position sizing
    - Pyramiding with N-spacing
    - Trailing stops
    - Donchian exits
    """
    
    def __init__(self, params: StrategyParams = None):
        self.params = params or DEFAULT_PARAMS
        self.trades: List[TradeRecord] = []
        self.equity_curve: Optional[pd.Series] = None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = df.copy()
        
        # Donchian channels (using previous bar's high/low, shifted by 1)
        df['upper_entry'] = df['high'].shift(1).rolling(self.params.entry_len).max()
        df['lower_entry'] = df['low'].shift(1).rolling(self.params.entry_len).min()
        df['upper_exit'] = df['high'].shift(1).rolling(self.params.exit_len).max()
        df['lower_exit'] = df['low'].shift(1).rolling(self.params.exit_len).min()
        
        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(self.params.atr_len).mean()
        
        # 200 EMA for regime filter
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        return df
    
    def calculate_unit_size(
        self,
        equity: float,
        atr: float,
        price: float
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            equity: Current account equity
            atr: Current ATR value
            price: Current price (for BTC sizing)
        
        Returns:
            Position size in BTC
        """
        unit_dollar_risk = equity * (self.params.risk_percent / 100)
        stop_usd = atr * self.params.size_stop_mult
        
        if stop_usd <= 0:
            return 0.0
        
        unit_size_raw = unit_dollar_risk / stop_usd
        unit_size = max(
            self.params.lot_step,
            round(unit_size_raw / self.params.lot_step) * self.params.lot_step
        )
        
        return unit_size
    
    def apply_costs(
        self,
        price: float,
        direction: str,
        is_entry: bool
    ) -> float:
        """Apply commission and slippage to execution price"""
        total_slippage_pct = self.params.commission_pct + self.params.slippage_pct
        
        if is_entry:
            if direction == 'long':
                return price * (1 + total_slippage_pct / 100)
            else:
                return price * (1 - total_slippage_pct / 100)
        else:
            if direction == 'long':
                return price * (1 - total_slippage_pct / 100)
            else:
                return price * (1 + total_slippage_pct / 100)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Run the full backtest simulation.
        
        Args:
            df: DataFrame with OHLCV data
            initial_capital: Starting capital in USD
            verbose: Print trade details
        
        Returns:
            DataFrame with equity curve and trade markers
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize state
        equity = initial_capital
        position_size = 0.0  # In BTC
        position_direction = None  # 'long' or 'short'
        avg_entry_price = 0.0
        units_count = 0
        last_add_price = 0.0
        highest_since_entry = 0.0
        lowest_since_entry = float('inf')
        
        self.trades = []
        current_trades: List[TradeRecord] = []  # Open positions
        
        # Results tracking
        equity_values = []
        position_values = []
        
        # Skip warmup period
        warmup = max(self.params.entry_len, self.params.atr_len, 200) + 1
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Skip warmup
            if i < warmup:
                equity_values.append(equity)
                position_values.append(0.0)
                continue
            
            close = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            upper_entry = row['upper_entry']
            lower_entry = row['lower_entry']
            upper_exit = row['upper_exit']
            lower_exit = row['lower_exit']
            ema200 = row['ema200']
            
            # Skip if any indicator is NaN
            if pd.isna(atr) or pd.isna(upper_entry):
                equity_values.append(equity)
                position_values.append(position_size * close if position_size != 0 else 0)
                continue
            
            # === Exit Logic First ===
            
            # Check trailing stop exits
            if position_size > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high)
                trail_stop = highest_since_entry - self.params.trail_mult * atr
                
                if low <= trail_stop:
                    # Trailing stop hit
                    exit_price = self.apply_costs(trail_stop, 'long', False)
                    pnl = position_size * (exit_price - avg_entry_price)
                    equity += pnl
                    
                    for trade in current_trades:
                        trade.exit_time = timestamp
                        trade.exit_price = exit_price
                        trade.pnl = trade.quantity * (exit_price - trade.entry_price)
                        trade.exit_reason = "Trailing Stop"
                        self.trades.append(trade)
                    
                    if verbose:
                        print(f"{timestamp}: TRAIL EXIT LONG @ {exit_price:.2f}, PnL: {pnl:.2f}")
                    
                    position_size = 0.0
                    position_direction = None
                    current_trades = []
                    units_count = 0
                    highest_since_entry = 0.0
            
            elif position_size < 0:  # Short position
                lowest_since_entry = min(lowest_since_entry, low)
                trail_stop = lowest_since_entry + self.params.trail_mult * atr
                
                if high >= trail_stop:
                    exit_price = self.apply_costs(trail_stop, 'short', False)
                    pnl = abs(position_size) * (avg_entry_price - exit_price)
                    equity += pnl
                    
                    for trade in current_trades:
                        trade.exit_time = timestamp
                        trade.exit_price = exit_price
                        trade.pnl = trade.quantity * (trade.entry_price - exit_price)
                        trade.exit_reason = "Trailing Stop"
                        self.trades.append(trade)
                    
                    if verbose:
                        print(f"{timestamp}: TRAIL EXIT SHORT @ {exit_price:.2f}, PnL: {pnl:.2f}")
                    
                    position_size = 0.0
                    position_direction = None
                    current_trades = []
                    units_count = 0
                    lowest_since_entry = float('inf')
            
            # Check Donchian exits (only if position still open)
            if position_size > 0 and close < lower_exit:
                exit_price = self.apply_costs(close, 'long', False)
                pnl = position_size * (exit_price - avg_entry_price)
                equity += pnl
                
                for trade in current_trades:
                    trade.exit_time = timestamp
                    trade.exit_price = exit_price
                    trade.pnl = trade.quantity * (exit_price - trade.entry_price)
                    trade.exit_reason = "Donchian Exit"
                    self.trades.append(trade)
                
                if verbose:
                    print(f"{timestamp}: DONCHIAN EXIT LONG @ {exit_price:.2f}, PnL: {pnl:.2f}")
                
                position_size = 0.0
                position_direction = None
                current_trades = []
                units_count = 0
                highest_since_entry = 0.0
            
            elif position_size < 0 and close > upper_exit:
                exit_price = self.apply_costs(close, 'short', False)
                pnl = abs(position_size) * (avg_entry_price - exit_price)
                equity += pnl
                
                for trade in current_trades:
                    trade.exit_time = timestamp
                    trade.exit_price = exit_price
                    trade.pnl = trade.quantity * (trade.entry_price - exit_price)
                    trade.exit_reason = "Donchian Exit"
                    self.trades.append(trade)
                
                if verbose:
                    print(f"{timestamp}: DONCHIAN EXIT SHORT @ {exit_price:.2f}, PnL: {pnl:.2f}")
                
                position_size = 0.0
                position_direction = None
                current_trades = []
                units_count = 0
                lowest_since_entry = float('inf')
            
            # === Entry Logic ===
            
            # Long entry condition
            long_condition = close > upper_entry
            
            # Short entry condition
            short_condition = (
                not self.params.long_only and
                close < lower_entry and
                (not self.params.use_regime_filter or close < ema200)
            )
            
            # New long entry
            if long_condition and position_size <= 0:
                # Close short first if exists
                if position_size < 0:
                    exit_price = self.apply_costs(close, 'short', False)
                    pnl = abs(position_size) * (avg_entry_price - exit_price)
                    equity += pnl
                    
                    for trade in current_trades:
                        trade.exit_time = timestamp
                        trade.exit_price = exit_price
                        trade.pnl = trade.quantity * (trade.entry_price - exit_price)
                        trade.exit_reason = "Reversal"
                        self.trades.append(trade)
                    
                    position_size = 0.0
                    current_trades = []
                
                # Enter long
                unit_size = self.calculate_unit_size(equity, atr, close)
                entry_price = self.apply_costs(close, 'long', True)
                
                position_size = unit_size
                position_direction = 'long'
                avg_entry_price = entry_price
                units_count = 1
                last_add_price = close
                highest_since_entry = high
                
                trade = TradeRecord(
                    entry_time=timestamp,
                    exit_time=None,
                    direction='long',
                    entry_price=entry_price,
                    exit_price=None,
                    quantity=unit_size,
                    pnl=None,
                    exit_reason=None,
                    unit_number=1
                )
                current_trades = [trade]
                
                if verbose:
                    print(f"{timestamp}: ENTER LONG @ {entry_price:.2f}, Size: {unit_size:.4f}")
            
            # New short entry
            elif short_condition and position_size >= 0:
                # Close long first if exists
                if position_size > 0:
                    exit_price = self.apply_costs(close, 'long', False)
                    pnl = position_size * (exit_price - avg_entry_price)
                    equity += pnl
                    
                    for trade in current_trades:
                        trade.exit_time = timestamp
                        trade.exit_price = exit_price
                        trade.pnl = trade.quantity * (exit_price - trade.entry_price)
                        trade.exit_reason = "Reversal"
                        self.trades.append(trade)
                    
                    position_size = 0.0
                    current_trades = []
                
                # Enter short
                unit_size = self.calculate_unit_size(equity, atr, close)
                entry_price = self.apply_costs(close, 'short', True)
                
                position_size = -unit_size
                position_direction = 'short'
                avg_entry_price = entry_price
                units_count = 1
                last_add_price = close
                lowest_since_entry = low
                
                trade = TradeRecord(
                    entry_time=timestamp,
                    exit_time=None,
                    direction='short',
                    entry_price=entry_price,
                    exit_price=None,
                    quantity=unit_size,
                    pnl=None,
                    exit_reason=None,
                    unit_number=1
                )
                current_trades = [trade]
                
                if verbose:
                    print(f"{timestamp}: ENTER SHORT @ {entry_price:.2f}, Size: {unit_size:.4f}")
            
            # === Pyramiding Logic ===
            
            # Long pyramiding
            if position_size > 0 and units_count < self.params.max_units:
                pyramid_trigger = last_add_price + (self.params.pyramid_spacing_n * atr)
                if high >= pyramid_trigger:
                    unit_size = self.calculate_unit_size(equity, atr, close)
                    entry_price = self.apply_costs(close, 'long', True)
                    
                    # Update average entry price
                    total_cost = avg_entry_price * position_size + entry_price * unit_size
                    position_size += unit_size
                    avg_entry_price = total_cost / position_size
                    units_count += 1
                    last_add_price = close
                    
                    trade = TradeRecord(
                        entry_time=timestamp,
                        exit_time=None,
                        direction='long',
                        entry_price=entry_price,
                        exit_price=None,
                        quantity=unit_size,
                        pnl=None,
                        exit_reason=None,
                        unit_number=units_count
                    )
                    current_trades.append(trade)
                    
                    if verbose:
                        print(f"{timestamp}: PYRAMID LONG #{units_count} @ {entry_price:.2f}")
            
            # Short pyramiding
            if position_size < 0 and units_count < self.params.max_units:
                pyramid_trigger = last_add_price - (self.params.pyramid_spacing_n * atr)
                if low <= pyramid_trigger:
                    unit_size = self.calculate_unit_size(equity, atr, close)
                    entry_price = self.apply_costs(close, 'short', True)
                    
                    # Update average entry price
                    abs_pos = abs(position_size)
                    total_cost = avg_entry_price * abs_pos + entry_price * unit_size
                    position_size -= unit_size
                    avg_entry_price = total_cost / abs(position_size)
                    units_count += 1
                    last_add_price = close
                    
                    trade = TradeRecord(
                        entry_time=timestamp,
                        exit_time=None,
                        direction='short',
                        entry_price=entry_price,
                        exit_price=None,
                        quantity=unit_size,
                        pnl=None,
                        exit_reason=None,
                        unit_number=units_count
                    )
                    current_trades.append(trade)
                    
                    if verbose:
                        print(f"{timestamp}: PYRAMID SHORT #{units_count} @ {entry_price:.2f}")
            
            # Track equity including unrealized PnL
            if position_size > 0:
                unrealized = position_size * (close - avg_entry_price)
            elif position_size < 0:
                unrealized = abs(position_size) * (avg_entry_price - close)
            else:
                unrealized = 0.0
            
            equity_values.append(equity + unrealized)
            position_values.append(position_size * close if position_size != 0 else 0)
        
        # Close any remaining position at end
        if position_size != 0:
            final_close = df.iloc[-1]['close']
            if position_size > 0:
                exit_price = self.apply_costs(final_close, 'long', False)
                pnl = position_size * (exit_price - avg_entry_price)
            else:
                exit_price = self.apply_costs(final_close, 'short', False)
                pnl = abs(position_size) * (avg_entry_price - exit_price)
            
            equity += pnl
            equity_values[-1] = equity
            
            for trade in current_trades:
                trade.exit_time = df.index[-1]
                trade.exit_price = exit_price
                if trade.direction == 'long':
                    trade.pnl = trade.quantity * (exit_price - trade.entry_price)
                else:
                    trade.pnl = trade.quantity * (trade.entry_price - exit_price)
                trade.exit_reason = "End of Backtest"
                self.trades.append(trade)
        
        # Build results DataFrame
        results = df.copy()
        results['equity'] = equity_values
        results['position_value'] = position_values
        results['returns'] = pd.Series(equity_values).pct_change()
        
        self.equity_curve = results['equity']
        
        return results
    
    def get_trade_stats(self) -> dict:
        """Calculate trading statistics from backtest results"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason,
                'unit_number': t.unit_number
            }
            for t in self.trades
        ])
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Calculate holding periods
        trades_df['holding_period'] = (
            trades_df['exit_time'] - trades_df['entry_time']
        ).dt.total_seconds() / 3600  # In hours
        
        stats = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'profit_factor': (
                abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
            ),
            'avg_holding_hours': trades_df['holding_period'].mean(),
            'trades_by_direction': trades_df.groupby('direction')['pnl'].sum().to_dict(),
            'trades_by_exit_reason': trades_df.groupby('exit_reason')['pnl'].sum().to_dict(),
        }
        
        return stats
    
    def get_equity_stats(self, initial_capital: float = 100_000.0) -> dict:
        """Calculate equity curve statistics"""
        if self.equity_curve is None:
            return {}
        
        equity = self.equity_curve
        returns = equity.pct_change().dropna()
        
        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Calculate time to recovery
        in_drawdown = False
        dd_start = None
        recovery_times = []
        
        for i in range(len(equity)):
            if drawdown.iloc[i] < 0 and not in_drawdown:
                in_drawdown = True
                dd_start = equity.index[i]
            elif drawdown.iloc[i] >= 0 and in_drawdown:
                in_drawdown = False
                if dd_start is not None:
                    recovery_time = (equity.index[i] - dd_start).total_seconds() / (24 * 3600)
                    recovery_times.append(recovery_time)
        
        # Annualized metrics (assuming 4H bars = 6 bars per day)
        bars_per_year = 365 * 6 if '4h' in str(equity.index.freq) else 365 * 24
        total_return = (equity.iloc[-1] / initial_capital) - 1
        years = len(equity) / bars_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
        else:
            sharpe = 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino = returns.mean() / negative_returns.std() * np.sqrt(bars_per_year)
        else:
            sortino = 0
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        stats = {
            'initial_capital': initial_capital,
            'final_equity': equity.iloc[-1],
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'avg_recovery_days': np.mean(recovery_times) if recovery_times else 0,
            'max_recovery_days': max(recovery_times) if recovery_times else 0,
            'volatility_ann_pct': returns.std() * np.sqrt(bars_per_year) * 100,
        }
        
        return stats


if __name__ == "__main__":
    # Quick test
    from data_fetcher import download_btc_data
    
    print("Loading BTC data...")
    df = download_btc_data(timeframe="4h")
    
    print("\nRunning backtest...")
    strategy = TurtleDonchianStrategy()
    results = strategy.run_backtest(df, initial_capital=100_000.0, verbose=False)
    
    print("\n=== TRADE STATISTICS ===")
    trade_stats = strategy.get_trade_stats()
    for key, value in trade_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== EQUITY STATISTICS ===")
    equity_stats = strategy.get_equity_stats()
    for key, value in equity_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

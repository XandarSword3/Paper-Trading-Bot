"""
Bollinger/RSI Mean-Reversion Strategy — a genuinely different hypothesis from
TurtleDonchianStrategy (strategy.py). Where the turtle system bets that
breakouts continue (trend-following), this bets that extreme short-term
deviations from a rolling mean tend to snap back (mean-reversion). Built to
be tested through the SAME out-of-sample discipline as the rest of this repo
(generate_folds / stitch_oos_equity / compute_equity_stats from walk_forward.py,
deflated_sharpe.py for significance) — never optimized or scored on the
frozen holdout (data_splits.get_holdout).

Signal:
  - Rolling SMA (`ma_len`) as the reversion target ("basis"), with a
    Bollinger-style band at +/- `band_mult` standard deviations.
  - RSI(`rsi_len`) as a momentum-exhaustion confirmation filter, so entries
    require both a band touch AND a momentum extreme, not just one.

Entry:
  - Long:  close <= lower_band AND RSI <= rsi_oversold
  - Short: close >= upper_band AND RSI >= rsi_overbought   (skipped if long_only)
  Only ever one open position at a time — no pyramiding. Averaging into a
  losing mean-reversion trade is exactly how these strategies blow up in a
  strong trend, so this deliberately does not do it.

Exit (first of these to trigger, checked every bar):
  - Reversion target hit: close crosses back to the basis (SMA).
  - Hard ATR stop: adverse move of `stop_atr_mult` x ATR from entry. This is
    the strategy's main defense against catching a falling knife in a real
    downtrend/uptrend rather than a genuine range-bound reversion.
  - Time stop: held longer than `max_hold_bars` without reverting — if the
    reversion thesis hasn't played out by then, it's more likely a trend the
    strategy misread as an extreme, not a range.

Position sizing and cost model mirror strategy.py's (equity-at-risk / ATR
stop distance, then commission+slippage on both legs) so results are
comparable on the same footing, not because this reuses that strategy's
logic.
Regime filter (optional, off by default — see `adx_threshold`):
  - ADX(`adx_len`) measures trend strength regardless of direction. Entries
    only fire when ADX <= `adx_threshold`, i.e. the market is range-bound
    rather than trending — mean-reversion's actual edge case. The grid
    search (mr_robustness.py) includes both filtered and effectively
    unfiltered (threshold=100) values, so whether this filter genuinely
    helps out-of-sample is something the walk-forward harness decides per
    fold, not something hand-picked in advance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from metrics_utils import infer_bars_per_year


@dataclass
class MeanReversionParams:
    ma_len: int = 20            # SMA/std lookback for the basis + bands
    band_mult: float = 2.0      # Bollinger band width, in std devs
    rsi_len: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_len: int = 14
    stop_atr_mult: float = 2.5  # hard stop distance, in ATRs from entry
    max_hold_bars: int = 20     # force-exit if reversion hasn't happened
    risk_percent: float = 1.0   # % of equity risked per trade (to the stop)
    long_only: bool = False
    adx_len: int = 14
    adx_threshold: float = 100.0  # >=100 = filter effectively off (ADX rarely exceeds ~60-70)
    lot_step: float = 0.001
    commission_pct: float = 0.08
    slippage_pct: float = 0.05


DEFAULT_MR_PARAMS = MeanReversionParams()


@dataclass
class MRTradeRecord:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    exit_reason: Optional[str]


class BollingerRSIMeanReversion:
    """Single-position (no pyramiding) Bollinger-band + RSI mean-reversion
    strategy with an ATR hard stop and a time stop."""

    def __init__(self, params: MeanReversionParams = None):
        self.params = params or DEFAULT_MR_PARAMS
        self.trades: List[MRTradeRecord] = []
        self.equity_curve: Optional[pd.Series] = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df['basis'] = df['close'].rolling(p.ma_len).mean()
        std = df['close'].rolling(p.ma_len).std()
        df['upper'] = df['basis'] + p.band_mult * std
        df['lower'] = df['basis'] - p.band_mult * std

        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / p.rsi_len, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / p.rsi_len, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)  # no movement yet -> neutral

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(p.atr_len).mean()

        # ADX (Wilder's smoothing) — trend-strength regime filter
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_wilder = true_range.ewm(alpha=1.0 / p.adx_len, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / p.adx_len, adjust=False).mean() / atr_wilder.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / p.adx_len, adjust=False).mean() / atr_wilder.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.ewm(alpha=1.0 / p.adx_len, adjust=False).mean().fillna(0.0)

        return df

    def calculate_size(self, equity: float, atr: float) -> float:
        p = self.params
        if equity <= 0 or atr <= 0:
            return 0.0
        dollar_risk = equity * (p.risk_percent / 100)
        stop_usd = atr * p.stop_atr_mult
        if stop_usd <= 0:
            return 0.0
        raw = dollar_risk / stop_usd
        return max(p.lot_step, round(raw / p.lot_step) * p.lot_step)

    def apply_costs(self, price: float, direction: str, is_entry: bool) -> float:
        p = self.params
        pct = (p.commission_pct + p.slippage_pct) / 100
        if is_entry:
            return price * (1 + pct) if direction == 'long' else price * (1 - pct)
        return price * (1 - pct) if direction == 'long' else price * (1 + pct)

    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 100_000.0,
                      verbose: bool = False) -> pd.DataFrame:
        p = self.params
        df = self.calculate_indicators(df)

        equity = initial_capital
        position_size = 0.0
        direction = None
        entry_price = 0.0
        entry_time = None
        bars_held = 0
        ruined = False

        self.trades = []
        equity_values = []

        warmup = max(p.ma_len, p.rsi_len, p.atr_len, p.adx_len) + 1

        for i in range(len(df)):
            row = df.iloc[i]
            ts = df.index[i]

            if i < warmup:
                equity_values.append(equity)
                continue

            close, high, low = row['close'], row['high'], row['low']
            basis, upper, lower = row['basis'], row['upper'], row['lower']
            rsi, atr = row['rsi'], row['atr']
            adx = row['adx']

            if pd.isna(atr) or pd.isna(basis) or pd.isna(rsi):
                equity_values.append(equity + (
                    position_size * (close - entry_price) if position_size > 0 else
                    abs(position_size) * (entry_price - close) if position_size < 0 else 0.0
                ))
                continue

            if ruined:
                equity_values.append(0.0)
                continue

            # === Manage open position: exits checked before new entries ===
            if position_size != 0:
                bars_held += 1
                exit_price = None
                exit_reason = None

                if direction == 'long':
                    stop_level = entry_price - p.stop_atr_mult * atr
                    if low <= stop_level:
                        exit_price = self.apply_costs(stop_level, 'long', False)
                        exit_reason = 'ATR Stop'
                    elif high >= basis:
                        exit_price = self.apply_costs(basis, 'long', False)
                        exit_reason = 'Mean Reversion Target'
                    elif bars_held >= p.max_hold_bars:
                        exit_price = self.apply_costs(close, 'long', False)
                        exit_reason = 'Time Stop'
                else:  # short
                    stop_level = entry_price + p.stop_atr_mult * atr
                    if high >= stop_level:
                        exit_price = self.apply_costs(stop_level, 'short', False)
                        exit_reason = 'ATR Stop'
                    elif low <= basis:
                        exit_price = self.apply_costs(basis, 'short', False)
                        exit_reason = 'Mean Reversion Target'
                    elif bars_held >= p.max_hold_bars:
                        exit_price = self.apply_costs(close, 'short', False)
                        exit_reason = 'Time Stop'

                if exit_price is not None:
                    if direction == 'long':
                        pnl = position_size * (exit_price - entry_price)
                    else:
                        pnl = abs(position_size) * (entry_price - exit_price)
                    equity += pnl
                    self.trades.append(MRTradeRecord(
                        entry_time=entry_time, exit_time=ts, direction=direction,
                        entry_price=entry_price, exit_price=exit_price,
                        quantity=abs(position_size), pnl=pnl, exit_reason=exit_reason,
                    ))
                    if verbose:
                        print(f"{ts}: EXIT {direction.upper()} @ {exit_price:.2f} "
                              f"({exit_reason}) pnl={pnl:.2f}")
                    position_size = 0.0
                    direction = None
                    bars_held = 0

            # === New entry (only if flat) ===
            if position_size == 0:
                if close <= lower and rsi <= p.rsi_oversold and adx <= p.adx_threshold:
                    size = self.calculate_size(equity, atr)
                    if size > 0:
                        entry_price = self.apply_costs(close, 'long', True)
                        position_size = size
                        direction = 'long'
                        entry_time = ts
                        bars_held = 0
                        if verbose:
                            print(f"{ts}: ENTER LONG @ {entry_price:.2f}")
                elif not p.long_only and close >= upper and rsi >= p.rsi_overbought and adx <= p.adx_threshold:
                    size = self.calculate_size(equity, atr)
                    if size > 0:
                        entry_price = self.apply_costs(close, 'short', True)
                        position_size = -size
                        direction = 'short'
                        entry_time = ts
                        bars_held = 0
                        if verbose:
                            print(f"{ts}: ENTER SHORT @ {entry_price:.2f}")

            unrealized = 0.0
            if position_size > 0:
                unrealized = position_size * (close - entry_price)
            elif position_size < 0:
                unrealized = abs(position_size) * (entry_price - close)

            bar_equity = equity + unrealized
            if bar_equity <= 0:
                ruined = True
                equity = 0.0
                bar_equity = 0.0
                position_size = 0.0
                direction = None
                if verbose:
                    print(f"{ts}: ACCOUNT RUINED — equity hit 0, halting further trading")

            equity_values.append(bar_equity)

        # Close any remaining position at the end of the data
        if position_size != 0:
            final_close = df.iloc[-1]['close']
            if direction == 'long':
                exit_price = self.apply_costs(final_close, 'long', False)
                pnl = position_size * (exit_price - entry_price)
            else:
                exit_price = self.apply_costs(final_close, 'short', False)
                pnl = abs(position_size) * (entry_price - exit_price)
            equity += pnl
            equity_values[-1] = equity
            self.trades.append(MRTradeRecord(
                entry_time=entry_time, exit_time=df.index[-1], direction=direction,
                entry_price=entry_price, exit_price=exit_price,
                quantity=abs(position_size), pnl=pnl, exit_reason='End of Backtest',
            ))

        results = df.copy()
        results['equity'] = equity_values
        self.equity_curve = results['equity']
        return results

    def get_trade_stats(self) -> dict:
        if not self.trades:
            return {}
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time, 'exit_time': t.exit_time, 'direction': t.direction,
            'entry_price': t.entry_price, 'exit_price': t.exit_price, 'quantity': t.quantity,
            'pnl': t.pnl, 'exit_reason': t.exit_reason,
        } for t in self.trades])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'profit_factor': (
                abs(wins['pnl'].sum() / losses['pnl'].sum())
                if len(losses) and losses['pnl'].sum() != 0 else float('inf')
            ),
            'trades_by_exit_reason': trades_df.groupby('exit_reason')['pnl'].sum().to_dict(),
            'trades_by_direction': trades_df.groupby('direction')['pnl'].sum().to_dict(),
        }

    def get_equity_stats(self, initial_capital: float = 100_000.0) -> dict:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return {}
        equity = self.equity_curve
        returns = equity.pct_change().dropna()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        bars_per_year = infer_bars_per_year(equity.index)
        total_return = (equity.iloc[-1] / initial_capital) - 1
        years = len(equity) / bars_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        sharpe = (returns.mean() / returns.std() * np.sqrt(bars_per_year)) if returns.std() > 0 else 0
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'initial_capital': initial_capital,
            'final_equity': equity.iloc[-1],
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
        }

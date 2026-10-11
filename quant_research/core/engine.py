"""
High-Precision Quantitative Backtesting Engine.
Enforces:
1. Signal evaluation on closed bars, fills strictly on next bar open (zero lookahead).
2. Taker fee + slippage cost model (14 bps round trip base, 18 bps stress).
3. Exact historical funding settlement using empirical funding timestamps.
4. Intrabar stop vs. target resolution via 1m kline traversal (pessimistic stop-first fallback).
5. Strict pyramid invariant gating (at most 1 unit entered/added per bar).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter


def assert_no_same_bar_pyramiding(trades: List[Any]):
    """Invariant validator: No two units in a trade list may share the same entry_time."""
    if not trades:
        return
    entry_times = [
        t.entry_time if hasattr(t, "entry_time") else t["entry_time"]
        for t in trades
    ]
    counts = Counter(entry_times)
    duplicates = {ts: c for ts, c in counts.items() if c > 1}
    assert not duplicates, (
        f"INVARIANT VIOLATION: Same-bar pyramid duplication on {len(duplicates)} bar(s): {duplicates}"
    )


@dataclass
class CostModel:
    taker_fee_pct: float = 0.05    # 0.05% per side (5 bps)
    slippage_pct: float = 0.02     # 0.02% per side (2 bps) -> 14 bps round trip
    apply_funding: bool = True
    fill_on_next_open: bool = True # True = open[t+1], False = close[t] (legacy)

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.taker_fee_pct + self.slippage_pct) * 100.0


@dataclass
class EngineTrade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: str = "long"  # 'long' or 'short'
    entry_price: float = 0.0
    executed_entry_price: float = 0.0
    exit_price: Optional[float] = None
    executed_exit_price: Optional[float] = None
    quantity: float = 0.0
    gross_pnl: float = 0.0
    fee_and_slip_cost: float = 0.0
    funding_pnl: float = 0.0
    net_pnl: float = 0.0
    return_bps: float = 0.0
    exit_reason: Optional[str] = None
    unit_number: int = 1


class QuantEngine:
    def __init__(self, cost_model: Optional[CostModel] = None):
        self.costs = cost_model or CostModel()

    @staticmethod
    def position_side(direction: Any) -> float:
        """
        Canonical direction sign: +1 for a long, -1 for a short, 0 for flat.
        Accepts 'long'/'short' (case-insensitive, prefix match) and numeric
        directions (1 / -1 / 0) so every simulator can share one convention.
        """
        if isinstance(direction, str):
            d = direction.strip().lower()
            if d.startswith("long"):
                return 1.0
            if d.startswith("short"):
                return -1.0
            return 0.0
        val = float(direction)
        if val > 0:
            return 1.0
        if val < 0:
            return -1.0
        return 0.0

    def funding_payment(self, direction: Any, quantity: float, close_price: float, funding_rate: float) -> float:
        """
        Cash flow of ONE funding settlement, per PREREG section 3:
        long positions PAY rate * notional, short positions RECEIVE it
        (both reversed when the rate is negative).

        Returns  -side * quantity * close_price * funding_rate
        with side = +1 long / -1 short, and quantity supplied as a positive magnitude.

        NOTE: this is the single source of truth for funding direction. The Turtle
        simulators previously charged  -quantity * close * rate  regardless of
        direction, which made SHORTS pay long-style funding; that defect is fixed
        here and every caller below uses this method.
        """
        side = self.position_side(direction)
        return -side * float(quantity) * float(close_price) * float(funding_rate)

    def apply_execution_costs(self, price: float, direction: str, is_entry: bool) -> float:
        """
        Applies slippage and commission penalty to execution prices.
        """
        slip = self.costs.slippage_pct / 100.0
        fee = self.costs.taker_fee_pct / 100.0
        total_friction = slip + fee

        if is_entry:
            if direction == "long":
                return price * (1.0 + total_friction)
            else:
                return price * (1.0 - total_friction)
        else:
            if direction == "long":
                return price * (1.0 - total_friction)
            else:
                return price * (1.0 + total_friction)

    def calculate_trade_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        funding_pnl: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Returns:
            (exec_entry_price, exec_exit_price, gross_pnl, friction_cost, net_pnl, return_bps)
        """
        exec_entry = self.apply_execution_costs(entry_price, direction, is_entry=True)
        exec_exit = self.apply_execution_costs(exit_price, direction, is_entry=False)

        if direction == "long":
            gross_pnl = quantity * (exit_price - entry_price)
            exec_pnl = quantity * (exec_exit - exec_entry)
        else:
            gross_pnl = quantity * (entry_price - exit_price)
            exec_pnl = quantity * (exec_entry - exec_exit)

        friction_cost = gross_pnl - exec_pnl
        net_pnl = exec_pnl + funding_pnl
        notional = quantity * entry_price
        return_bps = (net_pnl / notional * 10_000.0) if notional > 0 else 0.0

        return exec_entry, exec_exit, gross_pnl, friction_cost, net_pnl, return_bps

    def resolve_intrabar_exit(
        self,
        stop_price: float,
        target_price: Optional[float],
        direction: str,
        bar_high: float,
        bar_low: float,
        sub_bars_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Resolves whether stop loss or take profit was hit first.
        If both touched, inspects sub_bars_df (1m klines).
        If ambiguous or sub_bars_df absent, defaults to conservative STOP LOSS FIRST.
        """
        stop_hit = False
        target_hit = False

        if direction == "long":
            if bar_low <= stop_price:
                stop_hit = True
            if target_price is not None and bar_high >= target_price:
                target_hit = True
        else:
            if bar_high >= stop_price:
                stop_hit = True
            if target_price is not None and bar_low <= target_price:
                target_hit = True

        if stop_hit and not target_hit:
            return stop_price, "Stop Loss"
        if target_hit and not stop_hit:
            return target_price, "Take Profit"
        if not stop_hit and not target_hit:
            return None, None

        # Both touched in this bar!
        if sub_bars_df is not None and len(sub_bars_df) > 0:
            for _, sub in sub_bars_df.iterrows():
                s_high = sub["high"]
                s_low = sub["low"]
                if direction == "long":
                    s_stop = s_low <= stop_price
                    s_target = target_price is not None and s_high >= target_price
                else:
                    s_stop = s_high >= stop_price
                    s_target = target_price is not None and s_low <= target_price

                if s_stop and not s_target:
                    return stop_price, "Stop Loss (1m resolved)"
                if s_target and not s_stop:
                    return target_price, "Take Profit (1m resolved)"
                if s_stop and s_target:
                    # Ambiguous even at 1m -> pessimistic stop loss
                    return stop_price, "Stop Loss (1m same-bar fallback)"

        # Pessimistic default rule: Stop loss hit first
        return stop_price, "Stop Loss (Pessimistic fallback)"

    def resolve_intrabar_exit_levels(
        self,
        stop_price: float,
        level_price: float,
        direction: str,
        bar_high: float,
        bar_low: float,
        sub_bars_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Resolves the ordering of TWO SAME-SIDE levels inside a single bar:
        the protective stop and a second level (e.g. the Donchian channel-exit line).

        Both levels sit on the losing side for the position (a long's stop and its
        channel-exit line are both BELOW price), so resolve_intrabar_exit()'s
        stop/target asymmetry cannot be reused here.

        Returns (exit_price, side, reason) where side is "stop" or "level":
          - exactly one level touched -> that level
          - both touched, sub-bars supplied -> whichever the 1m sub-bars reach first
          - both touched and a single sub-bar touches both -> pessimistic "stop"
          - both touched and no sub-bars            -> pessimistic "stop"
        """
        if direction == "long":
            stop_touched = bar_low <= stop_price
            level_touched = bar_low <= level_price
        else:
            stop_touched = bar_high >= stop_price
            level_touched = bar_high >= level_price

        if stop_touched and not level_touched:
            return stop_price, "stop", "Stop Loss"
        if level_touched and not stop_touched:
            return level_price, "level", "Level"
        if not stop_touched and not level_touched:
            return None, None, None

        # Ambiguous bar: both levels were inside the range.
        if sub_bars_df is not None and len(sub_bars_df) > 0:
            for _, sub in sub_bars_df.iterrows():
                s_high = sub["high"]
                s_low = sub["low"]
                if direction == "long":
                    s_stop = s_low <= stop_price
                    s_level = s_low <= level_price
                else:
                    s_stop = s_high >= stop_price
                    s_level = s_high >= level_price

                if s_stop and not s_level:
                    return stop_price, "stop", "Stop Loss (1m resolved)"
                if s_level and not s_stop:
                    return level_price, "level", "Level (1m resolved)"
                if s_stop and s_level:
                    # Ambiguous even at 1m -> pessimistic stop loss first
                    return stop_price, "stop", "Stop Loss (1m same-bar fallback)"

        # Pessimistic default rule: stop loss hit first
        return stop_price, "stop", "Stop Loss (Pessimistic fallback)"

    def calculate_metrics(
        self,
        trades: List[EngineTrade],
        equity_series: pd.Series,
        initial_capital: float,
        bars_per_year: float = 2190.0,
    ) -> Dict[str, Any]:
        """Calculates institutional performance metrics."""
        if not trades or len(equity_series) < 2:
            return {
                "total_trades": len(trades),
                "total_return_pct": 0.0,
                "expectancy_bps": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "profit_factor": 0.0,
                "win_rate_pct": 0.0,
                "round_trip_cost_bps": self.costs.round_trip_bps,
            }

        total_return_pct = (equity_series.iloc[-1] / initial_capital - 1.0) * 100.0
        net_pnls = [t.net_pnl for t in trades]
        returns_bps = [t.return_bps for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p < 0]

        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        expectancy_bps = np.mean(returns_bps) if returns_bps else 0.0

        # Equity drawdown
        running_max = equity_series.expanding().max()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown_pct = abs(drawdowns.min()) * 100.0

        # Annualized Sharpe of bar returns
        bar_returns = equity_series.pct_change().dropna()
        if len(bar_returns) > 1 and bar_returns.std() > 0:
            sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0

        years = len(equity_series) / max(1.0, bars_per_year)
        cagr_pct = ((equity_series.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and equity_series.iloc[-1] > 0) else -100.0

        return {
            "total_trades": len(trades),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "expectancy_bps": round(expectancy_bps, 2),
            "profit_factor": round(profit_factor, 2),
            "win_rate_pct": round(win_rate, 2),
            "round_trip_cost_bps": round(self.costs.round_trip_bps, 1),
            "final_equity": round(equity_series.iloc[-1], 2),
        }

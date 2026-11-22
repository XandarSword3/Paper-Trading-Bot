"""OmniTrader: High-speed automated execution engine with dynamic risk controls.

Manages real-time multi-coin positions with automatic trailing profit stops,
hard loss guards, time-decay stops, and realistic exchange fee deductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

from strategy.omni_scanner import OmniSignal
from data.omni_feed import OmniFeed

logger = logging.getLogger("omni_trader")


@dataclass
class Position:
    symbol: str
    side: str            # "Buy" (Long) or "Sell" (Short)
    entry_price: float
    current_price: float
    quantity: float
    notional: float
    entry_time: float
    stop_loss: float
    take_profit: float
    peak_pnl_pct: float = 0.0
    trailing_active: bool = False
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: float
    exit_time: float
    gross_pnl: float
    fee_cost: float
    net_pnl: float
    net_pnl_pct: float
    exit_reason: str


@dataclass
class RiskConfig:
    max_positions: int = 5
    risk_per_trade_pct: float = 0.015       # 1.5% capital risk per trade
    max_position_notional_pct: float = 0.20 # 20% max notional per position
    hard_stop_loss_pct: float = 0.015       # 1.5% max loss
    take_profit_pct: float = 0.035          # 3.5% initial target
    trailing_start_pct: float = 0.010       # Activate trailing at +1.0% profit
    trailing_distance_pct: float = 0.005    # Trail 0.5% behind highest profit peak
    max_hold_seconds: float = 1800.0        # 30-minute time-stop
    fee_rate_bps: float = 5.0               # 5 bps taker fee
    slippage_bps: float = 3.0               # 3 bps realistic altcoin slippage


class OmniTrader:
    """Real-time paper trading engine executing multi-coin high-velocity setups."""

    def __init__(
        self,
        feed: OmniFeed,
        initial_capital: float = 10_000.0,
        risk_config: Optional[RiskConfig] = None,
        state_file: Optional[Path] = None,
    ) -> None:
        self.feed = feed
        self.risk = risk_config or RiskConfig()
        self.state_file = state_file or Path("logs/trendlab/omni_paper_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []

        self.total_signals_received: int = 0
        self.total_orders_executed: int = 0

    @property
    def equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.open_positions.values())
        return self.cash + unrealized

    def process_signal(self, signal: OmniSignal) -> Optional[Position]:
        """Evaluates an inbound signal against risk limits and opens position if approved."""
        self.total_signals_received += 1

        if len(self.open_positions) >= self.risk.max_positions:
            return None
        if signal.symbol in self.open_positions:
            return None
        if signal.price <= 0:
            return None

        # Calculate position sizing
        current_equity = self.equity
        notional_allocation = current_equity * self.risk.max_position_notional_pct
        qty = notional_allocation / signal.price

        # Fee deductions for opening order (taker + slippage)
        open_fee = notional_allocation * ((self.risk.fee_rate_bps + self.risk.slippage_bps) / 10000.0)
        self.cash -= open_fee
        self.total_fees_paid += open_fee

        now = time.time()
        if signal.side == "Buy":
            stop_px = signal.price * (1.0 - self.risk.hard_stop_loss_pct)
            tp_px = signal.price * (1.0 + self.risk.take_profit_pct)
        else:
            stop_px = signal.price * (1.0 + self.risk.hard_stop_loss_pct)
            tp_px = signal.price * (1.0 - self.risk.take_profit_pct)

        pos = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.price,
            current_price=signal.price,
            quantity=qty,
            notional=notional_allocation,
            entry_time=now,
            stop_loss=stop_px,
            take_profit=tp_px,
        )

        self.open_positions[signal.symbol] = pos
        self.total_orders_executed += 1
        logger.info("OPENED %s %s: qty=%.4f @ %.4f ($%.1f notional) | SL: %.4f, TP: %.4f",
                    pos.side, pos.symbol, pos.quantity, pos.entry_price, pos.notional, pos.stop_loss, pos.take_profit)
        return pos

    def update_positions(self) -> List[ClosedTrade]:
        """Updates prices, adjusts trailing stops, and exits positions hitting thresholds."""
        now = time.time()
        to_close: List[tuple[Position, str, float]] = []

        for sym, pos in list(self.open_positions.items()):
            snap = self.feed.latest.get(sym)
            if not snap or snap.last_price <= 0:
                continue

            current_px = snap.last_price
            pos.current_price = current_px

            # Compute current return
            if pos.side == "Buy":
                pnl_pct = (current_px - pos.entry_price) / pos.entry_price
                gross_pnl = (current_px - pos.entry_price) * pos.quantity
            else:
                pnl_pct = (pos.entry_price - current_px) / pos.entry_price
                gross_pnl = (pos.entry_price - current_px) * pos.quantity

            pos.unrealized_pnl = gross_pnl
            pos.peak_pnl_pct = max(pos.peak_pnl_pct, pnl_pct)

            # 1. Trailing Stop Management
            if not pos.trailing_active and pnl_pct >= self.risk.trailing_start_pct:
                pos.trailing_active = True
                logger.info("TRAILING ACTIVATED on %s at +%.2f%% profit", sym, pnl_pct * 100)

            if pos.trailing_active:
                if pos.side == "Buy":
                    # Stop trails below highest recorded price
                    trailing_stop = pos.entry_price * (1.0 + (pos.peak_pnl_pct - self.risk.trailing_distance_pct))
                    pos.stop_loss = max(pos.stop_loss, trailing_stop)
                else:
                    trailing_stop = pos.entry_price * (1.0 - (pos.peak_pnl_pct - self.risk.trailing_distance_pct))
                    pos.stop_loss = min(pos.stop_loss, trailing_stop)

            # 2. Check Exit Conditions
            exit_reason: Optional[str] = None
            if pos.side == "Buy":
                if current_px <= pos.stop_loss:
                    exit_reason = "Trailing Stop" if pos.trailing_active else "Stop Loss"
                elif current_px >= pos.take_profit:
                    exit_reason = "Take Profit Target"
            else:
                if current_px >= pos.stop_loss:
                    exit_reason = "Trailing Stop" if pos.trailing_active else "Stop Loss"
                elif current_px <= pos.take_profit:
                    exit_reason = "Take Profit Target"

            # 3. Check Time Stop
            if not exit_reason and (now - pos.entry_time) >= self.risk.max_hold_seconds:
                exit_reason = "Time Stop (Max Hold Expired)"

            if exit_reason:
                to_close.append((pos, exit_reason, current_px))

        closed_list: List[ClosedTrade] = []
        for pos, reason, exit_px in to_close:
            closed = self._close_position(pos, reason, exit_px)
            closed_list.append(closed)

        return closed_list

    def _close_position(self, pos: Position, reason: str, exit_px: float) -> ClosedTrade:
        now = time.time()
        if pos.side == "Buy":
            gross_pnl = (exit_px - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_px) * pos.quantity

        close_notional = exit_px * pos.quantity
        close_fee = close_notional * ((self.risk.fee_rate_bps + self.risk.slippage_bps) / 10000.0)
        net_pnl = gross_pnl - close_fee
        net_pnl_pct = net_pnl / pos.notional

        self.cash += gross_pnl - close_fee
        self.realized_pnl += net_pnl
        self.total_fees_paid += close_fee

        closed = ClosedTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=now,
            gross_pnl=gross_pnl,
            fee_cost=close_fee,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            exit_reason=reason,
        )

        del self.open_positions[pos.symbol]
        self.closed_trades.append(closed)
        self.save_state()

        logger.info(
            "CLOSED %s %s: net PnL: %+.2f USDT (%+.2f%%) | Reason: %s (Duration: %.0fs)",
            closed.side, closed.symbol, closed.net_pnl, closed.net_pnl_pct * 100,
            closed.exit_reason, closed.exit_time - closed.entry_time,
        )
        return closed

    def get_summary(self) -> dict:
        wins = [t for t in self.closed_trades if t.net_pnl > 0]
        losses = [t for t in self.closed_trades if t.net_pnl <= 0]
        win_rate = (len(wins) / len(self.closed_trades) * 100.0) if self.closed_trades else 0.0

        return {
            "initial_capital": self.initial_capital,
            "equity": self.equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "total_fees_paid": self.total_fees_paid,
            "open_positions": len(self.open_positions),
            "closed_trades": len(self.closed_trades),
            "win_rate_pct": win_rate,
            "signals_detected": self.total_signals_received,
        }

    def save_state(self) -> None:
        state = self.get_summary()
        state["open_positions_detail"] = [
            {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "quantity": p.quantity,
                "notional": p.notional,
                "unrealized_pnl": p.unrealized_pnl,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "trailing_active": p.trailing_active,
                "hold_seconds": time.time() - p.entry_time,
            }
            for p in self.open_positions.values()
        ]
        state["recent_trades"] = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "net_pnl": t.net_pnl,
                "net_pnl_pct": t.net_pnl_pct,
                "exit_reason": t.exit_reason,
                "duration_seconds": t.exit_time - t.entry_time,
            }
            for t in self.closed_trades[-20:]
        ]
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

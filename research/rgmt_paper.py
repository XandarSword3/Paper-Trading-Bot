"""Pure accounting for the RGMT daily paper trader.

The daily runner used to mix market data, state mutation and reporting in one
function.  That made the money math impossible to test and hid several real
bugs (no funding, no drawdown enforcement, wrong average entry on adds).  This
module is deliberately I/O-free: it takes plain dictionaries and returns new
ones, so every accounting rule can be unit-tested without a network or a clock.

Conventions
-----------
* ``positions`` maps symbol -> signed quantity (positive long, negative short).
* ``entry_prices`` maps symbol -> average entry price.
* ``cash`` is the perp wallet balance; equity = cash + unrealised PnL.
* Funding is charged on signed notional: a long pays positive funding, a short
  receives it.  This matches the research engine in ``research/trendlab``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

# Frozen research parameters.  Do not tune these in the live path.
MAX_GROSS = 3.0
MAX_ASSET_WEIGHT = 0.15
MAX_DRAWDOWN_GUARD = 0.30
FEE_BPS = 5.0
SLIPPAGE_BPS = 3.0
COST_RATE = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
MIN_ORDER_NOTIONAL = 10.0
MIN_WEIGHT_DELTA = 0.0005


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str
    qty: float
    price: float
    notional: float
    fee: float
    realized_pnl: float
    target_weight: float

    def as_dict(self) -> dict:
        return asdict(self)


def new_state(initial_capital: float = 10_000.0) -> dict:
    capital = float(initial_capital)
    return {
        "strategy": "RGMT",
        "initial_capital": capital,
        "cash": capital,
        "equity": capital,
        "peak_equity": capital,
        "realized_pnl": 0.0,
        "total_fees_paid": 0.0,
        "total_funding_cost": 0.0,
        "positions": {},
        "entry_prices": {},
        "runs": 0,
        "history": [],
        "trades": [],
        "halted": False,
        "halt_reason": None,
        "halted_at": None,
        "last_run": None,
        "last_regime": None,
    }


def _finite_positive(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _mark_price(
    symbol: str,
    prices: Mapping[str, float],
    fallback_prices: Optional[Mapping[str, float]],
) -> Optional[float]:
    price = _finite_positive(prices.get(symbol))
    if price is None and fallback_prices is not None:
        price = _finite_positive(fallback_prices.get(symbol))
    return price


def mark_to_market(
    state: Mapping,
    prices: Mapping[str, float],
    fallback_prices: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Return ``(equity, unrealised, marks)`` without mutating ``state``."""
    unrealised = 0.0
    marks: Dict[str, float] = {}
    for symbol, raw_qty in state.get("positions", {}).items():
        qty = float(raw_qty)
        if abs(qty) < 1e-12:
            continue
        price = _mark_price(symbol, prices, fallback_prices)
        if price is None:
            continue
        marks[symbol] = price
        entry = float(state.get("entry_prices", {}).get(symbol, price))
        unrealised += qty * (price - entry)
    equity = float(state.get("cash", 0.0)) + unrealised
    return equity, unrealised, marks


def accrue_funding(
    state: Mapping,
    funding_rates: Mapping[str, float],
    prices: Mapping[str, float],
    fallback_prices: Optional[Mapping[str, float]] = None,
) -> Tuple[dict, float]:
    """Charge/credit funding on signed notional.

    Returns a new state and the net funding cost (positive = paid).  A short
    position with positive funding produces a negative cost (a rebate).
    """
    new = deepcopy(dict(state))
    total = 0.0
    for symbol, raw_qty in new.get("positions", {}).items():
        qty = float(raw_qty)
        if abs(qty) < 1e-12:
            continue
        try:
            rate = float(funding_rates.get(symbol, 0.0) or 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        if rate == 0.0 or not math.isfinite(rate):
            continue
        price = _mark_price(symbol, prices, fallback_prices)
        if price is None:
            continue
        total += qty * price * rate
    new["cash"] = float(new.get("cash", 0.0)) - total
    new["total_funding_cost"] = float(new.get("total_funding_cost", 0.0)) + total
    return new, total


def rebalance(
    state: Mapping,
    targets: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    *,
    cost_rate: float = COST_RATE,
    min_order_notional: float = MIN_ORDER_NOTIONAL,
    min_weight_delta: float = MIN_WEIGHT_DELTA,
    fallback_prices: Optional[Mapping[str, float]] = None,
) -> Tuple[dict, list[Order]]:
    """Move positions to ``targets`` and return the new state plus orders.

    The realised PnL and fee rules are the important part:

    * full close: realise ``qty * (price - entry)``;
    * partial close: realise only the closed slice, keep the original entry;
    * increase: use the notional-weighted average entry;
    * flip: realise the old position and reset the entry to the fill price.

    Fees are charged on the traded notional (close + open for a flip).
    """
    new = deepcopy(dict(state))
    orders: list[Order] = []
    positions: dict = new.setdefault("positions", {})
    entries: dict = new.setdefault("entry_prices", {})
    threshold = max(float(min_order_notional), float(min_weight_delta) * float(equity))

    for symbol in sorted(set(targets) | set(positions)):
        target_weight = float(targets.get(symbol, 0.0) or 0.0)
        old_qty = float(positions.get(symbol, 0.0) or 0.0)
        price = _mark_price(symbol, prices, fallback_prices)
        if price is None:
            # Cannot price the symbol.  Leave it alone; the caller reports it.
            continue

        target_qty = target_weight * float(equity) / price
        delta = target_qty - old_qty
        notional = abs(delta) * price
        if notional < threshold:
            continue

        fee = notional * float(cost_rate)
        entry = float(entries.get(symbol, price))
        realized = 0.0

        if old_qty != 0.0 and target_qty != 0.0 and (old_qty > 0.0) != (target_qty > 0.0):
            realized = old_qty * (price - entry)
            entry = price
        elif target_qty == 0.0:
            realized = old_qty * (price - entry)
        elif abs(target_qty) < abs(old_qty):
            closed = old_qty - target_qty
            realized = closed * (price - entry)
        else:
            if old_qty == 0.0:
                entry = price
            else:
                entry = (abs(old_qty) * entry + abs(delta) * price) / abs(target_qty)

        new["cash"] = float(new.get("cash", 0.0)) + realized - fee
        new["realized_pnl"] = float(new.get("realized_pnl", 0.0)) + realized
        new["total_fees_paid"] = float(new.get("total_fees_paid", 0.0)) + fee

        if target_qty == 0.0:
            positions.pop(symbol, None)
            entries.pop(symbol, None)
        else:
            positions[symbol] = target_qty
            entries[symbol] = entry

        orders.append(
            Order(
                symbol=symbol,
                side="BUY" if delta > 0.0 else "SELL",
                qty=abs(delta),
                price=price,
                notional=notional,
                fee=fee,
                realized_pnl=realized,
                target_weight=target_weight,
            )
        )

    return new, orders


def validate_targets(
    targets: Mapping[str, float],
    *,
    max_gross: float = MAX_GROSS,
    max_asset_weight: float = MAX_ASSET_WEIGHT,
    tolerance: float = 1e-6,
) -> Tuple[float, float]:
    """Fail closed if a target set violates the frozen research limits."""
    gross = 0.0
    worst = 0.0
    for raw in targets.values():
        weight = abs(float(raw or 0.0))
        gross += weight
        worst = max(worst, weight)
    if gross > max_gross + tolerance:
        raise ValueError(f"target gross {gross:.6f} exceeds max_gross {max_gross}")
    if worst > max_asset_weight + tolerance:
        raise ValueError(
            f"target asset weight {worst:.6f} exceeds max_asset_weight {max_asset_weight}"
        )
    return gross, worst


def update_risk(
    state: Mapping,
    *,
    guard: float = MAX_DRAWDOWN_GUARD,
) -> Tuple[dict, float, bool]:
    """Update peak equity and drawdown.  Returns (state, drawdown, breached).

    ``breached`` is True only on the transition into a halt, so callers can
    alert once instead of every day.
    """
    new = deepcopy(dict(state))
    equity = float(new.get("equity", 0.0))
    peak = max(float(new.get("peak_equity", equity)), equity)
    new["peak_equity"] = peak
    drawdown = 0.0 if peak <= 0.0 else max(1.0 - equity / peak, 0.0)
    breached = False
    if drawdown >= guard and not new.get("halted", False):
        new["halted"] = True
        new["halt_reason"] = f"max_drawdown_guard:{drawdown:.4f}"
        breached = True
    return new, drawdown, breached


def flatten_targets(symbols: Iterable[str]) -> Dict[str, float]:
    return {str(symbol): 0.0 for symbol in symbols}


def already_ran_today(last_run: Optional[str], now: Optional[datetime] = None) -> bool:
    """True when ``last_run`` is on the same UTC calendar day as ``now``."""
    if not last_run:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(last_run).replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date() == current.astimezone(timezone.utc).date()


def append_trade(state: dict, order: Order, timestamp: str, equity_after: float) -> None:
    """Append an auditable trade record, capped to keep the state file small."""
    trades = state.setdefault("trades", [])
    trades.append(
        {
            "time": timestamp,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": order.price,
            "notional": order.notional,
            "fee": order.fee,
            "realized_pnl": order.realized_pnl,
            "target_weight": order.target_weight,
            "equity_after": equity_after,
        }
    )
    if len(trades) > 5_000:
        del trades[:-5_000]


def append_history(state: dict, entry: dict, cap: int = 2_000) -> None:
    history = state.setdefault("history", [])
    history.append(entry)
    if len(history) > cap:
        del history[:-cap]

"""
Classical Turtle / Donchian Breakout Strategy Implementation.
Implements:
1. Donchian Channel Breakouts: N-bar entry, M-bar exit.
2. Wilders ATR(N) volatility sizing and 2N initial stop.
3. Pyramiding: Up to 4 units maximum, spaced at 0.5N intervals with stop ratcheting.
4. Strict invariant: At most 1 unit added per candle bar (assert_no_same_bar_pyramiding).
5. 2.0x portfolio gross notional leverage cap.
6. Execution: Fills strictly at next bar open (open[t+1]) with 14 bps RT base / 18 bps RT stress.
7. Exact empirical funding settlements.
8. 1m sub-bar intrabar resolution for candle exits.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from quant_research.core.engine import CostModel, QuantEngine, EngineTrade, assert_no_same_bar_pyramiding
from quant_research.core.metrics import calculate_daily_sharpe


def prepare_turtle_features(df: pd.DataFrame, N: int, M: int) -> Dict[str, Any]:
    """
    Computes lookahead-free Donchian breakout channels and Wilders ATR.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_p = df["open"]

    # Entry channels: prior N bars strictly (excluding current bar)
    entry_high = high.shift(1).rolling(N).max()
    entry_low = low.shift(1).rolling(N).min()

    # Exit channels: prior M bars strictly (excluding current bar)
    exit_high = high.shift(1).rolling(M).max()
    exit_low = low.shift(1).rolling(M).min()

    # True Range & Wilders ATR(N)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilders smoothing: alpha = 1 / N
    atr_n = tr.ewm(alpha=1.0 / float(N), adjust=False).mean()

    return {
        "open": open_p.to_numpy(dtype=np.float64),
        "high": high.to_numpy(dtype=np.float64),
        "low": low.to_numpy(dtype=np.float64),
        "close": close.to_numpy(dtype=np.float64),
        "entry_high": entry_high.to_numpy(dtype=np.float64),
        "entry_low": entry_low.to_numpy(dtype=np.float64),
        "exit_high": exit_high.to_numpy(dtype=np.float64),
        "exit_low": exit_low.to_numpy(dtype=np.float64),
        "atr": atr_n.to_numpy(dtype=np.float64),
        "timestamps": df.index,
    }


def run_turtle_simulation(
    features: Dict[str, Any],
    funding_series: Optional[pd.Series],
    params: Dict[str, Any],
    cost_model: CostModel,
    initial_capital: float = 100_000.0,
    risk_percent: float = 1.0,
    max_units: int = 4,
    bars_per_year: float = 105120.0,
    df_1m: Optional[pd.DataFrame] = None,
    return_trades: bool = False,
) -> Any:
    """
    Simulates Classical Turtle trading with 4-unit pyramiding and ratcheting stops.
    """
    engine = QuantEngine(cost_model)

    open_arr = features["open"]
    high_arr = features["high"]
    low_arr = features["low"]
    close_arr = features["close"]
    entry_high_arr = features["entry_high"]
    entry_low_arr = features["entry_low"]
    exit_high_arr = features["exit_high"]
    exit_low_arr = features["exit_low"]
    atr_arr = features["atr"]
    timestamps = features["timestamps"]
    n_bars = len(close_arr)

    N = int(params["N"])
    M = int(params["M"])
    warmup = max(N, M, 30)

    # Pre-align funding rates
    if funding_series is not None and cost_model.apply_funding:
        f_df = funding_series.to_frame("funding_rate")
        f_df["t_round"] = f_df.index.round("1min")
        f_map = f_df.groupby("t_round")["funding_rate"].sum()
        funding_arr = f_map.reindex(timestamps, fill_value=0.0).to_numpy(dtype=np.float64)
    else:
        funding_arr = np.zeros(n_bars, dtype=np.float64)

    # Pre-align 1m sub-bars if passed
    has_subbars = df_1m is not None and len(df_1m) > 0
    sub_index = df_1m.index if has_subbars else None
    ambiguous_bars = 0

    equity = initial_capital
    units: List[Dict[str, Any]] = []  # active units
    direction = 0  # 1: long, -1: short, 0: flat
    active_stop = 0.0
    funding_accrued = 0.0
    last_action_bar = -1  # for assert_no_same_bar_pyramiding

    completed_trades: List[EngineTrade] = []
    equity_curve = [equity]
    # Marked-to-market equity (cash + unrealized PnL of open units at the
    # bar close) — accounting output ONLY. ``equity`` itself is untouched:
    # position sizing and the leverage cap keep using cash equity exactly
    # as before, so strategy rules are unchanged.
    marked_equity_curve = [equity]

    def _unrealized(close_px: float) -> float:
        if direction == 0 or not units:
            return 0.0
        return float(sum(
            (close_px - u["entry_price"]) * u["quantity"] * direction
            for u in units))

    def _append_equity(close_px: float) -> None:
        equity_curve.append(equity)
        marked_equity_curve.append(equity + _unrealized(close_px))

    for i in range(n_bars - 1):
        if equity <= 5.0:
            # Portfolio ruined (< $10 notional order capacity at 2x leverage)
            _append_equity(close_arr[i])
            continue

        ts = timestamps[i]
        next_open = open_arr[i + 1]
        next_ts = timestamps[i + 1]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        atr = atr_arr[i]

        if i < warmup or np.isnan(atr) or np.isnan(entry_high_arr[i]):
            _append_equity(close)
            continue

        # 1. Accrue funding if holding units (direction-aware: PREREG section 3)
        if direction != 0 and len(units) > 0:
            f_rate = funding_arr[i]
            if f_rate != 0.0:
                tot_pos = sum(u["quantity"] for u in units)
                payment = engine.funding_payment(direction, tot_pos, close, f_rate)
                equity += payment
                funding_accrued += payment

        # 2. Check Exits for active units
        exit_triggered = False
        exit_price = 0.0
        exit_reason = ""

        if direction != 0 and len(units) > 0:
            # A. Intrabar Stop Loss Check
            stop_hit = False
            if direction == 1 and low <= active_stop:
                stop_hit = True
            elif direction == -1 and high >= active_stop:
                stop_hit = True

            # B. Closed-bar Channel Exit Check
            channel_hit = False
            if direction == 1 and close < exit_low_arr[i]:
                channel_hit = True
            elif direction == -1 and close > exit_high_arr[i]:
                channel_hit = True

            if stop_hit and channel_hit:
                # AMBIGUOUS BAR: the protective stop AND the M-bar channel-exit line
                # both lie inside this bar's range.
                ambiguous_bars += 1
                channel_level = exit_low_arr[i] if direction == 1 else exit_high_arr[i]
                dir_str = "long" if direction == 1 else "short"

                sub_bars = None
                if has_subbars:
                    lo = int(np.searchsorted(sub_index, ts, side="left"))
                    hi = int(np.searchsorted(sub_index, next_ts, side="left"))
                    if hi > lo:
                        sub_bars = df_1m.iloc[lo:hi]

                if sub_bars is not None and len(sub_bars) > 0:
                    # 1m sub-bars are handed to the exit logic; the first level they
                    # actually reach decides the exit.
                    _, side, _ = engine.resolve_intrabar_exit_levels(
                        stop_price=active_stop,
                        level_price=channel_level,
                        direction=dir_str,
                        bar_high=high,
                        bar_low=low,
                        sub_bars_df=sub_bars,
                    )
                    if side == "level":
                        exit_triggered = True
                        exit_price = next_open
                        exit_reason = f"{M}-Bar Donchian Exit (1m resolved)"
                    else:
                        exit_triggered = True
                        exit_price = active_stop
                        exit_reason = "2N Stop Loss (1m resolved)"
                else:
                    # No sub-bars available -> pessimistic stop-loss-first fallback
                    exit_triggered = True
                    exit_price = active_stop
                    exit_reason = "2N Stop Loss"
            elif stop_hit:
                # Stop loss hit intrabar!
                exit_triggered = True
                exit_price = active_stop
                exit_reason = "2N Stop Loss"
            elif channel_hit:
                # Channel breakout exit triggered at close, fills at next open
                exit_triggered = True
                exit_price = next_open
                exit_reason = f"{M}-Bar Donchian Exit"

            if exit_triggered:
                # Close all active units
                unit_funding = funding_accrued / len(units) if len(units) > 0 else 0.0
                dir_str = "long" if direction == 1 else "short"

                for u in units:
                    exec_entry, exec_exit, gross, cost, net, bps = engine.calculate_trade_pnl(
                        entry_price=u["entry_price"],
                        exit_price=exit_price,
                        quantity=u["quantity"],
                        direction=dir_str,
                        funding_pnl=unit_funding,
                    )
                    equity += net
                    completed_trades.append(
                        EngineTrade(
                            entry_time=u["entry_time"],
                            exit_time=ts if exit_reason.startswith("2N Stop Loss") else next_ts,
                            direction=dir_str,
                            entry_price=u["entry_price"],
                            executed_entry_price=exec_entry,
                            exit_price=exit_price,
                            executed_exit_price=exec_exit,
                            quantity=u["quantity"],
                            gross_pnl=gross,
                            fee_and_slip_cost=cost,
                            funding_pnl=unit_funding,
                            net_pnl=net,
                            return_bps=bps,
                            exit_reason=exit_reason,
                            unit_number=u["unit_num"],
                        )
                    )

                units = []
                direction = 0
                active_stop = 0.0
                funding_accrued = 0.0

        # 3. Check Entries & Pyramids (only if no exit occurred on this bar)
        if not exit_triggered:
            # Sizing calculation: 1% risk per unit, stop distance = 2.0 * ATR
            risk_usd = equity * (risk_percent / 100.0)
            stop_dist = 2.0 * atr
            unit_qty = (risk_usd / stop_dist) if stop_dist > 0 else 0.0

            # Leverage cap: Max portfolio notional <= 2.0 * equity
            cur_notional = sum(u["quantity"] * next_open for u in units)
            max_add_notional = max(0.0, (equity * 2.0) - cur_notional)
            max_unit_qty = max_add_notional / next_open if next_open > 0 else 0.0
            unit_qty = min(unit_qty, max_unit_qty)

            # Minimum order size $10
            viable_order = (unit_qty * next_open) >= 10.0

            # CASE A: Flat -> Check initial entry signal on closed bar
            if direction == 0 and viable_order and i > last_action_bar:
                long_signal = close > entry_high_arr[i]
                short_signal = close < entry_low_arr[i]

                if long_signal:
                    direction = 1
                    active_stop = next_open - stop_dist
                    units.append({
                        "unit_num": 1,
                        "entry_price": next_open,
                        "entry_time": next_ts,
                        "quantity": unit_qty,
                    })
                    last_action_bar = i + 1

                elif short_signal:
                    direction = -1
                    active_stop = next_open + stop_dist
                    units.append({
                        "unit_num": 1,
                        "entry_price": next_open,
                        "entry_time": next_ts,
                        "quantity": unit_qty,
                    })
                    last_action_bar = i + 1

            # CASE B: Open position -> Check Pyramid additions (up to max_units units;
            # max_units=1 disables pyramiding entirely. The default 4 preserves the
            # committed section-7 behaviour so existing runners are untouched.)
            elif direction != 0 and len(units) < max_units and viable_order and i > last_action_bar:
                last_unit = units[-1]
                last_price = last_unit["entry_price"]
                pyramid_step = 0.5 * atr

                add_unit = False
                if direction == 1 and close >= (last_price + pyramid_step):
                    add_unit = True
                elif direction == -1 and close <= (last_price - pyramid_step):
                    add_unit = True

                if add_unit:
                    new_unit_num = len(units) + 1
                    units.append({
                        "unit_num": new_unit_num,
                        "entry_price": next_open,
                        "entry_time": next_ts,
                        "quantity": unit_qty,
                    })
                    # Ratchet stop for ALL units: 2N from newest unit entry price
                    if direction == 1:
                        active_stop = next_open - stop_dist
                    else:
                        active_stop = next_open + stop_dist

                    last_action_bar = i + 1

        _append_equity(close)

    # Invariant validator
    assert_no_same_bar_pyramiding(completed_trades)

    # Calculate metrics. The equity curve is indexed by the bar timestamps so
    # that the canonical daily-equity Sharpe (calculate_daily_sharpe, sqrt(365))
    # is well defined; the legacy per-bar Sharpe is left untouched for
    # comparability with previously committed numbers.
    eq_series = pd.Series(
        np.asarray(equity_curve, dtype=np.float64),
        index=pd.DatetimeIndex(timestamps[: len(equity_curve)]),
    )
    metrics = engine.calculate_metrics(
        trades=completed_trades,
        equity_series=eq_series,
        initial_capital=initial_capital,
        bars_per_year=bars_per_year,
    )
    metrics["daily_sharpe"] = round(calculate_daily_sharpe(eq_series), 3)
    metrics["final_equity"] = round(equity, 2)
    # Marked-to-market drawdown: includes unrealized PnL of open units,
    # fees/slippage/funding exactly once (all realized in ``equity``).
    marked_series = pd.Series(
        np.asarray(marked_equity_curve, dtype=np.float64),
        index=pd.DatetimeIndex(timestamps[: len(marked_equity_curve)]),
    )
    marked_dd = (marked_series - marked_series.expanding().max()) \
        / marked_series.expanding().max()
    metrics["max_drawdown_marked_pct"] = round(
        float(abs(marked_dd.min()) * 100.0), 4)
    metrics["final_marked_equity"] = round(float(marked_series.iloc[-1]), 2)
    metrics["ambiguous_bars"] = ambiguous_bars
    metrics["config"] = params

    return (metrics, completed_trades) if return_trades else metrics

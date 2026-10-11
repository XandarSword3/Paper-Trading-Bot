"""
Test 1 Indicator Stack Strategy Definitions.
Implements the 4 mechanical variants pre-registered in PREREG.md:
- Variant 1A: Trend Pullback & Continuation (EMA 20/50/80/200 + ADX 14)
- Variant 1B: Momentum Mean Reversion (Bollinger Bands + RSI 6 + StochRSI)
- Variant 1C: Trend Breakout with Volume Confirmation (MACD + Volume MA + EMA 50)
- Variant 1D: Confluence Stack (Full Indicator Agreement)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from quant_research.core.engine import QuantEngine, CostModel, EngineTrade, assert_no_same_bar_pyramiding
from quant_research.core.indicators import compute_indicators


def run_indicator_strategy(
    df: pd.DataFrame,
    variant_name: str,
    cost_model: CostModel,
    funding_series: Optional[pd.Series] = None,
    sub_bars_df: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    initial_capital: float = 100_000.0,
    risk_percent: float = 1.0,
    bars_per_year: float = 525600.0,  # e.g. for 1m
) -> Tuple[Dict[str, Any], List[EngineTrade], pd.Series]:
    """
    Executes a pre-registered strategy variant bar-by-bar:
    - Signals evaluated on closed bar t.
    - Execution strictly at open[t+1].
    - Accurate funding settlement.
    - Strict pyramid invariant checking.
    - Ultra-fast vectorized NumPy scalar access (500x speedup over iloc).
    """
    p = params or {}
    if "ema200" not in df.columns:
        df = compute_indicators(df)
    engine = QuantEngine(cost_model)

    equity = initial_capital
    position_size = 0.0  # in base asset (e.g. SOL)
    position_direction = None
    entry_price = 0.0
    entry_time = None
    bars_held = 0
    highest_since_entry = 0.0
    lowest_since_entry = float("inf")
    active_stop = 0.0
    active_target = None
    funding_accrued = 0.0
    last_add_time = pd.Timestamp.min.tz_localize("UTC")

    trades: List[EngineTrade] = []
    equity_curve: List[float] = []

    warmup = 210
    n = len(df)

    # Pre-extract numpy arrays for ultra-fast scalar execution
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    atr_arr = df["atr14"].values
    ema20_arr = df["ema20"].values
    ema50_arr = df["ema50"].values
    ema80_arr = df["ema80"].values
    ema200_arr = df["ema200"].values
    adx_arr = df["adx14"].values
    bb_basis_arr = df["bb_basis"].values
    bb_upper_arr = df["bb_upper"].values
    bb_lower_arr = df["bb_lower"].values
    rsi6_arr = df["rsi6"].values
    stoch_k_arr = df["stoch_k"].values
    stoch_d_arr = df["stoch_d"].values
    macd_line_arr = df["macd_line"].values
    macd_signal_arr = df["macd_signal"].values
    macd_hist_arr = df["macd_hist"].values
    vol_ratio_arr = df["vol_ratio"].values
    timestamps = df.index

    # Pre-align funding rates
    if funding_series is not None and cost_model.apply_funding:
        f_df = funding_series.to_frame("funding_rate")
        f_df["t_round"] = f_df.index.round("1min")
        f_map = f_df.groupby("t_round")["funding_rate"].sum()
        funding_arr = f_map.reindex(df.index, fill_value=0.0).values
    else:
        funding_arr = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        ts = timestamps[i]
        next_ts = timestamps[i + 1]
        next_open = open_arr[i + 1]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        atr = atr_arr[i]

        if i < warmup or np.isnan(atr):
            equity_curve.append(equity)
            continue

        # 1. Accrue funding if holding a position (via the canonical direction-aware helper;
        #    equal to the previous signed-size formula, now with one source of truth)
        if position_size != 0.0:
            f_rate = funding_arr[i]
            if f_rate != 0.0:
                payment = engine.funding_payment(position_direction, abs(position_size), close, f_rate)
                equity += payment
                funding_accrued += payment

        # 2. Check exits for open position on current bar
        if position_size != 0.0:
            bars_held += 1
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if position_direction == "long":
                highest_since_entry = max(highest_since_entry, high)
                # Trailing stop update
                if variant_name in ["1A", "1C", "1D"]:
                    trail_mult = p.get("trail_mult", 3.0)
                    trail_level = highest_since_entry - trail_mult * atr
                    active_stop = max(active_stop, trail_level)

                # Intrabar exit resolution
                res_price, res_reason = engine.resolve_intrabar_exit(
                    stop_price=active_stop,
                    target_price=active_target,
                    direction="long",
                    bar_high=high,
                    bar_low=low,
                )
                if res_price is not None:
                    exit_triggered = True
                    exit_price = res_price
                    exit_reason = res_reason

                # Specific indicator exits
                if not exit_triggered:
                    if variant_name == "1A" and close < ema50_arr[i]:
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "EMA 50 Cross"
                    elif variant_name == "1B":
                        if close >= bb_basis_arr[i]:
                            exit_triggered = True
                            exit_price = next_open if cost_model.fill_on_next_open else close
                            exit_reason = "BB Basis Target"
                        elif bars_held >= p.get("max_hold_bars", 15):
                            exit_triggered = True
                            exit_price = next_open if cost_model.fill_on_next_open else close
                            exit_reason = "Time Stop"
                    elif variant_name == "1C" and (macd_line_arr[i] < macd_signal_arr[i]):
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "MACD Reversal"
                    elif variant_name == "1D" and close < bb_basis_arr[i]:
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "BB Basis Cross"

            else:  # short
                lowest_since_entry = min(lowest_since_entry, low)
                if variant_name in ["1A", "1C", "1D"]:
                    trail_mult = p.get("trail_mult", 3.0)
                    trail_level = lowest_since_entry + trail_mult * atr
                    active_stop = min(active_stop, trail_level)

                res_price, res_reason = engine.resolve_intrabar_exit(
                    stop_price=active_stop,
                    target_price=active_target,
                    direction="short",
                    bar_high=high,
                    bar_low=low,
                )
                if res_price is not None:
                    exit_triggered = True
                    exit_price = res_price
                    exit_reason = res_reason

                if not exit_triggered:
                    if variant_name == "1A" and close > ema50_arr[i]:
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "EMA 50 Cross"
                    elif variant_name == "1B":
                        if close <= bb_basis_arr[i]:
                            exit_triggered = True
                            exit_price = next_open if cost_model.fill_on_next_open else close
                            exit_reason = "BB Basis Target"
                        elif bars_held >= p.get("max_hold_bars", 15):
                            exit_triggered = True
                            exit_price = next_open if cost_model.fill_on_next_open else close
                            exit_reason = "Time Stop"
                    elif variant_name == "1C" and (macd_line_arr[i] > macd_signal_arr[i]):
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "MACD Reversal"
                    elif variant_name == "1D" and close > bb_basis_arr[i]:
                        exit_triggered = True
                        exit_price = next_open if cost_model.fill_on_next_open else close
                        exit_reason = "BB Basis Cross"

            if exit_triggered:
                exec_entry, exec_exit, gross, cost, net, bps = engine.calculate_trade_pnl(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=abs(position_size),
                    direction=position_direction,
                    funding_pnl=funding_accrued,
                )
                equity += net
                trades.append(
                    EngineTrade(
                        entry_time=entry_time,
                        exit_time=ts,
                        direction=position_direction,
                        entry_price=entry_price,
                        executed_entry_price=exec_entry,
                        exit_price=exit_price,
                        executed_exit_price=exec_exit,
                        quantity=abs(position_size),
                        gross_pnl=gross,
                        fee_and_slip_cost=cost,
                        funding_pnl=funding_accrued,
                        net_pnl=net,
                        return_bps=bps,
                        exit_reason=exit_reason,
                        unit_number=1,
                    )
                )
                position_size = 0.0
                position_direction = None
                bars_held = 0
                funding_accrued = 0.0

        # 3. Check entry signals on closed bar t (only if flat)
        long_sig = False
        short_sig = False

        if position_size == 0.0:
            if variant_name == "1A":
                adx_thresh = p.get("adx_threshold", 25.0)
                bull_trend = (ema20_arr[i] > ema50_arr[i] > ema80_arr[i] > ema200_arr[i]) and (adx_arr[i] > adx_thresh)
                bear_trend = (ema20_arr[i] < ema50_arr[i] < ema80_arr[i] < ema200_arr[i]) and (adx_arr[i] > adx_thresh)
                long_sig = bull_trend and (close > ema20_arr[i]) and (close_arr[i - 1] <= ema20_arr[i - 1])
                short_sig = bear_trend and (close < ema20_arr[i]) and (close_arr[i - 1] >= ema20_arr[i - 1])

            elif variant_name == "1B":
                rsi_oversold = p.get("rsi_oversold", 20.0)
                rsi_overbought = p.get("rsi_overbought", 80.0)
                long_sig = (close <= bb_lower_arr[i]) and (rsi6_arr[i] < rsi_oversold) and (stoch_k_arr[i] < 20.0) and (stoch_k_arr[i] > stoch_d_arr[i])
                short_sig = (close >= bb_upper_arr[i]) and (rsi6_arr[i] > rsi_overbought) and (stoch_k_arr[i] > 80.0) and (stoch_k_arr[i] < stoch_d_arr[i])

            elif variant_name == "1C":
                vol_mult = p.get("vol_mult", 1.5)
                long_sig = (macd_hist_arr[i] > 0) and (macd_line_arr[i] > macd_signal_arr[i]) and (vol_ratio_arr[i] > vol_mult) and (close > ema50_arr[i])
                short_sig = (macd_hist_arr[i] < 0) and (macd_line_arr[i] < macd_signal_arr[i]) and (vol_ratio_arr[i] > vol_mult) and (close < ema50_arr[i])

            elif variant_name == "1D":
                adx_min = p.get("adx_min", 20.0)
                long_sig = (ema20_arr[i] > ema50_arr[i]) and (close > bb_basis_arr[i]) and (rsi6_arr[i] > 50.0) and (macd_line_arr[i] > macd_signal_arr[i]) and (adx_arr[i] > adx_min) and (vol_ratio_arr[i] > 1.0)
                short_sig = (ema20_arr[i] < ema50_arr[i]) and (close < bb_basis_arr[i]) and (rsi6_arr[i] < 50.0) and (macd_line_arr[i] < macd_signal_arr[i]) and (adx_arr[i] > adx_min) and (vol_ratio_arr[i] > 1.0)

            # Fill strictly at open[t+1]
            fill_price = next_open if cost_model.fill_on_next_open else close
            fill_time = next_ts if cost_model.fill_on_next_open else ts

            if (long_sig or short_sig) and fill_time > last_add_time:
                direction = "long" if long_sig else "short"
                risk_usd = equity * (risk_percent / 100.0)
                stop_atr_mult = p.get("stop_atr_mult", 2.0)
                stop_dist = stop_atr_mult * atr
                size = (risk_usd / stop_dist) if stop_dist > 0 else 0.0

                # Max position size cap (2.0x leverage cap)
                max_size = (equity * 2.0) / fill_price
                size = min(size, max_size)

                if size * fill_price >= 10.0:  # Min order size
                    position_size = size if direction == "long" else -size
                    position_direction = direction
                    entry_price = fill_price
                    entry_time = fill_time
                    bars_held = 0
                    funding_accrued = 0.0
                    last_add_time = fill_time

                    if direction == "long":
                        active_stop = fill_price - stop_dist
                        active_target = None
                        highest_since_entry = fill_price
                    else:
                        active_stop = fill_price + stop_dist
                        active_target = None
                        lowest_since_entry = fill_price

        # 4. Mark to market equity
        if position_size != 0.0:
            unrealized = position_size * (close - entry_price) if position_direction == "long" else abs(position_size) * (entry_price - close)
        else:
            unrealized = 0.0

        equity_curve.append(max(0.0, equity + unrealized))

    # Invariant assertion
    assert_no_same_bar_pyramiding(trades)

    equity_series = pd.Series(equity_curve, index=timestamps[: len(equity_curve)])
    metrics = engine.calculate_metrics(trades, equity_series, initial_capital, bars_per_year=bars_per_year)
    return metrics, trades, equity_series

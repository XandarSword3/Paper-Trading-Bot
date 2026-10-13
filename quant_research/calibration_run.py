"""
Calibration Run: Benchmarking V1 and Trail-After-Profit strategies under:
1. Old Rules: close-bar fills, no funding, 26 bps round-trip friction (0.08% fee + 0.05% slip per side).
2. New Rules: next-bar-open fills, real historical funding settlement, 14 bps round-trip friction (0.05% fee + 0.02% slip per side).

Evaluates across:
- In-Sample Development Window (2018-05-15 to 2024-12-31)
- Full Historical BTC 4H Data (2018-05-15 to 2026-07-21)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Path setup
sys.path.insert(0, os.path.abspath("research/strategies"))
sys.path.insert(0, os.path.abspath("research/data"))
sys.path.insert(0, os.path.abspath("docs/archive/backups"))
sys.path.insert(0, os.path.abspath("."))

from quant_research.core.engine import QuantEngine, CostModel, EngineTrade, assert_no_same_bar_pyramiding
from config import DEFAULT_PARAMS, StrategyParams


def load_btc_4h_and_funding():
    df = pd.read_csv("data/BTCUSDT_4h.csv", index_col=0, parse_dates=True)
    funding_path = "research/strategies/data/BTCUSDT_funding_rate.csv"
    funding_series = None
    if os.path.exists(funding_path):
        fdf = pd.read_csv(funding_path, index_col=0, parse_dates=True)
        funding_series = fdf["funding_rate"].sort_index()
    return df, funding_series


def run_v1_simulation(df: pd.DataFrame, funding_series: pd.Series, cost_model: CostModel) -> Dict[str, Any]:
    """
    Runs V1 Turtle-Donchian with specified cost model, fill timing, and funding.
    """
    df = df.copy()
    entry_len = 40
    exit_len = 16
    atr_len = 20
    trail_mult = 4.0
    risk_percent = 1.0
    max_units = 4
    pyramid_spacing_n = 1.5

    # Indicators
    df["upper_entry"] = df["high"].shift(1).rolling(entry_len).max()
    df["lower_exit"] = df["low"].shift(1).rolling(exit_len).min()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    engine = QuantEngine(cost_model)
    initial_capital = 100_000.0
    equity = initial_capital
    units = []  # list of dicts: {entry_time, entry_price, size, unit_num}
    trades: List[EngineTrade] = []
    equity_curve = []
    last_add_time = pd.Timestamp.min
    highest_since_entry = 0.0

    warmup = max(entry_len, atr_len, 200) + 1

    # Funding lookup helper
    def get_funding_between(t_start, t_end):
        if funding_series is None or not cost_model.apply_funding:
            return 0.0
        slice_f = funding_series[(funding_series.index > t_start) & (funding_series.index <= t_end)]
        return float(slice_f.sum()) if len(slice_f) > 0 else 0.0

    prev_ts = df.index[0]

    for i in range(len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        close, high, low, open_p = row["close"], row["high"], row["low"], row["open"]
        atr = row["atr"]

        if i < warmup or pd.isna(atr) or pd.isna(row["upper_entry"]):
            equity_curve.append(equity)
            prev_ts = ts
            continue

        # Position tracking & funding accrual
        pos_size = sum(u["size"] for u in units)
        f_rate = get_funding_between(prev_ts, ts)
        funding_cashflow = 0.0
        if pos_size > 0 and f_rate != 0.0:
            # This strategy is long-only; the helper keeps the sign convention in one place.
            funding_cashflow = engine.funding_payment("long", pos_size, close, f_rate)
            equity += funding_cashflow

        # Check exits
        if pos_size > 0:
            highest_since_entry = max(highest_since_entry, high)
            trail_stop = highest_since_entry - trail_mult * atr
            donchian_exit = row["lower_exit"]

            exit_triggered = False
            exit_price = None
            exit_reason = None

            if low <= trail_stop:
                exit_triggered = True
                exit_price = trail_stop
                exit_reason = "Trailing Stop"
            elif close < donchian_exit:
                exit_triggered = True
                exit_price = close
                exit_reason = "Donchian Exit"

            if exit_triggered:
                for u in units:
                    exec_entry, exec_exit, gross, cost, net, bps = engine.calculate_trade_pnl(
                        entry_price=u["entry_price"],
                        exit_price=exit_price,
                        quantity=u["size"],
                        direction="long",
                        funding_pnl=u.get("funding_accrued", 0.0),
                    )
                    equity += net
                    trades.append(
                        EngineTrade(
                            entry_time=u["entry_time"],
                            exit_time=ts,
                            direction="long",
                            entry_price=u["entry_price"],
                            executed_entry_price=exec_entry,
                            exit_price=exit_price,
                            executed_exit_price=exec_exit,
                            quantity=u["size"],
                            gross_pnl=gross,
                            fee_and_slip_cost=cost,
                            funding_pnl=u.get("funding_accrued", 0.0),
                            net_pnl=net,
                            return_bps=bps,
                            exit_reason=exit_reason,
                            unit_number=u["unit_num"],
                        )
                    )
                units = []
                pos_size = 0.0
                highest_since_entry = 0.0

        # Check entries / adds
        # Long signal on closed bar t
        long_signal = (close > row["upper_entry"]) and (close > row["ema200"])
        
        # Determine execution price and timestamp
        if cost_model.fill_on_next_open:
            if i + 1 < len(df):
                exec_price = df.iloc[i + 1]["open"]
                fill_ts = df.index[i + 1]
            else:
                exec_price = close
                fill_ts = ts
        else:
            exec_price = close
            fill_ts = ts

        # New entry
        if long_signal and pos_size == 0 and fill_ts > last_add_time:
            risk_usd = equity * (risk_percent / 100.0)
            stop_usd = atr * 2.0
            size = max(0.001, round((risk_usd / stop_usd) / 0.001) * 0.001) if stop_usd > 0 else 0.0
            if size > 0 and size * exec_price <= equity * 0.95:
                units.append({
                    "entry_time": fill_ts,
                    "entry_price": exec_price,
                    "size": size,
                    "unit_num": 1,
                    "funding_accrued": 0.0,
                })
                last_add_time = fill_ts
                highest_since_entry = high

        # Pyramiding
        elif pos_size > 0 and len(units) < max_units and fill_ts > last_add_time:
            last_entry = units[-1]["entry_price"]
            pyramid_trigger = last_entry + (pyramid_spacing_n * atr)
            if high >= pyramid_trigger:
                risk_usd = equity * (risk_percent / 100.0)
                stop_usd = atr * 2.0
                size = max(0.001, round((risk_usd / stop_usd) / 0.001) * 0.001) if stop_usd > 0 else 0.0
                if size > 0 and (pos_size + size) * exec_price <= equity * 0.95:
                    units.append({
                        "entry_time": fill_ts,
                        "entry_price": exec_price,
                        "size": size,
                        "unit_num": len(units) + 1,
                        "funding_accrued": 0.0,
                    })
                    last_add_time = fill_ts

        # Mark-to-market unrealized
        unrealized = sum(u["size"] * (close - u["entry_price"]) for u in units) if units else 0.0
        bar_equity = max(0.0, equity + unrealized)
        equity_curve.append(bar_equity)
        prev_ts = ts

    # Invariant assertion
    assert_no_same_bar_pyramiding(trades)
    metrics = engine.calculate_metrics(trades, pd.Series(equity_curve, index=df.index), initial_capital, bars_per_year=2190)
    return metrics


def run_trail_after_profit_simulation(df: pd.DataFrame, funding_series: pd.Series, cost_model: CostModel) -> Dict[str, Any]:
    """
    Runs the Enhanced 'Trail-After-Profit' Strategy (from enhanced_strategy.py).
    Key features:
    - Initial stop loss: 2.0 * ATR below entry.
    - Trailing stop: current_price - 4.0 * ATR, updated ONLY when it exceeds active stop loss (dormant until +2.0 ATR profit).
    - Regime filter: Close > 200 SMA and 200 SMA slope > 0.
    """
    df = df.copy()
    entry_len = 40
    exit_len = 16
    atr_len = 20
    trail_mult = 4.0
    size_stop_mult = 2.0
    risk_percent = 1.0

    df["donchian_high"] = df["high"].shift(1).rolling(entry_len).max()
    df["exit_low"] = df["low"].shift(1).rolling(exit_len).min()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma_rising"] = df["sma200"].diff(20) > 0

    engine = QuantEngine(cost_model)
    initial_capital = 100_000.0
    equity = initial_capital
    active_trade = None
    trades: List[EngineTrade] = []
    equity_curve = []
    last_add_time = pd.Timestamp.min

    warmup = 220

    def get_funding_between(t_start, t_end):
        if funding_series is None or not cost_model.apply_funding:
            return 0.0
        slice_f = funding_series[(funding_series.index > t_start) & (funding_series.index <= t_end)]
        return float(slice_f.sum()) if len(slice_f) > 0 else 0.0

    prev_ts = df.index[0]

    for i in range(len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        close, high, low, open_p = row["close"], row["high"], row["low"], row["open"]
        atr = row["atr"]

        if i < warmup or pd.isna(atr) or pd.isna(row["donchian_high"]):
            equity_curve.append(equity)
            prev_ts = ts
            continue

        # Position tracking & funding
        if active_trade is not None:
            f_rate = get_funding_between(prev_ts, ts)
            if f_rate != 0.0:
                # Long-only strategy: helper keeps the sign convention in one place.
                payment = engine.funding_payment("long", active_trade["size"], close, f_rate)
                equity += payment
                active_trade["funding_accrued"] += payment

        # Check exits
        if active_trade is not None:
            # Trailing stop logic:
            # Current trailing candidate
            trail_candidate = close - trail_mult * atr
            # In the original code, stop_loss = max(stop_loss, trail_candidate)
            # This only moves up once trail_candidate > initial stop (after profit)
            active_trade["stop_loss"] = max(active_trade["stop_loss"], trail_candidate)

            exit_triggered = False
            exit_price = None
            exit_reason = None

            if low <= active_trade["stop_loss"]:
                exit_triggered = True
                exit_price = active_trade["stop_loss"]
                exit_reason = "Trailing Stop"
            elif low <= row["exit_low"]:
                exit_triggered = True
                exit_price = row["exit_low"]
                exit_reason = "Donchian Exit"

            if exit_triggered:
                exec_entry, exec_exit, gross, cost, net, bps = engine.calculate_trade_pnl(
                    entry_price=active_trade["entry_price"],
                    exit_price=exit_price,
                    quantity=active_trade["size"],
                    direction="long",
                    funding_pnl=active_trade["funding_accrued"],
                )
                equity += net
                trades.append(
                    EngineTrade(
                        entry_time=active_trade["entry_time"],
                        exit_time=ts,
                        direction="long",
                        entry_price=active_trade["entry_price"],
                        executed_entry_price=exec_entry,
                        exit_price=exit_price,
                        executed_exit_price=exec_exit,
                        quantity=active_trade["size"],
                        gross_pnl=gross,
                        fee_and_slip_cost=cost,
                        funding_pnl=active_trade["funding_accrued"],
                        net_pnl=net,
                        return_bps=bps,
                        exit_reason=exit_reason,
                        unit_number=1,
                    )
                )
                active_trade = None

        # Check entries
        can_trade = (close > row["sma200"]) and row["sma_rising"]
        entry_sig = can_trade and (high >= row["donchian_high"])

        if cost_model.fill_on_next_open:
            if i + 1 < len(df):
                exec_price = df.iloc[i + 1]["open"]
                fill_ts = df.index[i + 1]
            else:
                exec_price = close
                fill_ts = ts
        else:
            exec_price = row["donchian_high"] if high >= row["donchian_high"] else close
            fill_ts = ts

        if entry_sig and active_trade is None and fill_ts > last_add_time:
            risk_usd = equity * (risk_percent / 100.0)
            stop_distance = size_stop_mult * atr
            size = max(0.001, round((risk_usd / stop_distance) / 0.001) * 0.001) if stop_distance > 0 else 0.0
            if size > 0 and size * exec_price <= equity * 0.95:
                active_trade = {
                    "entry_time": fill_ts,
                    "entry_price": exec_price,
                    "size": size,
                    "stop_loss": exec_price - stop_distance,
                    "funding_accrued": 0.0,
                }
                last_add_time = fill_ts

        unrealized = active_trade["size"] * (close - active_trade["entry_price"]) if active_trade else 0.0
        equity_curve.append(max(0.0, equity + unrealized))
        prev_ts = ts

    assert_no_same_bar_pyramiding(trades)
    metrics = engine.calculate_metrics(trades, pd.Series(equity_curve, index=df.index), initial_capital, bars_per_year=2190)
    return metrics


def main(out_path: str = "quant_research/results/calibration_results.json", only_new_rules: bool = False):
    df, funding = load_btc_4h_and_funding()
    print("=" * 80)
    print("CALIBRATION BENCHMARK RUN (BTCUSDT 4H)")
    print(f"Dataset: {df.index[0]} to {df.index[-1]} ({len(df)} candles)")
    print("=" * 80)

    # Cost Models
    old_costs = CostModel(taker_fee_pct=0.08, slippage_pct=0.05, apply_funding=False, fill_on_next_open=False)
    new_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)

    # Split windows:
    # 1. Full window
    # 2. In-sample development window (up to 2024-12-31)
    df_dev = df[df.index <= "2024-12-31 23:59:59"]

    print("\n--- RESULTS ON IN-SAMPLE DEVELOPMENT WINDOW (2018-05-15 to 2024-12-31) ---")
    v1_old_dev = run_v1_simulation(df_dev, funding, old_costs)
    v1_new_dev = run_v1_simulation(df_dev, funding, new_costs)
    tap_old_dev = run_trail_after_profit_simulation(df_dev, funding, old_costs)
    tap_new_dev = run_trail_after_profit_simulation(df_dev, funding, new_costs)

    rows_dev = [
        {"Model": "V1 Turtle (Old Rules: Close fill, no funding, 26 bps)", **v1_old_dev},
        {"Model": "V1 Turtle (New Rules: Next open fill, funding, 14 bps)", **v1_new_dev},
        {"Model": "Trail-After-Profit (Old Rules: Donchian fill, no funding, 26 bps)", **tap_old_dev},
        {"Model": "Trail-After-Profit (New Rules: Next open fill, funding, 14 bps)", **tap_new_dev},
    ]
    dev_table = pd.DataFrame(rows_dev)
    print(dev_table[["Model", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "expectancy_bps", "total_trades", "win_rate_pct"]].to_string(index=False))

    print("\n--- RESULTS ON FULL DATA WINDOW (2018-05-15 to 2026-07-21) ---")
    v1_old_full = run_v1_simulation(df, funding, old_costs)
    v1_new_full = run_v1_simulation(df, funding, new_costs)
    tap_old_full = run_trail_after_profit_simulation(df, funding, old_costs)
    tap_new_full = run_trail_after_profit_simulation(df, funding, new_costs)

    rows_full = [
        {"Model": "V1 Turtle (Old Rules: Close fill, no funding, 26 bps)", **v1_old_full},
        {"Model": "V1 Turtle (New Rules: Next open fill, funding, 14 bps)", **v1_new_full},
        {"Model": "Trail-After-Profit (Old Rules: Donchian fill, no funding, 26 bps)", **tap_old_full},
        {"Model": "Trail-After-Profit (New Rules: Next open fill, funding, 14 bps)", **tap_new_full},
    ]
    full_table = pd.DataFrame(rows_full)
    print(full_table[["Model", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "expectancy_bps", "total_trades", "win_rate_pct"]].to_string(index=False))

    # Save calibration results to json for reporting
    calib_out = {
        "development_window": rows_dev,
        "full_window": rows_full,
    }
    os.makedirs("quant_research/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calib_out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

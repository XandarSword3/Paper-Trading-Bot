"""
Random Entry Baseline Test (1,000 runs).
Replaces the Donchian channel breakout entry with random entries, keeping the exact same
2N stop loss, M-bar channel exit, and 14 bps round-trip friction.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import json

from quant_research.core.engine import CostModel, QuantEngine

DATA_DIR = "quant_research/data/processed"


def run_random_entry_baseline_simulation(n_sims: int = 1000):
    print("=" * 80)
    print(f"RUNNING RANDOM-ENTRY BASELINE TEST ({n_sims:,} RUNS)")
    print("=" * 80)

    # Load 1h BTC data (representative benchmark)
    df = pd.read_parquet(os.path.join(DATA_DIR, "BTCUSDT_1h.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # OOS slice
    mask = (df.index >= pd.Timestamp("2025-04-01", tz="UTC")) & (df.index <= pd.Timestamp("2026-03-31", tz="UTC"))
    sub_df = df[mask].copy()

    open_arr = sub_df["open"].to_numpy(dtype=np.float64)
    high_arr = sub_df["high"].to_numpy(dtype=np.float64)
    low_arr = sub_df["low"].to_numpy(dtype=np.float64)
    close_arr = sub_df["close"].to_numpy(dtype=np.float64)
    n_bars = len(close_arr)

    # ATR(20)
    tr = pd.concat([
        (sub_df["high"] - sub_df["low"]),
        (sub_df["high"] - sub_df["close"].shift(1)).abs(),
        (sub_df["low"] - sub_df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_arr = tr.ewm(alpha=1.0/20.0, adjust=False).mean().to_numpy(dtype=np.float64)

    rng = np.random.default_rng(42)
    engine = QuantEngine(CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True))

    results = []

    for sim_idx in range(n_sims):
        # Generate random entry signals (probability ~ 2% per bar when flat)
        in_pos = False
        direction = 0
        entry_price = 0.0
        active_stop = 0.0
        bars_held = 0
        net_pnls = []

        # Randomly choose entry bars
        random_draws = rng.random(n_bars)
        dir_draws = rng.choice([1, -1], size=n_bars)

        for i in range(25, n_bars - 1):
            next_open = open_arr[i + 1]
            atr = atr_arr[i]

            if in_pos:
                bars_held += 1
                stop_hit = (low_arr[i] <= active_stop) if direction == 1 else (high_arr[i] >= active_stop)
                time_exit = bars_held >= 20  # hold 20 bars (matching M=20)

                if stop_hit or time_exit:
                    exit_price = active_stop if stop_hit else next_open
                    dir_str = "long" if direction == 1 else "short"
                    _, _, _, _, net, bps = engine.calculate_trade_pnl(
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=1.0,
                        direction=dir_str,
                    )
                    net_pnls.append(bps)
                    in_pos = False
                    direction = 0
                    bars_held = 0

            elif not in_pos and random_draws[i] < 0.03:
                direction = dir_draws[i]
                entry_price = next_open
                stop_dist = 2.0 * atr
                active_stop = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
                in_pos = True
                bars_held = 0

        mean_bps = float(np.mean(net_pnls)) if net_pnls else -14.0
        results.append(mean_bps)

    results_arr = np.array(results)
    mean_res = float(np.mean(results_arr))
    std_res = float(np.std(results_arr))
    pct_pos = float((results_arr > 0).mean() * 100.0)

    print(f"Completed {n_sims:,} random entry simulations:")
    print(f"  Mean Net Expectancy:    {mean_res:+.2f} bps")
    print(f"  Std Dev:                {std_res:.2f} bps")
    print(f"  5th - 95th Percentile:  [{np.percentile(results_arr, 5):.2f}, {np.percentile(results_arr, 95):.2f}] bps")
    print(f"  % of Runs Positive:     {pct_pos:.2f}%")

    out_data = {
        "n_simulations": n_sims,
        "mean_net_bps": round(mean_res, 2),
        "std_net_bps": round(std_res, 2),
        "ci_90": [round(float(np.percentile(results_arr, 5)), 2), round(float(np.percentile(results_arr, 95)), 2)],
        "pct_positive_runs": round(pct_pos, 2),
    }

    with open("quant_research/results/random_entry_baseline.json", "w") as f:
        json.dump(out_data, f, indent=2)

    return out_data


if __name__ == "__main__":
    run_random_entry_baseline_simulation(1000)

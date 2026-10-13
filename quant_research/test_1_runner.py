"""
Test 1 Runner: Mechanical reconstruction of the indicator stack on SOL (1m, 3m, 5m).
Variants:
- Variant 1A: Multi-EMA Trend Alignment & ADX Filter (EMAs 20/50/80/200 + ADX 14)
- Variant 1B: Bollinger & RSI Momentum Mean Reversion (BB 20,2 + RSI 6 + StochRSI)
- Variant 1C: MACD Trend Breakout with Volume Confirmation (MACD 12,26,9 + Vol MA 20)
- Variant 1D: Confluence Stack (Full Indicator Agreement)

Splits evaluated:
- In-Sample: 2024-01-01 to 2025-03-31
- Walk-Forward OOS: 2025-04-01 to 2026-03-31
- User Live Window: 2025-04-04 to 2025-05-12

Costs applied:
- Base: 14 bps RT (5 bps taker + 2 bps slip per side) + Funding
- Stress: 18 bps RT (5 bps taker + 4 bps slip per side) + Funding
"""
import os
import sys
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel, QuantEngine
from quant_research.strategies.indicator_stack import run_indicator_strategy

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"


def load_data_and_funding(timeframe: str):
    kline_path = os.path.join(DATA_DIR, f"SOLUSDT_{timeframe}.parquet")
    fund_path = os.path.join(DATA_DIR, "SOLUSDT_funding.parquet")
    if not os.path.exists(kline_path):
        raise FileNotFoundError(f"Missing {kline_path}")
    df = pd.read_parquet(kline_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    funding = None
    if os.path.exists(fund_path):
        fdf = pd.read_parquet(fund_path)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding = fdf.set_index("timestamp")["funding_rate"].sort_index()
    return df, funding


def log_trial(trial_record: Dict[str, Any], log: bool = True):
    if not log:
        return
    df_row = pd.DataFrame([trial_record])
    df_row.to_csv(TRIALS_FILE, mode="a", header=not os.path.exists(TRIALS_FILE), index=False)


def run_test_1(out_path: str = "quant_research/results/test_1_results.json", log_trials: bool = True):
    print("=" * 100)
    print("TEST 1: MECHANICAL RECONSTRUCTION OF INDICATOR STACK (SOLUSDT)")
    print("=" * 100)

    timeframes = ["1m", "3m", "5m"]
    variants = ["1A", "1B", "1C", "1D"]
    variant_names = {
        "1A": "EMA Trend + ADX",
        "1B": "BB + RSI(6) + StochRSI MeanRev",
        "1C": "MACD + Vol MA Breakout",
        "1D": "Confluence Stack",
    }

    # Bars per year estimation
    bpy_map = {"1m": 525600.0, "3m": 175200.0, "5m": 105120.0}

    # Cost models
    base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)
    stress_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.04, apply_funding=True, fill_on_next_open=True)

    results_table = []
    trial_count = 0

    for tf in timeframes:
        print(f"\nLoading SOLUSDT {tf} data...", flush=True)
        df, funding = load_data_and_funding(tf)
        bpy = bpy_map[tf]
        print(f"[{tf}] Computing indicator stack...", flush=True)
        from quant_research.core.indicators import compute_indicators
        df = compute_indicators(df)

        # Windows
        is_df = df[(df.index >= "2024-01-01 00:00:00+00:00") & (df.index <= "2025-03-31 23:59:59+00:00")]
        oos_df = df[(df.index >= "2025-04-01 00:00:00+00:00") & (df.index <= "2026-03-31 23:59:59+00:00")]
        live_df = df[(df.index >= "2025-04-04 00:00:00+00:00") & (df.index <= "2025-05-12 23:59:59+00:00")]

        for var in variants:
            # 1. In-Sample Base Run
            m_is_base, trades_is, _ = run_indicator_strategy(is_df, var, base_costs, funding, bars_per_year=bpy)
            # 2. In-Sample Stress Run
            m_is_stress, _, _ = run_indicator_strategy(is_df, var, stress_costs, funding, bars_per_year=bpy)

            # 3. Walk-Forward OOS Base Run
            m_oos_base, trades_oos, _ = run_indicator_strategy(oos_df, var, base_costs, funding, bars_per_year=bpy)
            # 4. Walk-Forward OOS Stress Run
            m_oos_stress, _, _ = run_indicator_strategy(oos_df, var, stress_costs, funding, bars_per_year=bpy)

            # 5. User Live Window Base Run
            m_live_base, _, _ = run_indicator_strategy(live_df, var, base_costs, funding, bars_per_year=bpy)

            # Decision criteria
            pass_is = (m_is_base["expectancy_bps"] > 0) and (m_is_base["total_trades"] >= 100)
            pass_oos = (m_oos_base["expectancy_bps"] > 0) and (m_oos_base["total_trades"] >= 50)
            edge_status = "PASS" if (pass_is and pass_oos and m_oos_stress["expectancy_bps"] > -2.0) else "FAIL"

            trial_count += 1
            trial_id = f"T1-{tf}-{var}-{trial_count}"

            # Log to trials.csv
            log_trial({
                "trial_id": trial_id,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "test_name": "Test1_Indicator_Stack",
                "asset": "SOLUSDT",
                "timeframe": tf,
                "strategy_name": f"Variant_{var}_{variant_names[var]}",
                "params_json": json.dumps({"variant": var, "timeframe": tf}),
                "is_return_pct": m_is_base["total_return_pct"],
                "is_expectancy_bps": m_is_base["expectancy_bps"],
                "oos_return_pct": m_oos_base["total_return_pct"],
                "oos_expectancy_bps": m_oos_base["expectancy_bps"],
                "status": edge_status,
            }, log=log_trials)

            results_table.append({
                "TF": tf,
                "Variant": f"{var} ({variant_names[var][:15]})",
                "IS Trades": m_is_base["total_trades"],
                "IS Exp (bps)": m_is_base["expectancy_bps"],
                "IS Return %": m_is_base["total_return_pct"],
                "OOS Trades": m_oos_base["total_trades"],
                "OOS Exp (bps)": m_oos_base["expectancy_bps"],
                "OOS Return %": m_oos_base["total_return_pct"],
                "OOS Stress Exp": m_oos_stress["expectancy_bps"],
                "Live Window Return %": m_live_base["total_return_pct"],
                "Cost RT (bps)": m_oos_base["round_trip_cost_bps"],
                "Verdict": edge_status,
            })

    table_df = pd.DataFrame(results_table)
    print("\n" + "=" * 110)
    print("TEST 1 FULL RESULTS SUMMARY")
    print("=" * 110)
    print(table_df.to_string(index=False))

    # Save results to json
    os.makedirs("quant_research/results", exist_ok=True)
    table_df.to_json(out_path, orient="records", indent=2)
    print(f"\nLogged {trial_count if log_trials else 0} trials to quant_research/trials.csv")
    print(f"Saved {out_path}")
    return table_df


if __name__ == "__main__":
    run_test_1()

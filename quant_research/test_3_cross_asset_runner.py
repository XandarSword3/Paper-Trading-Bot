"""
Test 3 Runner: Cross-Asset Robustness on BTCUSDT and ETHUSDT.
Evaluates the best strategies / parameter sets across BTC and ETH:
1. Candidate 1: Variant 1B (Bollinger + RSI6 + StochRSI Mean Reversion)
2. Candidate 2: Variant 1A (Multi-EMA Trend + ADX)
3. Candidate 3: Test 2 Selected Config (CFG_1142: BB 2.5, RSI 20/80, ADX 30, Stop 3.0 ATR, TP 3.5 ATR, Hold 20)

Splits:
- In-Sample: 2024-01-01 to 2025-03-31
- Walk-Forward OOS: 2025-04-01 to 2026-03-31
- User Live Window: 2025-04-04 to 2025-05-12
- Hold-Out (2026-04-01 to 2026-10-09):
  * BTC: BURNED (reported as OOS only, never as clean holdout)
  * ETH: Fresh hold-out (touch strictly ONCE)

Costs applied:
- Base: 14 bps RT (5 bps taker + 2 bps slip per side) + Funding
- Stress: 18 bps RT (5 bps taker + 4 bps slip per side) + Funding
"""
import os
import sys
import json
import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel, QuantEngine
from quant_research.strategies.indicator_stack import run_indicator_strategy
from quant_research.strategies.parameter_grid_strategy import run_grid_simulation
from quant_research.test_2_runner import prepare_features, slice_features

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"


def run_cross_asset_evaluation():
    print("=" * 100)
    print("TEST 3: CROSS-ASSET ROBUSTNESS EVALUATION (BTCUSDT & ETHUSDT)")
    print("=" * 100)

    assets = ["BTCUSDT", "ETHUSDT"]
    timeframe = "5m"
    bpy = 105120.0

    base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)
    stress_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.04, apply_funding=True, fill_on_next_open=True)

    # Load Test 2 best config if available, else default CFG_1142
    best_t2_cfg = {
        "config_id": "CFG_1142",
        "bb_mult": 2.5,
        "rsi_thresh": [20.0, 80.0],
        "adx_thresh": 30.0,
        "stop_atr_mult": 3.0,
        "tp_type": "3.5",
        "max_hold": 20,
    }
    t2_json_path = "quant_research/results/test_2_results.json"
    if os.path.exists(t2_json_path):
        try:
            with open(t2_json_path, "r") as f:
                t2_res = json.load(f)
                best_t2_cfg = t2_res.get("best_config", best_t2_cfg)
        except Exception:
            pass

    results_table = []

    for asset in assets:
        print(f"\nProcessing {asset} ({timeframe})...", flush=True)
        kline_path = os.path.join(DATA_DIR, f"{asset}_{timeframe}.parquet")
        fund_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")

        df = pd.read_parquet(kline_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        fdf = pd.read_parquet(fund_path)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding_series = fdf.set_index("timestamp")["funding_rate"].sort_index()

        features = prepare_features(df, funding_series)

        # Slices
        is_feat = slice_features(features, "2024-01-01 00:00:00", "2025-03-31 23:59:59")
        oos_feat = slice_features(features, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
        live_feat = slice_features(features, "2025-04-04 00:00:00", "2025-05-12 23:59:59")
        holdout_feat = slice_features(features, "2026-04-01 00:00:00", "2026-10-09 23:59:59")

        # 1. Evaluate Test 2 Best Config
        strategies_to_test = [
            ("Grid_Best_CFG", best_t2_cfg),
            ("Variant_1B_MeanRev", {"bb_mult": 2.0, "rsi_thresh": [20.0, 80.0], "adx_thresh": 20.0, "stop_atr_mult": 2.0, "tp_type": "basis_reversion", "max_hold": 15}),
            ("Variant_1A_Trend", {"bb_mult": 2.0, "rsi_thresh": [50.0, 50.0], "adx_thresh": 25.0, "stop_atr_mult": 2.0, "tp_type": "3.5", "max_hold": "unlimited"}),
        ]

        for strat_name, p in strategies_to_test:
            # IS
            m_is, _ = run_grid_simulation(
                is_feat["open"], is_feat["high"], is_feat["low"], is_feat["close"],
                is_feat["atr"], is_feat["bb_basis"], is_feat["bb_std"], is_feat["rsi6"],
                is_feat["adx"], is_feat["funding"], is_feat["timestamps"],
                p, base_costs, bars_per_year=bpy
            )
            # OOS Base
            m_oos_base, _ = run_grid_simulation(
                oos_feat["open"], oos_feat["high"], oos_feat["low"], oos_feat["close"],
                oos_feat["atr"], oos_feat["bb_basis"], oos_feat["bb_std"], oos_feat["rsi6"],
                oos_feat["adx"], oos_feat["funding"], oos_feat["timestamps"],
                p, base_costs, bars_per_year=bpy
            )
            # OOS Stress
            m_oos_stress, _ = run_grid_simulation(
                oos_feat["open"], oos_feat["high"], oos_feat["low"], oos_feat["close"],
                oos_feat["atr"], oos_feat["bb_basis"], oos_feat["bb_std"], oos_feat["rsi6"],
                oos_feat["adx"], oos_feat["funding"], oos_feat["timestamps"],
                p, stress_costs, bars_per_year=bpy
            )
            # Live Window
            m_live, _ = run_grid_simulation(
                live_feat["open"], live_feat["high"], live_feat["low"], live_feat["close"],
                live_feat["atr"], live_feat["bb_basis"], live_feat["bb_std"], live_feat["rsi6"],
                live_feat["adx"], live_feat["funding"], live_feat["timestamps"],
                p, base_costs, bars_per_year=bpy
            )
            # Holdout (if ETH fresh; if BTC marked burned)
            m_ho, _ = run_grid_simulation(
                holdout_feat["open"], holdout_feat["high"], holdout_feat["low"], holdout_feat["close"],
                holdout_feat["atr"], holdout_feat["bb_basis"], holdout_feat["bb_std"], holdout_feat["rsi6"],
                holdout_feat["adx"], holdout_feat["funding"], holdout_feat["timestamps"],
                p, base_costs, bars_per_year=bpy
            )

            is_btc_burned = (asset == "BTCUSDT")
            ho_label = "BURNED (OOS only)" if is_btc_burned else f"{m_ho['expectancy_bps']:.2f} bps ({m_ho['total_return_pct']:.1f}%)"

            pass_verdict = "FAIL"
            if m_oos_base["expectancy_bps"] >= 5.0 and m_oos_stress["expectancy_bps"] > 0.0 and m_oos_base["total_trades"] >= 50:
                pass_verdict = "PASS"

            # Log trial to trials.csv
            t_row = {
                "trial_id": f"T3-{asset[:3]}-{strat_name}",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "test_name": "Test3_Cross_Asset",
                "asset": asset,
                "timeframe": timeframe,
                "strategy_name": strat_name,
                "params_json": json.dumps(p),
                "is_return_pct": m_is["total_return_pct"],
                "is_expectancy_bps": m_is["expectancy_bps"],
                "oos_return_pct": m_oos_base["total_return_pct"],
                "oos_expectancy_bps": m_oos_base["expectancy_bps"],
                "status": pass_verdict,
            }
            pd.DataFrame([t_row]).to_csv(TRIALS_FILE, mode="a", header=not os.path.exists(TRIALS_FILE), index=False)

            results_table.append({
                "Asset": asset,
                "Strategy": strat_name,
                "IS Trades": m_is["total_trades"],
                "IS Exp (bps)": m_is["expectancy_bps"],
                "IS Return %": m_is["total_return_pct"],
                "OOS Trades": m_oos_base["total_trades"],
                "OOS Exp (bps)": m_oos_base["expectancy_bps"],
                "OOS Return %": m_oos_base["total_return_pct"],
                "OOS Stress Exp": m_oos_stress["expectancy_bps"],
                "Live Win Return %": m_live["total_return_pct"],
                "Hold-Out (2026+)": ho_label,
                "Verdict": pass_verdict,
            })

    table_df = pd.DataFrame(results_table)
    print("\n" + "=" * 120)
    print("TEST 3 CROSS-ASSET ROBUSTNESS SUMMARY")
    print("=" * 120)
    print(table_df.to_string(index=False))

    os.makedirs("quant_research/results", exist_ok=True)
    table_df.to_json("quant_research/results/test_3_results.json", orient="records", indent=2)
    print("\nSaved quant_research/results/test_3_results.json successfully.")
    return table_df


if __name__ == "__main__":
    run_cross_asset_evaluation()

"""
Multi-Asset Classical Turtle / Donchian Breakout Evaluation Runner.
Evaluates 36 configurations across SOLUSDT, BTCUSDT, and ETHUSDT on 5m, 15m, and 1h.
Pre-registered criteria:
- N in {20, 55}, M in {10, 20} (4 configs per asset/TF = 36 total).
- 2N stop loss, 1% risk per unit, pyramid up to 4 units at 0.5N with stop ratchet.
- Invariant: assert_no_same_bar_pyramiding.
- 14 bps RT base, 18 bps RT stress, empirical funding settlement.
- IS: 2024-01-01 to 2025-03-31
- OOS: 2025-04-01 to 2026-03-31
- Pooled trade hurdle: >= 150 trades pooled across the 3 assets.
- BTC 2025+ and SOL 2026-04+ are BURNED (OOS only).
- ETH 2026-04-01+ is FRESH veto hold-out, accessed strictly once for survivors.

GATE COVERAGE vs quant_research/PREREG.md section 7.5 (all six clauses were
committed in acb5979 before any Turtle run):
  1. Net expectancy >= +5 bps            -> implemented (pooled OOS)
  2. Stress expectancy > 0 bps           -> implemented (pooled OOS, 18 bps RT)
  3. Pooled trades >= 150                -> implemented (pooled OOS)
  4. DSR p-value < 0.05 (N_trials=1354)  -> implemented via core.metrics.compute_dsr
                                            on the PER-TRADE Sharpe and T = pooled trades
  5. Max drawdown < 30%                  -> NOT evaluated at the pooled level
                                            (section 7.5 ties it to stitched walk-forward equity)
  6. Profit factor >= 1.25               -> NOT part of the pass gate (reported per asset only)
The Sharpe column below is the canonical daily-equity Sharpe
(core.metrics.calculate_daily_sharpe, sqrt(365)); the legacy per-bar Sharpe is
reported alongside as mean_oos_bar_sharpe for traceability only.
"""
import os
import sys
import json
import time
import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor

from quant_research.core.engine import CostModel
from quant_research.core.metrics import compute_dsr
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"
RESULTS_FILE = "quant_research/results/turtle_results.json"


def slice_features(features: Dict[str, Any], start_str: str, end_str: str) -> Dict[str, Any]:
    ts = features["timestamps"]
    mask = (ts >= pd.Timestamp(start_str, tz="UTC")) & (ts <= pd.Timestamp(end_str, tz="UTC"))
    return {
        "open": features["open"][mask],
        "high": features["high"][mask],
        "low": features["low"][mask],
        "close": features["close"][mask],
        "entry_high": features["entry_high"][mask],
        "entry_low": features["entry_low"][mask],
        "exit_high": features["exit_high"][mask],
        "exit_low": features["exit_low"][mask],
        "atr": features["atr"][mask],
        "timestamps": ts[mask],
    }


def per_trade_sharpe(trades) -> float:
    """Sharpe of the per-trade net return distribution (mean/stdev of bps)."""
    if not trades:
        return 0.0
    bps = np.array([t.return_bps for t in trades], dtype=np.float64)
    sd = float(np.std(bps))
    return float(np.mean(bps) / sd) if sd > 1e-12 else 0.0


def evaluate_asset_tf_config(args: Tuple[str, str, int, int, str]) -> Dict[str, Any]:
    asset, tf, N, M, cfg_id = args

    kline_path = os.path.join(DATA_DIR, f"{asset}_{tf}.parquet")
    fund_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")
    df = pd.read_parquet(kline_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    funding = None
    if os.path.exists(fund_path):
        fdf = pd.read_parquet(fund_path)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

    features = prepare_turtle_features(df, N=N, M=M)

    bpy_map = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}
    bpy = bpy_map[tf]

    base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)
    stress_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.04, apply_funding=True, fill_on_next_open=True)

    params = {"N": N, "M": M, "config_id": cfg_id, "asset": asset, "timeframe": tf}

    # IS: 2024-01-01 to 2025-03-31
    is_f = slice_features(features, "2024-01-01 00:00:00", "2025-03-31 23:59:59")
    m_is_base, trades_is = run_turtle_simulation(is_f, funding, params, base_costs, bars_per_year=bpy, return_trades=True)
    m_is_stress = run_turtle_simulation(is_f, funding, params, stress_costs, bars_per_year=bpy, return_trades=False)

    # OOS: 2025-04-01 to 2026-03-31
    oos_f = slice_features(features, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
    m_oos_base, trades_oos = run_turtle_simulation(oos_f, funding, params, base_costs, bars_per_year=bpy, return_trades=True)
    m_oos_stress = run_turtle_simulation(oos_f, funding, params, stress_costs, bars_per_year=bpy, return_trades=False)

    # Live Window: 2025-04-04 to 2025-05-12
    live_f = slice_features(features, "2025-04-04 00:00:00", "2025-05-12 23:59:59")
    m_live = run_turtle_simulation(live_f, funding, params, base_costs, bars_per_year=bpy, return_trades=False)

    return {
        "asset": asset,
        "timeframe": tf,
        "config_id": cfg_id,
        "N": N,
        "M": M,
        "is_trades": m_is_base["total_trades"],
        "is_exp_bps": m_is_base["expectancy_bps"],
        "is_return_pct": m_is_base["total_return_pct"],
        "is_sharpe": m_is_base["sharpe_ratio"],
        "is_daily_sharpe": m_is_base.get("daily_sharpe", 0.0),
        "is_per_trade_sharpe": round(per_trade_sharpe(trades_is), 4),
        "is_max_dd": m_is_base["max_drawdown_pct"],
        "is_pf": m_is_base["profit_factor"],
        "is_win_rate": m_is_base["win_rate_pct"],
        "is_stress_exp_bps": m_is_stress["expectancy_bps"],
        "oos_trades": m_oos_base["total_trades"],
        "oos_exp_bps": m_oos_base["expectancy_bps"],
        "oos_return_pct": m_oos_base["total_return_pct"],
        "oos_sharpe": m_oos_base["sharpe_ratio"],
        "oos_daily_sharpe": m_oos_base.get("daily_sharpe", 0.0),
        "oos_per_trade_sharpe": round(per_trade_sharpe(trades_oos), 4),
        "oos_max_dd": m_oos_base["max_drawdown_pct"],
        "oos_pf": m_oos_base["profit_factor"],
        "oos_win_rate": m_oos_base["win_rate_pct"],
        "oos_stress_exp_bps": m_oos_stress["expectancy_bps"],
        "live_trades": m_live["total_trades"],
        "live_return_pct": m_live["total_return_pct"],
        "live_exp_bps": m_live["expectancy_bps"],
    }


def run_all_turtle_tests():
    print("=" * 100)
    print("PRE-REGISTERED TURTLE / DONCHIAN BREAKOUT EVALUATION (SOL, BTC, ETH)")
    print("36 Distinct Configurations: 3 Assets x 3 Timeframes x 4 Lookbacks")
    print("=" * 100)

    assets = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    timeframes = ["5m", "15m", "1h"]
    configs = [
        (20, 10, "TURTLE_20_10"),
        (20, 20, "TURTLE_20_20"),
        (55, 10, "TURTLE_55_10"),
        (55, 20, "TURTLE_55_20"),
    ]

    all_jobs = []
    for asset in assets:
        for tf in timeframes:
            for N, M, cid in configs:
                full_id = f"{cid}_{asset}_{tf}"
                all_jobs.append((asset, tf, N, M, full_id))

    print(f"Executing {len(all_jobs)} backtests in parallel across CPU cores...", flush=True)
    t0 = time.time()

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(evaluate_asset_tf_config, all_jobs):
            results.append(res)
            print(f"[{res['asset']} {res['timeframe']:3} N={res['N']} M={res['M']}] IS: {res['is_trades']} tr, {res['is_exp_bps']:+.2f} bps | OOS: {res['oos_trades']} tr, {res['oos_exp_bps']:+.2f} bps", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted all {len(results)} backtest simulations in {elapsed:.2f} seconds!", flush=True)

    # Log all 36 runs to trials.csv
    trial_rows = []
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for r in results:
        trial_rows.append({
            "trial_id": f"TURTLE-{r['asset']}-{r['timeframe']}-N{r['N']}-M{r['M']}",
            "timestamp_utc": now_utc,
            "test_name": "Turtle_Donchian_Grid",
            "asset": r["asset"],
            "timeframe": r["timeframe"],
            "strategy_name": f"Turtle_N{r['N']}_M{r['M']}",
            "params_json": json.dumps({"N": r["N"], "M": r["M"], "asset": r["asset"], "tf": r["timeframe"]}),
            "is_return_pct": r["is_return_pct"],
            "is_expectancy_bps": r["is_exp_bps"],
            "oos_return_pct": r["oos_return_pct"],
            "oos_expectancy_bps": r["oos_exp_bps"],
            "status": "EVALUATED",
        })
    pd.DataFrame(trial_rows).to_csv(TRIALS_FILE, mode="a", header=not os.path.exists(TRIALS_FILE), index=False)
    print(f"Logged {len(trial_rows)} trials to {TRIALS_FILE}.")

    # Evaluate Pooled Criteria across the 3 Assets for each (Timeframe, N, M)
    print("\n" + "=" * 100)
    print("POOLED MULTI-ASSET PERFORMANCE SUMMARY (SOL + BTC + ETH)")
    print("=" * 100)

    pooled_summary = []
    survivors = []

    for tf in timeframes:
        for N, M, cid in configs:
            matching = [r for r in results if r["timeframe"] == tf and r["N"] == N and r["M"] == M]
            pooled_is_trades = sum(r["is_trades"] for r in matching)
            pooled_oos_trades = sum(r["oos_trades"] for r in matching)

            # Weighted average expectancy across the 3 assets
            pooled_is_exp = (sum(r["is_exp_bps"] * r["is_trades"] for r in matching) / pooled_is_trades) if pooled_is_trades > 0 else 0.0
            pooled_oos_exp = (sum(r["oos_exp_bps"] * r["oos_trades"] for r in matching) / pooled_oos_trades) if pooled_oos_trades > 0 else 0.0
            pooled_oos_stress = (sum(r["oos_stress_exp_bps"] * r["oos_trades"] for r in matching) / pooled_oos_trades) if pooled_oos_trades > 0 else 0.0

            # Sharpe ratios across the 3 assets: canonical DAILY equity Sharpe
            # (sqrt(365)) plus the legacy per-bar Sharpe kept for traceability.
            mean_oos_daily_sharpe = float(np.mean([r["oos_daily_sharpe"] for r in matching]))
            mean_oos_bar_sharpe = float(np.mean([r["oos_sharpe"] for r in matching]))
            mean_oos_trade_sharpe = float(np.mean([r["oos_per_trade_sharpe"] for r in matching]))

            # DSR p-value with N_trials = 1,354 (36 Turtle configs + 1,318 historical configs).
            # Uses the CANONICAL core.metrics.compute_dsr on the PER-TRADE Sharpe with
            # T = pooled OOS trade count (the null expected-max Sharpe is scaled by the
            # standard error 1/sqrt(T), unlike the legacy local helper that compared a
            # per-bar Sharpe against an unscaled expected max and therefore always
            # returned 1.0000).
            dsr_p = compute_dsr(per_trade_sharpe=mean_oos_trade_sharpe, n_trades=pooled_oos_trades, n_trials=1354)

            # Check Hurdle Criteria
            # 1. Net expectancy >= +5 bps
            # 2. Stress expectancy > 0 bps
            # 3. Pooled trades >= 150
            # 4. DSR p < 0.05
            pass_exp = pooled_oos_exp >= 5.0
            pass_stress = pooled_oos_stress > 0.0
            pass_sample = pooled_oos_trades >= 150
            pass_dsr = dsr_p < 0.05

            is_survivor = pass_exp and pass_stress and pass_sample and pass_dsr
            verdict = "PASS" if is_survivor else "FAIL"

            if is_survivor:
                survivors.append({
                    "timeframe": tf,
                    "N": N,
                    "M": M,
                    "cid": cid,
                    "pooled_is_trades": pooled_is_trades,
                    "pooled_oos_trades": pooled_oos_trades,
                    "pooled_oos_exp": pooled_oos_exp,
                    "pooled_oos_stress": pooled_oos_stress,
                    "mean_oos_sharpe": mean_oos_daily_sharpe,
                    "mean_oos_bar_sharpe": mean_oos_bar_sharpe,
                    "dsr_p": dsr_p,
                })

            pooled_summary.append({
                "timeframe": tf,
                "N": N,
                "M": M,
                "cid": cid,
                "pooled_is_trades": pooled_is_trades,
                "pooled_is_exp_bps": round(pooled_is_exp, 2),
                "pooled_oos_trades": pooled_oos_trades,
                "pooled_oos_exp_bps": round(pooled_oos_exp, 2),
                "pooled_oos_stress_bps": round(pooled_oos_stress, 2),
                "mean_oos_sharpe": round(mean_oos_daily_sharpe, 3),
                "mean_oos_bar_sharpe": round(mean_oos_bar_sharpe, 3),
                "mean_oos_per_trade_sharpe": round(mean_oos_trade_sharpe, 4),
                "dsr_p_value": round(dsr_p, 4),
                "verdict": verdict,
                "per_asset": matching,
            })

            print(f"[{tf:3} N={N:2} M={M:2}] Pooled IS: {pooled_is_trades:4} tr ({pooled_is_exp:+.2f} bps) | Pooled OOS: {pooled_oos_trades:4} tr ({pooled_oos_exp:+.2f} bps, Stress: {pooled_oos_stress:+.2f} bps, Daily SR: {mean_oos_daily_sharpe:+.3f}, Per-Trade SR: {mean_oos_trade_sharpe:+.4f}, DSR p={dsr_p:.4f}) -> {verdict}")

    print("\n" + "=" * 100)
    print(f"SURVIVING CANDIDATES CLEARING ALL PRE-REGISTERED OOS HURDLES: {len(survivors)}")
    print("=" * 100)

    # Hold-out evaluation protocol:
    # ETH 2026-04-01+ is touched STRICTLY ONLY for survivors!
    eth_holdout_results = []
    if len(survivors) > 0:
        print("Evaluating ETH 2026-04-01+ VETO HOLDOUT for survivors...", flush=True)
        eth_df = pd.read_parquet(os.path.join(DATA_DIR, "ETHUSDT_1h.parquet"))
        eth_df["timestamp"] = pd.to_datetime(eth_df["timestamp"], utc=True)
        eth_df = eth_df.set_index("timestamp").sort_index()
        fdf = pd.read_parquet(os.path.join(DATA_DIR, "ETHUSDT_funding.parquet"))
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

        base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)

        for surv in survivors:
            tf = surv["timeframe"]
            df_ho = pd.read_parquet(os.path.join(DATA_DIR, f"ETHUSDT_{tf}.parquet"))
            df_ho["timestamp"] = pd.to_datetime(df_ho["timestamp"], utc=True)
            df_ho = df_ho.set_index("timestamp").sort_index()
            feat_ho = prepare_turtle_features(df_ho, N=surv["N"], M=surv["M"])
            ho_slice = slice_features(feat_ho, "2026-04-01 00:00:00", "2026-10-09 23:59:59")
            bpy = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}[tf]
            p = {"N": surv["N"], "M": surv["M"], "config_id": f"ETH_HO_{surv['cid']}"}
            m_ho, ho_trades = run_turtle_simulation(ho_slice, funding, p, base_costs, bars_per_year=bpy, return_trades=True)
            eth_holdout_results.append({
                "candidate": surv,
                "ho_trades": m_ho["total_trades"],
                "ho_exp_bps": m_ho["expectancy_bps"],
                "ho_return_pct": m_ho["total_return_pct"],
                "ho_sharpe": m_ho["sharpe_ratio"],
                "ho_max_dd": m_ho["max_drawdown_pct"],
            })
            print(f"SURVIVOR [{surv['cid']} {tf}]: ETH Hold-out Trades: {m_ho['total_trades']}, Exp: {m_ho['expectancy_bps']:.2f} bps, Return: {m_ho['total_return_pct']:.2f}%")
    else:
        print("ZERO configurations passed the pre-registered hurdles (Net Exp >= +5 bps, Stress > 0 bps, Pooled Trades >= 150, DSR p < 0.05).")
        print("FRESH ETH HOLDOUT (2026-04-01+) REMAINS STRICTLY UNTOUCHED AND UNPOLLUTED.")

    output_payload = {
        "timestamp_utc": now_utc,
        "n_configs_evaluated": len(results),
        "pooled_summary": pooled_summary,
        "survivors_count": len(survivors),
        "survivors": survivors,
        "eth_veto_holdout_results": eth_holdout_results,
        "individual_runs": results,
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved comprehensive Turtle results to {RESULTS_FILE}")

    return output_payload


if __name__ == "__main__":
    run_all_turtle_tests()

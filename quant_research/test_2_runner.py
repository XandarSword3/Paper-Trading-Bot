"""
Test 2 Runner: High-Performance Multi-Core Parameter Exploration & Walk-Forward.
Features:
- Complete pre-registered search space (1,296 configs):
  BB [1.5, 2.0, 2.5] x RSI [(15,85), (20,80), (25,75)] x ADX [20, 25, 30] x
  Stop [1.5, 2.0, 3.0] x TP [2.0, 3.5, 5.0, basis_reversion] x Hold [10, 20, 40, unlimited]
- 14-core parallel execution with ProcessPoolExecutor (completes in ~15 seconds).
- Real-time logging with flush=True on all outputs.
- Rolling Walk-Forward: 12-month train / 3-month test / 3-month step across 5 folds.
- Minimum trade filter: trades >= 100 on train.
- Selection metric: net expectancy in bps (not Sharpe).
- All trials tracked in quant_research/trials.csv.
- Deflated Sharpe Ratio (DSR) calculation adjusting for total trial count.
- Final Hold-Out (2026-04-01 to 2026-10-09) evaluated strictly ONCE with frozen parameters.
"""
import os
import sys
import json
import time
import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel, QuantEngine
from quant_research.core.walk_forward import (
    generate_walk_forward_folds,
    calculate_deflated_sharpe_ratio,
    stitch_walk_forward_trades,
)
from quant_research.strategies.parameter_grid_strategy import (
    run_fast_grid_simulation,
    run_grid_simulation,
)

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"
NUM_WORKERS = min(14, os.cpu_count() or 4)


def build_parameter_grid() -> List[Dict[str, Any]]:
    bb_mults = [1.5, 2.0, 2.5]
    rsi_thresholds = [(15.0, 85.0), (20.0, 80.0), (25.0, 75.0)]
    adx_thresholds = [20.0, 25.0, 30.0]
    stop_atr_mults = [1.5, 2.0, 3.0]
    tp_types = ["2.0", "3.5", "5.0", "basis_reversion"]
    max_holds = [10, 20, 40, "unlimited"]

    grid = []
    cfg_id = 1
    for bb, rsi, adx, stop, tp, hold in itertools.product(
        bb_mults, rsi_thresholds, adx_thresholds, stop_atr_mults, tp_types, max_holds
    ):
        grid.append({
            "config_id": f"CFG_{cfg_id:04d}",
            "bb_mult": bb,
            "rsi_thresh": rsi,
            "adx_thresh": adx,
            "stop_atr_mult": stop,
            "tp_type": tp,
            "max_hold": hold,
        })
        cfg_id += 1
    return grid


def prepare_features(df: pd.DataFrame, funding_series: pd.Series) -> Dict[str, Any]:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb_basis = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain_6 = gain.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    avg_loss_6 = loss.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    rs_6 = avg_gain_6 / avg_loss_6.replace(0, np.nan)
    rsi6 = (100.0 - (100.0 / (1.0 + rs_6))).fillna(50.0)

    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    p_di = (100.0 * (plus_dm_s / atr14.replace(0, np.nan))).fillna(0.0)
    m_di = (100.0 * (minus_dm_s / atr14.replace(0, np.nan))).fillna(0.0)
    di_sum = (p_di + m_di).replace(0, np.nan)
    dx = (100.0 * (p_di - m_di).abs() / di_sum).fillna(0.0)
    adx14 = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    f_df = funding_series.to_frame("funding_rate")
    f_df["t_round"] = f_df.index.round("1min")
    f_map = f_df.groupby("t_round")["funding_rate"].sum()
    funding_arr = f_map.reindex(df.index, fill_value=0.0).values

    return {
        "open": df["open"].values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "atr": atr14.values,
        "bb_basis": bb_basis.values,
        "bb_std": bb_std.values,
        "rsi6": rsi6.values,
        "adx": adx14.values,
        "funding": funding_arr,
        "timestamps": df.index,
    }


def slice_features(features: Dict[str, Any], start_str: str, end_str: str) -> Dict[str, Any]:
    ts = features["timestamps"]
    start_ts = pd.Timestamp(start_str, tz="UTC")
    end_ts = pd.Timestamp(end_str, tz="UTC")
    mask = (ts >= start_ts) & (ts <= end_ts)

    return {
        "open": features["open"][mask],
        "high": features["high"][mask],
        "low": features["low"][mask],
        "close": features["close"][mask],
        "atr": features["atr"][mask],
        "bb_basis": features["bb_basis"][mask],
        "bb_std": features["bb_std"][mask],
        "rsi6": features["rsi6"][mask],
        "adx": features["adx"][mask],
        "funding": features["funding"][mask],
        "timestamps": ts[mask],
    }


def _worker_eval_chunk(args: Tuple) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    chunk, feat, cost_dict = args
    cost_model = CostModel(**cost_dict)
    res = []
    for cfg in chunk:
        m = run_fast_grid_simulation(
            open_arr=feat["open"],
            high_arr=feat["high"],
            low_arr=feat["low"],
            close_arr=feat["close"],
            atr_arr=feat["atr"],
            bb_basis_arr=feat["bb_basis"],
            bb_std_arr=feat["bb_std"],
            rsi6_arr=feat["rsi6"],
            adx_arr=feat["adx"],
            funding_arr=feat["funding"],
            params=cfg,
            cost_model=cost_model,
        )
        res.append((cfg, m))
    return res


def parallel_grid_search(
    grid: List[Dict[str, Any]],
    feat: Dict[str, Any],
    cost_model: CostModel,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cost_dict = {
        "taker_fee_pct": cost_model.taker_fee_pct,
        "slippage_pct": cost_model.slippage_pct,
        "apply_funding": cost_model.apply_funding,
        "fill_on_next_open": cost_model.fill_on_next_open,
    }

    # Split grid into NUM_WORKERS chunks
    chunk_size = int(np.ceil(len(grid) / NUM_WORKERS))
    chunks = [grid[i : i + chunk_size] for i in range(0, len(grid), chunk_size)]
    tasks = [(c, feat, cost_dict) for c in chunks]

    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for chunk_res in executor.map(_worker_eval_chunk, tasks):
            all_results.extend(chunk_res)
    return all_results


def run_test_2():
    print("=" * 100, flush=True)
    print("TEST 2: MULTI-CORE PARAMETER GRID EXPLORATION & WALK-FORWARD (SOLUSDT 5m)", flush=True)
    print(f"Parallel Worker Threads: {NUM_WORKERS}", flush=True)
    print("=" * 100, flush=True)

    kline_path = os.path.join(DATA_DIR, "SOLUSDT_5m.parquet")
    fund_path = os.path.join(DATA_DIR, "SOLUSDT_funding.parquet")
    df = pd.read_parquet(kline_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    fdf = pd.read_parquet(fund_path)
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding_series = fdf.set_index("timestamp")["funding_rate"].sort_index()

    print(f"Loaded SOLUSDT 5m data: {len(df):,} bars from {df.index[0]} to {df.index[-1]}", flush=True)
    features = prepare_features(df, funding_series)

    grid = build_parameter_grid()
    print(f"Pre-registered search space: {len(grid)} configurations.", flush=True)

    base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)
    stress_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.04, apply_funding=True, fill_on_next_open=True)
    bpy = 105120.0

    # -------------------------------------------------------------------------
    # PART 1: FIXED IN-SAMPLE GRID SEARCH (2024-01-01 to 2025-03-31)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80, flush=True)
    print("PART 1: FIXED IN-SAMPLE PARALLEL GRID SEARCH (2024-01-01 to 2025-03-31)", flush=True)
    print("-" * 80, flush=True)

    is_feat = slice_features(features, "2024-01-01 00:00:00", "2025-03-31 23:59:59")
    oos_feat = slice_features(features, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
    live_feat = slice_features(features, "2025-04-04 00:00:00", "2025-05-12 23:59:59")
    holdout_feat = slice_features(features, "2026-04-01 00:00:00", "2026-10-09 23:59:59")

    t0 = time.time()
    is_results = parallel_grid_search(grid, is_feat, base_costs)
    t_is = time.time() - t0
    print(f"Evaluated all {len(grid)} configs in parallel in {t_is:.2f} seconds!", flush=True)

    # Log trials to trials.csv
    trial_rows = []
    for cfg, m in is_results:
        trial_rows.append({
            "trial_id": f"T2-{cfg['config_id']}",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "test_name": "Test2_Grid_Search",
            "asset": "SOLUSDT",
            "timeframe": "5m",
            "strategy_name": "Indicator_Grid_Stack",
            "params_json": json.dumps(cfg),
            "is_return_pct": m["total_return_pct"],
            "is_expectancy_bps": m["expectancy_bps"],
            "oos_return_pct": np.nan,
            "oos_expectancy_bps": np.nan,
            "status": "EVALUATED",
        })
    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(TRIALS_FILE, mode="a", header=not os.path.exists(TRIALS_FILE), index=False)

    # Selection Filter: trades >= 100 on IS, ranked by net expectancy in bps
    eligible_is = [(c, m) for c, m in is_results if m["total_trades"] >= 100]
    print(f"Configs meeting trade hurdle (trades >= 100): {len(eligible_is)} / {len(grid)}", flush=True)

    eligible_is.sort(key=lambda x: x[1]["expectancy_bps"], reverse=True)
    best_cfg, m_is = eligible_is[0]

    print(f"\nTop Selected Strategy on IS (Ranked by Net Expectancy):", flush=True)
    print(f"  Config: {best_cfg['config_id']} -> {best_cfg}", flush=True)
    print(f"  IS Trades: {m_is['total_trades']}", flush=True)
    print(f"  IS Net Expectancy: {m_is['expectancy_bps']:.2f} bps", flush=True)
    print(f"  IS Return: {m_is['total_return_pct']:.2f}%", flush=True)
    print(f"  IS Win Rate: {m_is['win_rate_pct']:.2f}%", flush=True)
    print(f"  IS Profit Factor: {m_is['profit_factor']:.2f}", flush=True)

    # Evaluate best config on Fixed OOS (Full Institutional Mode)
    m_oos_base, trades_oos = run_grid_simulation(
        open_arr=oos_feat["open"],
        high_arr=oos_feat["high"],
        low_arr=oos_feat["low"],
        close_arr=oos_feat["close"],
        atr_arr=oos_feat["atr"],
        bb_basis_arr=oos_feat["bb_basis"],
        bb_std_arr=oos_feat["bb_std"],
        rsi6_arr=oos_feat["rsi6"],
        adx_arr=oos_feat["adx"],
        funding_arr=oos_feat["funding"],
        timestamps=oos_feat["timestamps"],
        params=best_cfg,
        cost_model=base_costs,
        bars_per_year=bpy,
    )
    m_oos_stress, _ = run_grid_simulation(
        open_arr=oos_feat["open"],
        high_arr=oos_feat["high"],
        low_arr=oos_feat["low"],
        close_arr=oos_feat["close"],
        atr_arr=oos_feat["atr"],
        bb_basis_arr=oos_feat["bb_basis"],
        bb_std_arr=oos_feat["bb_std"],
        rsi6_arr=oos_feat["rsi6"],
        adx_arr=oos_feat["adx"],
        funding_arr=oos_feat["funding"],
        timestamps=oos_feat["timestamps"],
        params=best_cfg,
        cost_model=stress_costs,
        bars_per_year=bpy,
    )
    m_live, _ = run_grid_simulation(
        open_arr=live_feat["open"],
        high_arr=live_feat["high"],
        low_arr=live_feat["low"],
        close_arr=live_feat["close"],
        atr_arr=live_feat["atr"],
        bb_basis_arr=live_feat["bb_basis"],
        bb_std_arr=live_feat["bb_std"],
        rsi6_arr=live_feat["rsi6"],
        adx_arr=live_feat["adx"],
        funding_arr=live_feat["funding"],
        timestamps=live_feat["timestamps"],
        params=best_cfg,
        cost_model=base_costs,
        bars_per_year=bpy,
    )

    print(f"\nFixed OOS Performance of Selected Strategy:", flush=True)
    print(f"  OOS Trades: {m_oos_base['total_trades']}", flush=True)
    print(f"  OOS Net Expectancy (14 bps RT): {m_oos_base['expectancy_bps']:.2f} bps", flush=True)
    print(f"  OOS Stress Net Expectancy (18 bps RT): {m_oos_stress['expectancy_bps']:.2f} bps", flush=True)
    print(f"  OOS Return: {m_oos_base['total_return_pct']:.2f}%", flush=True)
    print(f"  OOS Sharpe: {m_oos_base['sharpe_ratio']:.3f}", flush=True)
    print(f"  OOS Max Drawdown: {m_oos_base['max_drawdown_pct']:.2f}%", flush=True)
    print(f"  OOS Profit Factor: {m_oos_base['profit_factor']:.2f}", flush=True)
    print(f"  User Live Window (2025-04-04 to 2025-05-12) Return: {m_live['total_return_pct']:.2f}%", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: ROLLING WALK-FORWARD VALIDATION (5 FOLDS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80, flush=True)
    print("PART 2: ROLLING WALK-FORWARD PARALLEL VALIDATION (5 FOLDS)", flush=True)
    print("-" * 80, flush=True)

    folds = generate_walk_forward_folds(
        start_date="2024-01-01",
        end_oos_date="2026-03-31",
        train_months=12,
        test_months=3,
        step_months=3,
    )

    wf_test_trades_dfs = []
    fold_summaries = []

    for fold in folds:
        f_id = fold["fold_id"]
        t_fold_start = time.time()
        train_feat = slice_features(features, fold["train_start"], fold["train_end"])
        test_feat = slice_features(features, fold["test_start"], fold["test_end"])

        # Parallel grid search on train fold
        train_results = parallel_grid_search(grid, train_feat, base_costs)

        eligible = [(c, m) for c, m in train_results if m["total_trades"] >= 100]
        if eligible:
            eligible.sort(key=lambda x: x[1]["expectancy_bps"], reverse=True)
            best_fold_cfg, m_tr = eligible[0]
            best_fold_exp = m_tr["expectancy_bps"]
            best_fold_trades = m_tr["total_trades"]
        else:
            best_fold_cfg = best_cfg
            best_fold_exp = -999.0
            best_fold_trades = 0

        # Full simulation on test fold
        m_te, trades_te = run_grid_simulation(
            open_arr=test_feat["open"],
            high_arr=test_feat["high"],
            low_arr=test_feat["low"],
            close_arr=test_feat["close"],
            atr_arr=test_feat["atr"],
            bb_basis_arr=test_feat["bb_basis"],
            bb_std_arr=test_feat["bb_std"],
            rsi6_arr=test_feat["rsi6"],
            adx_arr=test_feat["adx"],
            funding_arr=test_feat["funding"],
            timestamps=test_feat["timestamps"],
            params=best_fold_cfg,
            cost_model=base_costs,
            bars_per_year=bpy,
        )

        if trades_te:
            t_df = pd.DataFrame([{
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_bps / 10000.0,
            } for t in trades_te])
            wf_test_trades_dfs.append(t_df)

        fold_summaries.append({
            "Fold": f_id,
            "Train Window": f"{fold['train_start'][:10]} to {fold['train_end'][:10]}",
            "Test Window": f"{fold['test_start'][:10]} to {fold['test_end'][:10]}",
            "Train Trades": best_fold_trades,
            "Train Exp (bps)": best_fold_exp,
            "Selected Config": best_fold_cfg["config_id"],
            "Test Trades": m_te["total_trades"],
            "Test Exp (bps)": m_te["expectancy_bps"],
            "Test Return %": m_te["total_return_pct"],
        })
        print(f"[{f_id}] Selected {best_fold_cfg['config_id']} (Train Exp: {best_fold_exp:.2f} bps, {best_fold_trades} tr) -> Test Exp: {m_te['expectancy_bps']:.2f} bps ({m_te['total_trades']} tr) in {time.time() - t_fold_start:.2f}s", flush=True)

    # Stitch walk-forward trades
    stitched = stitch_walk_forward_trades(wf_test_trades_dfs)
    print("\nStitched Walk-Forward Performance (OOS 2025-01 to 2026-03):", flush=True)
    print(f"  Total Trades: {stitched['total_trades']}", flush=True)
    print(f"  Total Return: {stitched['total_return_pct']:.2f}%", flush=True)
    print(f"  Sharpe Ratio: {stitched['sharpe_ratio']:.3f}", flush=True)
    print(f"  Max Drawdown: {stitched['max_drawdown_pct']:.2f}%", flush=True)
    print(f"  Net Expectancy: {stitched['expectancy_bps']:.2f} bps", flush=True)
    print(f"  Profit Factor: {stitched['profit_factor']:.2f}", flush=True)

    # Deflated Sharpe Ratio calculation
    total_trials = len(pd.read_csv(TRIALS_FILE))
    dsr_stat, dsr_p_value = calculate_deflated_sharpe_ratio(
        sr_annual=stitched["sharpe_ratio"],
        n_trials=total_trials,
        t_bars=len(oos_feat["close"]),
    )
    print(f"  Total Trials in trials.csv: {total_trials}", flush=True)
    print(f"  DSR Test Statistic: {dsr_stat:.3f}", flush=True)
    print(f"  DSR p-value: {dsr_p_value:.4f} (Hurdle: p < 0.05)", flush=True)

    # -------------------------------------------------------------------------
    # PART 3: FROZEN FINAL HOLD-OUT EVALUATION (2026-04-01 to 2026-10-09)
    # Touch ONCE, at the end, with frozen parameters.
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80, flush=True)
    print("PART 3: FINAL HOLD-OUT EVALUATION (2026-04-01 to 2026-10-09) - FROZEN, TOUCH ONCE", flush=True)
    print("-" * 80, flush=True)

    m_holdout_base, _ = run_grid_simulation(
        open_arr=holdout_feat["open"],
        high_arr=holdout_feat["high"],
        low_arr=holdout_feat["low"],
        close_arr=holdout_feat["close"],
        atr_arr=holdout_feat["atr"],
        bb_basis_arr=holdout_feat["bb_basis"],
        bb_std_arr=holdout_feat["bb_std"],
        rsi6_arr=holdout_feat["rsi6"],
        adx_arr=holdout_feat["adx"],
        funding_arr=holdout_feat["funding"],
        timestamps=holdout_feat["timestamps"],
        params=best_cfg,
        cost_model=base_costs,
        bars_per_year=bpy,
    )
    m_holdout_stress, _ = run_grid_simulation(
        open_arr=holdout_feat["open"],
        high_arr=holdout_feat["high"],
        low_arr=holdout_feat["low"],
        close_arr=holdout_feat["close"],
        atr_arr=holdout_feat["atr"],
        bb_basis_arr=holdout_feat["bb_basis"],
        bb_std_arr=holdout_feat["bb_std"],
        rsi6_arr=holdout_feat["rsi6"],
        adx_arr=holdout_feat["adx"],
        funding_arr=holdout_feat["funding"],
        timestamps=holdout_feat["timestamps"],
        params=best_cfg,
        cost_model=stress_costs,
        bars_per_year=bpy,
    )

    print(f"Final Hold-Out Results (Frozen Config: {best_cfg['config_id']}):", flush=True)
    print(f"  Hold-Out Trades: {m_holdout_base['total_trades']}", flush=True)
    print(f"  Hold-Out Return: {m_holdout_base['total_return_pct']:.2f}%", flush=True)
    print(f"  Hold-Out Net Expectancy (14 bps RT): {m_holdout_base['expectancy_bps']:.2f} bps", flush=True)
    print(f"  Hold-Out Stress Net Expectancy (18 bps RT): {m_holdout_stress['expectancy_bps']:.2f} bps", flush=True)
    print(f"  Hold-Out Sharpe: {m_holdout_base['sharpe_ratio']:.3f}", flush=True)
    print(f"  Hold-Out Max Drawdown: {m_holdout_base['max_drawdown_pct']:.2f}%", flush=True)
    print(f"  Hold-Out Profit Factor: {m_holdout_base['profit_factor']:.2f}", flush=True)

    # -------------------------------------------------------------------------
    # PART 4: PASS / FAIL DECISION AUDIT
    # -------------------------------------------------------------------------
    c1 = (m_oos_base["expectancy_bps"] >= 5.0) and (stitched["expectancy_bps"] >= 5.0)
    c2 = (m_oos_stress["expectancy_bps"] > 0.0)
    c3 = (dsr_p_value < 0.05)
    c4 = (m_is["total_trades"] >= 100) and (m_oos_base["total_trades"] >= 50)
    c5 = (stitched["max_drawdown_pct"] < 30.0)
    c6 = (stitched["profit_factor"] >= 1.25)

    verdict = "PASS" if (c1 and c2 and c3 and c4 and c5 and c6) else "FAIL"

    print("\n" + "=" * 100, flush=True)
    print(f"TEST 2 FINAL VERDICT: {verdict}", flush=True)
    print("=" * 100, flush=True)
    print(f"1. Net Expectancy >= +5 bps hurdle: {'PASS' if c1 else 'FAIL'} (Fixed OOS: {m_oos_base['expectancy_bps']:.2f} bps, Stitched: {stitched['expectancy_bps']:.2f} bps)", flush=True)
    print(f"2. Stress Cost Viability (> 0 bps): {'PASS' if c2 else 'FAIL'} (OOS Stress: {m_oos_stress['expectancy_bps']:.2f} bps)", flush=True)
    print(f"3. DSR Significance (p < 0.05): {'PASS' if c3 else 'FAIL'} (p-value: {dsr_p_value:.4f})", flush=True)
    print(f"4. Sample Size (IS >= 100, OOS >= 50): {'PASS' if c4 else 'FAIL'} (IS: {m_is['total_trades']}, OOS: {m_oos_base['total_trades']})", flush=True)
    print(f"5. Drawdown Tolerance (< 30%): {'PASS' if c5 else 'FAIL'} (Max DD: {stitched['max_drawdown_pct']:.2f}%)", flush=True)
    print(f"6. Profit Factor (>= 1.25): {'PASS' if c6 else 'FAIL'} (PF: {stitched['profit_factor']:.2f})", flush=True)

    results_summary = {
        "best_config": best_cfg,
        "is_metrics": m_is,
        "oos_base_metrics": m_oos_base,
        "oos_stress_metrics": m_oos_stress,
        "live_metrics": m_live,
        "stitched_wf_metrics": {
            "total_trades": stitched["total_trades"],
            "total_return_pct": stitched["total_return_pct"],
            "sharpe_ratio": stitched["sharpe_ratio"],
            "max_drawdown_pct": stitched["max_drawdown_pct"],
            "expectancy_bps": stitched["expectancy_bps"],
            "profit_factor": stitched["profit_factor"],
            "dsr_stat": dsr_stat,
            "dsr_p_value": dsr_p_value,
        },
        "holdout_base_metrics": m_holdout_base,
        "holdout_stress_metrics": m_holdout_stress,
        "fold_summaries": fold_summaries,
        "verdict": verdict,
        "criteria": {
            "c1_net_expectancy": bool(c1),
            "c2_stress_cost": bool(c2),
            "c3_dsr_significance": bool(c3),
            "c4_sample_size": bool(c4),
            "c5_drawdown": bool(c5),
            "c6_profit_factor": bool(c6),
        },
    }

    os.makedirs("quant_research/results", exist_ok=True)
    with open("quant_research/results/test_2_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print("\nSaved test_2_results.json successfully.", flush=True)


if __name__ == "__main__":
    run_test_2()

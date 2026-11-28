"""
Amendment C runner: PREREG section 10, committed BEFORE running.

Two arms, both on the exact section-7 Donchian machinery (same features, same
costs, same funding, same 1m intrabar resolution, same 1%-risk sizing and 2x
leverage cap), differing ONLY in what section 10 registers:

  C0  "1-unit"       : max_units=1, NO daily filter   -> isolates pyramiding
  C1  "DT-Filter"    : max_units=1, daily-trend filter -> C1 vs C0 = filter effect

72 new configurations (2 arms x 3 assets x 3 timeframes x 4 channel pairs),
appended to trials.csv; DSR evaluated at N_trials = 1,354 + 72 = 1,426.

The daily filter is the audited condition VERBATIM (entry_autopsy.py): previous
COMPLETED daily close versus its own EMA(20) on the 1m close series resampled to
1D. Long entries are admitted only on trend == +1, short only on trend == -1;
bars before the first completed prior daily bar admit no signal.

No ETH 2026-04+ hold-out window is touched by this runner.
"""
import os
import sys
import json
import time
import datetime

import numpy as np
import pandas as pd
from typing import Dict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel
from quant_research.core.metrics import compute_dsr
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_runner import slice_features, per_trade_sharpe

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"
DEFECTIVE_RESULTS_FILE = "quant_research/results/amendment_c_results.json"   # v1: 4-unit (defect, preserved)
RESULTS_FILE = "quant_research/results/amendment_c_results_v2.json"          # corrected 1-unit re-run

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]
CHANNELS = [(20, 10, "DT_20_10"), (20, 20, "DT_20_20"), (55, 10, "DT_55_10"), (55, 20, "DT_55_20")]
ARMS = [("C0_no_filter_1unit", False), ("C1_daily_filter_1unit", True)]
BARS_PER_YEAR = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}
N_TRIALS_DSR = 1354 + 72          # PREREG section 10: 1,354 historical + 72 new
DAILY_EMA_SPAN = 20

BASE_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)
STRESS_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.04, apply_funding=True, fill_on_next_open=True)

WINDOWS = {
    "is":  ("2024-01-01 00:00:00", "2025-03-31 23:59:59"),
    "oos": ("2025-04-01 00:00:00", "2026-03-31 23:59:59"),
    "stitched": ("2024-01-01 00:00:00", "2026-03-31 23:59:59"),   # PREREG C.3 hurdle 5
}


def daily_trend_from_daily_close(daily_close: pd.Series,
                                 span: int = DAILY_EMA_SPAN) -> pd.Series:
    """The audited daily EMAn trend from a daily close series.

    This is the single source of truth for the trend VALUES: EMA(span,
    adjust=False) seeded at the series start, +1 if close > EMA else -1.
    ``daily_trend_map`` (archive parquet path) and the V4 adapter both
    delegate here, so there is exactly one implementation.
    """
    daily = daily_close.sort_index().dropna()
    ema = daily.ewm(span=span, adjust=False).mean()
    return pd.Series(np.where(daily > ema, 1.0, -1.0), index=daily.index)


def daily_trend_map(asset: str) -> pd.Series:
    """The audited daily EMAn(20) trend: index = daily timestamps, values +1/-1.

    Identical construction to entry_autopsy.load_features: last close of each UTC
    day resampled from the 1m series, EMA(span=20, adjust=False) seeded at the
    archive start, +1 if close > EMA else -1. A missing entry (warmup) blocks
    trading rather than defaulting to a direction.
    """
    m1 = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1m.parquet"))
    daily = m1.set_index("timestamp")["close"].sort_index().resample("1D").last().dropna()
    return daily_trend_from_daily_close(daily)


def apply_daily_mask(features: Dict, trend: pd.Series) -> Dict:
    """Gate entry signals by the daily trend, direction-aware, on a COPY.

    Blocking a direction is done by pushing its channel level out of reach, so the
    simulation code itself (shared with the committed section-7 runner) never
    changes: long_signal = close > entry_high cannot fire on +inf, short cannot
    fire on -inf. Exits are untouched: a position opened while the filter agreed
    is managed to its stop or channel exit even if the daily trend flips.
    """
    ts = pd.DatetimeIndex(features["timestamps"])
    prev_day = ts.normalize() - pd.Timedelta(days=1)
    pos = trend.index.searchsorted(prev_day)
    pos = np.clip(pos, 0, len(trend.index) - 1)
    hit = trend.index[pos] == prev_day
    trend_vals = trend.to_numpy(dtype=np.float64)
    t = np.where(hit, trend_vals[pos], np.nan)

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in features.items()}
    blocked_long = np.isnan(t) | (t != 1.0)
    blocked_short = np.isnan(t) | (t != -1.0)
    out["entry_high"] = np.where(blocked_long, np.inf, out["entry_high"])
    out["entry_low"] = np.where(blocked_short, -np.inf, out["entry_low"])
    return out


def evaluate_job(job):
    asset, tf, N, M, arm, cfg_id = job
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_{tf}.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

    features = prepare_turtle_features(df, N=N, M=M)
    if arm == "C1_daily_filter_1unit":
        features = apply_daily_mask(features, daily_trend_map(asset))

    params = {"N": N, "M": M, "max_units": 1, "asset": asset, "timeframe": tf,
              "arm": arm, "config_id": cfg_id}
    bpy = BARS_PER_YEAR[tf]

    is_f = slice_features(features, *WINDOWS["is"])
    oos_f = slice_features(features, *WINDOWS["oos"])
    stitched_f = slice_features(features, *WINDOWS["stitched"])

    # max_units=1 is passed EXPLICITLY (the params dict is not read by the engine).
    # Defect history: the original Amendment C run (results/amendment_c_results.json)
    # omitted this argument, so all 72 configurations silently ran at the engine
    # default max_units=4; the artifact below is the corrected 1-unit re-run.
    m_is_base, trades_is = run_turtle_simulation(is_f, funding, params, BASE_COSTS, bars_per_year=bpy, return_trades=True,
                                                 max_units=1)
    m_is_stress = run_turtle_simulation(is_f, funding, params, STRESS_COSTS, bars_per_year=bpy, return_trades=False,
                                        max_units=1)
    m_oos_base, trades_oos = run_turtle_simulation(oos_f, funding, params, BASE_COSTS, bars_per_year=bpy, return_trades=True,
                                                   max_units=1)
    m_oos_stress = run_turtle_simulation(oos_f, funding, params, STRESS_COSTS, bars_per_year=bpy, return_trades=False,
                                         max_units=1)
    m_stitched = run_turtle_simulation(stitched_f, funding, params, BASE_COSTS, bars_per_year=bpy, return_trades=False,
                                       max_units=1)

    def _gross(trades):
        win = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        loss = -sum(t.gross_pnl for t in trades if t.gross_pnl < 0)
        return win, loss

    is_win, is_loss = _gross(trades_is)
    oos_win, oos_loss = _gross(trades_oos)

    return {
        "arm": arm, "asset": asset, "timeframe": tf, "N": N, "M": M, "config_id": cfg_id,
        "is_trades": m_is_base["total_trades"], "is_exp_bps": m_is_base["expectancy_bps"],
        "is_return_pct": m_is_base["total_return_pct"], "is_daily_sharpe": m_is_base["daily_sharpe"],
        "is_max_dd": m_is_base["max_drawdown_pct"], "is_pf": m_is_base["profit_factor"],
        "is_win_rate": m_is_base["win_rate_pct"], "is_stress_exp_bps": m_is_stress["expectancy_bps"],
        "is_gross_win": is_win, "is_gross_loss": is_loss,
        "is_per_trade_sharpe": round(per_trade_sharpe(trades_is), 6),
        "oos_trades": m_oos_base["total_trades"], "oos_exp_bps": m_oos_base["expectancy_bps"],
        "oos_return_pct": m_oos_base["total_return_pct"], "oos_daily_sharpe": m_oos_base["daily_sharpe"],
        "oos_max_dd": m_oos_base["max_drawdown_pct"], "oos_win_rate": m_oos_base["win_rate_pct"],
        "oos_stress_exp_bps": m_oos_stress["expectancy_bps"],
        "oos_gross_win": oos_win, "oos_gross_loss": oos_loss,
        "oos_per_trade_sharpe": round(per_trade_sharpe(trades_oos), 6),
        "stitched_max_dd": m_stitched["max_drawdown_pct"], "stitched_return_pct": m_stitched["total_return_pct"],
    }


def run_and_write(out_path: str = RESULTS_FILE, log_trials: bool = True):
    print("=" * 100)
    print("AMENDMENT C: DT-FILTERED DONCHIAN (1 UNIT) vs UNFILTERED DONCHIAN (1 UNIT)")
    print(f"72 configurations | DSR N_trials = {N_TRIALS_DSR} | ETH hold-out NOT touched")
    print("=" * 100, flush=True)

    jobs = []
    for arm, _ in ARMS:
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                for N, M, cid in CHANNELS:
                    jobs.append((asset, tf, N, M, arm, f"{cid}_{asset}_{tf}_{arm}"))

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor() as ex:
        for res in ex.map(evaluate_job, jobs):
            rows.append(res)
            print(f"[{res['arm']} {res['asset']} {res['timeframe']:>3} N={res['N']:>2} M={res['M']}] "
                  f"IS {res['is_trades']:>4} tr {res['is_exp_bps']:+8.2f} bps | "
                  f"OOS {res['oos_trades']:>4} tr {res['oos_exp_bps']:+8.2f} bps", flush=True)
    print(f"\n{len(rows)} backtests in {time.time() - t0:.1f}s", flush=True)

    # pooled per (arm, tf, N, M) over the three assets - same convention as the
    # committed section-7 runner (trade-weighted expectancies, mean per-trade Sharpe)
    pooled = []
    for arm, _ in ARMS:
        for tf in TIMEFRAMES:
            for N, M, cid in CHANNELS:
                g = [r for r in rows if r["arm"] == arm and r["timeframe"] == tf
                     and r["N"] == N and r["M"] == M]
                is_tr = sum(r["is_trades"] for r in g)
                oos_tr = sum(r["oos_trades"] for r in g)
                exp_is = sum(r["is_exp_bps"] * r["is_trades"] for r in g) / is_tr if is_tr else 0.0
                exp_oos = sum(r["oos_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
                exp_stress = sum(r["oos_stress_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
                trade_sharpe = float(np.mean([r["oos_per_trade_sharpe"] for r in g])) if g else 0.0
                dsr_p = compute_dsr(per_trade_sharpe=trade_sharpe, n_trades=oos_tr, n_trials=N_TRIALS_DSR)
                pooled_pf = (sum(r["oos_gross_win"] for r in g)
                             / sum(r["oos_gross_loss"] for r in g)
                             if sum(r["oos_gross_loss"] for r in g) > 0 else 0.0)
                worst_dd = max(r["stitched_max_dd"] for r in g)
                hurdles = {
                    "net_expectancy_ge_5bps": bool(exp_oos >= 5.0),
                    "stress_expectancy_gt_0bps": bool(exp_stress > 0.0),
                    "pooled_oos_trades_ge_150": bool(oos_tr >= 150),
                    "dsr_p_lt_0_05": bool(dsr_p < 0.05),
                    "stitched_max_dd_lt_30pct": bool(worst_dd < 30.0),
                    "pooled_oos_pf_ge_1_25": bool(pooled_pf >= 1.25),
                }
                verdict = "PASS" if all(hurdles.values()) else "FAIL"
                pooled.append({
                    "arm": arm, "timeframe": tf, "N": N, "M": M, "config_id": cid,
                    "pooled_is_trades": is_tr, "pooled_is_exp_bps": exp_is,
                    "pooled_oos_trades": oos_tr, "pooled_oos_exp_bps": exp_oos,
                    "pooled_oos_stress_exp_bps": exp_stress,
                    "mean_oos_per_trade_sharpe": trade_sharpe, "dsr_p_value": dsr_p,
                    "pooled_oos_profit_factor": pooled_pf,
                    "worst_asset_stitched_max_dd_pct": worst_dd,
                    "hurdles": hurdles, "verdict": verdict,
                    "failed_hurdles": [k for k, ok in hurdles.items() if not ok],
                })
                print(f"{arm} {tf:>3} N={N:>2} M={M}: OOS {oos_tr:>5} tr {exp_oos:+8.2f} bps "
                      f"stress {exp_stress:+8.2f} PF {pooled_pf:5.2f} DSR {dsr_p:.4f} "
                      f"DD {worst_dd:5.1f}% -> {verdict}", flush=True)

    # attribution: C1 - C0 pooled OOS expectancy at equal risk layering
    attr = []
    pmap = {(p["arm"], p["timeframe"], p["N"], p["M"]): p for p in pooled}
    for tf in TIMEFRAMES:
        for N, M, cid in CHANNELS:
            c0, c1 = pmap[("C0_no_filter_1unit", tf, N, M)], pmap[("C1_daily_filter_1unit", tf, N, M)]
            attr.append({
                "timeframe": tf, "N": N, "M": M,
                "c0_oos_exp_bps": c0["pooled_oos_exp_bps"], "c1_oos_exp_bps": c1["pooled_oos_exp_bps"],
                "filter_delta_bps": c1["pooled_oos_exp_bps"] - c0["pooled_oos_exp_bps"],
                "c0_oos_trades": c0["pooled_oos_trades"], "c1_oos_trades": c1["pooled_oos_trades"],
                "c0_verdict": c0["verdict"], "c1_verdict": c1["verdict"],
            })
    attr.sort(key=lambda r: -r["filter_delta_bps"])

    survivors = [p for p in pooled if p["verdict"] == "PASS"]
    payload = {
        "registered_by": "quant_research/PREREG.md section 10 (Amendment C)",
        "runner": "quant_research/candidate_runner.py",
        "correction_note": "v1 of this artifact (amendment_c_results.json) was generated "
                           "before the max_units=1 argument was passed to the engine, so "
                           "its 72 rows silently ran at the 4-unit default. This is the "
                           "corrected 1-unit re-run of the same 72 registered configurations; "
                           "v1 is preserved untouched.",
        "n_trials_dsr": N_TRIALS_DSR,
        "arms": {"C1_daily_filter_1unit": "Donchian + daily-trend filter + 1 unit (primary)",
                 "C0_no_filter_1unit": "Donchian, 1 unit, no daily filter (yardstick)"},
        "windows": WINDOWS,
        "holdout_touched": False,
        "rows": rows,
        "pooled": pooled,
        "attribution_filter_vs_no_filter": attr,
        "n_survivors": len(survivors),
        "survivors": survivors,
        "verdict": "SURVIVORS_AWAIT_HOLDOUT" if survivors else "NO_EDGE",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")

    if log_trials:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trial_rows = [{
            "trial_id": f"{r['arm']}-{r['asset']}-{r['timeframe']}-N{r['N']}-M{r['M']}",
            "timestamp_utc": now,            "test_name": "AmendmentC_DTDonchian_1unit_fix",
            "asset": r["asset"], "timeframe": r["timeframe"],
            "strategy_name": f"{r['arm']}_N{r['N']}_M{r['M']}",
            "params_json": json.dumps({"N": r["N"], "M": r["M"], "max_units": 1,
                                       "daily_filter": r["arm"] == "C1_daily_filter_1unit",
                                       "asset": r["asset"], "tf": r["timeframe"]}),
            "is_return_pct": r["is_return_pct"], "is_expectancy_bps": r["is_exp_bps"],
            "oos_return_pct": r["oos_return_pct"], "oos_expectancy_bps": r["oos_exp_bps"],
            "status": "EVALUATED",
        } for r in rows]
        pd.DataFrame(trial_rows).to_csv(TRIALS_FILE, mode="a",
                                        header=not os.path.exists(TRIALS_FILE), index=False)
        print(f"logged {len(trial_rows)} trials to {TRIALS_FILE}")
    return payload


if __name__ == "__main__":
    run_and_write()

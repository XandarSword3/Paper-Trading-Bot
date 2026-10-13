"""
Random-entry null (v2) for the Turtle/Donchian family.

Difference from the retired artifacts:
  * The synthetic baseline in audit_results.json (random_baseline_per_config) and the
    placeholder in turtle_diagnostics.py drew constants and are retracted; this module
    is a real simulation.
  * Every null run uses the SAME exit logic, 2N stop, 0.5N pyramid ladder, 4-unit cap,
    2x leverage cap, $10 minimum order, next-open fills, 14 bps RT friction and
    direction-aware funding as the evaluated strategy: only the ENTRY TIMES and the
    ENTRY DIRECTIONS are randomised.

Matching rule (per config, equity-independent):
  * entry rate   : the null places exactly as many entry signals as the strategy
                   completed CYCLES over the same OOS window, drawn uniformly without
                   replacement from the eligible bars (after warmup). m null runs share
                   one rate parameter, so the mean signal rate equals the strategy's.
    direction mix: each signal is long with probability equal to the strategy's
                   realised long-cycle share on that config.

Signals that arrive while a position is already open are ignored, exactly as the
strategy ignores breakout signals during a position; the realised entry count per run
is reported next to the strategy's so the match can be inspected.

Statistic: the equal-weighted mean NET bps of the completed CYCLE (mean over the
1-4 units in the cycle), i.e. the same cycle-level statistic used in item 4.
p-value: one-sided right tail, p = P(null cycle mean >= strategy cycle mean);
small p means the strategy is WORSE than random entries. Holm-adjusted across the
24 evaluated configs (12 x 1h + 12 x 15m; 5m skipped by instruction).
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

from quant_research.core.engine import CostModel
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation

DATA_DIR = "quant_research/data/processed"
OUT_PATH = "quant_research/results/random_entry_baseline_v2.json"

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
CONFIGS = [(20, 10), (20, 20), (55, 10), (55, 20)]
TIMEFRAMES = ["1h", "15m"]          # 5m skipped by instruction
N_RUNS = 1000
CHUNK = 50
OOS_START = "2025-04-01 00:00:00"
OOS_END = "2026-03-31 23:59:59"
WARMUP = 55  # matches max(N, M, 30) upper bound of the strategy warmup

BASE_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)


def _slice(features, start_str, end_str):
    ts = features["timestamps"]
    mask = (ts >= pd.Timestamp(start_str, tz="UTC")) & (ts <= pd.Timestamp(end_str, tz="UTC"))
    return {k: (features[k][mask] if isinstance(features[k], np.ndarray) else ts[mask]) for k in features}


def load_features(asset, tf):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_{tf}.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding = fdf.set_index("timestamp")["funding_rate"].sort_index()
    return df, funding


def cycles_from_trades(trades):
    """Group completed units by exit_time (one exit closes a whole cycle)."""
    by_exit = {}
    for t in trades:
        by_exit.setdefault(pd.Timestamp(t.exit_time), []).append(t)
    cycles = []
    for _, units in sorted(by_exit.items()):
        cycles.append({
            "exit_time": units[0].exit_time,
            "direction": units[0].direction,
            "units": len(units),
            "net_bps": float(np.mean([u.return_bps for u in units])),
            "net_usd": float(sum(u.net_pnl for u in units)),
        })
    return cycles


def strategy_reference(asset, tf, N, M):
    df, funding = load_features(asset, tf)
    feat = prepare_turtle_features(df, N=N, M=M)
    oos = _slice(feat, OOS_START, OOS_END)
    metrics, trades = run_turtle_simulation(
        oos, funding, {"N": N, "M": M}, BASE_COSTS,
        bars_per_year=8760.0 if tf == "1h" else 35040.0, return_trades=True,
    )
    cycles = cycles_from_trades(trades)
    n_long = sum(1 for c in cycles if c["direction"] == "long")
    return {
        "n_bars": int(len(oos["close"])),
        "strategy_cycles": len(cycles),
        "strategy_trades": len(trades),
        "strategy_cycle_mean_bps": float(np.mean([c["net_bps"] for c in cycles])) if cycles else 0.0,
        "long_share": (n_long / len(cycles)) if cycles else 0.5,
    }


def _random_entry_features(feat, rng, n_signals, long_share, warmup):
    n = len(feat["close"])
    close = feat["close"]
    entry_high = np.full(n, np.inf)
    entry_low = np.full(n, -np.inf)
    bars = rng.choice(np.arange(warmup, n - 1), size=min(n_signals, max(0, n - 1 - warmup)), replace=False)
    is_long = rng.random(bars.size) < long_share
    for b, lng in zip(bars, is_long):
        eps = 1e-9 * max(1.0, abs(close[b]))
        if lng:
            entry_high[b] = close[b] - eps
        else:
            entry_low[b] = close[b] + eps
    out = dict(feat)
    out["entry_high"] = entry_high
    out["entry_low"] = entry_low
    return out


def _chunk_worker(args):
    asset, tf, N, M, chunk_idx, seed, n_signals, long_share = args
    df, funding = load_features(asset, tf)
    feat = prepare_turtle_features(df, N=N, M=M)
    oos = _slice(feat, OOS_START, OOS_END)
    bpy = 8760.0 if tf == "1h" else 35040.0

    rng = np.random.default_rng(seed)
    means, realized = [], []
    for _ in range(CHUNK):
        rand_feat = _random_entry_features(oos, rng, n_signals, long_share, WARMUP)
        _, trades = run_turtle_simulation(
            rand_feat, funding, {"N": N, "M": M}, BASE_COSTS,
            bars_per_year=bpy, return_trades=True,
        )
        cycles = cycles_from_trades(trades)
        means.append(float(np.mean([c["net_bps"] for c in cycles])) if cycles else np.nan)
        realized.append(len(cycles))
    return {"config": f"{asset} {tf} {N}/{M}", "chunk": chunk_idx, "means": means, "realized": realized}


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main():
    print("=" * 100)
    print(f"RANDOM-ENTRY NULL (REAL SIMULATION): {N_RUNS} RUNS x {len(ASSETS) * len(CONFIGS)} CONFIGS PER TIMEFRAME")
    print("Timeframes: 1h and 15m (5m intentionally skipped). Matched entry rate + direction mix.")
    print("=" * 100)

    results = []
    jobs_meta = []
    for tf in TIMEFRAMES:
        for asset in ASSETS:
            for N, M in CONFIGS:
                ref = strategy_reference(asset, tf, N, M)
                jobs_meta.append((asset, tf, N, M, ref))
                print(f"[ref] {asset:8} {tf:3} N={N:2} M={M:2} cycles={ref['strategy_cycles']:4d} "
                      f"bars={ref['n_bars']:6d} long_share={ref['long_share']:.3f} "
                      f"cycle_mean={ref['strategy_cycle_mean_bps']:+.3f} bps", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = []
        for asset, tf, N, M, ref in jobs_meta:
            n_chunks = N_RUNS // CHUNK
            for ci in range(n_chunks):
                seed = hash((asset, tf, N, M, ci)) % (2 ** 31)
                futures.append(ex.submit(
                    _chunk_worker,
                    (asset, tf, N, M, ci, seed, ref["strategy_cycles"], ref["long_share"]),
                ))
        raw = [f.result() for f in futures]
    print(f"Completed {len(raw)} chunks ({len(raw) * CHUNK} runs) in {time.time() - t0:.1f}s", flush=True)

    by_cfg = {}
    for r in raw:
        by_cfg.setdefault(r["config"], {"means": [], "realized": []})
        by_cfg[r["config"]]["means"].extend(r["means"])
        by_cfg[r["config"]]["realized"].extend(r["realized"])

    for asset, tf, N, M, ref in jobs_meta:
        key = f"{asset} {tf} {N}/{M}"
        means = np.asarray([m for m in by_cfg[key]["means"] if not np.isnan(m)], dtype=float)
        realized = np.asarray(by_cfg[key]["realized"], dtype=float)
        obs = ref["strategy_cycle_mean_bps"]
        # Two one-sided tails of the SAME null distribution:
        #   p_beats  = P(null mean >= strategy) -> small = strategy BEATS random entries
        #   p_worse  = P(null mean <= strategy) -> small = strategy is WORSE than random
        # P(null >= obs) + P(null <= obs) = 1 + P(null == obs), and the null means are
        # means of many runs, so exact ties are measure-zero.
        p_beats = float((means >= obs).mean()) if means.size else 1.0
        p_worse = float((means <= obs).mean()) if means.size else 1.0
        results.append({
            "config": key,
            "asset": asset,
            "timeframe": tf,
            "N": N,
            "M": M,
            "strategy_cycles": ref["strategy_cycles"],
            "strategy_trades": ref["strategy_trades"],
            "strategy_cycle_mean_bps": round(obs, 4),
            "null_runs": int(means.size),
            "null_cycle_mean_bps_mean": round(float(means.mean()), 4) if means.size else None,
            "null_cycle_mean_bps_sd": round(float(means.std(ddof=1)), 4) if means.size > 1 else None,
            "null_realized_entries_mean": round(float(realized.mean()), 2) if realized.size else None,
            "p_beats_random (P(null >= strategy))": round(p_beats, 4),
            "p_worse_than_random (P(null <= strategy))": round(p_worse, 4),
        })

    p_beats_arr = np.array([r["p_beats_random (P(null >= strategy))"] for r in results], dtype=float)
    p_worse_arr = np.array([r["p_worse_than_random (P(null <= strategy))"] for r in results], dtype=float)
    holm_beats = holm_adjust(p_beats_arr)
    holm_worse = holm_adjust(p_worse_arr)
    for r, hb, hw in zip(results, holm_beats, holm_worse):
        r["p_beats_holm"] = round(float(hb), 4)
        r["p_worse_holm"] = round(float(hw), 4)

    print("\n" + "=" * 100)
    print("PER-CONFIG NULL RESULTS (cycle-level net bps; p = P(null mean >= strategy))")
    print("=" * 100)
    print(f"{'config':22} {'cycles':>6} {'strat bps':>10} {'null mean':>10} {'null SD':>9} {'null n':>7} "
          f"{'realiz.ent':>10} {'p_beats':>8} {'p_beatsH':>9} {'p_worse':>8} {'p_worseH':>9}")
    for r in results:
        print(f"{r['config']:22} {r['strategy_cycles']:6d} {r['strategy_cycle_mean_bps']:10.3f} "
              f"{r['null_cycle_mean_bps_mean']:10.3f} {r['null_cycle_mean_bps_sd']:9.3f} {r['null_runs']:7d} "
              f"{r['null_realized_entries_mean']:10.1f} {r['p_beats_random (P(null >= strategy))']:8.4f} "
              f"{r['p_beats_holm']:9.4f} {r['p_worse_than_random (P(null <= strategy))']:8.4f} {r['p_worse_holm']:9.4f}")

    sig_worse = [r for r in results if r["p_worse_holm"] < 0.05]
    sig_better = [r for r in results if r["p_beats_holm"] < 0.05]
    print(f"\nConfigs where the strategy is significantly WORSE than matched random entries "
          f"(Holm p_worse < 0.05): {len(sig_worse)} / {len(results)}")
    for r in sig_worse:
        print(f"  {r['config']:22} strategy {r['strategy_cycle_mean_bps']:+.2f} bps vs null "
              f"{r['null_cycle_mean_bps_mean']:+.2f} bps (Holm p_worse {r['p_worse_holm']:.4f})")
    print(f"Configs where the strategy significantly BEATS matched random entries "
          f"(Holm p_beats < 0.05): {len(sig_better)} / {len(results)}")
    for r in sig_better:
        print(f"  {r['config']:22} strategy {r['strategy_cycle_mean_bps']:+.2f} bps vs null "
              f"{r['null_cycle_mean_bps_mean']:+.2f} bps (Holm p_beats {r['p_beats_holm']:.4f})")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "real simulation; matched entry rate (strategy cycles) and direction mix; "
                  "identical exits/stops/pyramid/costs/funding; both one-sided tails, Holm over 24 configs",
        "n_runs_per_config": N_RUNS,
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    return payload


if __name__ == "__main__":
    main()

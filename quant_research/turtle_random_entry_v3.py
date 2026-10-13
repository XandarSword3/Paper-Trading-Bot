"""
Random-entry null (v3) for the Turtle/Donchian family - PRE-REGISTERED STATISTIC.

Why a v3:
  * v2 (`turtle_random_entry_v2.py`, `results/random_entry_baseline_v2.json`) ran only
    the 12 x 1h + 12 x 15m configs and judged them on the EQUAL-WEIGHTED CYCLE MEAN
    (net bps per cycle), which is not the statistic the pre-registration evaluates.
  * The pre-registered statistic is the per-trade / per-unit net expectancy pooled
    across units (PREREG sections 6.1 / 7.5: "Net expectancy >= +5 bps per trade",
    pooled sample size across assets), plus the total net P&L in USD.
  * v2 also seeded its null runs with Python's process-randomised `hash()`, so v2 is
    not bit-reproducible. v3 uses zlib.crc32 seeds: same numbers on every machine.

Null construction (unchanged from v2, only the entry times/directions are random):
  * entry rate  : exactly as many random entry signals as the strategy completed
                  CYCLES over the same OOS window, drawn uniformly without
                  replacement from eligible bars (after warmup).
  * direction   : each signal is long with probability equal to the strategy's
                  realised long-cycle share on that config.
  * everything else (exits, 2N stop, 0.5N pyramid ladder, 4-unit cap, 2x leverage
    cap, $10 minimum order, next-open fills, 14 bps RT friction, sign-corrected
    funding) is the SAME code path as the evaluated strategy.

Statistics (per config = asset x timeframe x N/M, 36 configs):
  S1 unit-pooled net bps = sum(unit net_bps) / n_units          [trade-weighted]
  S2 total net P&L (USD) = sum(unit net_pnl)
Pooled-across-assets variants (12 rows) are the same two statistics computed on the
union of the three assets' units, which is the unit the PREREG section 7.5 pass
criteria are stated on ("pooled across the three assets").

p-values: two one-sided tails of the same null distribution.
  p_beats = P(null >= strategy) -> small means the strategy BEATS matched random entries
  p_worse = P(null <= strategy) -> small means the strategy is WORSE than random
Holm-Bonferroni across the 36 configs, separately for each statistic.

Because the pooled statistics combine three independent asset-level nulls, the pooled
null is formed by SUMMING the same run index across assets (run i of SOL + run i of
BTC + run i of ETH), which is valid for independent Monte-Carlo streams and costs no
extra simulation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import time
import zlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

from quant_research.core.engine import CostModel
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation

DATA_DIR = "quant_research/data/processed"
OUT_PATH = "quant_research/results/random_entry_baseline_v3.json"

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
CONFIGS = [(20, 10), (20, 20), (55, 10), (55, 20)]
TIMEFRAMES = ["5m", "15m", "1h"]
N_RUNS = 1000
CHUNK = 50
OOS_START = "2025-04-01 00:00:00"
OOS_END = "2026-03-31 23:59:59"
WARMUP = 55  # matches max(N, M, 30) upper bound of the strategy warmup
BARS_PER_YEAR = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}

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


def unit_stats(trades):
    """Pre-registered statistics from a list of completed units."""
    if not trades:
        return 0.0, 0.0, 0
    bps = float(np.mean([t.return_bps for t in trades]))
    usd = float(np.sum([t.net_pnl for t in trades]))
    return bps, usd, len(trades)


def cycles_from_trades(trades):
    by_exit = {}
    for t in trades:
        by_exit.setdefault(pd.Timestamp(t.exit_time), []).append(t)
    n_long = sum(1 for units in by_exit.values() if units[0].direction == "long")
    return len(by_exit), n_long


def strategy_reference(asset, tf, N, M):
    df, funding = load_features(asset, tf)
    feat = prepare_turtle_features(df, N=N, M=M)
    oos = _slice(feat, OOS_START, OOS_END)
    _, trades = run_turtle_simulation(
        oos, funding, {"N": N, "M": M}, BASE_COSTS,
        bars_per_year=BARS_PER_YEAR[tf], return_trades=True,
    )
    bps, usd, n_units = unit_stats(trades)
    n_cycles, n_long = cycles_from_trades(trades)
    return {
        "n_bars": int(len(oos["close"])),
        "strategy_cycles": n_cycles,
        "strategy_units": n_units,
        "strategy_pooled_net_bps": bps,
        "strategy_total_net_usd": usd,
        "long_share": (n_long / n_cycles) if n_cycles else 0.5,
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
    bpy = BARS_PER_YEAR[tf]

    rng = np.random.default_rng(seed)
    bps_runs, usd_runs, units_runs = [], [], []
    for _ in range(CHUNK):
        rand_feat = _random_entry_features(oos, rng, n_signals, long_share, WARMUP)
        _, trades = run_turtle_simulation(
            rand_feat, funding, {"N": N, "M": M}, BASE_COSTS,
            bars_per_year=bpy, return_trades=True,
        )
        bps, usd, n_units = unit_stats(trades)
        bps_runs.append(bps)
        usd_runs.append(usd)
        units_runs.append(n_units)
    return {
        "config": f"{asset} {tf} {N}/{M}", "asset": asset, "timeframe": tf, "N": N, "M": M,
        "chunk": chunk_idx, "bps": bps_runs, "usd": usd_runs, "units": units_runs,
    }


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def two_tails(null, obs):
    null = np.asarray(null, dtype=float)
    return float((null >= obs).mean()), float((null <= obs).mean())


def main():
    print("=" * 110)
    print(f"RANDOM-ENTRY NULL v3 (REAL SIMULATION): {N_RUNS} RUNS x 36 CONFIGS, PRE-REGISTERED STATISTIC")
    print("S1 = unit-pooled net bps (sum of unit net_bps / n_units); S2 = total net P&L (USD)")
    print("Timeframes: 5m, 15m, 1h. Matched entry rate (strategy cycles) + direction mix. Holm over 36.")
    print("=" * 110)

    jobs_meta = []
    for tf in TIMEFRAMES:
        for asset in ASSETS:
            for N, M in CONFIGS:
                ref = strategy_reference(asset, tf, N, M)
                jobs_meta.append((asset, tf, N, M, ref))
                print(f"[ref] {asset:8} {tf:3} N={N:2} M={M:2} cycles={ref['strategy_cycles']:5d} "
                      f"units={ref['strategy_units']:5d} bars={ref['n_bars']:6d} "
                      f"long_share={ref['long_share']:.3f} pooled_bps={ref['strategy_pooled_net_bps']:+.4f} "
                      f"total_usd={ref['strategy_total_net_usd']:+.2f}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = []
        for asset, tf, N, M, ref in jobs_meta:
            for ci in range(N_RUNS // CHUNK):
                seed = zlib.crc32(f"{asset}|{tf}|{N}|{M}|{ci}".encode())  # deterministic across processes
                futures.append(ex.submit(
                    _chunk_worker,
                    (asset, tf, N, M, ci, seed, ref["strategy_cycles"], ref["long_share"]),
                ))
        raw = [f.result() for f in futures]
    print(f"\nCompleted {len(raw)} chunks ({len(raw) * CHUNK} runs) in {time.time() - t0:.1f}s", flush=True)

    # ---- assemble per-asset null streams -------------------------------------
    streams = {}   # key -> dict of arrays
    for asset, tf, N, M, ref in jobs_meta:
        key = f"{asset} {tf} {N}/{M}"
        rows = sorted([r for r in raw if r["config"] == key], key=lambda r: r["chunk"])
        streams[key] = {
            "bps": np.concatenate([np.asarray(r["bps"], dtype=float) for r in rows]),
            "usd": np.concatenate([np.asarray(r["usd"], dtype=float) for r in rows]),
            "units": np.concatenate([np.asarray(r["units"], dtype=float) for r in rows]),
        }

    # ---- 36 per-asset configs -------------------------------------------------
    results = []
    for asset, tf, N, M, ref in jobs_meta:
        key = f"{asset} {tf} {N}/{M}"
        st = streams[key]
        obs_bps = ref["strategy_pooled_net_bps"]
        obs_usd = ref["strategy_total_net_usd"]
        p_b, p_w = two_tails(st["bps"], obs_bps)
        q_b, q_w = two_tails(st["usd"], obs_usd)
        results.append({
            "config": key, "asset": asset, "timeframe": tf, "N": N, "M": M,
            "strategy_cycles": ref["strategy_cycles"],
            "strategy_units": ref["strategy_units"],
            "strategy_pooled_net_bps": round(obs_bps, 4),
            "strategy_total_net_usd": round(obs_usd, 2),
            "null_runs": int(st["bps"].size),
            "null_pooled_net_bps_mean": round(float(st["bps"].mean()), 4),
            "null_pooled_net_bps_sd": round(float(st["bps"].std(ddof=1)), 4),
            "null_total_net_usd_mean": round(float(st["usd"].mean()), 2),
            "null_total_net_usd_sd": round(float(st["usd"].std(ddof=1)), 2),
            "null_units_mean": round(float(st["units"].mean()), 2),
            "p_beats_bps": round(p_b, 4),
            "p_worse_bps": round(p_w, 4),
            "p_beats_usd": round(q_b, 4),
            "p_worse_usd": round(q_w, 4),
        })

    for stat in ["bps", "usd"]:
        for tail, col in [("beats", f"p_beats_{stat}"), ("worse", f"p_worse_{stat}")]:
            adj = holm_adjust([r[col] for r in results])
            for r, a in zip(results, adj):
                r[f"{col}_holm"] = round(float(a), 4)

    # ---- 12 pooled-across-assets configs -------------------------------------
    pooled = []
    for tf in TIMEFRAMES:
        for N, M in CONFIGS:
            keys = [f"{a} {tf} {N}/{M}" for a in ASSETS]
            n_units_sum = np.zeros(N_RUNS)
            bps_num = np.zeros(N_RUNS)
            usd_sum = np.zeros(N_RUNS)
            obs_units = 0
            obs_bps_num = 0.0
            obs_usd = 0.0
            for k in keys:
                st = streams[k]
                n_units_sum += st["units"]
                bps_num += st["bps"] * st["units"]     # denominator-correct pooling
                usd_sum += st["usd"]
                ref = [j for j in jobs_meta if f"{j[0]} {j[1]} {j[2]}/{j[3]}" == k][0]
                obs_units += ref[4]["strategy_units"]
                obs_bps_num += ref[4]["strategy_pooled_net_bps"] * ref[4]["strategy_units"]
                obs_usd += ref[4]["strategy_total_net_usd"]
            with np.errstate(invalid="ignore", divide="ignore"):
                null_bps = np.where(n_units_sum > 0, bps_num / np.maximum(n_units_sum, 1), np.nan)
            valid = ~np.isnan(null_bps)
            obs_bps = (obs_bps_num / obs_units) if obs_units else 0.0
            p_b, p_w = two_tails(null_bps[valid], obs_bps)
            q_b, q_w = two_tails(usd_sum[valid], obs_usd)
            pooled.append({
                "timeframe": tf, "N": N, "M": M,
                "strategy_units": int(obs_units),
                "strategy_pooled_net_bps": round(obs_bps, 4),
                "strategy_total_net_usd": round(obs_usd, 2),
                "null_runs": int(valid.sum()),
                "null_pooled_net_bps_mean": round(float(null_bps[valid].mean()), 4),
                "null_pooled_net_bps_sd": round(float(null_bps[valid].std(ddof=1)), 4),
                "null_total_net_usd_mean": round(float(usd_sum[valid].mean()), 2),
                "null_total_net_usd_sd": round(float(usd_sum[valid].std(ddof=1)), 2),
                "p_beats_bps": round(p_b, 4), "p_worse_bps": round(p_w, 4),
                "p_beats_usd": round(q_b, 4), "p_worse_usd": round(q_w, 4),
            })
    for stat in ["bps", "usd"]:
        for col in [f"p_beats_{stat}", f"p_worse_{stat}"]:
            adj = holm_adjust([r[col] for r in pooled])
            for r, a in zip(pooled, adj):
                r[f"{col}_holm"] = round(float(a), 4)

    print("\n" + "=" * 110)
    print("36 PER-ASSET CONFIGS - unit-pooled net bps (S1) and total net USD (S2)")
    print("=" * 110)
    print(f"{'config':22} {'units':>6} {'strat bps':>10} {'null bps':>10} {'nullSD':>8} "
          f"{'p_beat':>7} {'Holm':>7} {'p_worse':>8} {'Holm':>7} | "
          f"{'strat $':>10} {'null $':>10} {'p_beat':>7} {'p_worse':>8}")
    for r in results:
        print(f"{r['config']:22} {r['strategy_units']:6d} {r['strategy_pooled_net_bps']:10.3f} "
              f"{r['null_pooled_net_bps_mean']:10.3f} {r['null_pooled_net_bps_sd']:8.3f} "
              f"{r['p_beats_bps']:7.4f} {r['p_beats_bps_holm']:7.4f} {r['p_worse_bps']:8.4f} {r['p_worse_bps_holm']:7.4f} | "
              f"{r['strategy_total_net_usd']:10.2f} {r['null_total_net_usd_mean']:10.2f} "
              f"{r['p_beats_usd']:7.4f} {r['p_worse_usd']:8.4f}")

    print("\n" + "=" * 110)
    print("12 POOLED-ACROSS-ASSETS CONFIGS (PREREG 7.5 unit of inference)")
    print("=" * 110)
    print(f"{'config':16} {'units':>6} {'strat bps':>10} {'null bps':>10} {'nullSD':>8} "
          f"{'p_beat':>7} {'Holm':>7} {'p_worse':>8} {'Holm':>7} | "
          f"{'strat $':>12} {'null $':>12} {'p_beat':>7} {'p_worse':>8}")
    for r in pooled:
        print(f"{r['timeframe']+' '+str(r['N'])+'/'+str(r['M']):16} {r['strategy_units']:6d} "
              f"{r['strategy_pooled_net_bps']:10.3f} {r['null_pooled_net_bps_mean']:10.3f} "
              f"{r['null_pooled_net_bps_sd']:8.3f} {r['p_beats_bps']:7.4f} {r['p_beats_bps_holm']:7.4f} "
              f"{r['p_worse_bps']:8.4f} {r['p_worse_bps_holm']:7.4f} | {r['strategy_total_net_usd']:12.2f} "
              f"{r['null_total_net_usd_mean']:12.2f} {r['p_beats_usd']:7.4f} {r['p_worse_usd']:8.4f}")

    for label, coll, keyf in [("36 per-asset configs", results, lambda r: r["config"]),
                              ("12 pooled configs", pooled, lambda r: f"{r['timeframe']} {r['N']}/{r['M']}")]:
        sw = [r for r in coll if r["p_worse_bps_holm"] < 0.05]
        sb = [r for r in coll if r["p_beats_bps_holm"] < 0.05]
        print(f"\n[{label}] beats matched random entries (S1, Holm p_beats<0.05): {len(sb)}/{len(coll)}; "
              f"worse than matched random entries (S1, Holm p_worse<0.05): {len(sw)}/{len(coll)}")
        for r in sw:
            print(f"   WORSE(S1) {keyf(r):22} strategy {r['strategy_pooled_net_bps']:+.2f} bps vs null "
                  f"{r['null_pooled_net_bps_mean']:+.2f} bps (Holm p {r['p_worse_bps_holm']:.4f})")
        sw_usd = [r for r in coll if r["p_worse_usd_holm"] < 0.05]
        sb_usd = [r for r in coll if r["p_beats_usd_holm"] < 0.05]
        print(f"[{label}] same two tails on S2 (total net USD): beats {len(sb_usd)}/{len(coll)}, "
              f"worse {len(sw_usd)}/{len(coll)}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": "results/random_entry_baseline_v2.json (24 configs, equal-weighted cycle-mean statistic)",
        "method": "real simulation; matched entry rate (strategy cycles) and direction mix; identical exits/stops/"
                  "pyramid/costs/sign-corrected funding; statistics = unit-pooled net bps (S1) and total net P&L "
                  "USD (S2); both one-sided tails; Holm over 36 per-asset configs (and over 12 pooled configs "
                  "reported separately); deterministic crc32 seeds",
        "n_runs_per_config": N_RUNS,
        "oos_window": [OOS_START, OOS_END],
        "timeframes": TIMEFRAMES,
        "configs": results,
        "pooled_across_assets": pooled,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    return payload


if __name__ == "__main__":
    main()

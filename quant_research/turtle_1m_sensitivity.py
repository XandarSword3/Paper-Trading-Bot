"""
Item 5 - intrabar sensitivity: rerun the 36 Turtle configs on the OOS window with the
1m sub-bar resolver WIRED IN (df_1m passed to run_turtle_simulation) and compare with
the close-based baseline.

Statement of the committed convention (answered by this script):
  * The M-bar Donchian exit is CLOSE-BASED in both the main simulator
    (strategies/turtle_strategy.run_turtle_simulation) and the ambiguity counter
    (turtle_diagnostics.run_turtle_diagnostic_sim): it fires when the bar's CLOSE
    crosses the prior-M-bar channel and fills at the next open.
  * The 2N protective stop is an INTRABAR TOUCH: it fires when the bar's low/high
    reaches the stop level.
  * The ambiguity counter counts bars where BOTH hold on the same bar. Only on those
    bars does sub-bar resolution make a difference.
  * The committed main results use the pessimistic stop-first fallback on ambiguous
    bars (no df_1m was wired). This file provides the 1m-wired sensitivity numbers.
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
OUT_PATH = "quant_research/results/turtle_1m_sensitivity.json"
ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]
CONFIGS = [(20, 10), (20, 20), (55, 10), (55, 20)]
OOS_START = "2025-04-01 00:00:00"
OOS_END = "2026-03-31 23:59:59"
BARS_PER_YEAR = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}
BASE_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)


def _slice(features, start_str, end_str):
    ts = features["timestamps"]
    mask = (ts >= pd.Timestamp(start_str, tz="UTC")) & (ts <= pd.Timestamp(end_str, tz="UTC"))
    return {k: (features[k][mask] if isinstance(features[k], np.ndarray) else ts[mask]) for k in features}


def run_config(job):
    asset, tf, N, M = job
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_{tf}.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

    sub = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1m.parquet"))
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.set_index("timestamp").sort_index()
    sub_oos = sub.loc[(sub.index >= pd.Timestamp(OOS_START, tz="UTC")) & (sub.index <= pd.Timestamp(OOS_END, tz="UTC"))]

    feat = prepare_turtle_features(df, N=N, M=M)
    oos = _slice(feat, OOS_START, OOS_END)
    p = {"N": N, "M": M}
    bpy = BARS_PER_YEAR[tf]

    base = run_turtle_simulation(oos, funding, p, BASE_COSTS, bars_per_year=bpy)
    res_1m = run_turtle_simulation(oos, funding, p, BASE_COSTS, bars_per_year=bpy, df_1m=sub_oos)

    return {
        "asset": asset, "timeframe": tf, "N": N, "M": M,
        "baseline_trades": base["total_trades"],
        "baseline_net_bps": base["expectancy_bps"],
        "baseline_ambiguous_bars": base.get("ambiguous_bars"),
        "wired_trades": res_1m["total_trades"],
        "wired_net_bps": res_1m["expectancy_bps"],
        "wired_ambiguous_bars": res_1m.get("ambiguous_bars"),
        "delta_trades": res_1m["total_trades"] - base["total_trades"],
        "delta_net_bps": round(res_1m["expectancy_bps"] - base["expectancy_bps"], 4),
    }


def main():
    print("=" * 100)
    print("ITEM 5: 36-CONFIG 1M SUB-BAR SENSITIVITY (df_1m wired into the OOS run)")
    print("=" * 100)
    jobs = [(a, tf, N, M) for a in ASSETS for tf in TIMEFRAMES for N, M in CONFIGS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(run_config, jobs))
    print(f"Completed {len(rows)} configs in {time.time() - t0:.1f}s\n")

    print(f"{'config':22} {'base tr':>7} {'wired tr':>8} {'dtr':>5} {'base net':>9} {'wired net':>9} {'dnet':>8} {'ambig':>6}")
    label_width = 22
    for r in rows:
        label = f"{r['asset']} {r['timeframe']} {r['N']}/{r['M']}"
        print(f"{label:{label_width}} {r['baseline_trades']:7d} "
              f"{r['wired_trades']:8d} {r['delta_trades']:5d} {r['baseline_net_bps']:9.3f} "
              f"{r['wired_net_bps']:9.3f} {r['delta_net_bps']:8.3f} {r['baseline_ambiguous_bars']:6d}")

    changed = [r for r in rows if r["delta_trades"] != 0 or abs(r["delta_net_bps"]) > 1e-9]
    print(f"\nConfigs whose trade count or net bps CHANGED when 1m sub-bars were wired: {len(changed)} / {len(rows)}")
    for r in changed:
        print(f"  {r['asset']} {r['timeframe']} {r['N']}/{r['M']}: trades {r['baseline_trades']}->{r['wired_trades']}, "
              f"net {r['baseline_net_bps']:+.3f}->{r['wired_net_bps']:+.3f} bps (delta {r['delta_net_bps']:+.3f})")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "convention": "close-based Donchian exit + intrabar-touch stop; ambiguous bars resolved "
                      "pessimistically (stop first) in the committed baseline; this file wires df_1m",
        "oos_window": [OOS_START, OOS_END],
        "configs": rows,
        "n_configs_changed": len(changed),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    return payload


if __name__ == "__main__":
    main()

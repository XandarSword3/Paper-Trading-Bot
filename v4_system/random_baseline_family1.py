"""Family 1 random-entry null baseline (PREREG III.10, 2026-11-28).

Construction adapted from ``quant_research/turtle_random_entry_v3.py``
(same null philosophy, V4 data + Family 1 folds):

- For each (asset, timeframe, N, M) configuration and each of the five
  OOS folds, ``R`` random-entry simulations are run through the SAME
  ``run_turtle_simulation`` code path as the strategy.
- Random signals: entry levels are replaced by sparse triggers at bars
  drawn UNIFORMLY WITHOUT REPLACEMENT from the fold's TEST window only
  (warm-up bars never receive signals). Count per fold = the strategy's
  OOS trade count in that fold; direction mix = strategy long share.
- Everything else is identical to the strategy: Donchian M-bar exits,
  2N ATR stop, pyramiding disabled (max_units=1), 1% risk, 2x leverage
  cap, next-open fills, 1m sub-bar exit resolution, direction-aware
  funding, base costs (14 bps RT).
- The daily trend filter is NOT applied to the null (it is part of the
  strategy being tested, not of the random-entry null).
- Statistic per config x asset: pooled OOS net expectancy (bps/trade)
  across the five folds, base costs. ``p_beats`` = fraction of null
  runs whose pooled expectancy >= the strategy's.
- Hurdle (III.10): ``p_beats <= 0.05`` (one-sided).

Seeds are deterministic: ``zlib.crc32("family1|asset|tf|N|M|run")``.
"""
from __future__ import annotations

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v4_system.adapters import quant_research_data as qr
from v4_system import run_family1 as r1

from quant_research.core.engine import CostModel
from quant_research.strategies.turtle_strategy import (
    prepare_turtle_features, run_turtle_simulation)

N_RANDOM_RUNS = 200          # PREREG III.10
P_BEATS_HURDLE = 0.05        # PREREG III.10
# Each worker holds a full 1m bundle (~150 MB); 16 workers exhausted RAM
# (ArrayMemoryError). 6 workers ~= 1 GB peak, still ~4x speedup.
MAX_WORKERS = 6


def _random_entry_features(feat: Dict[str, Any], rng: np.random.Generator,
                           test_start_idx: int, n_signals: int,
                           long_share: float) -> Dict[str, Any]:
    """Replace entry channels with sparse random triggers in the test
    window only. Exit channels, ATR and OHLCV are untouched."""
    n = len(feat["close"])
    close = feat["close"]
    entry_high = np.full(n, np.inf)
    entry_low = np.full(n, -np.inf)
    lo = max(test_start_idx, 1)
    hi = n - 1
    n_draw = min(n_signals, max(0, hi - lo))
    if n_draw > 0:
        bars = rng.choice(np.arange(lo, hi), size=n_draw, replace=False)
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


def _fold_test_start_idx(features: Dict[str, Any], test_start: str) -> int:
    ts = features["timestamps"]
    return int(np.searchsorted(ts, pd.Timestamp(test_start, tz="UTC")))


def baseline_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run R random-entry simulations for one (asset, tf, N, M) config.

    ``task`` keys: asset, timeframe, N, M, folds (list of dicts with
    fold_id/test_start/test_end/n_signals/long_share), n_runs.
    Returns pooled null expectancy per run (bps) plus diagnostics.
    """
    asset, tf, N, M = task["asset"], task["timeframe"], task["N"], task["M"]
    dev_end = r1.ASSETS[asset]["dev_end"]
    as_of = f"{dev_end}T23:59:59Z"
    # The null does not use the daily trend filter, so skip its second
    # 1m load (memory) — features are unmasked anyway.
    bundle = qr.load_turtle_bundle(asset, tf, as_of, strict=False,
                                   include_1m_subbars=True,
                                   include_daily_trend=False)
    # UNMASKED features: the daily trend filter is part of the strategy,
    # not of the random-entry null.
    feats = prepare_turtle_features(bundle["df"], N, M)
    bpy = r1.BARS_PER_YEAR[tf]

    null_bps: List[float] = []
    for run_i in range(task["n_runs"]):
        seed = zlib.crc32(
            f"family1|{asset}|{tf}|{N}|{M}|{run_i}".encode())
        rng = np.random.default_rng(seed)
        run_bps: List[float] = []
        for fold in task["folds"]:
            warm_start = r1._warmup_start(fold["test_start"])
            f_slice = r1.slice_features(feats, warm_start, fold["test_end"])
            test_idx = _fold_test_start_idx(f_slice, fold["test_start"])
            rand_feat = _random_entry_features(
                f_slice, rng, test_idx, fold["n_signals"],
                fold["long_share"])
            m1_slice = r1._slice_1m(bundle["df_1m"], warm_start,
                                    fold["test_end"])
            funding_slice = r1._slice_funding(bundle["funding"], warm_start,
                                              fold["test_end"])
            _, trades = run_turtle_simulation(
                rand_feat, funding_slice,
                {"N": N, "M": M, "max_units": r1.MAX_UNITS,
                 "risk_percent": r1.RISK_PERCENT},
                r1.BASE_COSTS, bars_per_year=bpy,
                df_1m=m1_slice, return_trades=True,
                max_units=r1.MAX_UNITS, risk_percent=r1.RISK_PERCENT)
            test_start = pd.Timestamp(fold["test_start"], tz="UTC")
            oos = [t for t in trades
                   if pd.Timestamp(t.entry_time) >= test_start]
            if oos:
                run_bps.extend(t.return_bps for t in oos)
        null_bps.append(float(np.mean(run_bps)) if run_bps else 0.0)

    return {"asset": asset, "timeframe": tf, "N": N, "M": M,
            "null_pooled_bps": null_bps}


def _fold_refs(fold_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-fold matched-entry references from strategy OOS trades."""
    refs = []
    for fr in fold_results:
        trades = fr["trades"]
        n = len(trades)
        long_share = (sum(1 for t in trades if t.direction == "long") / n
                      if n else 0.5)
        refs.append({"fold_id": fr["fold_id"],
                     "test_start": fr["test_start"],
                     "test_end": fr["test_end"],
                     "n_signals": n, "long_share": long_share})
    return refs


def run_baselines(strategy_by_key: Dict[tuple, Dict[str, Any]],
                  n_runs: int = N_RANDOM_RUNS,
                  max_workers: int = MAX_WORKERS) -> Dict[tuple, Dict[str, Any]]:
    """Run the random-entry null for every config x asset.

    ``strategy_by_key`` maps (asset, tf, N, M) -> {"fold_results": [...],
    "strategy_bps": float}. Returns (asset, tf, N, M) -> baseline stats.
    """
    tasks = []
    for (asset, tf, N, M), ref in strategy_by_key.items():
        tasks.append({"asset": asset, "timeframe": tf, "N": N, "M": M,
                      "folds": _fold_refs(ref["fold_results"]),
                      "n_runs": n_runs})
    out: Dict[tuple, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(baseline_worker, tasks):
            key = (res["asset"], res["timeframe"], res["N"], res["M"])
            strat_bps = strategy_by_key[key]["strategy_bps"]
            null = np.asarray(res["null_pooled_bps"], dtype=float)
            p_beats = float((null >= strat_bps).mean()) if len(null) else 1.0
            out[key] = {
                "n_runs": n_runs,
                "strategy_pooled_bps": strat_bps,
                "null_mean_bps": float(null.mean()) if len(null) else 0.0,
                "null_p5_bps": float(np.percentile(null, 5)) if len(null) else 0.0,
                "null_p95_bps": float(np.percentile(null, 95)) if len(null) else 0.0,
                "p_beats": p_beats,
                "passes": p_beats <= P_BEATS_HURDLE,
            }
    return out

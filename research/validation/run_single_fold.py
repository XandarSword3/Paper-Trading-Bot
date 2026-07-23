"""
Optimizes and scores exactly ONE walk-forward fold for a given strategy.

This is the unit of work behind the walk_forward_validation.yml GitHub
Actions matrix: splitting the 9 folds across 9 parallel runners (each also
parallelizing its own grid search across --jobs workers) is what makes a
full, non---fast grid search finish inside each job's runtime limit instead
of one process working through all 9 folds sequentially for ~30 hours.

To add a new strategy: add an entry to strategy_registry.STRATEGY_REGISTRY.
This script never needs to change.

Usage:
    python run_single_fold.py --strategy v1 --fold-index 0 --jobs 4 \
        --out results/folds/v1/fold_0.json
"""
import argparse
import json
import os

import pandas as pd

from config import DEFAULT_SPLIT, DEFAULT_WALK_FORWARD
from data_fetcher import download_btc_data
from data_splits import get_development
from strategy_registry import get_strategy_config
from walk_forward import generate_folds, run_fold
from walk_forward_test import FAST_RANGES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="Key in strategy_registry.STRATEGY_REGISTRY")
    ap.add_argument("--fold-index", type=int, required=True)
    ap.add_argument("--fast", action="store_true", help="Use the small sanity-check grid instead of the full one")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers within this fold's own grid search")
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = get_strategy_config(args.strategy)
    timeframe = cfg["timeframe"]
    ranges = FAST_RANGES if args.fast else cfg["ranges"]

    print(f"[{args.strategy}] timeframe={timeframe} fold_index={args.fold_index} "
          f"fast={args.fast} jobs={args.jobs}")

    df = get_development(download_btc_data(timeframe=timeframe))
    folds = generate_folds(df, DEFAULT_WALK_FORWARD, DEFAULT_SPLIT)

    if args.fold_index >= len(folds):
        raise SystemExit(
            f"fold-index {args.fold_index} out of range — only {len(folds)} "
            f"folds exist for {args.strategy} ({timeframe})."
        )
    fold = folds[args.fold_index]
    print(f"Fold {fold.index}: train {fold.train_start} -> {fold.test_start}, "
          f"test {fold.test_start} -> {fold.test_end}")

    fr = run_fold(
        df, fold, ranges, DEFAULT_WALK_FORWARD, args.initial_capital,
        n_jobs=args.jobs, show_progress=True,
    )

    payload = {
        "strategy": args.strategy,
        "timeframe": timeframe,
        "fast": args.fast,
        "fold_index": fold.index,
        "fold_train_start": str(fold.train_start),
        "fold_test_start": str(fold.test_start),
        "fold_test_end": str(fold.test_end),
        "best_params": fr.best_params,
        "in_sample_metric_name": fr.in_sample_metric_name,
        "in_sample_metric_value": fr.in_sample_metric_value,
        "in_sample_combos_tested": fr.in_sample_combos_tested,
        "oos_return_pct": fr.oos_return_pct,
        "oos_trades": fr.oos_trades,
        "oos_win_rate": fr.oos_win_rate,
        "oos_max_drawdown_pct": fr.oos_max_drawdown_pct,
        "oos_equity": {str(k): float(v) for k, v in fr.oos_equity.items()},
        "oos_trade_pnls": fr.oos_trade_pnls,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"Fold {fold.index} done: OOS {fr.oos_return_pct:+.2f}% | {fr.oos_trades} trades | "
          f"win rate {fr.oos_win_rate:.1f}% | max DD {fr.oos_max_drawdown_pct:.2f}%")


if __name__ == "__main__":
    main()

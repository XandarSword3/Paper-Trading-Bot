"""
Combines the per-fold JSON files written by run_single_fold.py into exactly
the same artifacts walk_forward_test.py itself writes when run as one
process: results/walk_forward_oos_equity.csv, results/walk_forward_fold_
results.csv, and the committed walk_forward_results_<strategy>.json summary
that build_readiness_gates.py and edge_validation_test.py already read.

Usage:
    python aggregate_folds.py --strategy v1 --fold-dir results/folds/v1
"""
import argparse
import glob
import json
import os

import pandas as pd

from config import DEFAULT_SPLIT, DEFAULT_WALK_FORWARD, RESULTS_DIR
from strategy_registry import get_strategy_config
from walk_forward import Fold, FoldResult, compute_equity_stats, parameter_stability, stitch_oos_equity
from mr_walk_forward import mr_parameter_stability


def load_fold_result(path: str) -> FoldResult:
    with open(path) as f:
        d = json.load(f)
    fold = Fold(
        index=d["fold_index"],
        train_start=pd.Timestamp(d["fold_train_start"]),
        test_start=pd.Timestamp(d["fold_test_start"]),
        test_end=pd.Timestamp(d["fold_test_end"]),
    )
    oos_equity = pd.Series({pd.Timestamp(k): v for k, v in d["oos_equity"].items()}).sort_index()
    return FoldResult(
        fold=fold,
        best_params=d["best_params"],
        in_sample_metric_name=d["in_sample_metric_name"],
        in_sample_metric_value=d["in_sample_metric_value"],
        in_sample_combos_tested=d["in_sample_combos_tested"],
        oos_return_pct=d["oos_return_pct"],
        oos_trades=d["oos_trades"],
        oos_win_rate=d["oos_win_rate"],
        oos_max_drawdown_pct=d["oos_max_drawdown_pct"],
        oos_equity=oos_equity,
        oos_trade_pnls=d["oos_trade_pnls"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    args = ap.parse_args()

    cfg = get_strategy_config(args.strategy)
    timeframe = cfg["timeframe"]
    family = cfg["family"]

    paths = sorted(glob.glob(os.path.join(args.fold_dir, "*.json")))
    if not paths:
        raise SystemExit(f"No fold result files found in {args.fold_dir}")

    fold_results = [load_fold_result(p) for p in paths]
    fold_results.sort(key=lambda fr: fr.fold.index)

    expected = fold_results[0].fold.index  # sanity: indices should be 0..N-1 contiguous
    for fr in fold_results:
        if fr.fold.index != expected:
            raise SystemExit(f"Missing or duplicate fold index: expected {expected}, found {fr.fold.index}")
        expected += 1

    stitched = stitch_oos_equity(fold_results, args.initial_capital)
    overall_stats = compute_equity_stats(stitched, args.initial_capital)
    stability = (
        mr_parameter_stability(fold_results) if family == "mean_reversion"
        else parameter_stability(fold_results)
    )

    all_pnls = [p for fr in fold_results for p in fr.oos_trade_pnls]
    total_trades = len(all_pnls)
    win_rate = (sum(1 for p in all_pnls if p > 0) / total_trades * 100) if total_trades else 0.0

    # --- acceptance check: no OOS bar reaches the holdout -------------------
    holdout_start = pd.Timestamp(DEFAULT_SPLIT.holdout_start)
    leaked = len(stitched) > 0 and stitched.index[-1] >= holdout_start
    if leaked:
        raise SystemExit(
            f"ACCEPTANCE CHECK FAILED — last OOS bar {stitched.index[-1]} >= "
            f"holdout_start ({holdout_start})"
        )
    print(f"ACCEPTANCE CHECK PASSED — every OOS bar is strictly before holdout_start ({holdout_start})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    equity_path = os.path.join(RESULTS_DIR, "walk_forward_oos_equity.csv")
    folds_path = os.path.join(RESULTS_DIR, "walk_forward_fold_results.csv")

    stitched.rename("equity").to_frame().assign(
        drawdown_pct=lambda d: (d["equity"] / d["equity"].expanding().max() - 1) * 100
    ).to_csv(equity_path, index_label="timestamp")

    rows = []
    for fr in fold_results:
        rows.append({
            "fold": fr.fold.index,
            "train_start": fr.fold.train_start,
            "test_start": fr.fold.test_start,
            "test_end": fr.fold.test_end,
            **fr.best_params,
            f"in_sample_{fr.in_sample_metric_name}": fr.in_sample_metric_value,
            "in_sample_combos_tested": fr.in_sample_combos_tested,
            "oos_return_pct": fr.oos_return_pct,
            "oos_trades": fr.oos_trades,
            "oos_win_rate": fr.oos_win_rate,
            "oos_max_drawdown_pct": fr.oos_max_drawdown_pct,
        })
    pd.DataFrame(rows).to_csv(folds_path, index=False)

    # Named after the registry key itself (not remapped by timeframe) so a
    # new strategy sharing v1's/v4's timeframe — e.g. this mean-reversion
    # strategy on 4h — writes its own walk_forward_results_mr_4h.json instead
    # of silently overwriting walk_forward_results_v1.json. In existing usage
    # args.strategy is already "v1"/"v4", so this is behavior-preserving for
    # both of those.
    strategy_name = args.strategy
    summary = {
        "strategy": strategy_name,
        "timeframe": timeframe,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "grid_mode": "fast (reduced grid, sanity check only)" if args.fast else "full",
        "config": {
            "train_years": DEFAULT_WALK_FORWARD.train_years,
            "test_months": DEFAULT_WALK_FORWARD.test_months,
            "step_months": DEFAULT_WALK_FORWARD.step_months,
            "expanding": DEFAULT_WALK_FORWARD.expanding,
        },
        "acceptance_check_passed": not leaked,
        "num_folds": len(fold_results),
        "total_oos_trades": total_trades,
        "oos_win_rate_pct": win_rate,
        "oos_sharpe": overall_stats.get("sharpe_ratio"),
        "oos_total_return_pct": overall_stats.get("total_return_pct"),
        "oos_cagr_pct": overall_stats.get("cagr_pct"),
        "oos_max_drawdown_pct": overall_stats.get("max_drawdown_pct"),
        "oos_calmar_ratio": overall_stats.get("calmar_ratio"),
        "oos_coverage_start": str(overall_stats.get("start")) if overall_stats else None,
        "oos_coverage_end": str(overall_stats.get("end")) if overall_stats else None,
        "parameter_stability": {
            name: {"is_stable": info["is_stable"], "distinct_values": info["distinct_values"], "sequence": info["sequence"]}
            for name, info in stability.items()
        },
    }
    summary_path = f"walk_forward_results_{strategy_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote {equity_path}, {folds_path}, {summary_path}")
    print(f"Stitched OOS: return={overall_stats.get('total_return_pct'):.2f}% "
          f"sharpe={overall_stats.get('sharpe_ratio'):.3f} "
          f"maxDD={overall_stats.get('max_drawdown_pct'):.2f}% trades={total_trades}")


if __name__ == "__main__":
    main()

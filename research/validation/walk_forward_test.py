"""
Phase 2 acceptance-check script — VALIDATION_REMEDIATION_PLAN.md.

Runs the rolling walk-forward pipeline end to end and writes the two
artifacts the plan's Phase 2 acceptance check asks for:

  results/walk_forward_oos_equity.csv   — the stitched, OOS-only equity curve
  results/walk_forward_fold_results.csv — per-fold params + in-sample metric
                                           + out-of-sample performance

Every number printed under "STITCHED OUT-OF-SAMPLE EQUITY CURVE" comes from
bars that were never inside the training window used to pick that bar's
parameters. Compare this to V1's in-sample-only 855% headline from
STRATEGY_VERSION_AUDIT.md — expect it to be far lower. That's the point:
a validated, honest number, not a bigger one.

Usage:
    python walk_forward_test.py                   # full 2800-combo grid per fold
    python walk_forward_test.py --fast             # small grid, quick sanity run
    python walk_forward_test.py --jobs 8            # parallelize each fold's grid search
    python walk_forward_test.py --train-years 1.5 --test-months 4
"""

import argparse
import json
import os
import sys

import pandas as pd

from config import DEFAULT_ROBUSTNESS, DEFAULT_SPLIT, DEFAULT_WALK_FORWARD, RESULTS_DIR, WalkForwardConfig, RobustnessRanges
from data_splits import describe_split
from walk_forward import run_walk_forward


FAST_RANGES = RobustnessRanges(
    entry_len=(15, 40, 10),
    exit_len=(7, 20, 6),
    trail_mult=(2.0, 4.0, 1.0),
    risk_percent=(0.5, 1.0, 0.5),
    pyramid_spacing_n=(0.5, 1.5, 0.5),
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 walk-forward validation")
    p.add_argument("--timeframe", default="4h", choices=["1h", "4h"])
    p.add_argument("--train-years", type=float, default=DEFAULT_WALK_FORWARD.train_years)
    p.add_argument("--test-months", type=int, default=DEFAULT_WALK_FORWARD.test_months)
    p.add_argument("--step-months", type=int, default=DEFAULT_WALK_FORWARD.step_months)
    p.add_argument("--expanding", action="store_true", help="anchored walk-forward instead of rolling")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers per fold's grid search")
    p.add_argument("--fast", action="store_true", help="small parameter grid for a quick sanity run")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    return p.parse_args()


def main():
    args = parse_args()

    from data_fetcher import download_btc_data
    print(f"Loading BTC {args.timeframe} data...")
    df = download_btc_data(timeframe=args.timeframe)

    print(f"\n{describe_split()}")

    config = WalkForwardConfig(
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
        expanding=args.expanding,
    )
    ranges = FAST_RANGES if args.fast else DEFAULT_ROBUSTNESS
    if args.fast:
        print("\n--fast: using a reduced parameter grid for a quick sanity run, "
              "not a real validation result. Drop --fast for the real one.")

    report = run_walk_forward(
        df, config=config, ranges=ranges,
        initial_capital=args.initial_capital, n_jobs=args.jobs, verbose=True,
    )

    print(report.summary_report())

    # --- Acceptance check: re-verify no OOS bar reaches the holdout ------
    holdout_start = pd.Timestamp(DEFAULT_SPLIT.holdout_start)
    if len(report.stitched_equity) > 0:
        last_oos_bar = report.stitched_equity.index[-1]
        leaked = last_oos_bar >= holdout_start
    else:
        leaked = False
    print("\n" + "=" * 80)
    if leaked:
        print(f"ACCEPTANCE CHECK: FAILED — last OOS bar ({last_oos_bar}) is at or "
              f"past holdout_start ({holdout_start}).")
        print("=" * 80)
        sys.exit(1)
    else:
        print("ACCEPTANCE CHECK: PASSED — every OOS bar falls strictly before "
              f"holdout_start ({holdout_start}); the true holdout was never scored.")
        print("=" * 80)

    # --- Save artifacts ----------------------------------------------------
    equity_path = os.path.join(RESULTS_DIR, "walk_forward_oos_equity.csv")
    folds_path = os.path.join(RESULTS_DIR, "walk_forward_fold_results.csv")

    report.stitched_equity.rename("equity").to_frame().assign(
        drawdown_pct=lambda d: (d["equity"] / d["equity"].expanding().max() - 1) * 100
    ).to_csv(equity_path, index_label="timestamp")

    report.fold_results_df().to_csv(folds_path, index=False)

    print(f"\nSaved:\n  {equity_path}\n  {folds_path}")

    # --- Publish a small, git-tracked summary ------------------------------
    # results/ is gitignored (regenerable); this is the durable, committed
    # pointer to it — read_only degradation check in build_readiness_gates.py
    # (Phase 5) looks for exactly this filename/schema.
    strategy_name = {"4h": "v1", "1h": "v4"}.get(args.timeframe, args.timeframe)
    summary = {
        "strategy": strategy_name,
        "timeframe": args.timeframe,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "grid_mode": "fast (reduced grid, sanity check only)" if args.fast else "full",
        "config": {
            "train_years": config.train_years,
            "test_months": config.test_months,
            "step_months": config.step_months,
            "expanding": config.expanding,
        },
        "acceptance_check_passed": not leaked,
        "num_folds": len(report.fold_results),
        "total_oos_trades": report.total_oos_trades,
        "oos_win_rate_pct": report.oos_win_rate,
        "oos_sharpe": report.overall_stats.get("sharpe_ratio"),
        "oos_total_return_pct": report.overall_stats.get("total_return_pct"),
        "oos_cagr_pct": report.overall_stats.get("cagr_pct"),
        "oos_max_drawdown_pct": report.overall_stats.get("max_drawdown_pct"),
        "oos_calmar_ratio": report.overall_stats.get("calmar_ratio"),
        "oos_coverage_start": str(report.overall_stats.get("start")) if report.overall_stats else None,
        "oos_coverage_end": str(report.overall_stats.get("end")) if report.overall_stats else None,
        "parameter_stability": {
            name: {"is_stable": info["is_stable"], "distinct_values": info["distinct_values"],
                   "sequence": info["sequence"]}
            for name, info in report.stability.items()
        },
    }
    summary_path = f"walk_forward_results_{strategy_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()

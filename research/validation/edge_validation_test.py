"""
Phase 3 driver — VALIDATION_REMEDIATION_PLAN.md: "Make the Monte Carlo test
what it claims to test."

Runs two analyses and prints them side by side so the difference is obvious:

  1. OOS block-bootstrap (monte_carlo.OOSBlockBootstrap) — resamples the
     walk-forward out-of-sample RETURNS from Phase 2. Answers: "how much
     would the OOS result plausibly vary?" A genuine forward-reliability
     question.

  2. Deflated Sharpe ratio (deflated_sharpe.py) — corrects the in-sample
     robustness_test.py grid search's best Sharpe for having tried 2,800+
     combinations. Answers: "is the best-of-2800 number actually good, or
     just the best of 2800 noisy tries?"

Prerequisites (run these first if the files below don't exist):
    python walk_forward_test.py            # produces walk_forward_oos_equity.csv
    python robustness_test.py              # produces robustness_results.csv

Usage:
    python edge_validation_test.py
    python edge_validation_test.py --n-simulations 2000 --block-size 60
"""

import argparse
import os

import pandas as pd

from config import RESULTS_DIR, BlockBootstrapConfig, StrategyParams, DEFAULT_PARAMS
from monte_carlo import OOSBlockBootstrap
from deflated_sharpe import (
    annualized_to_periodic,
    returns_skew_kurtosis,
    deflated_sharpe_ratio,
)
from walk_forward import _infer_bars_per_year


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 edge validation")
    p.add_argument("--n-simulations", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=42, help="bootstrap block size in bars")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def run_oos_bootstrap_section(args) -> str:
    equity_path = os.path.join(RESULTS_DIR, "walk_forward_oos_equity.csv")
    if not os.path.exists(equity_path):
        msg = (f"\nSKIPPED OOS block-bootstrap: {equity_path} not found.\n"
               f"Run `python walk_forward_test.py` first.")
        print(msg)
        return msg

    returns = OOSBlockBootstrap.load_oos_returns(equity_path)
    bootstrap = OOSBlockBootstrap(BlockBootstrapConfig(
        block_size_bars=args.block_size,
        n_simulations=args.n_simulations,
        random_seed=args.random_seed,
    ))
    bootstrap.run(returns, initial_capital=args.initial_capital)
    report = bootstrap.get_summary_report()
    print(report)

    out_path = os.path.join(RESULTS_DIR, "oos_bootstrap_equities.csv")
    pd.DataFrame({"final_equity": bootstrap.results.equity_distribution}).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return report


def run_deflated_sharpe_section(args) -> str:
    grid_path = os.path.join(RESULTS_DIR, "robustness_results.csv")
    if not os.path.exists(grid_path):
        msg = (f"\nSKIPPED deflated Sharpe: {grid_path} not found.\n"
               f"Run `python robustness_test.py` first.")
        print(msg)
        return msg

    grid_df = pd.read_csv(grid_path)
    if "sharpe" in grid_df.columns:
        sharpe_col = "sharpe"
    elif "sharpe_ratio" in grid_df.columns:
        sharpe_col = "sharpe_ratio"
    else:
        msg = f"\nSKIPPED deflated Sharpe: no sharpe column found in {grid_path}."
        print(msg)
        return msg

    n_trials = len(grid_df)
    best_row = grid_df.loc[grid_df[sharpe_col].idxmax()]
    best_sr_annualized = float(best_row[sharpe_col])

    try:
        from data_fetcher import download_btc_data
        from data_splits import get_development
        from strategy import TurtleDonchianStrategy

        print("\nRe-running the winning combo's backtest to get its own return series "
              "(needed for skew/kurtosis/n_obs — not stored in robustness_results.csv)...")
        df = download_btc_data(timeframe="4h")
        dev_df = get_development(df)

        params = StrategyParams(
            entry_len=int(best_row.get("entry_len", DEFAULT_PARAMS.entry_len)),
            exit_len=int(best_row.get("exit_len", DEFAULT_PARAMS.exit_len)),
            trail_mult=float(best_row.get("trail_mult", DEFAULT_PARAMS.trail_mult)),
            risk_percent=float(best_row.get("risk_pct", DEFAULT_PARAMS.risk_percent)),
            pyramid_spacing_n=float(best_row.get("pyramid_spacing", DEFAULT_PARAMS.pyramid_spacing_n)),
            atr_len=DEFAULT_PARAMS.atr_len,
            size_stop_mult=DEFAULT_PARAMS.size_stop_mult,
            max_units=DEFAULT_PARAMS.max_units,
            long_only=DEFAULT_PARAMS.long_only,
            use_regime_filter=DEFAULT_PARAMS.use_regime_filter,
            lot_step=DEFAULT_PARAMS.lot_step,
            commission_pct=DEFAULT_PARAMS.commission_pct,
            slippage_pct=DEFAULT_PARAMS.slippage_pct,
        )
        strategy = TurtleDonchianStrategy(params)
        results = strategy.run_backtest(dev_df, initial_capital=args.initial_capital, verbose=False)
        bar_returns = results["equity"].pct_change().dropna()

        bars_per_year = _infer_bars_per_year(bar_returns.index)
        n_obs = len(bar_returns)
        skew, kurt = returns_skew_kurtosis(bar_returns.to_numpy())
    except Exception as e:
        print(f"\nCouldn't re-run the winning backtest to get its return series ({e}). "
              f"Falling back to a normal-distribution assumption (skew=0, kurtosis=3) and "
              f"n_obs estimated from the grid's bars-per-trial, if available — treat the "
              f"resulting DSR as an approximation, not the precise figure.")
        bars_per_year = 365 * 6  # 4h default
        n_obs = int(best_row.get("trades", 100)) * 20  # crude fallback, clearly approximate
        skew, kurt = 0.0, 3.0

    trial_sharpes_annualized = grid_df[sharpe_col].to_numpy()
    trial_sharpes_periodic = annualized_to_periodic(trial_sharpes_annualized, bars_per_year)
    best_sr_periodic = annualized_to_periodic(best_sr_annualized, bars_per_year)

    result = deflated_sharpe_ratio(
        sr_observed_periodic=best_sr_periodic,
        n_obs=n_obs,
        n_trials=n_trials,
        trial_sharpes_periodic=trial_sharpes_periodic,
        skew=skew,
        kurtosis=kurt,
        sr_observed_annualized=best_sr_annualized,
    )
    report = result.report()
    print(report)
    return report


def main():
    args = parse_args()

    print(f"\n{'=' * 80}\nPHASE 3: EDGE VALIDATION\n{'=' * 80}")
    print("Two independent checks below. Neither depends on the other — read whichever "
          "prerequisite files exist.")

    section1 = run_oos_bootstrap_section(args)
    section2 = run_deflated_sharpe_section(args)

    report_path = os.path.join(RESULTS_DIR, "edge_validation_report.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("PHASE 3: EDGE VALIDATION\n")
        f.write("=" * 80 + "\n")
        f.write(section1 + "\n\n")
        f.write(section2 + "\n")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()

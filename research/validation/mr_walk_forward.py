"""
Walk-forward OOS validation for BollingerRSIMeanReversion (mean_reversion_strategy.py) —
a new, independent strategy, NOT a variant of TurtleDonchianStrategy.

Reuses, unchanged, from this repo's existing tooling:
  - generate_folds / stitch_oos_equity / compute_equity_stats (walk_forward.py):
    strategy-agnostic — they operate on plain pd.Series equity curves and
    Fold boundaries, so they apply just as well to this strategy.
  - get_development (data_splits.py): same frozen in-sample+validation window,
    same untouched holdout.
  - deflated_sharpe.py: same PSR/DSR significance testing applied to V4.

Does NOT reuse: RobustnessTester / optimize_fold / TurtleDonchianStrategy /
StrategyParams — those are specific to the Donchian-breakout parameter space.
mr_robustness.py provides the equivalent grid-search scorer for THIS
strategy's own parameter space.

Never touches config.DEFAULT_SPLIT.holdout_start onward — generate_folds()
enforces this structurally (see walk_forward.py's docstring/assertions).

Usage:
    python mr_walk_forward.py 4h
    python mr_walk_forward.py 1h
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from config import DEFAULT_SPLIT, DEFAULT_WALK_FORWARD, WalkForwardConfig
from data_fetcher import download_btc_data
from data_splits import get_development, describe_split
from mean_reversion_strategy import BollingerRSIMeanReversion, MeanReversionParams
from mr_robustness import MRRobustnessTester, MRRobustnessRanges, DEFAULT_MR_ROBUSTNESS, MRParameterResult
from walk_forward import (
    Fold,
    FoldResult,
    generate_folds,
    stitch_oos_equity,
    compute_equity_stats,
    _oos_equity_stats_local,
)

MR_PARAM_NAMES = ["ma_len", "band_mult", "rsi_oversold", "rsi_overbought",
                  "stop_atr_mult", "max_hold_bars"]


def _build_mr_params(best_params: dict) -> MeanReversionParams:
    return MeanReversionParams(
        ma_len=best_params["ma_len"],
        band_mult=best_params["band_mult"],
        rsi_oversold=best_params["rsi_oversold"],
        rsi_overbought=best_params["rsi_overbought"],
        stop_atr_mult=best_params["stop_atr_mult"],
        max_hold_bars=best_params["max_hold_bars"],
        rsi_len=14,
        atr_len=14,
        risk_percent=1.0,
        long_only=False,
        lot_step=0.001,
        commission_pct=0.08,
        slippage_pct=0.05,
    )


def optimize_fold_mr(
    train_df: pd.DataFrame,
    ranges: MRRobustnessRanges,
    config: WalkForwardConfig,
    initial_capital: float,
) -> Tuple[dict, MRParameterResult, pd.DataFrame]:
    """Grid-search this fold's training window only, same discipline as
    walk_forward.optimize_fold(): select by config.selection_metric, exclude
    combos below min_trades_for_selection so a lucky near-zero-trade combo
    can't top the ranking."""
    tester = MRRobustnessTester(ranges)
    grid = tester.generate_parameter_grid()

    results: List[MRParameterResult] = []
    for params in grid:
        try:
            results.append(tester.test_parameters(train_df, params, initial_capital))
        except Exception:
            continue

    results_df = pd.DataFrame([asdict(r) for r in results]) if results else pd.DataFrame()

    eligible = [r for r in results if r.num_trades >= config.min_trades_for_selection]
    if not eligible:
        eligible = results
    if not eligible:
        raise ValueError("No parameter combination produced a valid result for this fold.")

    metric = config.selection_metric
    field_name = "sharpe_ratio" if metric in ("sharpe", "sharpe_ratio") else metric
    best = max(eligible, key=lambda r: getattr(r, field_name))
    best_params = {name: getattr(best, name) for name in MR_PARAM_NAMES}
    return best_params, best, results_df


def run_fold_mr(
    df: pd.DataFrame,
    fold: Fold,
    ranges: MRRobustnessRanges,
    config: WalkForwardConfig,
    initial_capital: float,
) -> FoldResult:
    train_df = df[(df.index >= fold.train_start) & (df.index < fold.test_start)]
    combined_df = df[(df.index >= fold.train_start) & (df.index < fold.test_end)]

    if len(train_df) < 50:
        raise ValueError(f"Fold {fold.index}: training window has only {len(train_df)} bars — too short to optimize.")

    best_params, best_result, _combo_df = optimize_fold_mr(train_df, ranges, config, initial_capital)

    strategy = BollingerRSIMeanReversion(_build_mr_params(best_params))
    results = strategy.run_backtest(combined_df, initial_capital=initial_capital, verbose=False)

    oos_equity = results.loc[(results.index >= fold.test_start) & (results.index < fold.test_end), "equity"]
    oos_ret_pct, oos_dd_pct = _oos_equity_stats_local(oos_equity)

    oos_trades = [t for t in strategy.trades if fold.test_start <= t.entry_time < fold.test_end]
    oos_pnls = [t.pnl for t in oos_trades if t.pnl is not None]
    win_rate = (sum(1 for p in oos_pnls if p > 0) / len(oos_pnls) * 100) if oos_pnls else 0.0

    return FoldResult(
        fold=fold,
        best_params=best_params,
        in_sample_metric_name="sharpe_ratio",
        in_sample_metric_value=best_result.sharpe_ratio,
        in_sample_combos_tested=len(_combo_df),
        oos_return_pct=oos_ret_pct,
        oos_trades=len(oos_trades),
        oos_win_rate=win_rate,
        oos_max_drawdown_pct=oos_dd_pct,
        oos_equity=oos_equity,
        oos_trade_pnls=oos_pnls,
    )


def mr_parameter_stability(fold_results: List[FoldResult]) -> Dict[str, dict]:
    stability = {}
    for name in MR_PARAM_NAMES:
        sequence = [fr.best_params[name] for fr in fold_results]
        changes = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
        n_transitions = max(1, len(sequence) - 1)
        stability[name] = {
            "sequence": sequence,
            "distinct_values": len(set(sequence)),
            "changes": changes,
            "is_stable": (1 - (changes / n_transitions)) >= 0.5,
        }
    return stability


def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "4h"
    initial_capital = 100_000.0

    print(describe_split())
    print(f"Timeframe: {timeframe}")
    print("Strategy: BollingerRSIMeanReversion (new — independent of the turtle/Donchian family)")

    df = download_btc_data(timeframe=timeframe)
    df = get_development(df)  # never touch the 2025-01-01+ holdout
    print(f"Development window: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")

    folds = generate_folds(df, DEFAULT_WALK_FORWARD, DEFAULT_SPLIT)
    print(f"\n{len(folds)} folds (train={DEFAULT_WALK_FORWARD.train_years}y, "
          f"test={DEFAULT_WALK_FORWARD.test_months}mo, step={DEFAULT_WALK_FORWARD.step_months}mo)")
    print(f"Grid size per fold: {len(MRRobustnessTester(DEFAULT_MR_ROBUSTNESS).generate_parameter_grid())} combos")

    fold_results: List[FoldResult] = []
    for fold in folds:
        fr = run_fold_mr(df, fold, DEFAULT_MR_ROBUSTNESS, DEFAULT_WALK_FORWARD, initial_capital)
        fold_results.append(fr)
        print(f"  Fold {fold.index}: test {fold.test_start.date()} -> {fold.test_end.date()} | "
              f"best_params={fr.best_params} | in-sample Sharpe={fr.in_sample_metric_value:.2f} | "
              f"OOS return {fr.oos_return_pct:+.2f}% | trades {fr.oos_trades} | "
              f"win rate {fr.oos_win_rate:.1f}% | max DD {fr.oos_max_drawdown_pct:.2f}%")

    stitched = stitch_oos_equity(fold_results, initial_capital)
    stats = compute_equity_stats(stitched, initial_capital)
    stability = mr_parameter_stability(fold_results)

    total_trades = sum(fr.oos_trades for fr in fold_results)
    total_wins = sum(1 for fr in fold_results for p in fr.oos_trade_pnls if p > 0)

    print("\n" + "=" * 80)
    print(f"MEAN-REVERSION STRATEGY — STITCHED OUT-OF-SAMPLE EQUITY CURVE ({timeframe})")
    print("(per-fold grid search on training window only, scored strictly OOS)")
    print("=" * 80)
    for k in ["total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "calmar_ratio"]:
        print(f"  {k}: {stats.get(k)}")
    print(f"  total_oos_trades: {total_trades}")
    print(f"  oos_win_rate: {(total_wins/total_trades*100) if total_trades else 0:.2f}%")

    print("\n" + "-" * 80)
    print("PARAMETER STABILITY ACROSS FOLDS")
    print("-" * 80)
    for name, info in stability.items():
        status = "stable" if info["is_stable"] else "UNSTABLE"
        seq = " -> ".join(str(v) for v in info["sequence"])
        print(f"{name}: {status}  ({info['distinct_values']} distinct value(s))")
        print(f"  {seq}")

    stitched.to_csv(f"results/mean_reversion_oos_equity_{timeframe}.csv")
    print(f"\nSaved results/mean_reversion_oos_equity_{timeframe}.csv")
    return fold_results, stats


if __name__ == "__main__":
    main()

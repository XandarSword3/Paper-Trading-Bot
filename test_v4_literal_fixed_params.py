"""
Scores V4's EXACT literal live parameters (entry_len=8, exit_len=16,
atr_len=14, trail_mult=3.5, risk_percent=1.0%, pyramid_spacing_n=1.5 — taken
directly from github_bot_v4.py's ENTRY_LEN/EXIT_LEN/ATR_LEN/TRAIL_MULT/
RISK_PCT constants) across the same rolling walk-forward OOS folds that
walk_forward_test.py uses, WITHOUT re-optimizing per fold.

This is deliberately different from walk_forward_test.py: that script
re-picks the best parameters from a search grid on each fold's training
window, then scores the fold winner OOS — which tells you whether the
STRATEGY FAMILY has an edge. This script fixes V4's actual hardcoded
parameters for every fold, which tells you whether the CONFIGURATION
ACTUALLY RUNNING LIVE has one.

Never touches the holdout (2025-01-01 onward) — folds are generated only
inside the development window, same as walk_forward.generate_folds().
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import DEFAULT_SPLIT, DEFAULT_WALK_FORWARD, StrategyParams
from data_fetcher import download_btc_data
from data_splits import get_development, describe_split
from strategy import TurtleDonchianStrategy
from walk_forward import (
    generate_folds,
    stitch_oos_equity,
    compute_equity_stats,
    FoldResult,
    _oos_equity_stats_local,
)

V4_LITERAL_PARAMS = StrategyParams(
    entry_len=8,
    exit_len=16,
    atr_len=14,
    trail_mult=3.5,
    risk_percent=1.0,       # github_bot_v4.py's RISK_PCT=0.01 == 1% == 1.0 here
    pyramid_spacing_n=1.5,  # not in github_bot_v4.py's constants; kept at repo default
    max_units=4,
    # long_only / use_regime_filter / lot_step / commission_pct / slippage_pct: repo defaults
)


def run_fixed_param_fold(df: pd.DataFrame, fold, params: StrategyParams, initial_capital: float) -> FoldResult:
    combined_df = df[(df.index >= fold.train_start) & (df.index < fold.test_end)]

    strategy = TurtleDonchianStrategy(params)
    results = strategy.run_backtest(combined_df, initial_capital=initial_capital, verbose=False)

    oos_equity = results.loc[(results.index >= fold.test_start) & (results.index < fold.test_end), "equity"]
    oos_ret_pct, oos_dd_pct = _oos_equity_stats_local(oos_equity)

    oos_trades = [t for t in strategy.trades if fold.test_start <= t.entry_time < fold.test_end]
    oos_pnls = [t.pnl for t in oos_trades if t.pnl is not None]
    win_rate = (sum(1 for p in oos_pnls if p > 0) / len(oos_pnls) * 100) if oos_pnls else 0.0

    return FoldResult(
        fold=fold,
        best_params={
            "entry_len": params.entry_len, "exit_len": params.exit_len,
            "atr_len": params.atr_len, "trail_mult": params.trail_mult,
            "risk_percent": params.risk_percent, "pyramid_spacing_n": params.pyramid_spacing_n,
        },
        in_sample_metric_name="fixed_params_no_refit",
        in_sample_metric_value=0.0,
        in_sample_combos_tested=1,
        oos_return_pct=oos_ret_pct,
        oos_trades=len(oos_trades),
        oos_win_rate=win_rate,
        oos_max_drawdown_pct=oos_dd_pct,
        oos_equity=oos_equity,
        oos_trade_pnls=oos_pnls,
    )


def main():
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    initial_capital = 100_000.0
    print(describe_split())
    print(f"Timeframe: {timeframe}")

    df = download_btc_data(timeframe=timeframe)
    df = get_development(df)  # never touch the 2025-01-01+ holdout
    print(f"Development window: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")

    folds = generate_folds(df, DEFAULT_WALK_FORWARD, DEFAULT_SPLIT)
    print(f"\n{len(folds)} folds generated (train={DEFAULT_WALK_FORWARD.train_years}y, "
          f"test={DEFAULT_WALK_FORWARD.test_months}mo, step={DEFAULT_WALK_FORWARD.step_months}mo)")

    fold_results = []
    for fold in folds:
        fr = run_fixed_param_fold(df, fold, V4_LITERAL_PARAMS, initial_capital)
        fold_results.append(fr)
        print(f"  Fold {fold.index}: test {fold.test_start.date()} -> {fold.test_end.date()} | "
              f"OOS return {fr.oos_return_pct:+.2f}% | trades {fr.oos_trades} | "
              f"win rate {fr.oos_win_rate:.1f}% | max DD {fr.oos_max_drawdown_pct:.2f}%")

    stitched = stitch_oos_equity(fold_results, initial_capital)
    stats = compute_equity_stats(stitched, initial_capital)

    total_trades = sum(fr.oos_trades for fr in fold_results)
    total_wins = sum(1 for fr in fold_results for p in fr.oos_trade_pnls if p > 0)

    print("\n" + "=" * 80)
    print("V4 LITERAL LIVE PARAMETERS — STITCHED OUT-OF-SAMPLE EQUITY CURVE")
    print("(entry_len=8, exit_len=16, atr_len=14, trail_mult=3.5, risk=1.0%, no per-fold refitting)")
    print("=" * 80)
    for k in ["total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "calmar_ratio"]:
        print(f"  {k}: {stats.get(k)}")
    print(f"  total_oos_trades: {total_trades}")
    print(f"  oos_win_rate: {(total_wins/total_trades*100) if total_trades else 0:.2f}%")

    stitched.to_csv("results/v4_literal_params_oos_equity.csv")
    print("\nSaved results/v4_literal_params_oos_equity.csv")


if __name__ == "__main__":
    main()

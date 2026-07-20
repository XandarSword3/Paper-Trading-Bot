"""
Rolling walk-forward optimization — Phase 2 of VALIDATION_REMEDIATION_PLAN.md.

Replaces the question robustness_test.py's flat, single-shot grid search
could never actually answer: "would these parameters have made money on data
the search never saw?" Every fold here optimizes on one window and is scored
only on the window immediately after it, which the optimizer never touches.
Only the stitched-together, out-of-window results count as performance —
the in-sample number a fold's grid search produces is used purely to pick
that fold's parameters, and is reported separately, never blended into the
OOS figures.

This module never touches the true holdout (config.DEFAULT_SPLIT.holdout_start
onward). generate_folds() enforces that structurally: every fold is clipped to
data_splits.get_development(df), and a hard assertion re-checks every fold
boundary before any backtest runs. If that assertion trips, the caller passed
data that already leaked, so this raises rather than paper over it.

Usage:
    from data_fetcher import download_btc_data
    from walk_forward import run_walk_forward

    df = download_btc_data(timeframe="4h")
    report = run_walk_forward(df)
    print(report.summary_report())
"""

import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DEFAULT_PARAMS,
    DEFAULT_ROBUSTNESS,
    DEFAULT_SPLIT,
    DEFAULT_WALK_FORWARD,
    RobustnessRanges,
    StrategyParams,
    WalkForwardConfig,
)
from data_splits import get_development
from robustness_test import ParameterResult, RobustnessTester
from strategy import TurtleDonchianStrategy

warnings.filterwarnings("ignore")

PARAM_FIELDS = ["entry_len", "exit_len", "trail_mult", "risk_percent", "pyramid_spacing"]


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    """One walk-forward fold: optimize on [train_start, test_start), score on
    [test_start, test_end). test_start == train_end by construction."""
    index: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    df: pd.DataFrame,
    config: WalkForwardConfig = DEFAULT_WALK_FORWARD,
    split=DEFAULT_SPLIT,
) -> List[Fold]:
    """
    Build the rolling (or expanding) fold schedule, clipped to the
    development window so no fold's test window can ever reach the holdout.

    df is expected to already be development-only (see get_development), but
    every fold boundary is re-checked against split.holdout_start below
    regardless of what the caller passed in — this is the structural half of
    the plan's "grep for holdout_start" acceptance check, not a substitute
    for it.
    """
    dev_start = pd.Timestamp(split.in_sample_start)
    holdout_start = pd.Timestamp(split.holdout_start)

    data_start = max(dev_start, df.index[0])
    data_end = min(holdout_start, df.index[-1] + pd.Timedelta(seconds=1))

    train_len = pd.DateOffset(months=round(config.train_years * 12))
    test_len = pd.DateOffset(months=config.test_months)
    step_len = pd.DateOffset(months=config.step_months)

    folds: List[Fold] = []
    fold_train_start = data_start
    i = 0

    while True:
        train_end = fold_train_start + train_len if not config.expanding else data_start + train_len + i * step_len
        train_start = fold_train_start if not config.expanding else data_start
        test_start = train_end
        test_end = test_start + test_len

        if test_end > data_end:
            break

        # Hard leakage guard — never trust the caller, re-derive and assert.
        assert train_start < test_start <= test_end, "fold boundaries out of order"
        assert test_start >= dev_start, "fold train window starts before development window"
        assert test_end <= holdout_start, (
            f"fold {i} test window ends {test_end} — at or past holdout_start "
            f"{holdout_start}. Refusing to score on holdout data."
        )

        folds.append(Fold(index=i, train_start=train_start, test_start=test_start, test_end=test_end))

        fold_train_start = fold_train_start + step_len
        i += 1

    return folds


# ---------------------------------------------------------------------------
# Per-fold optimization (in-sample only) + scoring (out-of-sample only)
# ---------------------------------------------------------------------------

_worker_df: Optional[pd.DataFrame] = None


def _init_worker(df: pd.DataFrame) -> None:
    """ProcessPoolExecutor initializer: stash the fold's training data once
    per worker process instead of re-pickling it on every submitted task."""
    global _worker_df
    _worker_df = df


def _evaluate_in_worker(params_dict: dict, initial_capital: float) -> ParameterResult:
    return RobustnessTester().test_parameters(_worker_df, params_dict, initial_capital)


def optimize_fold(
    train_df: pd.DataFrame,
    ranges: RobustnessRanges,
    config: WalkForwardConfig,
    initial_capital: float,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> Tuple[dict, ParameterResult, pd.DataFrame]:
    """
    Grid-search this fold's training window only. Returns the winning
    parameter dict, its full ParameterResult, and a DataFrame of every combo
    tested (for per-fold plateau inspection, same shape as
    RobustnessTester.results_df).
    """
    tester = RobustnessTester(ranges)
    grid = tester.generate_parameter_grid()

    results: List[ParameterResult] = []

    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(train_df,)) as ex:
            futures = [ex.submit(_evaluate_in_worker, p, initial_capital) for p in grid]
            iterator = tqdm(futures, desc="  optimizing fold", disable=not show_progress)
            for fut in iterator:
                try:
                    results.append(fut.result())
                except Exception:
                    continue
    else:
        iterator = tqdm(grid, desc="  optimizing fold", disable=not show_progress)
        for params in iterator:
            try:
                results.append(tester.test_parameters(train_df, params, initial_capital))
            except Exception:
                continue

    results_df = pd.DataFrame([asdict(r) for r in results]) if results else pd.DataFrame()

    # Exclude near-riskless, near-zero-trade combos from selection — a combo
    # with 1 trade and no losses can post an inflated Sharpe that has no
    # bearing on whether the strategy actually works.
    eligible = [r for r in results if r.num_trades >= config.min_trades_for_selection]
    if not eligible:
        eligible = results  # fall back rather than fail the fold outright

    if not eligible:
        raise ValueError("No parameter combination produced a valid result for this fold.")

    metric = config.selection_metric
    best = max(eligible, key=lambda r: getattr(r, _map_metric_field(metric)))
    best_params = {
        "entry_len": best.entry_len,
        "exit_len": best.exit_len,
        "trail_mult": best.trail_mult,
        "risk_percent": best.risk_percent,
        "pyramid_spacing_n": best.pyramid_spacing,
    }
    return best_params, best, results_df


def _map_metric_field(metric: str) -> str:
    """ParameterResult uses 'risk_percent'/'sharpe_ratio' naming; accept a
    couple of common aliases so config typos fail loudly instead of quietly
    selecting on the wrong column."""
    aliases = {"sharpe": "sharpe_ratio", "calmar": "calmar_ratio", "cagr": "cagr_pct"}
    field_name = aliases.get(metric, metric)
    if not hasattr(ParameterResult, "__dataclass_fields__") or field_name not in ParameterResult.__dataclass_fields__:
        raise ValueError(f"Unknown selection_metric '{metric}' — not a ParameterResult field.")
    return field_name


def _build_params(best_params: dict) -> StrategyParams:
    return StrategyParams(
        entry_len=best_params["entry_len"],
        exit_len=best_params["exit_len"],
        trail_mult=best_params["trail_mult"],
        risk_percent=best_params["risk_percent"],
        pyramid_spacing_n=best_params["pyramid_spacing_n"],
        atr_len=DEFAULT_PARAMS.atr_len,
        size_stop_mult=DEFAULT_PARAMS.size_stop_mult,
        max_units=DEFAULT_PARAMS.max_units,
        long_only=DEFAULT_PARAMS.long_only,
        use_regime_filter=DEFAULT_PARAMS.use_regime_filter,
        lot_step=DEFAULT_PARAMS.lot_step,
        commission_pct=DEFAULT_PARAMS.commission_pct,
        slippage_pct=DEFAULT_PARAMS.slippage_pct,
    )


@dataclass
class FoldResult:
    fold: Fold
    best_params: dict
    in_sample_metric_name: str
    in_sample_metric_value: float
    in_sample_combos_tested: int
    oos_return_pct: float
    oos_trades: int
    oos_win_rate: float
    oos_max_drawdown_pct: float
    oos_equity: pd.Series = field(repr=False)
    oos_trade_pnls: List[float] = field(default_factory=list, repr=False)


def _oos_equity_stats_local(equity_slice: pd.Series) -> Tuple[float, float]:
    """Return-pct and max-drawdown-pct computed strictly within one fold's
    own OOS slice, rebased to that slice's own starting value — a fold's
    drawdown shouldn't be inherited from whatever happened during its
    training window."""
    if len(equity_slice) < 2:
        return 0.0, 0.0
    start = equity_slice.iloc[0]
    ret_pct = (equity_slice.iloc[-1] / start - 1) * 100
    running_max = equity_slice.expanding().max()
    dd = (equity_slice - running_max) / running_max
    return ret_pct, abs(dd.min()) * 100


def run_fold(
    df: pd.DataFrame,
    fold: Fold,
    ranges: RobustnessRanges,
    config: WalkForwardConfig,
    initial_capital: float,
    n_jobs: int,
    show_progress: bool,
) -> FoldResult:
    train_df = df[(df.index >= fold.train_start) & (df.index < fold.test_start)]
    combined_df = df[(df.index >= fold.train_start) & (df.index < fold.test_end)]

    if len(train_df) < 50:
        raise ValueError(f"Fold {fold.index}: training window has only {len(train_df)} bars — too short to optimize.")

    best_params, best_result, _combo_df = optimize_fold(
        train_df, ranges, config, initial_capital, n_jobs=n_jobs, show_progress=show_progress
    )

    strategy = TurtleDonchianStrategy(_build_params(best_params))
    results = strategy.run_backtest(combined_df, initial_capital=initial_capital, verbose=False)

    oos_equity = results.loc[(results.index >= fold.test_start) & (results.index < fold.test_end), "equity"]
    oos_ret_pct, oos_dd_pct = _oos_equity_stats_local(oos_equity)

    oos_trades = [t for t in strategy.trades if fold.test_start <= t.entry_time < fold.test_end]
    oos_pnls = [t.pnl for t in oos_trades if t.pnl is not None]
    win_rate = (sum(1 for p in oos_pnls if p > 0) / len(oos_pnls) * 100) if oos_pnls else 0.0

    metric_field = _map_metric_field(config.selection_metric)

    return FoldResult(
        fold=fold,
        best_params=best_params,
        in_sample_metric_name=metric_field,
        in_sample_metric_value=getattr(best_result, metric_field),
        in_sample_combos_tested=len(_combo_df),
        oos_return_pct=oos_ret_pct,
        oos_trades=len(oos_trades),
        oos_win_rate=win_rate,
        oos_max_drawdown_pct=oos_dd_pct,
        oos_equity=oos_equity,
        oos_trade_pnls=oos_pnls,
    )


# ---------------------------------------------------------------------------
# Stitching + overall report
# ---------------------------------------------------------------------------

def stitch_oos_equity(fold_results: List[FoldResult], initial_capital: float) -> pd.Series:
    """Concatenate each fold's OOS-only equity slice into one continuous,
    compounding curve. Each fold is rebased so it picks up exactly where the
    previous fold's OOS curve left off — the training-window portion of any
    fold's backtest never contributes a single value to this curve."""
    pieces = []
    running_equity = initial_capital
    for fr in fold_results:
        slice_ = fr.oos_equity
        if len(slice_) == 0:
            continue
        baseline = slice_.iloc[0]
        scale = running_equity / baseline if baseline else 1.0
        rebased = slice_ * scale
        pieces.append(rebased)
        running_equity = rebased.iloc[-1]
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Median bar spacing, not the (often-unset) .freq attribute, since real
    fetched data frequently has freq=None even when regularly spaced."""
    if len(index) < 3:
        return 365.0
    diffs = index.to_series().diff().dropna()
    median_seconds = diffs.dt.total_seconds().median()
    if not median_seconds or median_seconds <= 0:
        return 365.0
    return (365.25 * 86400) / median_seconds


def compute_equity_stats(equity: pd.Series, initial_capital: float) -> dict:
    """Standalone equity-curve stats for a stitched curve that has no single
    owning TurtleDonchianStrategy instance (mirrors strategy.get_equity_stats
    but with robust bars-per-year inference)."""
    if len(equity) < 2:
        return {}

    returns = equity.pct_change().dropna()
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    bars_per_year = _infer_bars_per_year(equity.index)
    total_return = (equity.iloc[-1] / initial_capital) - 1
    years = len(equity) / bars_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = (returns.mean() / returns.std() * np.sqrt(bars_per_year)) if returns.std() > 0 else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": equity.iloc[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "start": equity.index[0],
        "end": equity.index[-1],
    }


def parameter_stability(fold_results: List[FoldResult]) -> Dict[str, dict]:
    """Per-parameter stability across fold winners. A parameter that changes
    on nearly every fold is being re-fit to noise, not settling on a value —
    that instability is itself a finding, per Phase 2 of the plan."""
    param_names = ["entry_len", "exit_len", "trail_mult", "risk_percent", "pyramid_spacing_n"]
    stability = {}
    for name in param_names:
        sequence = [fr.best_params[name] for fr in fold_results]
        changes = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
        n_transitions = max(1, len(sequence) - 1)
        stability[name] = {
            "sequence": sequence,
            "distinct_values": len(set(sequence)),
            "changes": changes,
            "stability_score": 1 - (changes / n_transitions),
            "is_stable": (1 - (changes / n_transitions)) >= 0.5,
        }
    return stability


@dataclass
class WalkForwardReport:
    config: WalkForwardConfig
    fold_results: List[FoldResult]
    stitched_equity: pd.Series
    overall_stats: dict
    stability: Dict[str, dict]
    total_oos_trades: int
    oos_win_rate: float

    def fold_results_df(self) -> pd.DataFrame:
        rows = []
        for fr in self.fold_results:
            row = {
                "fold": fr.fold.index,
                "train_start": fr.fold.train_start,
                "test_start": fr.fold.test_start,
                "test_end": fr.fold.test_end,
                **fr.best_params,
                "in_sample_" + fr.in_sample_metric_name: fr.in_sample_metric_value,
                "in_sample_combos_tested": fr.in_sample_combos_tested,
                "oos_return_pct": fr.oos_return_pct,
                "oos_trades": fr.oos_trades,
                "oos_win_rate": fr.oos_win_rate,
                "oos_max_drawdown_pct": fr.oos_max_drawdown_pct,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("WALK-FORWARD VALIDATION SUMMARY (Phase 2, out-of-sample only)")
        lines.append("=" * 80)

        n = len(self.fold_results)
        lines.append(f"\nFolds: {n}  |  train={self.config.train_years}y  "
                      f"test={self.config.test_months}mo  step={self.config.step_months}mo  "
                      f"({'expanding' if self.config.expanding else 'rolling'})")

        s = self.overall_stats
        if s:
            lines.append("\n" + "-" * 80)
            lines.append("STITCHED OUT-OF-SAMPLE EQUITY CURVE")
            lines.append("-" * 80)
            lines.append(f"Coverage:        {s['start']} -> {s['end']}")
            lines.append(f"Total return:    {s['total_return_pct']:+.1f}%   "
                         f"(compare against V1's in-sample-only 855% headline)")
            lines.append(f"CAGR:            {s['cagr_pct']:+.1f}%")
            lines.append(f"Max drawdown:    {s['max_drawdown_pct']:.1f}%")
            lines.append(f"Sharpe:          {s['sharpe_ratio']:.2f}")
            lines.append(f"Calmar:          {s['calmar_ratio']:.2f}")
            lines.append(f"OOS trades:      {self.total_oos_trades}   win rate: {self.oos_win_rate:.1f}%")

        lines.append("\n" + "-" * 80)
        lines.append("PARAMETER STABILITY ACROSS FOLDS")
        lines.append("-" * 80)
        for name, info in self.stability.items():
            status = "stable" if info["is_stable"] else "UNSTABLE"
            seq = " -> ".join(str(v) for v in info["sequence"])
            lines.append(f"\n{name}: {status}  ({info['distinct_values']} distinct value(s) "
                          f"across {n} folds, {info['changes']} change(s))")
            lines.append(f"  {seq}")

        n_unstable = sum(1 for v in self.stability.values() if not v["is_stable"])
        lines.append("\n" + "-" * 80)
        if n_unstable == 0:
            lines.append("VERDICT: parameters settle across folds — a plateau, not a fluke.")
        else:
            lines.append(f"VERDICT: {n_unstable}/{len(self.stability)} parameter(s) never settle — "
                          f"treat the in-sample \"optimum\" as noise for those, not a real edge.")
        lines.append("-" * 80)

        return "\n".join(lines)


def run_walk_forward(
    df: pd.DataFrame,
    config: WalkForwardConfig = DEFAULT_WALK_FORWARD,
    ranges: RobustnessRanges = DEFAULT_ROBUSTNESS,
    initial_capital: float = 100_000.0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> WalkForwardReport:
    """Run the full rolling walk-forward pipeline and return a report whose
    stitched_equity is, by construction, built only from bars that fell
    inside some fold's out-of-sample window."""
    dev_df = get_development(df)
    folds = generate_folds(dev_df, config)

    if not folds:
        raise ValueError(
            "No folds fit inside the development window with this config "
            f"(train_years={config.train_years}, test_months={config.test_months}). "
            "Shrink train_years/test_months or check DEFAULT_SPLIT."
        )

    fold_results: List[FoldResult] = []
    for fold in folds:
        if verbose:
            print(f"\nFold {fold.index + 1}/{len(folds)}: "
                  f"train {fold.train_start.date()} -> {fold.test_start.date()}, "
                  f"test {fold.test_start.date()} -> {fold.test_end.date()}")
        fr = run_fold(dev_df, fold, ranges, config, initial_capital, n_jobs, show_progress=verbose)
        fold_results.append(fr)
        if verbose:
            print(f"  best params: {fr.best_params}")
            print(f"  OOS: {fr.oos_return_pct:+.1f}% over {fr.oos_trades} trades, "
                  f"max DD {fr.oos_max_drawdown_pct:.1f}%")

    stitched = stitch_oos_equity(fold_results, initial_capital)
    overall_stats = compute_equity_stats(stitched, initial_capital)
    stability = parameter_stability(fold_results)

    all_pnls = [p for fr in fold_results for p in fr.oos_trade_pnls]
    total_trades = len(all_pnls)
    win_rate = (sum(1 for p in all_pnls if p > 0) / total_trades * 100) if total_trades else 0.0

    return WalkForwardReport(
        config=config,
        fold_results=fold_results,
        stitched_equity=stitched,
        overall_stats=overall_stats,
        stability=stability,
        total_oos_trades=total_trades,
        oos_win_rate=win_rate,
    )

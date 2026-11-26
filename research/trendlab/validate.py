"""Deep validation for a candidate strategy: bootstrap, DSR, costs, placebo, plots."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_backtest
from .stats import (
    block_bootstrap_sharpe,
    compute_metrics,
    deflated_sharpe_ratio,
    drawdown,
    max_drawdown,
    monte_carlo_drawdown,
    monthly_table,
    yearly_table,
)

SPLITS = {
    "train": ("2020-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "holdout": ("2025-07-01", "2026-10-31"),
}


def period_metrics(res: BacktestResult) -> dict[str, dict]:
    out = {}
    for period, (a, b) in {**SPLITS, "full": ("2000-01-01", "2100-01-01")}.items():
        r = res.returns.loc[a:b]
        if r.empty:
            continue
        m = compute_metrics(
            r,
            res.gross.loc[a:b],
            res.turnover.loc[a:b],
            res.costs.loc[a:b],
            res.funding.loc[a:b],
        )
        out[period] = m.as_dict()
    return out


def placebo_sharpe(
    panel: dict[str, pd.DataFrame],
    signal_fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame],
    funding: dict[str, pd.Series] | None,
    cfg: BacktestConfig,
    n_iter: int = 20,
    seed: int = 123,
) -> dict:
    """Shuffle calendar rows, recompute the same rule, and measure Sharpe.

    A genuine rule cannot predict a time-shuffled future.  A materially positive
    placebo Sharpe would indicate look-ahead or an accounting bug.
    """
    rng = np.random.default_rng(seed)
    idx = panel["close"].index
    sharpes = []
    for _ in range(n_iter):
        perm = rng.permutation(len(idx))
        shuffled = {
            field: frame.iloc[perm].set_axis(idx)
            for field, frame in panel.items()
        }
        sig = signal_fn(shuffled)
        res = run_backtest(shuffled, sig, funding, cfg=cfg)
        sharpes.append(float(compute_metrics(res.returns).sharpe))
    arr = np.array(sharpes)
    return {
        "n": n_iter,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def shift_placebo(
    panel: dict[str, pd.DataFrame],
    signal: pd.DataFrame,
    funding: dict[str, pd.Series] | None,
    cfg: BacktestConfig,
    n_iter: int = 30,
    min_shift: int = 30,
    max_shift: int = 500,
    seed: int = 99,
) -> dict:
    """Circularly shift each symbol's signal in time and re-run.

    This keeps each signal's autocorrelation, turnover and cross-sectional
    character intact while destroying its alignment with future returns, so a
    genuine edge should collapse to roughly zero.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_iter):
        shifted = {
            sym: np.roll(signal[sym].to_numpy(), int(rng.integers(min_shift, max_shift)))
            for sym in signal.columns
        }
        sh = pd.DataFrame(shifted, index=signal.index)
        res = run_backtest(panel, sh, funding, cfg=cfg)
        vals.append(float(compute_metrics(res.returns).sharpe))
    arr = np.array(vals)
    return {
        "n": n_iter,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def cost_sensitivity(
    panel: dict[str, pd.DataFrame],
    sig: pd.DataFrame,
    funding: dict[str, pd.Series],
    base_cfg: BacktestConfig,
    costs: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0, 20.0),
) -> pd.DataFrame:
    rows = []
    for c in costs:
        res = run_backtest(panel, sig, funding, cfg=replace(base_cfg, cost_bps=c))
        m = compute_metrics(res.returns, res.gross, res.turnover, res.costs, res.funding)
        rows.append({"cost_bps_one_way": c, "sharpe": m.sharpe, "cagr": m.cagr,
                     "ann_vol": m.ann_vol, "max_dd": m.max_dd, "cost_drag": m.cost_drag_annual})
    return pd.DataFrame(rows)


def parameter_sensitivity(
    panel: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
    cfg: BacktestConfig,
    variants: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for label, sig in variants.items():
        res = run_backtest(panel, sig, funding, cfg=cfg)
        m = compute_metrics(res.returns, res.gross, res.turnover, res.costs, res.funding)
        tr = compute_metrics(res.returns.loc[SPLITS["train"][0]:SPLITS["train"][1]])
        va = compute_metrics(res.returns.loc[SPLITS["validation"][0]:SPLITS["validation"][1]])
        ho = compute_metrics(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
        rows.append({"variant": label, "sharpe_full": m.sharpe, "cagr_full": m.cagr,
                     "max_dd_full": m.max_dd, "sharpe_train": tr.sharpe,
                     "sharpe_validation": va.sharpe, "sharpe_holdout": ho.sharpe})
    return pd.DataFrame(rows).sort_values("sharpe_full", ascending=False)


def deep_dive(
    name: str,
    sig: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
    cfg: BacktestConfig,
    n_trials: int,
    signal_fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame] | None = None,
    placebo_iters: int = 10,
) -> dict:
    res = run_backtest(panel, sig, funding, cfg=cfg)
    per = period_metrics(res)
    full = per["full"]
    hold = per["holdout"]
    n_obs = len(res.returns)
    boot_full = block_bootstrap_sharpe(res.returns)
    boot_hold = block_bootstrap_sharpe(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
    skew = float(res.returns.skew())
    kurt = float(res.returns.kurtosis() + 3.0)
    dsr = deflated_sharpe_ratio(full["sharpe"], n_trials, n_obs, skew, kurt)
    dsr_hold = deflated_sharpe_ratio(
        hold["sharpe"], n_trials, int(hold["days"]), skew, kurt
    )
    mc = monte_carlo_drawdown(res.returns)
    placebo = None
    if signal_fn is not None:
        placebo = placebo_sharpe(panel, signal_fn, funding, cfg, n_iter=placebo_iters)
    costs = cost_sensitivity(panel, sig, funding, cfg)
    return {
        "name": name,
        "periods": per,
        "bootstrap_full": boot_full,
        "bootstrap_holdout": boot_hold,
        "deflated_sharpe": dsr,
        "deflated_sharpe_holdout": dsr_hold,
        "n_trials": n_trials,
        "monte_carlo": mc,
        "placebo": placebo,
        "cost_sensitivity": costs.to_dict("records"),
        "yearly": yearly_table(res.returns).to_dict("records"),
        "monthly": monthly_table(res.returns).reset_index().to_dict("records"),
        "weights": res.weights,
        "equity": res.equity,
        "returns": res.returns,
        "result": res,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_equity(report: dict, path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = report["equity"]
    dd = drawdown(eq)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(eq.index, eq.values, color="#0b6", lw=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Equity (log, start=1)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    for label, (a, b) in SPLITS.items():
        axes[0].axvspan(pd.Timestamp(a, tz=eq.index.tz), pd.Timestamp(b, tz=eq.index.tz),
                        alpha=0.07, label=label)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(dd.index, dd.values * 100, 0, color="#c33", alpha=0.45)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_yearly(report: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yt = pd.DataFrame(report["yearly"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#0b6" if v >= 0 else "#c33" for v in yt["return"]]
    ax.bar(yt["year"].astype(str), yt["return"] * 100, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Annual return %")
    ax.set_title("Calendar-year net returns")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

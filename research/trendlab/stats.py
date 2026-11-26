"""Performance and statistical validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS = 365


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


@dataclass
class Metrics:
    start: str
    end: str
    days: int
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    hit_rate: float
    monthly_hit_rate: float
    exposure: float
    avg_gross: float
    turnover_annual: float
    skew: float
    kurtosis: float
    best_day: float
    worst_day: float
    funding_drag_annual: float
    cost_drag_annual: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compute_metrics(
    returns: pd.Series,
    gross: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    funding: pd.Series | None = None,
    rf: float = 0.0,
) -> Metrics:
    r = returns.dropna()
    equity = (1.0 + r).cumprod()
    n = len(r)
    if n == 0:
        return Metrics("", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    span_days = max((r.index[-1] - r.index[0]).days, 1)
    years = span_days / 365.25
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(PERIODS))
    mean_ann = float(r.mean() * PERIODS)
    sharpe = float((mean_ann - rf) / vol) if vol > 0 else 0.0
    downside = r[r < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(PERIODS)) if len(downside) > 1 else 0.0
    sortino = float((mean_ann - rf) / dvol) if dvol > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    return Metrics(
        start=str(r.index[0].date()),
        end=str(r.index[-1].date()),
        days=int(n),
        total_return=total,
        cagr=cagr,
        ann_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_dd=mdd,
        calmar=calmar,
        hit_rate=float((r > 0).mean()),
        monthly_hit_rate=float((monthly > 0).mean()) if len(monthly) else 0.0,
        exposure=float((gross.fillna(0) > 0).mean()) if gross is not None else 0.0,
        avg_gross=float(gross.mean()) if gross is not None else 0.0,
        turnover_annual=float(turnover.mean() * PERIODS) if turnover is not None else 0.0,
        skew=float(r.skew()),
        kurtosis=float(r.kurtosis()),
        best_day=float(r.max()),
        worst_day=float(r.min()),
        funding_drag_annual=float(funding.mean() * PERIODS) if funding is not None else 0.0,
        cost_drag_annual=float(costs.mean() * PERIODS) if costs is not None else 0.0,
    )


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_iter: int = 5000,
    block: int = 20,
    seed: int = 7,
    periods: int = PERIODS,
) -> dict:
    """Stationary block bootstrap CI for the annualised Sharpe ratio."""
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block * 2:
        return {"sharpe": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_positive": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    # Draw block starts, then contiguous indices (wrap-around).
    starts = rng.integers(0, n, size=(n_iter, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_iter, -1)[:, :n]
    samples = r[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    sharpes = np.where(stds > 0, means / stds * np.sqrt(periods), 0.0)
    base = float(r.mean() / r.std(ddof=1) * np.sqrt(periods)) if r.std(ddof=1) > 0 else 0.0
    return {
        "sharpe": base,
        "ci_lo": float(np.percentile(sharpes, 2.5)),
        "ci_hi": float(np.percentile(sharpes, 97.5)),
        "p_positive": float((sharpes > 0).mean()),
        "boot_mean": float(sharpes.mean()),
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: float | None = None,
) -> float:
    """Bailey & Lopez de Prado deflated Sharpe ratio (probability SR > 0).

    ``n_trials`` is the number of independent-ish configurations tried, which
    penalises the selection of a lucky backtest.
    """
    from scipy.stats import norm

    if n_obs < 3 or n_trials < 1:
        return float("nan")
    emc = 0.5772156649
    if sr_variance is None or sr_variance <= 0:
        # Expected variance of the SR estimator under the null.
        sr_variance = 1.0 / max(n_obs - 1, 1)
    e_max = np.sqrt(sr_variance) * (
        (1 - emc) * norm.ppf(1 - 1.0 / n_trials)
        + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    sr0 = float(e_max) / np.sqrt(PERIODS)
    sr_hat = observed_sharpe / np.sqrt(PERIODS)
    denom = np.sqrt(
        (1 - skew * sr_hat + (kurtosis - 1) / 4.0 * sr_hat**2) / max(n_obs - 1, 1)
    )
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr_hat - sr0) / denom))


def yearly_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    rows = []
    for year, grp in r.groupby(r.index.year):
        eq = (1 + grp).cumprod()
        vol = grp.std(ddof=1) * np.sqrt(PERIODS)
        rows.append(
            {
                "year": int(year),
                "return": float(eq.iloc[-1] - 1),
                "vol": float(vol),
                "sharpe": float(grp.mean() / grp.std(ddof=1) * np.sqrt(PERIODS))
                if grp.std(ddof=1) > 0 else 0.0,
                "max_dd": max_drawdown(eq),
                "days": len(grp),
            }
        )
    return pd.DataFrame(rows)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    m = (1 + r).resample("ME").prod() - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    return tbl.pivot(index="year", columns="month", values="ret")


def monte_carlo_drawdown(
    returns: pd.Series, n_iter: int = 2000, horizon_days: int = 365, seed: int = 11
) -> dict:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 30:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_iter, horizon_days), replace=True)
    eq = np.cumprod(1 + draws, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak - 1).min(axis=1)
    finals = eq[:, -1] - 1
    return {
        "horizon_days": horizon_days,
        "final_return_p05": float(np.percentile(finals, 5)),
        "final_return_p50": float(np.percentile(finals, 50)),
        "final_return_p95": float(np.percentile(finals, 95)),
        "max_dd_p05": float(np.percentile(dd, 5)),
        "max_dd_p50": float(np.percentile(dd, 50)),
        "max_dd_p95": float(np.percentile(dd, 95)),
        "p_negative": float((finals < 0).mean()),
        "p_ruin_50": float((dd < -0.5).mean()),
    }

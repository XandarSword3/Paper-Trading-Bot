"""
Institutional Performance and Risk Metrics Module.
Implements:
1. Daily Equity Sharpe Ratio (sqrt(365)).
2. Bailey & Lopez de Prado Deflated Sharpe Ratio (DSR).
3. Moving-Block Bootstrap for 95% Confidence Intervals.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple, List, Dict, Any, Optional


def calculate_daily_sharpe(equity_series: pd.Series) -> float:
    """
    Computes annualized Sharpe ratio strictly from daily equity returns:
    SR_annual = (mean_daily_return / std_daily_return) * sqrt(365).
    """
    if len(equity_series) < 2:
        return 0.0

    # Resample to daily frequency (taking end of day equity)
    if isinstance(equity_series.index, pd.DatetimeIndex):
        daily_eq = equity_series.resample("1D").last().ffill().dropna()
    else:
        daily_eq = equity_series

    if len(daily_eq) < 3:
        return 0.0

    daily_returns = daily_eq.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0

    mean_d = float(daily_returns.mean())
    std_d = float(daily_returns.std(ddof=1))

    if std_d < 1e-12:
        return 0.0

    return float((mean_d / std_d) * np.sqrt(365.0))


def compute_dsr(
    per_trade_sharpe: float,
    n_trades: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Computes Deflated Sharpe Ratio (DSR) p-value using per-trade Sharpe ratio
    and sample size T = n_trades against n_trials competing hypotheses.
    """
    if n_trades < 5 or per_trade_sharpe <= 0 or n_trials < 1:
        return 1.0

    gamma = 0.57721566490153286  # Euler-Mascheroni constant
    # Standard error of Sharpe ratio across N null trials under H0
    sr_null_std = 1.0 / np.sqrt(float(n_trades))
    # Expected maximum Sharpe ratio under the null hypothesis of zero true edge
    if n_trials > 1:
        e_max_z = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        e_max_sr = sr_null_std * e_max_z
    else:
        e_max_sr = 0.0

    # Variance of the Sharpe ratio estimator
    var_sr = (1.0 + 0.5 * (per_trade_sharpe ** 2) - skew * per_trade_sharpe + ((kurt - 3.0) / 4.0) * (per_trade_sharpe ** 2)) / float(n_trades)
    std_sr = np.sqrt(max(1e-12, var_sr))

    z = (per_trade_sharpe - e_max_sr) / std_sr
    p_val = float(1.0 - stats.norm.cdf(z))
    return float(np.clip(p_val, 0.0, 1.0))


def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = 10,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Moving-block bootstrap for time-series dependent trades.
    Returns: (mean, lower_ci, upper_ci)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n < block_size:
        block_size = max(1, n // 2)

    rng = np.random.default_rng(random_seed)
    n_blocks = int(np.ceil(n / block_size))

    # All possible contiguous blocks of length block_size
    possible_starts = n - block_size + 1
    if possible_starts <= 0:
        block_starts = np.zeros((n_bootstraps, 1), dtype=int)
        boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstraps)])
    else:
        boot_means = np.empty(n_bootstraps)
        for b in range(n_bootstraps):
            starts = rng.integers(0, possible_starts, size=n_blocks)
            sampled_idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
            boot_means[b] = np.mean(values[sampled_idx])

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))
    sample_mean = float(np.mean(values))

    return sample_mean, lower, upper

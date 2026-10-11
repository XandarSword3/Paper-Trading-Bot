import pytest
import numpy as np
import pandas as pd
from quant_research.core.metrics import calculate_daily_sharpe, compute_dsr, block_bootstrap_ci


def test_known_series_daily_sharpe():
    """
    Unit test for daily Sharpe ratio:
    Daily mean return = 0.001, Daily std = 0.01.
    Daily SR = 0.001 / 0.01 = 0.1.
    Annualized SR (sqrt(365)) = 0.1 * sqrt(365) = 1.910497... ~ 1.91.
    """
    n_days = 1000
    dates = pd.date_range("2020-01-01", periods=n_days, freq="1D", tz="UTC")

    # Construct synthetic daily equity series with known mean and std
    rng = np.random.default_rng(42)
    # Generate daily returns with mean exactly 0.001 and std exactly 0.01
    z = rng.normal(0, 1, n_days)
    daily_returns = 0.001 + 0.01 * (z - np.mean(z)) / np.std(z, ddof=1)

    equity_vals = 1000.0 * np.cumprod(1.0 + daily_returns)
    eq_series = pd.Series(equity_vals, index=dates)

    annual_sr = calculate_daily_sharpe(eq_series)
    expected_sr = 0.1 * np.sqrt(365.0)

    # Annualized Sharpe ratio should match ~1.91 (within 0.01)
    assert abs(annual_sr - expected_sr) < 0.01, f"Expected {expected_sr:.4f}, got {annual_sr:.4f}"
    assert abs(annual_sr - 1.9105) < 0.01, f"Expected ~1.91, got {annual_sr:.4f}"


def test_dsr_internal_consistency_and_scipy_independent_parity():
    """
    Internal-consistency and independent parity unit test for Deflated Sharpe Ratio.
    Computes DSR using an independent from-scratch SciPy implementation and compares
    it directly against compute_dsr across multiple parameter sets.
    """
    import scipy.stats as sp_stats

    def independent_scipy_dsr(sr: float, t: int, n: int, skewness: float = 0.0, kurtosis: float = 3.0) -> float:
        """Independent reference implementation using raw scipy.stats."""
        if t < 5 or sr <= 0 or n < 1:
            return 1.0
        emc = 0.57721566490153286
        if n > 1:
            z_star = (1.0 - emc) * sp_stats.norm.ppf(1.0 - 1.0 / n) + emc * sp_stats.norm.ppf(1.0 - 1.0 / (n * np.e))
            e_max = z_star / np.sqrt(float(t))
        else:
            e_max = 0.0
        var_est = (1.0 + 0.5 * (sr ** 2) - skewness * sr + ((kurtosis - 3.0) / 4.0) * (sr ** 2)) / float(t)
        z_stat = (sr - e_max) / np.sqrt(max(1e-12, var_est))
        return float(1.0 - sp_stats.norm.cdf(z_stat))

    # Test Case 1: T=1250, N=100, SR=0.1575
    daily_sr = 2.5 / np.sqrt(252.0)
    ref_1 = independent_scipy_dsr(daily_sr, 1250, 100, -0.72, 5.78)
    cand_1 = compute_dsr(daily_sr, 1250, 100, -0.72, 5.78)
    assert abs(cand_1 - ref_1) < 1e-9, f"Mismatch: cand={cand_1}, ref={ref_1}"

    # Test Case 2: T=500, N=50, SR=0.08, normal skew/kurt
    ref_2 = independent_scipy_dsr(0.08, 500, 50, 0.0, 3.0)
    cand_2 = compute_dsr(0.08, 500, 50, 0.0, 3.0)
    assert abs(cand_2 - ref_2) < 1e-9, f"Mismatch: cand={cand_2}, ref={ref_2}"

    # Test Case 3: Edge cases (negative SR, n_trials=1, n_trades < 5)
    assert compute_dsr(per_trade_sharpe=-0.1, n_trades=100, n_trials=100) == 1.0
    assert compute_dsr(per_trade_sharpe=0.5, n_trades=3, n_trials=10) == 1.0
    ref_single = independent_scipy_dsr(0.2, 500, 1)
    cand_single = compute_dsr(0.2, 500, 1)
    assert abs(cand_single - ref_single) < 1e-9


def test_block_bootstrap_ci():
    """
    Unit test for block-bootstrap confidence interval:
    Verifies that for a normal series with mean 10.0, the bootstrap mean matches
    and 95% CI covers the true parameter.
    """
    rng = np.random.default_rng(123)
    vals = rng.normal(loc=10.0, scale=2.0, size=200)

    mean_val, lower_ci, upper_ci = block_bootstrap_ci(vals, block_size=10, n_bootstraps=1000)

    assert abs(mean_val - np.mean(vals)) < 1e-6
    assert lower_ci < mean_val < upper_ci
    assert lower_ci <= 10.0 <= upper_ci

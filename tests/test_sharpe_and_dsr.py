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


def test_known_value_dsr():
    """
    Unit test for Deflated Sharpe Ratio (DSR) against published benchmark
    from Bailey & López de Prado (2014), 'The Deflated Sharpe Ratio',
    Journal of Portfolio Management:

    Parameters:
    - T = 1250 (5 years of daily observations)
    - N = 100 trials
    - Annualized SR = 2.5 -> Daily SR = 2.5 / sqrt(252) ≈ 0.1574917
    - Skewness = -0.72
    - Kurtosis = 5.78

    Published Expected Max Z for N=100 trials:
    E[max Z] ≈ (1 - γ)*Φ^(-1)(1 - 1/N) + γ*Φ^(-1)(1 - 1/(N*e)) ≈ 2.530603
    Expected Max SR = 2.530603 / sqrt(1250) ≈ 0.071576
    Estimated Std(SR) ≈ 0.030239
    Z-score = (0.1574917 - 0.071576) / 0.030239 ≈ 2.84096
    Published DSR (PSR) = Φ(2.84096) ≈ 0.99775
    Corresponding p-value = 1 - DSR ≈ 0.00225 (0.0022489)
    """
    daily_sr = 2.5 / np.sqrt(252.0)
    p_val = compute_dsr(
        per_trade_sharpe=daily_sr,
        n_trades=1250,
        n_trials=100,
        skew=-0.72,
        kurt=5.78,
    )
    # Verify DSR p-value matches published value within 1e-4
    assert abs(p_val - 0.0022489) < 1e-4, f"Expected DSR p-value ~ 0.00225, got {p_val:.6f}"

    # Verify boundary edge cases:
    # 1. Negative SR must yield p-value 1.0 (no significance)
    assert compute_dsr(per_trade_sharpe=-0.1, n_trades=100, n_trials=100) == 1.0
    # 2. Insufficient samples (< 5) must yield p-value 1.0
    assert compute_dsr(per_trade_sharpe=0.5, n_trades=3, n_trials=10) == 1.0


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

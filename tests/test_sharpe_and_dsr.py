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
    Unit test for Deflated Sharpe Ratio (DSR):
    1. Single trial (n_trials=1): Standard Normal CDF of SR / SE.
    2. Zero or negative Sharpe: DSR p-value must equal 1.0 (no significance).
    3. Large number of trials (e.g. n_trials=1000): Expected max SR is ~3.0+,
       so an observed per-trade SR of 0.05 on 100 trades has p-value near 1.0.
    """
    # Negative SR -> p-val 1.0
    assert compute_dsr(per_trade_sharpe=-0.1, n_trades=100, n_trials=100) == 1.0

    # Small SR with huge hypothesis trial count -> p-val 1.0
    p_val_mined = compute_dsr(per_trade_sharpe=0.05, n_trades=100, n_trials=1354)
    assert p_val_mined > 0.99, f"Expected DSR p-value > 0.99 for data-mined SR, got {p_val_mined}"

    # Truly massive SR (e.g. per-trade SR = 0.5 on 500 trades) clears even 1000 trials
    p_val_strong = compute_dsr(per_trade_sharpe=0.5, n_trades=500, n_trials=1000)
    assert p_val_strong < 0.001, f"Expected DSR p-value < 0.001 for massive edge, got {p_val_strong}"


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

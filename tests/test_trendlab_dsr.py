"""Regression test for the trendlab deflated Sharpe ratio benchmark.

Bug fixed on fix-dsr-sr0: ``sr0`` was computed as ``e_max / sqrt(PERIODS)``
even though ``e_max`` is already in per-period Sharpe units. That shrank the
null benchmark ~19x and pushed DSR toward 1.0 for any positive Sharpe.

The defining property: a strategy whose Sharpe equals the *expected maximum*
of ``n_trials`` null trials should score DSR ~= 0.5 (it is exactly at the
selection-bias benchmark, no better, no worse).
"""

import numpy as np
import pytest
from scipy.stats import norm

from research.trendlab.stats import PERIODS, deflated_sharpe_ratio

EMC = 0.5772156649


def _expected_max_null_sharpe_annualized(n_trials: int, n_obs: int) -> float:
    """E[max SR] under the null, expressed as an annualized Sharpe."""
    sr_variance = 1.0 / (n_obs - 1)
    e_max = np.sqrt(sr_variance) * (
        (1 - EMC) * norm.ppf(1 - 1.0 / n_trials)
        + EMC * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return float(e_max * np.sqrt(PERIODS))


@pytest.mark.parametrize("n_trials,n_obs", [(210, 2500), (1354, 2000), (50, 500)])
def test_null_best_sharpe_scores_about_half(n_trials: int, n_obs: int):
    observed = _expected_max_null_sharpe_annualized(n_trials, n_obs)
    dsr = deflated_sharpe_ratio(observed, n_trials, n_obs, skew=0.0, kurtosis=3.0)
    assert dsr == pytest.approx(0.5, abs=0.01)


def test_dsr_increases_with_observed_sharpe():
    n_trials, n_obs = 210, 2500
    null_best = _expected_max_null_sharpe_annualized(n_trials, n_obs)
    d_null = deflated_sharpe_ratio(null_best, n_trials, n_obs)
    d_better = deflated_sharpe_ratio(null_best * 2.0, n_trials, n_obs)
    d_worse = deflated_sharpe_ratio(null_best * 0.5, n_trials, n_obs)
    assert d_better > d_null > d_worse


def test_dsr_not_inflated_for_marginal_sharpe():
    """A Sharpe barely above the null benchmark must not score ~1.0."""
    n_trials, n_obs = 210, 2500
    null_best = _expected_max_null_sharpe_annualized(n_trials, n_obs)
    dsr = deflated_sharpe_ratio(null_best * 1.2, n_trials, n_obs)
    assert dsr < 0.9

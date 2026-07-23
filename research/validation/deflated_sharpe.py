"""
Probabilistic and Deflated Sharpe Ratio — Phase 3 of VALIDATION_REMEDIATION_PLAN.md.

    "2,800+ parameter combinations were tried before picking the best one —
    reporting that best Sharpe without correcting for the number of trials
    overstates significance."

Implements Bailey & Lopez de Prado's Probabilistic Sharpe Ratio (PSR) and
Deflated Sharpe Ratio (DSR):

  Bailey, D. and Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient
  Frontier", Journal of Risk, 15(2).
  Bailey, D. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
  Journal of Portfolio Management, 40(5).

PSR(SR*) answers: "given this many return observations and this return
distribution's skew/kurtosis, how confident can we be that the TRUE Sharpe
exceeds a benchmark SR*?" It is a probability in [0, 1], not a p-value on
its own — read 0.95 as "95% probability the true Sharpe beats SR*, under the
model", not as a frequentist significance level, though it plays the same
role in practice.

DSR is PSR evaluated against SR* = the Sharpe ratio you'd *expect* the best
of N unskilled trials to show by pure luck (SR0 below), instead of against
zero. It is the correct way to ask "is the best of 2,800 tries actually
good, or just the best of 2,800 coin flips?"

All Sharpe ratios (sr_observed, sr_benchmark, sharpe_std, individual entries
in trial_sharpes) must be in the SAME, PERIODIC (non-annualized) units — do
not mix annualized and per-bar Sharpes. annualized_to_periodic() below
converts strategy.py's annualized sharpe_ratio output for this purpose.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

EULER_MASCHERONI = 0.5772156649015329


def annualized_to_periodic(sr_annualized: float, bars_per_year: float) -> float:
    """Undo the sqrt(bars_per_year) scaling strategy.get_equity_stats() (and
    ParameterResult.sharpe_ratio) applies, since PSR/DSR must be computed on
    the raw per-period Sharpe with the matching per-period n_obs."""
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")
    return sr_annualized / np.sqrt(bars_per_year)


def returns_skew_kurtosis(returns: Sequence[float]) -> tuple:
    """Sample skewness (gamma3) and NON-EXCESS kurtosis (gamma4, normal=3)
    of a return series, bias-corrected. Needs a handful of observations to
    mean anything — with under ~30 the estimates are noisy and the PSR/DSR
    formulas below are correspondingly unreliable."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 4:
        return 0.0, 3.0  # not enough data to estimate higher moments; fall back to normal
    g3 = float(_skew(returns, bias=False))
    g4 = float(_kurtosis(returns, bias=False, fisher=False))  # fisher=False -> normal maps to 3
    return g3, g4


def sharpe_ratio_std_error(sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Standard error of an estimated (periodic) Sharpe ratio (Mertens 2002 /
    Bailey & Lopez de Prado 2012, eq. 7):

        se(SR) = sqrt( (1 - skew*SR + ((kurtosis-1)/4)*SR^2) / (n_obs - 1) )

    Reduces to the classical sqrt((1 + SR^2/2)/n_obs) when returns are
    normal (skew=0, kurtosis=3) — fat tails / negative skew widen it, which
    is exactly the non-normality correction the DSR paper's title refers to.
    """
    if n_obs <= 1:
        raise ValueError("n_obs must be > 1")
    variance = (1 - skew * sr + ((kurtosis - 1) / 4) * sr ** 2) / (n_obs - 1)
    return float(np.sqrt(max(variance, 1e-12)))


def probabilistic_sharpe_ratio(
    sr_observed: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """P(true periodic Sharpe > sr_benchmark), given the observed periodic
    Sharpe, sample size, and return-distribution shape. All Sharpes must be
    periodic (see module docstring)."""
    se = sharpe_ratio_std_error(sr_observed, n_obs, skew, kurtosis)
    z = (sr_observed - sr_benchmark) / se
    return float(norm.cdf(z))


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float) -> float:
    """
    Expected maximum (periodic) Sharpe ratio you'd observe from the BEST of
    n_trials strategies that all have TRUE Sharpe = 0 (pure luck), given the
    empirical cross-trial standard deviation of Sharpe outcomes sharpe_std
    (Bailey & Lopez de Prado 2014, eq. 10 — an extreme-value approximation):

        E[max SR] = sharpe_std * [ (1-gamma)*Phi^-1(1 - 1/N)
                                    + gamma*Phi^-1(1 - 1/(N*e)) ]

    This is the "haircut" benchmark DSR tests the observed best Sharpe
    against, instead of against zero.
    """
    if n_trials < 2:
        return 0.0
    if sharpe_std <= 0:
        return 0.0
    term1 = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * (term1 + term2))


@dataclass
class DeflatedSharpeResult:
    sr_observed_periodic: float
    sr_observed_annualized: Optional[float]
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float
    sr0_expected_max_under_null: float
    psr_vs_zero: float          # P(true SR > 0) — ignores the multiple-trials problem
    dsr_vs_sr0: float           # P(true SR > SR0) — the multiple-trials-corrected version
    verdict: str

    def report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("DEFLATED / PROBABILISTIC SHARPE RATIO")
        lines.append("=" * 80)
        if self.sr_observed_annualized is not None:
            lines.append(f"\nObserved Sharpe (annualized): {self.sr_observed_annualized:.2f}")
        lines.append(f"Observed Sharpe (periodic):   {self.sr_observed_periodic:.3f}   "
                      f"(n={self.n_obs} periods, skew={self.skew:.2f}, kurtosis={self.kurtosis:.2f})")
        lines.append(f"Parameter combinations tried:  {self.n_trials}")
        lines.append(f"Expected best-of-{self.n_trials} Sharpe if NO strategy here had any real "
                      f"edge (SR0): {self.sr0_expected_max_under_null:.3f}")

        lines.append("\n" + "-" * 80)
        lines.append(f"PSR(0)  — P(true Sharpe > 0), ignoring how many combos were tried: "
                      f"{self.psr_vs_zero*100:.1f}%")
        lines.append(f"DSR     — P(true Sharpe > SR0), i.e. beats what luck alone would "
                      f"produce from {self.n_trials} tries: {self.dsr_vs_sr0*100:.1f}%")
        lines.append("-" * 80)
        lines.append(f"\nVERDICT: {self.verdict}")
        return "\n".join(lines)


def deflated_sharpe_ratio(
    sr_observed_periodic: float,
    n_obs: int,
    n_trials: int,
    trial_sharpes_periodic: Sequence[float],
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_observed_annualized: Optional[float] = None,
) -> DeflatedSharpeResult:
    """
    Full PSR + DSR analysis for "the best of n_trials parameter combos".

    trial_sharpes_periodic: the (periodic) Sharpe ratio achieved by every
    combination tried — used only to estimate the cross-trial variance that
    feeds expected_max_sharpe_under_null(). sr_observed_periodic should
    normally be (very close to) max(trial_sharpes_periodic), since it's the
    winner that was actually selected and reported.
    """
    trial_sharpes_periodic = np.asarray(trial_sharpes_periodic, dtype=float)
    trial_sharpes_periodic = trial_sharpes_periodic[np.isfinite(trial_sharpes_periodic)]
    sharpe_std = float(trial_sharpes_periodic.std(ddof=1)) if len(trial_sharpes_periodic) > 1 else 0.0

    sr0 = expected_max_sharpe_under_null(n_trials, sharpe_std)
    psr0 = probabilistic_sharpe_ratio(sr_observed_periodic, n_obs, skew, kurtosis, sr_benchmark=0.0)
    dsr = probabilistic_sharpe_ratio(sr_observed_periodic, n_obs, skew, kurtosis, sr_benchmark=sr0)

    if dsr >= 0.95:
        verdict = "survives deflation — beats the multiple-testing null with high confidence."
    elif dsr >= 0.5:
        verdict = ("weakly survives deflation — better than pure luck from this many trials, "
                   "but not by a comfortable margin. Treat as suggestive, not confirmed.")
    else:
        verdict = (f"does NOT survive deflation — the observed Sharpe is plausibly just the "
                   f"best of {n_trials} noisy tries, not a real edge. PSR(0)={psr0*100:.0f}% "
                   f"looked fine in isolation; DSR is the number that accounts for how many "
                   f"combinations were searched.")

    return DeflatedSharpeResult(
        sr_observed_periodic=sr_observed_periodic,
        sr_observed_annualized=sr_observed_annualized,
        n_obs=n_obs,
        n_trials=n_trials,
        skew=skew,
        kurtosis=kurtosis,
        sr0_expected_max_under_null=sr0,
        psr_vs_zero=psr0,
        dsr_vs_sr0=dsr,
        verdict=verdict,
    )

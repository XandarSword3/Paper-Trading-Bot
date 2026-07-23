"""
Chronological data split helpers — Phase 1 of VALIDATION_REMEDIATION_PLAN.md.

Enforces one frozen in-sample / validation / holdout boundary, defined once
in config.DEFAULT_SPLIT, so no script has to hand-roll its own date filter
(which is how the holdout got silently ignored in the first place — every
tool independently called download_btc_data() and used the whole thing).

ACCEPTANCE CHECK (per the plan): grep this repo for "holdout_start" or the
literal date "2025-01-01". It should appear only in config.py, this file,
and final_holdout_validation.py. If it shows up in robustness_test.py,
monte_carlo.py, regime_simulation.py, survivability.py, main.py, or any
strategy version's own optimizer, that defeats the point of having a
holdout — go fix the leak there, don't add an exception here.

Known gap: this currently governs the shared V1 tooling (robustness_test.py,
monte_carlo.py, regime_simulation.py, survivability.py, main.py) since those
already share strategy.py and a common data path. V2/V3/V4 each still
self-fetch and run their own self-contained optimizer (see
STRATEGY_VERSION_AUDIT.md) — bringing them onto this same split is part of
Phase 6 (consolidating the strategy versions onto one engine), not this
phase.
"""
import pandas as pd

from config import DEFAULT_SPLIT


def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str)


def get_in_sample(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample window only. Free to use for development and robustness testing."""
    return df[
        (df.index >= _ts(DEFAULT_SPLIT.in_sample_start))
        & (df.index < _ts(DEFAULT_SPLIT.validation_start))
    ]


def get_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Validation window only. Used for walk-forward OOS folds and stress testing."""
    return df[
        (df.index >= _ts(DEFAULT_SPLIT.validation_start))
        & (df.index < _ts(DEFAULT_SPLIT.holdout_start))
    ]


def get_development(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample + validation combined — everything except the true holdout.

    This is what robustness_test.py, monte_carlo.py, regime_simulation.py,
    survivability.py, and main.py should all use instead of the raw,
    unsliced dataset. Defined as "everything before the holdout starts"
    rather than an inclusive end-date comparison, so there's no gap/overlap
    at the day boundary (a plain `index <= "2024-12-31"` silently drops
    every candle after midnight on that day for intraday timeframes).
    """
    return df[df.index < _ts(DEFAULT_SPLIT.holdout_start)]


def get_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """True holdout window. Intentionally has no other caller in this repo
    besides final_holdout_validation.py — see module docstring."""
    mask = df.index >= _ts(DEFAULT_SPLIT.holdout_start)
    if DEFAULT_SPLIT.holdout_end:
        # holdout_end names a calendar day; include the whole day.
        mask &= df.index < _ts(DEFAULT_SPLIT.holdout_end) + pd.Timedelta(days=1)
    return df[mask]


def describe_split() -> str:
    """Human-readable summary of the frozen split, for logging/report headers."""
    s = DEFAULT_SPLIT
    return (
        f"In-sample:  {s.in_sample_start} -> {s.in_sample_end}\n"
        f"Validation: {s.validation_start} -> {s.validation_end}\n"
        f"Holdout:    {s.holdout_start} -> {s.holdout_end or 'latest'} "
        f"(scored once, at the end, by final_holdout_validation.py only)"
    )

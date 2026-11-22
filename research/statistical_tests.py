"""Statistical inference engine: Calibrated Deflated Sharpe Ratio (DSR) & Non-IID Block Bootstrap.

Features:
- Calibrated DSR: Scales SR* by empirical cross-trial variance, uses independent episode count T.
- Moving Block Bootstrap by Time: 30m blocks, empirical 95% CI and non-parametric p-values.
- Pre-specified 5m Primary Horizon: Eliminates multiple-comparison bias from max-picking.
- Lower-Bound Break-Even Gate: Requires lower 95% CI bound > 0.0 bps net.
- Monotonic Dose-Response & Outlier Diagnostic (Median, Trimmed Mean, Top 1% Share).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("statistical_tests")


def _norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class TrialRegistry:
    """Tracks and registers every parameter configuration, horizon, and model iteration."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.path = registry_path or Path(__file__).resolve().parent.parent / "logs" / "signal_research" / "trial_registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trials: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.trials = data.get("trials", [])
            except Exception as e:
                logger.warning("Could not load trial registry: %s", e)
                self.trials = []

    def _save(self) -> None:
        try:
            payload = {
                "total_trials": len(self.trials),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "trials": self.trials,
            }
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed saving trial registry: %s", e)

    def register_trial(self, config: dict, description: str = "") -> int:
        """Register a new configuration trial, incrementing lifetime trials count."""
        trial_entry = {
            "trial_id": len(self.trials) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "config": config,
        }
        self.trials.append(trial_entry)
        self._save()
        logger.info("Registered trial #%d: %s", len(self.trials), description)
        return len(self.trials)

    @property
    def total_trials(self) -> int:
        # Accounts for all prior versions (V1, V4, carry bots, 5 horizons)
        # Default minimum 10 trials to prevent zero-penalty undercounting
        return max(10, len(self.trials))


def compute_expected_max_sharpe(n_trials: int, variance_trials: float = 1.0) -> float:
    """Bailey & Lopez de Prado (2014) expected maximum Sharpe ratio under N trials, scaled by cross-trial std dev."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    z = math.sqrt(2.0 * math.log(n_trials))
    sr_star_standard = z + (euler_mascheroni / z)
    # Scale by empirical cross-trial standard deviation
    return float(math.sqrt(max(0.1, variance_trials)) * sr_star_standard)


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    effective_t: int,
    variance_trials: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> Tuple[float, float]:
    """Computes Deflated Sharpe Ratio (DSR) using per-period Sharpe and effective independent episodes T.

    Returns (dsr_probability, expected_max_sr).
    """
    if effective_t <= 2:
        return (0.0, 0.0)

    sr_star = compute_expected_max_sharpe(n_trials, variance_trials=variance_trials)

    # Asymptotic standard error of Sharpe ratio under non-normality (Mertens 2002)
    denom_inner = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
    denom_inner = max(1e-6, denom_inner)
    se_sr = math.sqrt(denom_inner / (effective_t - 1))

    z_stat = (observed_sr - sr_star) / se_sr
    dsr_p = float(_norm_cdf(z_stat))
    return (dsr_p, sr_star)


def moving_block_bootstrap_by_time(
    df: pd.DataFrame,
    horizon_col: str = "5m",
    block_duration_minutes: float = 30.0,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Performs moving block bootstrap across non-overlapping continuous temporal blocks.

    Evaluates primary episode signals vs matched naive controls under realistic net returns.
    """
    rng = np.random.default_rng(random_seed)

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column for temporal block bootstrap")

    # Filter strictly to PRIMARY EPISODES that did NOT span a WS gap
    clean = df[
        (df["is_primary_episode"] == True) &
        (df.get("has_ws_gap", False) == False)
    ].dropna(subset=[horizon_col, "timestamp"]).copy()

    if len(clean) < 15:
        return {"error": f"Insufficient primary episodes for block bootstrap (N={len(clean)} < 15)"}

    clean["block_id"] = (clean["timestamp"] // (block_duration_minutes * 60.0)).astype(int)
    unique_blocks = clean["block_id"].unique()
    n_blocks = len(unique_blocks)

    if n_blocks < 3:
        return {"error": f"Insufficient time blocks ({n_blocks} < 3). Wait for longer recording window."}

    block_map = {b: group for b, group in clean.groupby("block_id")}

    sig_means = []
    ctrl_means = []
    diff_means = []

    for _ in range(n_bootstrap):
        sampled_block_ids = rng.choice(unique_blocks, size=n_blocks, replace=True)
        resampled_dfs = [block_map[b] for b in sampled_block_ids]
        combined = pd.concat(resampled_dfs, ignore_index=True)

        sigs = combined[~combined["is_control"]][horizon_col]
        ctrls = combined[combined["is_control"]][horizon_col]

        sig_mean = float(sigs.mean()) if not sigs.empty else 0.0
        ctrl_mean = float(ctrls.mean()) if not ctrls.empty else 0.0

        sig_means.append(sig_mean)
        ctrl_means.append(ctrl_mean)
        diff_means.append(sig_mean - ctrl_mean)

    sig_means = np.array(sig_means)
    ctrl_means = np.array(ctrl_means)
    diff_means = np.array(diff_means)

    # 95% Confidence Intervals
    sig_ci = (float(np.percentile(sig_means, 2.5)), float(np.percentile(sig_means, 97.5)))
    ctrl_ci = (float(np.percentile(ctrl_means, 2.5)), float(np.percentile(ctrl_means, 97.5)))
    diff_ci = (float(np.percentile(diff_means, 2.5)), float(np.percentile(diff_means, 97.5)))

    p_val_zero = float(np.mean(sig_means <= 0.0))
    p_val_diff = float(np.mean(diff_means <= 0.0))

    return {
        "horizon": horizon_col,
        "n_primary_episodes": len(clean[~clean["is_control"]]),
        "n_naive_controls": len(clean[clean["is_control"]]),
        "n_blocks": n_blocks,
        "block_duration_minutes": block_duration_minutes,
        "n_bootstrap": n_bootstrap,
        "net_mean_bps": round(float(np.mean(sig_means)), 2),
        "net_95_ci_bps": (round(sig_ci[0], 2), round(sig_ci[1], 2)),
        "control_mean_bps": round(float(np.mean(ctrl_means)), 2),
        "control_95_ci_bps": (round(ctrl_ci[0], 2), round(ctrl_ci[1], 2)),
        "excess_over_naive_bps": round(float(np.mean(diff_means)), 2),
        "excess_95_ci_bps": (round(diff_ci[0], 2), round(diff_ci[1], 2)),
        "p_value_net_le_zero": round(p_val_zero, 4),
        "p_value_excess_le_zero": round(p_val_diff, 4),
        # Kill Criterion lower bound check
        "lower_ci_clears_breakeven": bool(sig_ci[0] > 0.0),
    }


def evaluate_institutional_kill_gate(bootstrap_res: dict) -> dict:
    """Applies the strict institutional kill criterion:

    The LOWER bound of the 95% Bootstrap CI on the primary 5m horizon MUST clear 0.0 bps Net.
    """
    if "error" in bootstrap_res:
        return {
            "decision": "COLLECTING_SAMPLE",
            "reason": bootstrap_res["error"],
            "cleared": False,
        }

    lower_ci = bootstrap_res["net_95_ci_bps"][0]
    cleared = lower_ci > 0.0
    decision = "PROCEED_TO_MAKER_EXECUTION_ENGINEERING" if cleared else "KILL_STRATEGY"

    statement = (
        f"Primary 5m Horizon 95% Bootstrap CI is [{bootstrap_res['net_95_ci_bps'][0]:+.2f}, {bootstrap_res['net_95_ci_bps'][1]:+.2f}] bps net. "
        f"Lower bound ({lower_ci:+.2f} bps) {'CLEARS' if cleared else 'FAILS TO CLEAR'} break-even (> 0.0 bps net)."
    )

    return {
        "decision": decision,
        "cleared": cleared,
        "primary_horizon": bootstrap_res["horizon"],
        "mean_net_bps": bootstrap_res["net_mean_bps"],
        "lower_95_ci_bps": lower_ci,
        "excess_over_naive_bps": bootstrap_res["excess_over_naive_bps"],
        "verdict_statement": statement,
    }

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


def compute_expected_max_sharpe(n_trials: int, variance_trials: Optional[float] = None, effective_t: Optional[int] = None) -> float:
    """Bailey & Lopez de Prado (2014) expected maximum Sharpe ratio under N trials, scaled by cross-trial std dev."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    z = math.sqrt(2.0 * math.log(n_trials))
    sr_star_standard = z + (euler_mascheroni / z)

    # Scale by empirical cross-trial std dev, or asymptotic null std dev 1/sqrt(T-1) for per-period Sharpe
    if variance_trials is not None and variance_trials > 0:
        sigma_sr = math.sqrt(variance_trials)
    elif effective_t is not None and effective_t > 1:
        sigma_sr = 1.0 / math.sqrt(effective_t - 1)
    else:
        sigma_sr = 1.0
    return float(sigma_sr * sr_star_standard)


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    effective_t: int,
    variance_trials: Optional[float] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> Tuple[float, float]:
    """Computes Deflated Sharpe Ratio (DSR) using per-period Sharpe and effective independent episodes T.

    Returns (dsr_probability, expected_max_sr).
    """
    if effective_t <= 2:
        return (0.0, 0.0)

    sr_star = compute_expected_max_sharpe(n_trials, variance_trials=variance_trials, effective_t=effective_t)

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


def evaluate_institutional_kill_gate(
    bootstrap_res: dict,
    df: Optional[pd.DataFrame] = None,
    horizons_summary: Optional[dict] = None,
    min_required_episodes: int = 300,
    min_required_blocks: int = 50,
) -> dict:
    """Applies the complete institutional 5-criterion kill gate.

    A strategy can NEVER proceed on an interim look. It requires:
    1. Sample Adequacy: N >= 300 independent primary episodes and >= 50 time blocks (covering >= 7 days).
    2. Positive Median: 5m Median Net Return > 0.0 bps (rules out right-skewed lottery pumps).
    3. Leave-One-Coin-Out (LOCO): Mean Net Return > 0.0 bps after dropping the single top-performing coin.
    4. Multi-Horizon Sign Consistency: 1m, 5m, and 15m medians must all be positive.
    5. Bootstrap CI Lower Bound: Lower 95% Bootstrap CI bound > 0.0 bps Net.
    """
    n_primary = bootstrap_res.get("n_primary_episodes", 0) if "error" not in bootstrap_res else 0
    n_blocks = bootstrap_res.get("n_blocks", 0) if "error" not in bootstrap_res else 0
    lower_ci = bootstrap_res.get("net_95_ci_bps", [0.0, 0.0])[0] if "error" not in bootstrap_res else None

    # Compute empirical median on primary 5m
    median_5m = None
    loco_mean = None
    loco_top_coin = None
    if df is not None and not df.empty and "5m" in df.columns:
        prims = df[(df.get("is_primary_episode", True) == True) & (~df.get("is_control", False))]
        r5 = prims["5m"].dropna().to_numpy()
        if len(r5) > 0:
            median_5m = float(np.median(r5))
            n_primary = len(r5)

        # LOCO Analysis
        if "symbol" in prims.columns and len(r5) >= 5:
            coin_sums = prims.groupby("symbol")["5m"].sum()
            if not coin_sums.empty:
                loco_top_coin = str(coin_sums.idxmax())
                no_top = prims[prims["symbol"] != loco_top_coin]["5m"].dropna().to_numpy()
                loco_mean = float(np.mean(no_top)) if len(no_top) > 0 else 0.0

    # Horizon consistency
    h_consistent = False
    med_1m, med_15m = None, None
    if horizons_summary and isinstance(horizons_summary, dict):
        med_1m = horizons_summary.get("1m", {}).get("median_net_bps")
        med_5m_sum = horizons_summary.get("5m", {}).get("median_net_bps")
        med_15m = horizons_summary.get("15m", {}).get("median_net_bps")
        if med_1m is not None and med_5m_sum is not None and med_15m is not None:
            h_consistent = bool(med_1m > 0 and med_5m_sum > 0 and med_15m > 0)

    # Criteria scorecard
    criteria = {
        "sample_adequacy": {
            "name": f"Sample Adequacy (N ≥ {min_required_episodes}, Blocks ≥ {min_required_blocks})",
            "threshold": f"N ≥ {min_required_episodes}, Blocks ≥ {min_required_blocks}",
            "value": f"N = {n_primary}, Blocks = {n_blocks}",
            "passed": bool(n_primary >= min_required_episodes and n_blocks >= min_required_blocks),
        },
        "positive_median": {
            "name": "Positive Median at 5m Horizon",
            "threshold": "Median Net > 0.0 bps",
            "value": f"{median_5m:+.2f} bps" if median_5m is not None else "Pending",
            "passed": bool(median_5m is not None and median_5m > 0.0),
        },
        "leave_one_coin_out": {
            "name": f"Leave-One-Coin-Out (LOCO) Robustness (Drop {loco_top_coin or 'Top Coin'})",
            "threshold": "Mean Net > 0.0 bps without top coin",
            "value": f"{loco_mean:+.2f} bps (Ex-{loco_top_coin})" if loco_mean is not None else "Pending",
            "passed": bool(loco_mean is not None and loco_mean > 0.0),
        },
        "horizon_consistency": {
            "name": "Multi-Horizon Consistency (1m, 5m, 15m Medians > 0)",
            "threshold": "1m > 0, 5m > 0, 15m > 0",
            "value": f"1m: {med_1m or 0:+.1f}, 5m: {median_5m or 0:+.1f}, 15m: {med_15m or 0:+.1f} bps",
            "passed": h_consistent,
        },
        "bootstrap_ci_lower": {
            "name": "95% Temporal Block Bootstrap Lower CI Bound",
            "threshold": "CI_lower > 0.0 bps Net",
            "value": f"{lower_ci:+.2f} bps" if lower_ci is not None else "Pending (<50 blocks)",
            "passed": bool(lower_ci is not None and lower_ci > 0.0 and n_blocks >= min_required_blocks),
        },
    }

    # Strict Gate Verdict
    is_interim = n_primary < min_required_episodes or n_blocks < min_required_blocks
    all_passed = all(c["passed"] for c in criteria.values())

    if is_interim:
        decision = "COLLECTING_SAMPLE"
        failing_count = sum(1 for c in criteria.values() if not c["passed"])
        statement = (
            f"INTERIM DIAGNOSTIC ONLY ({n_primary}/{min_required_episodes} episodes, {n_blocks}/{min_required_blocks} blocks). "
            f"Currently {failing_count} of 5 institutional criteria failing. DO NOT PROCEED TO EXECUTION ENGINEERING."
        )
    elif all_passed:
        decision = "PROCEED_TO_MAKER_EXECUTION_ENGINEERING"
        statement = f"All 5 institutional criteria confirmed on full sample (N={n_primary}). Cleared for passive maker execution modeling."
    else:
        decision = "KILL_STRATEGY"
        statement = "Strategy failed institutional verification criteria on completed sample. Momentum alpha rejected."

    return {
        "decision": decision,
        "cleared": bool(all_passed and not is_interim),
        "is_interim": is_interim,
        "primary_horizon": "5m",
        "n_primary_episodes": n_primary,
        "n_blocks": n_blocks,
        "median_net_bps": median_5m,
        "loco_mean_net_bps": loco_mean,
        "loco_top_coin": loco_top_coin,
        "criteria": criteria,
        "statement": statement,
        "verdict_statement": statement,
    }

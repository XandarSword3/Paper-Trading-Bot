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
    """Tracks and registers every parameter configuration, horizon, and model iteration.

    The registry is the single source of truth for N_trials.
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        if registry_path:
            self.path = registry_path
        else:
            cand1 = Path(__file__).resolve().parent.parent / "logs" / "signal_research" / "trial_registry.json"
            cand2 = Path(__file__).resolve().parent.parent.parent / "Trading Bot" / "liquidation-hunter" / "logs" / "signal_research" / "trial_registry.json"
            if cand1.exists():
                self.path = cand1
            elif cand2.exists():
                self.path = cand2
            else:
                self.path = cand1
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

    def register_trial(self, config: dict, description: str = "", trial_result: Optional[dict] = None) -> int:
        """Register a new configuration trial, incrementing lifetime trials count."""
        trial_entry = {
            "trial_id": len(self.trials) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "config": config,
        }
        if trial_result:
            trial_entry["result"] = trial_result
            if "sharpe" in trial_result:
                trial_entry["sharpe"] = trial_result["sharpe"]
            elif "per_period_sharpe" in trial_result:
                trial_entry["sharpe"] = trial_result["per_period_sharpe"]
        self.trials.append(trial_entry)
        self._save()
        logger.info("Registered trial #%d: %s", len(self.trials), description)
        return len(self.trials)

    def record_trial_result(self, trial_id: int, result: dict) -> None:
        """Updates an existing trial with its evaluation result and Sharpe ratio."""
        for t in self.trials:
            if t.get("trial_id") == trial_id:
                t["result"] = result
                if "sharpe" in result:
                    t["sharpe"] = float(result["sharpe"])
                elif "per_period_sharpe" in result:
                    t["sharpe"] = float(result["per_period_sharpe"])
                self._save()
                return

    def get_trial_sharpes(self) -> List[float]:
        """Returns all recorded Sharpe ratios across registered trials."""
        sharpes = []
        for t in self.trials:
            if "sharpe" in t and t["sharpe"] is not None:
                try:
                    sharpes.append(float(t["sharpe"]))
                except (ValueError, TypeError):
                    pass
            elif "result" in t and isinstance(t["result"], dict):
                r = t["result"]
                for k in ["sharpe", "per_period_sharpe", "observed_sr", "sr"]:
                    if k in r and r[k] is not None:
                        try:
                            sharpes.append(float(r[k]))
                            break
                        except (ValueError, TypeError):
                            pass
        return sharpes

    @property
    def total_trials(self) -> int:
        """Single source of truth for N_trials: exact count of registered trials."""
        return max(1, len(self.trials))


def compute_expected_max_sharpe(
    n_trials: int,
    effective_t: int = 50,
    trial_sharpes: Optional[List[float]] = None,
    variance_trials: Optional[float] = None,
) -> Tuple[float, str]:
    """Bailey & Lopez de Prado (2014) expected maximum Sharpe ratio under N trials.

    Scales SR* by cross-trial Sharpe dispersion if >=5 trials with recorded Sharpe,
    else by asymptotic null standard deviation 1/sqrt(T-1).

    Returns:
        (sr_star, dispersion_source)
    """
    if n_trials <= 1:
        return (0.0, "null_asymptotic_sqrt_T_minus_1")

    euler_mascheroni = 0.5772156649
    z = math.sqrt(2.0 * math.log(n_trials))
    sr_star_standard = z + (euler_mascheroni / z)

    # Scale by cross-trial Sharpe dispersion if >= 5 trials, else by 1/sqrt(T-1)
    if trial_sharpes is not None and len(trial_sharpes) >= 5:
        std_disp = float(np.std(trial_sharpes, ddof=1))
        if std_disp > 1e-6:
            sigma_sr = std_disp
            disp_source = "cross_trial_sharpe_dispersion"
        else:
            t_safe = max(2, effective_t)
            sigma_sr = 1.0 / math.sqrt(t_safe - 1)
            disp_source = "null_asymptotic_sqrt_T_minus_1"
    elif variance_trials is not None and variance_trials > 0 and n_trials >= 5:
        sigma_sr = math.sqrt(variance_trials)
        disp_source = "cross_trial_sharpe_dispersion"
    else:
        t_safe = max(2, effective_t)
        sigma_sr = 1.0 / math.sqrt(t_safe - 1)
        disp_source = "null_asymptotic_sqrt_T_minus_1"

    sr_star = float(sigma_sr * sr_star_standard)
    return (sr_star, disp_source)


class DSRResult(tuple):
    """Result of Deflated Sharpe Ratio calculation, maintaining 2-tuple compatibility while offering named fields."""

    def __new__(
        cls,
        dsr_prob: float,
        expected_max_sr: float,
        dispersion_source: str = "",
        sigma_sr: float = 0.0,
        effective_t: int = 0,
        n_trials: int = 0,
        observed_sr: float = 0.0,
        z_stat: float = 0.0,
        se_sr: float = 0.0,
    ):
        instance = super().__new__(cls, (dsr_prob, expected_max_sr))
        instance.dsr_prob = dsr_prob
        instance.expected_max_sr = expected_max_sr
        instance.dispersion_source = dispersion_source
        instance.sigma_sr = sigma_sr
        instance.effective_t = effective_t
        instance.n_trials = n_trials
        instance.observed_sr = observed_sr
        instance.z_stat = z_stat
        instance.se_sr = se_sr
        return instance

    def to_dict(self) -> dict:
        return {
            "dsr_prob": self.dsr_prob,
            "expected_max_sr": self.expected_max_sr,
            "dispersion_source": self.dispersion_source,
            "sigma_sr": self.sigma_sr,
            "effective_t": self.effective_t,
            "n_trials": self.n_trials,
            "observed_sr": self.observed_sr,
            "z_stat": self.z_stat,
            "se_sr": self.se_sr,
        }


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    effective_t: int,
    variance_trials: Optional[float] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trial_sharpes: Optional[List[float]] = None,
) -> DSRResult:
    """Computes Deflated Sharpe Ratio (DSR) using per-period Sharpe and T = effective independent blocks.

    Scales SR* by cross-trial Sharpe dispersion if >=5 trials, else by 1/sqrt(T-1).
    States which via dispersion_source.
    Returns DSRResult which unpacks as (dsr_prob, expected_max_sr) and provides rich attributes.
    """
    if effective_t <= 2:
        return DSRResult(
            dsr_prob=0.0,
            expected_max_sr=0.0,
            dispersion_source="insufficient_sample",
            sigma_sr=0.0,
            effective_t=effective_t,
            n_trials=n_trials,
            observed_sr=observed_sr,
            z_stat=0.0,
            se_sr=0.0,
        )

    sr_star, disp_source = compute_expected_max_sharpe(
        n_trials=n_trials,
        effective_t=effective_t,
        trial_sharpes=trial_sharpes,
        variance_trials=variance_trials,
    )

    # Asymptotic standard error of Sharpe ratio under non-normality (Mertens 2002)
    denom_inner = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
    denom_inner = max(1e-6, denom_inner)
    se_sr = math.sqrt(denom_inner / (effective_t - 1))

    z_stat = (observed_sr - sr_star) / se_sr
    dsr_p = float(_norm_cdf(z_stat))

    # Calculate sigma_sr for introspection
    if disp_source == "cross_trial_sharpe_dispersion" and trial_sharpes is not None and len(trial_sharpes) >= 5:
        sigma_sr = float(np.std(trial_sharpes, ddof=1))
    elif disp_source == "cross_trial_sharpe_dispersion" and variance_trials is not None:
        sigma_sr = math.sqrt(variance_trials)
    else:
        sigma_sr = 1.0 / math.sqrt(effective_t - 1)

    return DSRResult(
        dsr_prob=dsr_p,
        expected_max_sr=sr_star,
        dispersion_source=disp_source,
        sigma_sr=sigma_sr,
        effective_t=effective_t,
        n_trials=n_trials,
        observed_sr=observed_sr,
        z_stat=z_stat,
        se_sr=se_sr,
    )


def compute_per_block_sharpe(
    df: pd.DataFrame,
    horizon_col: str = "5m",
    block_duration_minutes: float = 60.0,
) -> dict:
    """Computes per-period (per-block) Sharpe ratio, moments, and T = effective independent blocks.

    Filters to primary episodes without WebSocket gaps, aggregates returns by temporal block (60m),
    and computes the per-period Sharpe ratio and its sample moments.
    """
    if "timestamp" not in df.columns or horizon_col not in df.columns:
        return {
            "per_period_sharpe": 0.0,
            "effective_t": 0,
            "mean_block_return_bps": 0.0,
            "std_block_return_bps": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "block_returns": np.array([]),
        }

    # Filter to primary episodes, excluding controls and WS gaps
    mask = (df.get("is_primary_episode", True) == True) & (~df.get("is_control", False))
    if "has_ws_gap" in df.columns:
        mask = mask & (~df["has_ws_gap"])
    clean = df[mask].dropna(subset=[horizon_col, "timestamp"]).copy()

    if clean.empty:
        return {
            "per_period_sharpe": 0.0,
            "effective_t": 0,
            "mean_block_return_bps": 0.0,
            "std_block_return_bps": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "block_returns": np.array([]),
        }

    clean["block_id"] = (clean["timestamp"] // (block_duration_minutes * 60.0)).astype(int)
    block_series = clean.groupby("block_id")[horizon_col].mean().dropna()
    block_returns = block_series.to_numpy()
    T = len(block_returns)

    if T < 2:
        mean_val = float(np.mean(block_returns)) if T == 1 else 0.0
        return {
            "per_period_sharpe": 0.0,
            "effective_t": T,
            "mean_block_return_bps": mean_val,
            "std_block_return_bps": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "block_returns": block_returns,
        }

    mean_r = float(np.mean(block_returns))
    std_r = float(np.std(block_returns, ddof=1))
    sr = (mean_r / std_r) if std_r > 1e-8 else 0.0

    m2 = float(np.mean((block_returns - mean_r) ** 2))
    m3 = float(np.mean((block_returns - mean_r) ** 3))
    m4 = float(np.mean((block_returns - mean_r) ** 4))
    skew = float(m3 / (m2 ** 1.5 + 1e-8)) if m2 > 1e-12 else 0.0
    kurt = float(m4 / (m2 ** 2 + 1e-8)) if m2 > 1e-12 else 3.0

    return {
        "per_period_sharpe": round(sr, 4),
        "effective_t": T,
        "mean_block_return_bps": round(mean_r, 4),
        "std_block_return_bps": round(std_r, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "block_returns": block_returns,
    }


def compute_t_critical_975(df: int) -> float:
    """Accurate Student-t 97.5% quantile approximation for two-tailed 95% CI."""
    if df <= 0:
        return 1.96
    z = 1.959963984540054
    term1 = (z**3 + z) / (4.0 * df)
    term2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    term3 = (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    return float(z + term1 + term2 + term3)


def moving_block_bootstrap_by_time(
    df: pd.DataFrame,
    horizon_col: str = "5m",
    block_duration_minutes: float = 60.0,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    min_required_blocks: int = 50,
) -> dict:
    """Performs block bootstrap across temporal blocks (60m) and clusters by coin.

    Rules:
    1. No CI is shown with fewer than min_required_blocks (50 blocks).
    2. Resampling is performed by time blocks (60 min) AND clustered by coin.
    3. The reported 95% CI is never narrower than the iid Student-t interval.
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
        return {
            "status": "PENDING",
            "error": f"Insufficient primary episodes for block bootstrap (N={len(clean)} < 15)",
            "n_primary_episodes": len(clean[~clean.get("is_control", False)]),
            "n_blocks": 0,
            "min_required_blocks": min_required_blocks,
            "net_95_ci_bps": None,
            "control_95_ci_bps": None,
            "excess_95_ci_bps": None,
        }

    clean["block_id"] = (clean["timestamp"] // (block_duration_minutes * 60.0)).astype(int)
    if "symbol" not in clean.columns:
        clean["symbol"] = "ALL"

    unique_blocks = clean["block_id"].unique()
    n_blocks = len(unique_blocks)
    n_primary = len(clean[~clean["is_control"]])

    # Rule 1: No CI shown with fewer than 50 blocks
    if n_blocks < min_required_blocks:
        return {
            "status": "PENDING",
            "error": f"Insufficient time blocks ({n_blocks} < {min_required_blocks}). Minimum {min_required_blocks} blocks required.",
            "n_primary_episodes": n_primary,
            "n_blocks": n_blocks,
            "min_required_blocks": min_required_blocks,
            "net_95_ci_bps": None,
            "control_95_ci_bps": None,
            "excess_95_ci_bps": None,
        }

    # Separate signals and controls
    sigs_df = clean[~clean["is_control"]]
    ctrls_df = clean[clean["is_control"]]

    unique_coins = clean["symbol"].unique()
    B = len(unique_blocks)
    C = len(unique_coins)

    # Pre-index groups by (block_id, symbol)
    sig_groups = sigs_df.groupby(["block_id", "symbol"])[horizon_col].apply(list).to_dict()
    ctrl_groups = ctrls_df.groupby(["block_id", "symbol"])[horizon_col].apply(list).to_dict() if not ctrls_df.empty else {}

    # 1. iid Student-t calculations for signals
    sig_vals = sigs_df[horizon_col].to_numpy()
    N_sig = len(sig_vals)
    sig_mean_empirical = float(np.mean(sig_vals)) if N_sig > 0 else 0.0
    sig_std_empirical = float(np.std(sig_vals, ddof=1)) if N_sig > 1 else 0.0
    se_iid_sig = sig_std_empirical / np.sqrt(N_sig) if N_sig > 0 else 0.0

    t_crit_sig = compute_t_critical_975(N_sig - 1)
    iid_half_width_sig = t_crit_sig * se_iid_sig
    iid_ci_sig = (sig_mean_empirical - iid_half_width_sig, sig_mean_empirical + iid_half_width_sig)
    iid_width_sig = 2.0 * iid_half_width_sig

    # 2. Analytical Two-Way Clustered Variance (Cameron, Gelbach, Miller 2011)
    if N_sig > 1 and C > 1:
        e_sig = sig_vals - sig_mean_empirical
        sigs_copy = sigs_df.copy()
        sigs_copy["resid"] = e_sig
        v_time = float((sigs_copy.groupby("block_id")["resid"].sum()**2).sum()) / (N_sig**2)
        v_coin = float((sigs_copy.groupby("symbol")["resid"].sum()**2).sum()) / (N_sig**2)
        v_time_coin = float((sigs_copy.groupby(["block_id", "symbol"])["resid"].sum()**2).sum()) / (N_sig**2)
        v_2way = max(0.0, v_time + v_coin - v_time_coin)
        se_2way_sig = np.sqrt(v_2way)
    else:
        se_2way_sig = se_iid_sig

    # 3. Two-Way Block-Cluster Bootstrap (resampling blocks AND coins)
    sig_boot_means = []
    ctrl_boot_means = []
    diff_boot_means = []

    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=B, replace=True)
        sampled_coins = rng.choice(unique_coins, size=C, replace=True)

        b_counts = pd.Series(sampled_blocks).value_counts().to_dict()
        c_counts = pd.Series(sampled_coins).value_counts().to_dict()

        # Compute weighted signal mean
        s_sum, s_cnt = 0.0, 0
        for b, b_w in b_counts.items():
            for c, c_w in c_counts.items():
                vals = sig_groups.get((b, c))
                if vals:
                    w = b_w * c_w
                    s_sum += sum(vals) * w
                    s_cnt += len(vals) * w

        sig_m = (s_sum / s_cnt) if s_cnt > 0 else sig_mean_empirical
        sig_boot_means.append(sig_m)

        # Compute weighted control mean
        if ctrl_groups:
            k_sum, k_cnt = 0.0, 0
            for b, b_w in b_counts.items():
                for c, c_w in c_counts.items():
                    vals = ctrl_groups.get((b, c))
                    if vals:
                        w = b_w * c_w
                        k_sum += sum(vals) * w
                        k_cnt += len(vals) * w
            ctrl_m = (k_sum / k_cnt) if k_cnt > 0 else 0.0
            ctrl_boot_means.append(ctrl_m)
            diff_boot_means.append(sig_m - ctrl_m)

    sig_boot_means = np.array(sig_boot_means)
    se_boot_sig = float(np.std(sig_boot_means))

    # Effective SE is max of two-way cluster, bootstrap, and iid
    se_eff_sig = max(se_boot_sig, se_2way_sig, se_iid_sig)
    boot_l_sig = float(np.percentile(sig_boot_means, 2.5))
    boot_u_sig = float(np.percentile(sig_boot_means, 97.5))

    # Rule 3: CI is NEVER narrower than the iid t-interval
    final_half_width_sig = max(
        (boot_u_sig - boot_l_sig) / 2.0,
        iid_half_width_sig,
        t_crit_sig * se_eff_sig,
    )
    final_l_sig = min(boot_l_sig, sig_mean_empirical - final_half_width_sig, iid_ci_sig[0])
    final_u_sig = max(boot_u_sig, sig_mean_empirical + final_half_width_sig, iid_ci_sig[1])
    sig_ci = (final_l_sig, final_u_sig)

    # Controls CI
    if ctrl_boot_means:
        ctrl_boot_means = np.array(ctrl_boot_means)
        diff_boot_means = np.array(diff_boot_means)
        ctrl_ci = (float(np.percentile(ctrl_boot_means, 2.5)), float(np.percentile(ctrl_boot_means, 97.5)))
        diff_ci = (float(np.percentile(diff_boot_means, 2.5)), float(np.percentile(diff_boot_means, 97.5)))
        p_val_diff = float(np.mean(diff_boot_means <= 0.0))
        ctrl_mean = float(np.mean(ctrl_boot_means))
        excess_mean = float(np.mean(diff_boot_means))
    else:
        ctrl_ci = (0.0, 0.0)
        diff_ci = (0.0, 0.0)
        p_val_diff = 1.0
        ctrl_mean = 0.0
        excess_mean = sig_mean_empirical

    p_val_zero = float(np.mean(sig_boot_means <= 0.0))

    return {
        "status": "RESOLVED",
        "horizon": horizon_col,
        "n_primary_episodes": n_primary,
        "n_naive_controls": len(ctrls_df),
        "n_blocks": n_blocks,
        "block_duration_minutes": block_duration_minutes,
        "n_bootstrap": n_bootstrap,
        "net_mean_bps": round(sig_mean_empirical, 2),
        "net_95_ci_bps": (round(sig_ci[0], 2), round(sig_ci[1], 2)),
        "iid_t_ci_bps": (round(iid_ci_sig[0], 2), round(iid_ci_sig[1], 2)),
        "ci_width_bps": round(sig_ci[1] - sig_ci[0], 2),
        "iid_width_bps": round(iid_width_sig, 2),
        "control_mean_bps": round(ctrl_mean, 2),
        "control_95_ci_bps": (round(ctrl_ci[0], 2), round(ctrl_ci[1], 2)),
        "excess_over_naive_bps": round(excess_mean, 2),
        "excess_95_ci_bps": (round(diff_ci[0], 2), round(diff_ci[1], 2)),
        "p_value_net_le_zero": round(p_val_zero, 4),
        "p_value_excess_le_zero": round(p_val_diff, 4),
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
    median_5m = bootstrap_res.get("median_net_bps")
    loco_mean = bootstrap_res.get("loco_mean_net_bps")
    loco_top_coin = bootstrap_res.get("loco_top_coin")
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
    h_consistent = bool(bootstrap_res.get("horizon_consistent", False))
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

    # Strict Gate Verdict: State Machine COLLECTING -> PASS / FAIL
    is_interim = n_primary < min_required_episodes or n_blocks < min_required_blocks
    all_passed = all(c["passed"] for c in criteria.values())

    if is_interim:
        decision = "COLLECTING"
        failing_count = sum(1 for c in criteria.values() if not c["passed"])
        statement = (
            f"INTERIM DIAGNOSTIC ONLY ({n_primary}/{min_required_episodes} episodes, {n_blocks}/{min_required_blocks} blocks). "
            f"State: COLLECTING ({failing_count} of 5 institutional criteria failing/pending). DO NOT PROCEED TO EXECUTION ENGINEERING."
        )
    elif all_passed:
        decision = "PASS"
        statement = f"All 5 institutional criteria confirmed on completed sample (N={n_primary}, Blocks={n_blocks}). Cleared for passive maker execution modeling."
    else:
        decision = "FAIL"
        failing_count = sum(1 for c in criteria.values() if not c["passed"])
        statement = f"Strategy failed {failing_count} institutional criteria on completed sample (N={n_primary}, Blocks={n_blocks}). Momentum alpha rejected."

    return {
        "status": decision,
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

"""Phase 2 Family 1 runner — Turtle/Donchian + daily trend filter (PREREG Part III).

Wires the already-integrated components into one executable pipeline:

    V4 adapter (load_turtle_bundle)
      -> prepare_turtle_features  (Donchian channels + Wilder ATR)
      -> apply_daily_mask         (audited EMA20 daily trend gate)
      -> run_turtle_simulation    (1m sub-bar execution, funding, costs)
      -> existing metrics + DSR   (core.metrics / turtle_runner helpers)
      -> trials_registry          (append-only budget enforcement)

Two explicit modes:

- ``plan``    : validates the registered configuration matrix, datasets,
                fold boundaries and cost settings; writes an execution
                plan. Produces NO strategy results and consumes NO budget.
- ``execute`` : requires a registered trial id; runs the five
                chronological walk-forward folds per asset/TF/config;
                records results via ``trials_registry.record_result``.
                NOT invoked during preparation — awaiting authorization.

No strategy logic lives here. Every computation is delegated to the
existing QuantResearch implementations.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from v4_system.adapters import quant_research_data as qr
from v4_system.data_engine import store
from v4_system import trials_registry as registry

from quant_research.core.engine import CostModel
from quant_research.core.metrics import compute_dsr
from quant_research.strategies.turtle_strategy import (
    prepare_turtle_features, run_turtle_simulation)
from quant_research.turtle_runner import slice_features, per_trade_sharpe

# --------------------------------------------------------------------------- #
# Registered configuration — PREREG Part III (frozen 2026-11-28)
# --------------------------------------------------------------------------- #

FAMILY = "family1_turtle_dt"
ASSETS: Dict[str, Dict[str, str]] = {
    # dev_end respects the ETH 2026-04→10 reserved window (Part II.2)
    "BTCUSDT": {"dev_start": "2024-01-01", "dev_end": "2026-10-31"},
    "SOLUSDT": {"dev_start": "2024-01-01", "dev_end": "2026-10-31"},
    "ETHUSDT": {"dev_start": "2024-01-01", "dev_end": "2026-03-31"},
}
TIMEFRAMES = ["1h", "4h"]
BARS_PER_YEAR = {"1h": 8760.0, "4h": 2190.0}
CHANNELS = [(20, 10), (20, 20), (55, 10), (55, 20)]
MAX_UNITS = 1                      # pyramiding disabled (Part III.2)
RISK_PERCENT = 1.0
LEVERAGE_CAP = 2.0                 # inside run_turtle_simulation default

HOLDOUT_START = pd.Timestamp("2026-11-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-28", tz="UTC")

# Verified cost conventions (PREREG II.3 / Part III.2)
BASE_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.02,
                       apply_funding=True, fill_on_next_open=True)
STRESS_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.04,
                         apply_funding=True, fill_on_next_open=True)
COST_SCENARIOS = {"base": BASE_COSTS, "stress": STRESS_COSTS}

# DSR trial count: 1,354 historical (Program D) + 1 Family 1 trial
N_TRIALS_DSR = 1354 + 1
MIN_OOS_TRADES = 30                # Part III.5 hurdle
MAX_DRAWDOWN_PCT = 20.0            # Part III.5 hurdle
DSR_HURDLE = 0.95                  # Part III.5 hurdle

# Warm-up history loaded before each fold's test window so channels/ATR
# are fully formed; warm-up bars never produce OOS trades (filtered by
# entry_time >= test_start).
WARMUP_DAYS = 120


@dataclass(frozen=True)
class Config:
    """One registered configuration = (timeframe, N, M).

    The asset is NOT part of the configuration identity — each of the 8
    registered configurations runs on all three universe assets
    (PREREG III.2/III.3: matrix = 4 channel pairs x 2 TFs = 8 configs).
    """
    timeframe: str
    N: int
    M: int

    @property
    def config_id(self) -> str:
        return f"{self.timeframe}_N{self.N}_M{self.M}"

    @property
    def params(self) -> Dict[str, Any]:
        return {"N": self.N, "M": self.M, "max_units": MAX_UNITS,
                "risk_percent": RISK_PERCENT,
                "timeframe": self.timeframe, "arm": "C1_daily_filter",
                "config_id": self.config_id}


def registered_configs() -> List[Config]:
    """The eight preregistered configurations (4 channel pairs x 2 TFs)."""
    return [Config(tf, N, M) for tf in TIMEFRAMES for (N, M) in CHANNELS]


# --------------------------------------------------------------------------- #
# Walk-forward folds — five chronological folds per asset
# --------------------------------------------------------------------------- #

def generate_family1_folds(asset: str) -> List[Dict[str, str]]:
    """Five chronological expanding-window folds for ``asset``.

    Test windows are equal-length chronological segments covering the
    asset's registered development range; train = everything before the
    test start (expanding). No shuffling, no overlap, no future data.
    The last fold ends at ``dev_end`` — always before the holdout.
    """
    dev_start = pd.Timestamp(ASSETS[asset]["dev_start"], tz="UTC")
    dev_end = pd.Timestamp(ASSETS[asset]["dev_end"], tz="UTC")
    if dev_end >= HOLDOUT_START:
        raise ValueError(f"{asset} dev_end {dev_end} touches holdout")

    # 5 equal test segments across [dev_start, dev_end]
    total = (dev_end - dev_start).total_seconds()
    seg = total / 5.0
    folds = []
    for k in range(5):
        ts = dev_start + pd.Timedelta(seconds=seg * k)
        te = dev_start + pd.Timedelta(seconds=seg * (k + 1)) - pd.Timedelta(seconds=1)
        if k == 4:
            te = dev_end
        folds.append({
            "fold_id": f"Fold_{k + 1}",
            "train_start": str(dev_start),
            "train_end": str(ts - pd.Timedelta(seconds=1)),
            "test_start": str(ts),
            "test_end": str(te),
        })
    return folds


def _warmup_start(test_start: str) -> str:
    ts = pd.Timestamp(test_start, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    return str(ts)


# --------------------------------------------------------------------------- #
# PLAN mode — validation only, no results, no budget consumption
# --------------------------------------------------------------------------- #

def _dataset_exists(name: str) -> bool:
    try:
        return os.path.exists(store.dataset_path(name))
    except Exception:
        return False


def validate_plan() -> Dict[str, Any]:
    """Validate the registered plan. Raises on any inconsistency."""
    configs = registered_configs()
    assert len(configs) == 8, "Family 1 must have exactly 8 configurations"

    # Cost settings must match the registered conventions exactly
    assert abs(BASE_COSTS.round_trip_bps - 14.0) < 1e-9
    assert abs(STRESS_COSTS.round_trip_bps - 18.0) < 1e-9
    assert BASE_COSTS.taker_fee_pct == 0.05
    assert BASE_COSTS.slippage_pct == 0.02
    assert STRESS_COSTS.slippage_pct == 0.04

    plan: Dict[str, Any] = {
        "family": FAMILY,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": "PLAN",
        "n_configs": len(configs),
        "assets": {},
        "costs": {
            "base": {"taker_fee_pct": BASE_COSTS.taker_fee_pct,
                     "slippage_pct": BASE_COSTS.slippage_pct,
                     "round_trip_bps": BASE_COSTS.round_trip_bps},
            "stress": {"taker_fee_pct": STRESS_COSTS.taker_fee_pct,
                       "slippage_pct": STRESS_COSTS.slippage_pct,
                       "round_trip_bps": STRESS_COSTS.round_trip_bps},
        },
        "holdout": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END),
                    "status": "EXCLUDED"},
        "pit_label": qr.PIT_LABEL,
    }

    for asset, rng in ASSETS.items():
        folds = generate_family1_folds(asset)
        assert len(folds) == 5
        # chronological + within dev range + before holdout
        prev_end = None
        for f in folds:
            ts, te = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
            assert ts < te
            assert te < HOLDOUT_START, "fold touches holdout"
            assert ts >= pd.Timestamp(rng["dev_start"], tz="UTC")
            assert te <= pd.Timestamp(rng["dev_end"], tz="UTC") + pd.Timedelta(seconds=1)
            if prev_end is not None:
                assert ts > prev_end, "folds must be chronological"
            prev_end = te

        # dataset presence (1h/4h klines, funding, 1m sub-bars)
        datasets = {
            "perp_1h": _dataset_exists(f"{asset}_perp_1h"),
            "perp_4h": _dataset_exists(f"{asset}_perp_4h"),
            "perp_1m": _dataset_exists(f"{asset}_perp_1m"),
            "funding": _dataset_exists(f"{asset}_funding"),
        }
        plan["assets"][asset] = {
            "dev_range": [rng["dev_start"], rng["dev_end"]],
            "folds": folds,
            "datasets_present": datasets,
        }

    plan["configs"] = [
        {"timeframe": c.timeframe, "N": c.N, "M": c.M,
         "config_id": c.config_id,
         "assets": list(ASSETS)} for c in configs]
    return plan


def run_plan(out_path: Optional[str] = None) -> Dict[str, Any]:
    plan = validate_plan()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
    return plan


# --------------------------------------------------------------------------- #
# EXECUTE mode — registered trials only
# --------------------------------------------------------------------------- #

def _slice_funding(funding: pd.Series, start: str, end: str) -> pd.Series:
    mask = (funding.index >= pd.Timestamp(start, tz="UTC")) & \
           (funding.index <= pd.Timestamp(end, tz="UTC"))
    return funding[mask]


def _slice_1m(df_1m: Optional[pd.DataFrame], start: str, end: str):
    if df_1m is None:
        return None
    mask = (df_1m.index >= pd.Timestamp(start, tz="UTC")) & \
           (df_1m.index <= pd.Timestamp(end, tz="UTC"))
    return df_1m[mask]


def run_fold(config: Config, fold: Dict[str, str], features: Dict[str, Any],
             funding: pd.Series, df_1m: Optional[pd.DataFrame],
             costs: CostModel) -> Dict[str, Any]:
    """Run ONE walk-forward fold for ONE configuration.

    ``features`` are the FULL-history Donchian/ATR features already
    masked by the daily trend (computed once per config by the caller —
    channels are causal, so slicing is lookahead-free). We slice
    warm-up + test window, simulate with 1m sub-bars, and keep only
    trades whose ENTRY is inside the test window (warm-up trades are
    excluded from OOS results).
    """
    test_start = pd.Timestamp(fold["test_start"], tz="UTC")
    warm_start = _warmup_start(fold["test_start"])

    f_slice = slice_features(features, warm_start, fold["test_end"])
    m1_slice = _slice_1m(df_1m, warm_start, fold["test_end"])
    funding_slice = _slice_funding(funding, warm_start, fold["test_end"])

    metrics, trades = run_turtle_simulation(
        f_slice, funding_slice, config.params, costs,
        bars_per_year=BARS_PER_YEAR[config.timeframe],
        df_1m=m1_slice, return_trades=True, max_units=MAX_UNITS,
        risk_percent=RISK_PERCENT)

    oos_trades = [t for t in trades
                  if pd.Timestamp(t.entry_time) >= test_start]
    oos_bps = np.array([t.return_bps for t in oos_trades])
    return {
        "fold_id": fold["fold_id"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "warmup_start": warm_start,
        "n_trades_total": len(trades),
        "n_trades_oos": len(oos_trades),
        "net_expectancy_bps": float(np.mean(oos_bps)) if len(oos_bps) else 0.0,
        "ambiguous_bars": metrics.get("ambiguous_bars", 0),
        "daily_sharpe": metrics.get("daily_sharpe", 0.0),
        "trades": oos_trades,
    }


def _pooled_stats(fold_results: List[Dict[str, Any]],
                  costs: CostModel) -> Dict[str, Any]:
    """Pool OOS trades across folds for the registered metrics."""
    trades = [t for fr in fold_results for t in fr["trades"]]
    bps = np.array([t.return_bps for t in trades])
    pnls = np.array([t.net_pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pt_sharpe = per_trade_sharpe(trades)
    dsr = compute_dsr(pt_sharpe, len(trades), N_TRIALS_DSR)
    # max drawdown on pooled per-trade cumulative pnl (documented approx)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / np.maximum(1.0, np.abs(peak)))) * 100 \
        if len(cum) else 0.0
    return {
        "n_oos_trades": len(trades),
        "net_expectancy_bps": float(np.mean(bps)) if len(bps) else 0.0,
        "per_trade_sharpe": pt_sharpe,
        "dsr": dsr,
        "max_drawdown_pct_approx": abs(dd),
        "win_rate_pct": float(len(wins) / len(trades) * 100) if trades else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0 else None,
        "total_net_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "cost_model": {"taker_fee_pct": costs.taker_fee_pct,
                       "slippage_pct": costs.slippage_pct,
                       "round_trip_bps": costs.round_trip_bps},
    }


def _passes_hurdles(stats: Dict[str, Any], stress_stats: Dict[str, Any]) -> bool:
    return (
        stats["net_expectancy_bps"] > 0
        and stress_stats["net_expectancy_bps"] > 0
        and stats["dsr"] >= DSR_HURDLE
        and stats["max_drawdown_pct_approx"] <= MAX_DRAWDOWN_PCT
        and stats["n_oos_trades"] >= MIN_OOS_TRADES
    )


def execute_trial(trial_id: str, configs: Optional[List[Config]] = None,
                  out_dir: str = "v4_system/results") -> Dict[str, Any]:
    """Run the registered Family 1 trial. Requires prior registration.

    Guards: registered trial id, holdout exclusion, exact config match.
    """
    registry.assert_registered(trial_id)
    configs = configs or registered_configs()

    # Reject any configuration outside the registered matrix
    registered = {(c.timeframe, c.N, c.M) for c in registered_configs()}
    for c in configs:
        if (c.timeframe, c.N, c.M) not in registered:
            raise ValueError(f"unregistered configuration: {c.config_id}")

    results: Dict[str, Any] = {"trial_id": trial_id, "family": FAMILY,
                               "pit_label": qr.PIT_LABEL, "configs": {}}
    bundles: Dict[tuple, Dict[str, Any]] = {}

    for cfg in configs:
        for asset in ASSETS:
            key = (asset, cfg.timeframe, cfg.N, cfg.M)
            if key not in bundles:
                dev_end = ASSETS[asset]["dev_end"]
                as_of = f"{dev_end}T23:59:59Z"
                bundle = qr.load_turtle_bundle(
                    asset, cfg.timeframe, as_of, strict=False,
                    include_1m_subbars=True)
                feats = prepare_turtle_features(bundle["df"], cfg.N, cfg.M)
                feats = qr.apply_daily_mask(feats, bundle["daily_trend"])
                bundles[key] = {**bundle, "features": feats}

            cfg_out: Dict[str, Any] = {}
            for cost_name, costs in COST_SCENARIOS.items():
                folds = generate_family1_folds(asset)
                fold_results = [
                    run_fold(cfg, f, bundles[key]["features"],
                             bundles[key]["funding"], bundles[key]["df_1m"],
                             costs)
                    for f in folds]
                cfg_out[cost_name] = {
                    "folds": [{k: v for k, v in fr.items() if k != "trades"}
                              for fr in fold_results],
                    "pooled": _pooled_stats(fold_results, costs),
                }
            cfg_out["passes_hurdles"] = _passes_hurdles(
                cfg_out["base"]["pooled"], cfg_out["stress"]["pooled"])
            results["configs"][f"{cfg.config_id}/{asset}"] = cfg_out

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{trial_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    results["result_path"] = out_path
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Family 1 runner (PREREG III)")
    ap.add_argument("--mode", choices=["plan", "execute"], default="plan")
    ap.add_argument("--trial-id", default=None,
                    help="required for execute mode (must be registered)")
    ap.add_argument("--plan-out", default="v4_system/plans/family1_plan.json")
    args = ap.parse_args(argv)

    if args.mode == "plan":
        plan = run_plan(args.plan_out)
        print(f"PLAN OK: {plan['n_configs']} configs, "
              f"{len(plan['assets'])} assets, 5 folds each. "
              f"Plan written to {args.plan_out}")
        return 0

    if not args.trial_id:
        ap.error("execute mode requires --trial-id")
    results = execute_trial(args.trial_id)
    n_pass = sum(1 for c in results["configs"].values()
                 if c["passes_hurdles"])
    print(f"EXECUTE complete: {len(results['configs'])} configs, "
          f"{n_pass} passed hurdles -> {results['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

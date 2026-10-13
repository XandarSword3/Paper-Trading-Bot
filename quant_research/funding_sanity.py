"""
Funding sanity check (round-2 item 3).

For every one of the 36 Turtle configurations the pre-registered funding rule is:

    funding_pnl = -side * quantity * close * rate       (QuantEngine.funding_payment)

and the pre-registered decision statistic is the per-UNIT net expectancy in bps.
This script makes the funding component of that statistic auditable by reporting,
per config:

  * mean_hold_hours                    - mean unit holding time, hours
  * mean_settlements_per_unit          - mean number of actual funding settlements the
                                         unit was exposed to (settlement bars inside the
                                         simulator's accrual window)
  * mean_rate_per_settlement_bps       - mean signed funding rate of those settlements,
                                         sign applied: -1 for a long (long pays a
                                         positive rate), +1 for a short
  * expected_funding_bps_meanfield     - hold_hours/8 * rate   (the mean-field estimator)
  * expected_funding_bps_settled       - sum of the signed settlement rates, the
                                         settlement-exact estimator
  * reported_funding_bps_corrected     - mean over units of funding_pnl / entry_notional,
                                         straight from the simulator
  * reported_funding_bps_inverted      - the same per-unit quantity with the historical
                                         (pre-fix) sign convention, i.e. the value the
                                         retired artifacts printed

Accrual-window convention (matches both committed simulators): funding is charged at the
start of bar j for every settlement stamped at ts[j], for every bar on which a position is
open. A unit is therefore exposed to settlements s with
    entry_time <= ts_s <= exit_time            (stop-loss exits)
    entry_time <= ts_s <  exit_time            (channel exits: the channel is detected on
                                                a closed bar and the exit fills at the NEXT
                                                open, so the exit-bar settlement is not
                                                charged against the closed position)
NOTE: the simulators split each cycle's accrued funding EQUALLY across the 1-4 units of
the cycle (funding_accrued / len(units)), independently of each unit's own exposure time;
reported_funding_bps is the mean over units of that split, so late pyramid units inherit a
share of funding accrued before they existed.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from quant_research.core.engine import CostModel
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_diagnostics import run_turtle_diagnostic_sim, slice_features

DATA_DIR = "quant_research/data/processed"
OUT_PATH = "quant_research/results/funding_sanity_v2.json"

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]
CONFIGS = [(20, 10), (20, 20), (55, 10), (55, 20)]
OOS_START = "2025-04-01 00:00:00"
OOS_END = "2026-03-31 23:59:59"
BARS_PER_YEAR = {"5m": 105120.0, "15m": 35040.0, "1h": 8760.0}
BASE_COSTS = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)


def load_features(asset, tf):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_{tf}.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    fdf["t_round"] = fdf["timestamp"].dt.round("1min")
    funding = fdf.groupby("t_round")["funding_rate"].sum().sort_index()
    funding = funding.rename_axis(None)  # keep the index unnamed: the simulators add a 't_round' column
    return df, funding


def slice_features_oos(features):
    ts = features["timestamps"]
    mask = (ts >= pd.Timestamp(OOS_START, tz="UTC")) & (ts <= pd.Timestamp(OOS_END, tz="UTC"))
    return {k: (features[k][mask] if isinstance(features[k], np.ndarray) else ts[mask]) for k in features}


def unit_funding_detail(trades, funding, bar_delta):
    """Per-unit hold / settlement / rate detail for one config."""
    f_index = funding.index
    rows = []
    for t in trades:
        side = 1.0 if t.direction == "long" else -1.0
        hold_h = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600.0
        notional = t.quantity * t.entry_price
        channel_exit = bool(t.exit_reason and "Donchian" in t.exit_reason)
        upper = pd.Timestamp(t.exit_time) - bar_delta if channel_exit else pd.Timestamp(t.exit_time)
        window = funding.loc[(f_index >= pd.Timestamp(t.entry_time)) & (f_index <= upper)]
        n_sett = int(len(window))
        rate_sum_bps = -side * float(window.sum()) * 1e4       # corrected convention
        rate_bps = rate_sum_bps / n_sett if n_sett else 0.0    # signed rate per settlement
        rows.append({
            "direction": t.direction,
            "hold_hours": hold_h,
            "n_settlements": n_sett,
            "rate_per_settlement_bps": rate_bps,
            "expected_funding_bps_meanfield": (hold_h / 8.0) * rate_bps,
            "expected_funding_bps_settled": rate_sum_bps,
            "reported_funding_bps_corrected": (t.funding_pnl / notional * 1e4) if notional > 0 else 0.0,
            "reported_funding_bps_inverted": (-t.funding_pnl / notional * 1e4) if notional > 0 else 0.0,
            "net_bps": t.return_bps,
            "channel_exit": channel_exit,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 110)
    print("FUNDING SANITY CHECK - 36 CONFIGS, OOS 2025-04-01 .. 2026-03-31")
    print("expected = hold_hours/8 * signed_rate_per_settlement (mean-field) and sum(signed rates) (settled)")
    print("=" * 110)

    per_config = []
    unit_frames = {}
    for tf in TIMEFRAMES:
        bar_delta = pd.Timedelta({"5m": "5min", "15m": "15min", "1h": "1h"}[tf])
        for asset in ASSETS:
            df, funding = load_features(asset, tf)
            for N, M in CONFIGS:
                feat = prepare_turtle_features(df, N=N, M=M)
                oos = slice_features_oos(feat)
                _, trades = run_turtle_simulation(
                    oos, funding, {"N": N, "M": M}, BASE_COSTS,
                    bars_per_year=BARS_PER_YEAR[tf], return_trades=True,
                )
                diag = run_turtle_diagnostic_sim(oos, funding, N, M, BASE_COSTS)
                ud = unit_funding_detail(trades, funding, bar_delta)
                unit_frames[(asset, tf, N, M)] = ud
                row = {
                    "asset": asset, "timeframe": tf, "N": N, "M": M,
                    "n_units": int(len(ud)),
                    "mean_hold_hours": round(float(ud["hold_hours"].mean()), 4),
                    "mean_settlements_per_unit": round(float(ud["n_settlements"].mean()), 4),
                    "mean_rate_per_settlement_bps": round(float(ud["rate_per_settlement_bps"].mean()), 4),
                    "expected_funding_bps_meanfield": round(float(ud["expected_funding_bps_meanfield"].mean()), 4),
                    "expected_funding_bps_settled": round(float(ud["expected_funding_bps_settled"].mean()), 4),
                    "reported_funding_bps_corrected": round(float(ud["reported_funding_bps_corrected"].mean()), 4),
                    "reported_funding_bps_inverted": round(float(ud["reported_funding_bps_inverted"].mean()), 4),
                    "net_bps_corrected": round(float(ud["net_bps"].mean()), 4),
                    "diagnostics_funding_bps": diag["funding_bps"],
                    "diagnostics_net_bps": diag["net_bps"],
                    "long_units_share_pct": round(float((ud["direction"] == "long").mean() * 100.0), 2),
                    "channel_exit_share_pct": round(float(ud["channel_exit"].mean() * 100.0), 2),
                }
                per_config.append(row)
                print(f"{asset:8} {tf:3} {N:2}/{M:2} units={row['n_units']:5d} hold={row['mean_hold_hours']:7.3f}h "
                      f"settl={row['mean_settlements_per_unit']:5.3f} rate={row['mean_rate_per_settlement_bps']:+7.3f}bps "
                      f"exp_mf={row['expected_funding_bps_meanfield']:+8.3f} exp_settled={row['expected_funding_bps_settled']:+8.3f} "
                      f"rep_corr={row['reported_funding_bps_corrected']:+8.3f} rep_inv={row['reported_funding_bps_inverted']:+8.3f} "
                      f"diag={row['diagnostics_funding_bps']:+8.3f}", flush=True)

    # unit-weighted aggregation by asset x timeframe (all 4 (N,M) configs pooled)
    by_asset_tf = []
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            frames = [unit_frames[(asset, tf, N, M)] for N, M in CONFIGS]
            ud = pd.concat(frames, ignore_index=True)
            by_asset_tf.append({
                "asset": asset, "timeframe": tf,
                "n_units": int(len(ud)),
                "mean_hold_hours": round(float(ud["hold_hours"].mean()), 4),
                "mean_settlements_per_unit": round(float(ud["n_settlements"].mean()), 4),
                "mean_rate_per_settlement_bps": round(float(ud["rate_per_settlement_bps"].mean()), 4),
                "expected_funding_bps_meanfield": round(float(ud["expected_funding_bps_meanfield"].mean()), 4),
                "expected_funding_bps_settled": round(float(ud["expected_funding_bps_settled"].mean()), 4),
                "reported_funding_bps_corrected": round(float(ud["reported_funding_bps_corrected"].mean()), 4),
                "long_units_share_pct": round(float((ud["direction"] == "long").mean() * 100.0), 2),
            })

    # parity between the strategy simulator's funding_bps and the diagnostics simulator's
    par = np.array([[r["reported_funding_bps_corrected"], r["diagnostics_funding_bps"]] for r in per_config], dtype=float)
    max_abs_delta = float(np.max(np.abs(par[:, 0] - par[:, 1])))
    print(f"\nStrategy-sim vs diagnostics-sim funding_bps: max |delta| = {max_abs_delta:.6f} bps over 36 configs")

    # example: BTC 15m 20/10
    ex = [r for r in per_config if r["asset"] == "BTCUSDT" and r["timeframe"] == "15m" and r["N"] == 20 and r["M"] == 10][0]
    ex_units = unit_frames[("BTCUSDT", "15m", 20, 10)]
    ex_long = ex_units[ex_units["direction"] == "long"]
    ex_short = ex_units[ex_units["direction"] == "short"]
    example = {
        "config": "BTCUSDT 15m 20/10",
        "row": ex,
        "n_long_units": int(len(ex_long)), "n_short_units": int(len(ex_short)),
        "long_mean_rate_per_settlement_bps": round(float(ex_long["rate_per_settlement_bps"].mean()), 4) if len(ex_long) else None,
        "short_mean_rate_per_settlement_bps": round(float(ex_short["rate_per_settlement_bps"].mean()), 4) if len(ex_short) else None,
        "long_reported_funding_bps": round(float(ex_long["reported_funding_bps_corrected"].mean()), 4) if len(ex_long) else None,
        "short_reported_funding_bps": round(float(ex_short["reported_funding_bps_corrected"].mean()), 4) if len(ex_short) else None,
        "long_expected_funding_bps_meanfield": round(float(ex_long["expected_funding_bps_meanfield"].mean()), 4) if len(ex_long) else None,
        "short_expected_funding_bps_meanfield": round(float(ex_short["expected_funding_bps_meanfield"].mean()), 4) if len(ex_short) else None,
    }
    print("\nBTC 15m 20/10 detail:", json.dumps(example, indent=2)[:800])

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Per-config funding decomposition for the sign-corrected v2 artifacts. expected_funding_bps_meanfield "
                "is hold/8 x signed rate per settlement; expected_funding_bps_settled is the sum of the signed "
                "settlement rates inside the simulator's accrual window; reported_funding_bps_corrected is the "
                "simulator's mean per-unit funding_pnl/notional. reported_funding_bps_inverted is the same quantity "
                "under the retired (pre-fix) sign convention.",
        "oos_window": [OOS_START, OOS_END],
        "accrual_window": "settlement bars s with entry_time <= ts_s <= exit_time (stop exits) or < exit_time (channel exits)",
        "split_convention": "each cycle's accrued funding is divided equally across its 1-4 units",
        "max_abs_delta_strategy_vs_diagnostics_funding_bps": round(max_abs_delta, 6),
        "per_config": per_config,
        "by_asset_tf": by_asset_tf,
        "btc_15m_20_10_example": example,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    return payload


if __name__ == "__main__":
    main()

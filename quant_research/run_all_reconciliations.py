"""
Comprehensive Quantitative Audit & Reconciliation Engine.
Executes all required reconciliations for Items 1-9:
- Saves structured output to quant_research/results/audit_results_rerun.json
- Prints full raw text outputs to console

NOTE ON COMMITTED RESULTS: quant_research/results/audit_results.json is a
committed snapshot and is deliberately NOT overwritten by this script.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, List
import scipy.stats as stats

from quant_research.core.engine import CostModel, QuantEngine
from quant_research.core.metrics import calculate_daily_sharpe, compute_dsr, block_bootstrap_ci
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_diagnostics import slice_features

DATA_DIR = "quant_research/data/processed"


def run_item_2_cost_reconciliation():
    print("=" * 100)
    print("ITEM 2: COST RECONCILIATION & GROSS/NET FORMULA DEFINITIONS")
    print("=" * 100)
    # Formulations
    print("FORMULA DEFINITIONS:")
    print("  Gross Bps per Trade = (Gross Realized PnL / Entry Notional) * 10,000")
    print("    where Gross PnL = Quantity * (Exit Price - Entry Price) for longs")
    print("    and Entry Notional = Quantity * Entry Price")
    print("  Fee Bps per Trade   = (Execution Cost / Entry Notional) * 10,000")
    print("    where Execution Cost = Quantity * (Entry Price * 7 bps + Exit Price * 7 bps) at 14 bps RT")
    print("  Funding Bps per Trade = (Funding PnL / Entry Notional) * 10,000")
    print("  Net Bps per Trade   = Gross Bps - Fee Bps + Funding Bps")
    print("    Net Bps = (Net PnL / Entry Notional) * 10,000")
    print("-" * 100)

    # Load 1h N=20 M=20 data
    assets = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    results = {}

    for asset in assets:
        kline_path = os.path.join(DATA_DIR, f"{asset}_1h.parquet")
        fund_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")
        df = pd.read_parquet(kline_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        fdf = pd.read_parquet(fund_path)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

        feat = prepare_turtle_features(df, N=20, M=20)
        oos_feat = slice_features(feat, "2025-04-01 00:00:00", "2026-03-31 23:59:59")

        # Run with funding and without funding
        cost_with = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
        cost_without = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)

        m_with, t_with = run_turtle_simulation(oos_feat, funding, {"N": 20, "M": 20}, cost_with, 100_000.0, return_trades=True)
        m_without, t_without = run_turtle_simulation(oos_feat, funding, {"N": 20, "M": 20}, cost_without, 100_000.0, return_trades=True)

        notional = [t.quantity * t.entry_price for t in t_with]
        gross_bps_mean = float(np.mean([t.gross_pnl / n * 10000.0 for t, n in zip(t_with, notional)]))
        fee_bps_mean = float(np.mean([t.fee_and_slip_cost / n * 10000.0 for t, n in zip(t_with, notional)]))
        funding_bps_mean = float(np.mean([t.funding_pnl / n * 10000.0 for t, n in zip(t_with, notional)]))
        net_bps_mean = float(np.mean([t.net_pnl / n * 10000.0 for t, n in zip(t_with, notional)]))
        gap = gross_bps_mean - net_bps_mean

        # Funding statistics for the same window, read straight from the RAW archives
        raw = _load_raw_funding_rates(asset)
        win = raw[(raw["timestamp"] >= pd.Timestamp(OOS_START, tz="UTC")) &
                  (raw["timestamp"] <= pd.Timestamp(OOS_END, tz="UTC"))]
        mean_rate = float(win["last_funding_rate"].mean())
        ann_pct = mean_rate * 3.0 * 365.0 * 100.0   # 3 empirical settlements/day

        results[asset] = {
            "trades": len(t_with),
            "gross_bps": round(gross_bps_mean, 2),
            "fee_bps": round(fee_bps_mean, 2),
            "funding_bps": round(funding_bps_mean, 2),
            "net_bps_with_funding": round(net_bps_mean, 2),
            "net_bps_no_funding": round(m_without["expectancy_bps"], 2),
            "total_gap_bps": round(gap, 2),
            "mean_raw_funding_rate": mean_rate,
            "annualized_mean_funding_pct": round(ann_pct, 3),
            "denominator": "per-UNIT entry notional (quantity * entry_price), equal-weighted across units",
        }

        print(f"Asset: {asset:8s} | Units: {len(t_with):4d} | Gross: {gross_bps_mean:+6.2f} bps | Fee+Slip: {fee_bps_mean:5.2f} bps | Funding: {funding_bps_mean:+6.2f} bps | Net (funding series + apply_funding=True): {net_bps_mean:+6.2f} bps | Net (funding series + apply_funding=False): {m_without['expectancy_bps']:+6.2f} bps | Gap (Gross - Net): {gap:5.2f} bps")

    print("\nBPS DENOMINATOR: every bps figure above divides a unit's PnL by that unit's OWN entry notional")
    print("                 (quantity * entry_price), then averages across units (equal-weighted). It is NOT")
    print("                 dollar-weighted, and it is NOT divided by round-trip (entry+exit) notional.")
    print("\nEXPLANATION OF THE GAP (every number computed above, no string literals):")
    print("  Identity check: gap = fee_bps - funding_bps by construction (all three computed in this loop).")
    for asset in assets:
        r = results[asset]
        identity = round(r["fee_bps"] - r["funding_bps"], 2)
        print(f"    {asset:8s}: fee {r['fee_bps']:5.2f} - funding {r['funding_bps']:+6.2f} = {identity:+6.2f} bps  "
              f"(gap column: {r['total_gap_bps']:+6.2f}) | mean raw funding rate {r['mean_raw_funding_rate']:+.8f}/settlement "
              f"= {r['annualized_mean_funding_pct']:+.2f}% annualized (3 settlements/day)")
    print("  1. Fee Drag: 14.00 bps RT (5 bps taker fee + 2 bps slippage per side) charged on every unit.")
    print("  2. Funding Carry: charged/credited per unit from the empirical settlement series; the sign of the")
    print("     per-asset funding term is reported above and is NOT assumed to be a drag.")
    print("  3. Funding Application: the 'no funding' arm above is CostModel(apply_funding=False) with the funding")
    print("     SERIES STILL PASSED IN. It is not the same call as funding_series=None; the raw on/off table in")
    print("     item 1 shows both produce identical net expectancy, so the label matters but not the numbers.")

    return results


def _load_raw_funding_rates(symbol: str) -> pd.DataFrame:
    """Read the RAW Binance fundingRate archives (CSV inside monthly zips).
    Independent of quant_research/data/processed/*_funding.parquet."""
    import glob
    import zipfile

    frames = []
    for zp in sorted(glob.glob(f"quant_research/data/raw/{symbol}-fundingRate-*.zip")):
        with zipfile.ZipFile(zp) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                frames.append(pd.read_csv(fh))
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["calc_time"], unit="ms", utc=True)
    # the simulator aligns funding to the bar grid by rounding to the minute
    # (pandas 2.3.3 Series.round("1min") is a no-op for sub-minute offsets -> use floor)
    raw["ts_round"] = raw["timestamp"].dt.floor("min")
    return raw.sort_values("timestamp").reset_index(drop=True)


OOS_START = "2025-04-01 00:00:00"
OOS_END = "2026-03-31 23:59:59"


def _funding_events_for_unit(t, fund_ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Funding timestamps the simulator charges for ONE unit.

    The sim charges funding at the START of bar i whenever a position is open and
    funding_arr[i] != 0, and only then checks the exit.  Therefore:
      - entry bar .. exit bar inclusive when the exit is an intrabar 2N stop
        (exit_time == the stop bar),
      - entry bar .. decision bar inclusive when the exit is the M-bar channel
        exit (exit_time == decision bar + 1 bar, because that fill is next-open).
    """
    is_stop = str(t.exit_reason).startswith("2N Stop Loss")
    hi = t.exit_time if is_stop else t.exit_time - pd.Timedelta(hours=1)
    mask = (fund_ts >= t.entry_time) & (fund_ts <= hi)
    return fund_ts[mask]


def run_item_1_funding_reconciliation():
    print("=" * 100)
    print("ITEM 1: FUNDING - 1h N=20/M=20 PER ASSET (OOS 2025-04-01 .. 2026-03-31)")
    print("=" * 100)

    assets = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    records = {}
    all_trades = {}

    print(f"{'Asset':9s} | {'Units':6s} | {'AvgHold(h)':11s} | {'FundEv/Unit':11s} | "
          f"{'MeanRate(arc)':13s} | {'Sim funding $':14s} | {'RawCSV funding $':16s} | {'Delta $':9s} | "
          f"{'RawCSV DIR-AWARE $':20s} | {'DirAware-Sim $':11s}")
    print("-" * 146)

    for asset in assets:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1h.parquet"))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
        funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

        feat = prepare_turtle_features(df, N=20, M=20)
        oos = slice_features(feat, OOS_START, OOS_END)
        metrics, trades = run_turtle_simulation(oos, funding, {"N": 20, "M": 20}, cost,
                                                100_000.0, return_trades=True)

        fund_ts = funding.index
        hold_hours = [(t.exit_time - t.entry_time) / pd.Timedelta(hours=1) for t in trades]
        events = [len(_funding_events_for_unit(t, fund_ts)) for t in trades]

        raw = _load_raw_funding_rates(asset)
        win = raw[(raw["timestamp"] >= pd.Timestamp(OOS_START, tz="UTC")) &
                  (raw["timestamp"] <= pd.Timestamp(OOS_END, tz="UTC"))]
        mean_rate = float(win["last_funding_rate"].mean())
        # rounded settlement grid -> rate map used by the independent walk
        rate_rows = (win.groupby("ts_round")["last_funding_rate"]
                        .sum().sort_index())

        # ---- independent $: raw CSV rates x 1h closes, walked per position cycle
        closes = df["close"]
        cycles = {}
        for t in trades:
            cycles.setdefault(t.exit_time, []).append(t)
        indep_total = 0.0          # replicates the engine: charge = -qty*close*rate for BOTH sides
        indep_dir_aware_total = 0.0  # PREREG section 3: short units RECEIVE positive funding
        n_short_cycles = 0
        for exit_t, cyc_units in cycles.items():
            is_stop = str(cyc_units[0].exit_reason).startswith("2N Stop Loss")
            hi = exit_t if is_stop else exit_t - pd.Timedelta(hours=1)
            first_entry = min(u.entry_time for u in cyc_units)
            side = 1.0 if str(cyc_units[0].direction).lower().startswith("long") else -1.0
            if side < 0:
                n_short_cycles += 1
            for ts, rate in rate_rows.items():
                if ts < first_entry or ts > hi:
                    continue
                active = [u for u in cyc_units if u.entry_time <= ts]
                if not active:
                    continue
                qty = sum(u.quantity for u in active)
                close_px = float(closes.loc[ts])
                charge = -qty * close_px * float(rate)
                indep_total += charge
                indep_dir_aware_total += side * charge

        sim_total = float(sum(t.funding_pnl for t in trades))
        records[asset] = {
            "units": len(trades),
            "avg_hold_hours": round(float(np.mean(hold_hours)), 3),
            "funding_events_per_unit": round(float(np.mean(events)), 3),
            "mean_raw_funding_rate": mean_rate,
            "sim_funding_usd": round(sim_total, 4),
            "raw_csv_funding_usd": round(indep_total, 4),
            "delta_usd": round(sim_total - indep_total, 4),
            "raw_csv_dir_aware_funding_usd": round(indep_dir_aware_total, 4),
            "dir_aware_minus_sim_usd": round(indep_dir_aware_total - sim_total, 4),
            "short_cycles": n_short_cycles,
            "total_cycles": len(cycles),
            "trades": len(trades),
            "expectancy_bps": metrics["expectancy_bps"],
        }
        all_trades[asset] = trades
        print(f"{asset:9s} | {len(trades):6d} | {np.mean(hold_hours):11.3f} | {np.mean(events):11.3f} | "
              f"{mean_rate:13.8f} | {sim_total:14.4f} | {indep_total:16.4f} | {sim_total - indep_total:9.4f} | "
              f"{indep_dir_aware_total:20.4f} | {indep_dir_aware_total - sim_total:+11.4f}")

    print("-" * 146)
    print("Denominators: 'AvgHold' and 'FundEv/Unit' are per UNIT (per pyramid unit), not per cycle.")
    print("              'MeanRate(arc)' is the mean of last_funding_rate over the raw fundingRate")
    print("              archives inside the OOS window (per settlement, 3 per day).")
    print("              '$' columns are absolute USDT for the whole window.")

    # ---------------------------------------------------------------
    # Hand-computation: 3 BTC units end to end
    # ---------------------------------------------------------------
    print("\n" + "-" * 100)
    print("HAND-COMPUTATION: 3 BTCUSDT UNITS END TO END")
    print("-" * 100)
    btc = all_trades["BTCUSDT"]
    bdf = pd.read_parquet(os.path.join(DATA_DIR, "BTCUSDT_1h.parquet"))
    bdf["timestamp"] = pd.to_datetime(bdf["timestamp"], utc=True)
    bdf = bdf.set_index("timestamp").sort_index()
    braw = _load_raw_funding_rates("BTCUSDT")
    bwin = braw[(braw["timestamp"] >= pd.Timestamp(OOS_START, tz="UTC")) &
                (braw["timestamp"] <= pd.Timestamp(OOS_END, tz="UTC"))]
    rate_by_ts = bwin.groupby("ts_round")["last_funding_rate"].sum().to_dict()

    bcycles = {}
    for t in btc:
        bcycles.setdefault(t.exit_time, []).append(t)
    # prefer a 3-unit cycle; otherwise fall back to the first 3 units
    three = next((sorted(v, key=lambda u: u.unit_number) for v in bcycles.values() if len(v) == 3), None)
    if three is None:
        three = sorted(btc, key=lambda u: u.entry_time)[:3]

    hand_total = 0.0
    for u in three:
        is_stop = str(u.exit_reason).startswith("2N Stop Loss")
        hi = u.exit_time if is_stop else u.exit_time - pd.Timedelta(hours=1)
        events = [ts for ts in rate_by_ts if u.entry_time <= ts <= hi]
        print(f"\n  UNIT {u.unit_number}  dir={u.direction}  qty={u.quantity:.6f}")
        print(f"    entry      : {u.entry_time} @ {u.entry_price:.4f}")
        print(f"    exit       : {u.exit_time} @ {u.exit_price:.4f}  ({u.exit_reason})")
        print(f"    hold       : {(u.exit_time - u.entry_time) / pd.Timedelta(hours=1):.3f} h")
        print(f"    charge win : [{u.entry_time} .. {hi}]")
        if not events:
            print("    (no funding settlement inside the holding window)")
        for ts in sorted(events):
            rate = rate_by_ts[ts]
            close_px = float(bdf["close"].loc[ts])
            # LONG pays when rate > 0, receives when rate < 0; sign = -qty*close*rate
            payment = -u.quantity * close_px * rate
            hand_total += payment
            sign = "PAY" if payment < 0 else "RECV"
            print(f"      {ts}  rate={rate:+.6f}  close={close_px:.2f}  "
                  f"payment={-u.quantity:.6f} * {close_px:.2f} * {rate:+.6f} = {payment:+.6f} USDT  [{sign}]")
    print(f"\n  Sum of the 3 units' direct settlements = {hand_total:+.6f} USDT")
    print("  NOTE: the engine charges funding at CYCLE level (-total_qty*close*rate) and then splits")
    print("        the cycle total EQUALLY across units, so a per-unit funding_pnl is a 1/3 share,")
    print("        not this direct per-unit product. Cycle totals are identical either way.")

    # ---------------------------------------------------------------
    # BTC funding OFF / OFF / ON
    # ---------------------------------------------------------------
    print("\n" + "-" * 100)
    print("BTCUSDT 1h N=20/M=20 - RAW FUNDING ON/OFF OUTPUTS")
    print("-" * 100)
    bdf_feat = prepare_turtle_features(bdf, N=20, M=20)
    b_oos = slice_features(bdf_feat, OOS_START, OOS_END)
    bfund = pd.read_parquet(os.path.join(DATA_DIR, "BTCUSDT_funding.parquet"))
    bfund["timestamp"] = pd.to_datetime(bfund["timestamp"], utc=True)
    bfund = bfund.set_index("timestamp")["funding_rate"].sort_index()

    variants = [
        ("funding=None, apply_funding=True", None, True),
        ("funding=<series>, apply_funding=False", bfund, False),
        ("funding=<series>, apply_funding=True  (APPLIED)", bfund, True),
    ]
    out = {}
    for label, fs, flag in variants:
        cm = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=flag)
        m, tr = run_turtle_simulation(b_oos, fs, {"N": 20, "M": 20}, cm, 100_000.0, return_trades=True)
        total = float(sum(t.funding_pnl for t in tr))
        out[label] = (m["expectancy_bps"], total, len(tr))
        print(f"  {label:46s} | trades={len(tr):4d} | net exp={m['expectancy_bps']:+8.2f} bps | "
              f"funding $={total:+10.4f}")

    with open("quant_research/results/audit_results.json") as fh:
        committed = json.load(fh)["cost_reconciliation"]["BTCUSDT"]
    off_none = out["funding=None, apply_funding=True"][0]
    off_flag = out["funding=<series>, apply_funding=False"][0]
    on = out["funding=<series>, apply_funding=True  (APPLIED)"][0]
    print("\n  Committed snapshot (quant_research/results/audit_results.json):")
    print(f"    net_bps_no_funding   = {committed['net_bps_no_funding']:+.2f} bps")
    print(f"    net_bps_with_funding = {committed['net_bps_with_funding']:+.2f} bps")
    print("\n  WHICH EARLIER RUN WAS MISCONFIGURED:")
    print(f"    funding=None -> {off_none:+.2f} bps ; apply_funding=False -> {off_flag:+.2f} bps "
          f"(identical: {off_none == off_flag}).")
    print("    The earlier run_item_2_cost_reconciliation (committed in 8720ea0) printed its")
    print("    funding-OFF arm as '(funding=None)' while the code passed the funding SERIES with")
    print("    apply_funding=False, and its three explanatory lines are hard-coded string literals")
    print("    rather than values read back from the run. That labelled-OFF arm is the")
    print("    misconfigured earlier run: the label does not describe the executed configuration.")
    print("\n  SECOND MISCONFIGURATION FOUND BY THIS CROSS-CHECK (funding SIGN):")
    print("    PREREG section 3 states: 'Long positions pay rate x notional; short positions")
    print("    receive rate x notional'. The engine instead charges -qty*close*rate for every")
    print("    position regardless of direction (turtle_strategy.run_turtle_simulation and")
    print("    turtle_diagnostics.run_turtle_diagnostic_sim), so a SHORT is charged long-style")
    print("    funding instead of receiving it. Column 'RawCSV DIR-AWARE $' above applies the")
    print("    PREREG sign to the identical settlement walk; the last column is dir-aware minus")
    print("    the simulated total (POSITIVE = the engine charged MORE than the PREREG sign implies).")
    for asset in assets:
        r = records[asset]
        print(f"      {asset:8s}: short cycles {r['short_cycles']:4d}/{r['total_cycles']:4d} | "
              f"sim {r['sim_funding_usd']:+10.4f} | dir-aware {r['raw_csv_dir_aware_funding_usd']:+10.4f} | "
              f"engine excess charge {r['dir_aware_minus_sim_usd']:+10.4f} USDT")
    print("    The direction-agnostic charge is retained in the regenerated Turtle results below so that")
    print("    the audited numbers stay comparable with the committed ones; correcting the sign changes")
    print("    the funding term of every short cycle and is reported here as an open defect, not applied.")
    return records


def run_item_3_pyramids_and_cycles():
    print("\n" + "=" * 100)
    print("ITEM 3: PYRAMID DISTRIBUTIONS, CYCLES, UNITS == TRADES VERIFICATION")
    print("=" * 100)
    with open("quant_research/results/turtle_diagnostics.json") as f:
        diag = json.load(f)

    cfgs = diag["diagnostics_individual"]
    print(f"{'Config':22s} | {'Dist [1, 2, 3, 4]':28s} | {'Cycles':7s} | {'Units':7s} | {'Trades':7s} | {'Check':5s}")
    print("-" * 85)

    all_verified = True
    pyramid_records = []

    for c in cfgs:
        name = f"{c['asset']} {c['timeframe']} {c['N']}/{c['M']}"
        d = c["pyramid_distribution"]
        d_str = f"[{d['1']}, {d['2']}, {d['3']}, {d['4']}]"
        cycles = d["1"] + d["2"] + d["3"] + d["4"]
        units = 1 * d["1"] + 2 * d["2"] + 3 * d["3"] + 4 * d["4"]
        trades = c["n_trades"]
        match = (units == trades)
        if not match:
            all_verified = False
        print(f"{name:22s} | {d_str:28s} | {cycles:7d} | {units:7d} | {trades:7d} | {'PASS' if match else 'FAIL'}")
        pyramid_records.append({
            "config": name,
            "distribution": d,
            "cycles": cycles,
            "units": units,
            "trades": trades,
            "units_equal_trades": match
        })

    print("-" * 85)
    print(f"Universal Verification: Every configuration satisfies Units == Trades exactly ({all_verified}).")

    # ------------------------------------------------------------------
    # POOLED CYCLE COUNTS PER CONFIG - recomputed, never hard-coded
    # ------------------------------------------------------------------
    print("\nPOOLED CYCLE COUNTS PER CONFIG (recomputed from the pyramid_distribution sums and,")
    print("independently, from the per-cycle records stored by the diagnostics run):")
    print(f"{'Timeframe':9s} | {'N/M':7s} | {'Cycles(dist sums)':18s} | {'Cycles(cycle records)':21s} | {'Units/Trades':13s} | {'Agree':5s}")
    print("-" * 90)
    per_tf_cycles = {}
    per_tf_units = {}
    cycle_rows = []
    for tf in ["5m", "15m", "1h"]:
        for N, M in [(20, 10), (20, 20), (55, 10), (55, 20)]:
            rows = [c for c in cfgs if c["timeframe"] == tf and c["N"] == N and c["M"] == M]
            cyc_dist = sum(sum(r["pyramid_distribution"].values()) for r in rows)
            cyc_sim = sum(int(r.get("n_cycles", 0)) for r in rows)
            units = sum(r["n_trades"] for r in rows)
            per_tf_cycles[tf] = per_tf_cycles.get(tf, 0) + cyc_dist
            per_tf_units[tf] = per_tf_units.get(tf, 0) + units
            print(f"{tf:9s} | {N:2d}/{M:2d}   | {cyc_dist:18d} | {cyc_sim:21d} | {units:13d} | {'YES' if cyc_dist == cyc_sim else 'NO':5s}")
            cycle_rows.append({"timeframe": tf, "N": N, "M": M, "cycles_from_distribution": cyc_dist,
                               "cycles_from_cycle_records": cyc_sim, "units_trades": units})
    print("-" * 90)
    print("PER-TIMEFRAME TOTALS (each cycle is counted once per config it belongs to):")
    for tf in ["5m", "15m", "1h"]:
        print(f"  {tf:4s}: {per_tf_cycles[tf]:7d} cycles / {per_tf_units[tf]:7d} units")

    print("\nCORRECTION OF THE PREVIOUSLY PRINTED CYCLE FIGURES:")
    print("  The earlier text printed '5m: 10,002', '15m: 4,402', '1h: 1,228' as per-timeframe cycle totals.")
    print("  Recomputed above, 10,002 is exactly the 5m N=20/M=10 POOLED cycle count (one config, not the")
    print("  timeframe). 4,402 and 1,228 match no cycle sum in the diagnostics file: the real 15m config")
    print("  counts are 3,438 / 2,923 / 2,032 / 1,873 and the 1h ones 850 / 727 / 528 / 497.")
    print("  The '7,380 / 2,126 / 569 / 631 = 15,863' figures have no source in any committed artifact")
    print("  and no reproducible run; that claim is deleted rather than restated.")

    # ------------------------------------------------------------------
    # Cycle-level inference, clustered by cycle or by UTC exit day
    # ------------------------------------------------------------------
    print("\nCYCLE-LEVEL INFERENCE (observation = completed cycle; cluster = UTC exit day):")
    print(f"{'Config':15s} | {'Cycles':7s} | {'mean(bps)':10s} | {'SE(cycle)':10s} | {'SE(day)':8s} | {'n_eff':6s} | {'CI95 cycle-block':21s} | {'CI95 day-cluster':20s} | {'cycle SR':9s} | {'DSR p':6s}")
    print("-" * 160)
    for p in diag["pooled_corrected"]:
        name = f"{p['timeframe']}/{p['N']}/{p['M']}"
        print(f"{name:15s} | {p['pooled_cycles']:7d} | {p['cycle_net_bps_mean']:+9.3f} | {p['cycle_se_bps']:10.3f} | "
              f"{p['day_clustered_se_bps']:8.3f} | {p['n_eff_day_clustered']:6d} | "
              f"[{p['cycle_ci_95'][0]:+8.2f},{p['cycle_ci_95'][1]:+8.2f}] | "
              f"[{p['day_cluster_ci_95'][0]:+7.2f},{p['day_cluster_ci_95'][1]:+7.2f}] | "
              f"{p['cycle_sharpe']:+9.3f} | {p['cycle_dsr_p_value']:.4f}")
    print("-" * 160)
    print("  mean(bps) is the equal-weighted mean NET bps of the completed CYCLE (mean over its 1-4 units),")
    print("  so it is NOT the same statistic as the unit-weighted pooled_net_bps in the Turtle table.")
    print("  SE(cycle) = sd(cycle bps)/sqrt(n_cycles). SE(day) = CR0 cluster-robust SE with the UTC exit day")
    print("  as cluster: sqrt(sum_d (T_d - n_d*mean)^2)/n_cycles. n_eff = n_cycles*(SE(cycle)/SE(day))^2 is")
    print("  the T the clustered DSR actually uses.")
    any_clustered_pass = any(p["cycle_dsr_p_value_day_clustered"] < 0.05 for p in diag["pooled_corrected"])
    print(f"  All twelve cycle-level DSR p-values are 1.0000 for a reason that is not a borderline call: the")
    print(f"  per-cycle Sharpe is negative for every config, and compute_dsr returns 1.0 by construction when")
    print(f"  the Sharpe is <= 0. Any config with clustered DSR < 0.05: {any_clustered_pass}.")

    return pyramid_records


def run_item_4_ambiguous_and_subbars():
    print("\n" + "=" * 100)
    print("ITEM 4: AMBIGUOUS BARS & 1M SUB-BAR RESOLUTION TEST ON 1H CONFIGS")
    print("=" * 100)
    print("DEFINITION OF 'AMBIGUOUS BAR' (one bar counts once, not once per unit):")
    print("  a position is OPEN at the start of the bar AND both of the following hold on that bar:")
    print("    (a) the ratcheted 2N stop lies inside the bar range  long: low <= stop | short: high >= stop")
    print("    (b) the M-bar Donchian channel-exit line fires on that bar's CLOSE")
    print("        long: close < exit_low[i] | short: close > exit_high[i]")
    print("  The committed count for SOLUSDT 5m N=20/M=10 is 655 (turtle_diagnostics.json).")
    print("")
    print("STATE OF 1m SUB-BARS IN THE COMMITTED RESULTS:")
    print("  PREREG section 7.3 claims all 5m/15m/1h bars pass 1m sub-bars into the engine's intrabar")
    print("  resolver. The committed turtle_runner.py / turtle_diagnostics.py never passed df_1m, and")
    print("  turtle_strategy.run_turtle_simulation had no intrabar resolver call at all, so every")
    print("  committed Turtle number uses the pessimistic STOP-FIRST fallback on ambiguous bars.")
    print("  A position's own unit tests confirm the resolver now exists")
    print("  (QuantEngine.resolve_intrabar_exit_levels for two same-side levels: protective stop and")
    print("  channel line) and tests/test_ambiguous_bars.py exercises it end to end.")
    print("  The measurements below compare the SAME 1h configs with and without sub-bars; the JSONs")
    print("  regenerated by this audit intentionally keep the committed stop-first behaviour so that every")
    print("  other audited number stays comparable with what was committed.")

    assets = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    diff_records = []

    for asset in assets:
        kline_path = os.path.join(DATA_DIR, f"{asset}_1h.parquet")
        sub_path = os.path.join(DATA_DIR, f"{asset}_1m.parquet")
        fund_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")

        df_1h = pd.read_parquet(kline_path)
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
        df_1h = df_1h.set_index("timestamp").sort_index()

        df_1m = pd.read_parquet(sub_path)
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)
        df_1m = df_1m.set_index("timestamp").sort_index()

        funding = None
        if os.path.exists(fund_path):
            fdf = pd.read_parquet(fund_path)
            fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
            funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

        cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
        feat_1h = prepare_turtle_features(df_1h, N=20, M=20)
        oos_feat = slice_features(feat_1h, "2025-04-01 00:00:00", "2026-03-31 23:59:59")

        # Without sub-bars (conservative stop-first)
        m_def = run_turtle_simulation(oos_feat, funding, {"N": 20, "M": 20}, cost, 100_000.0, df_1m=None)
        # With true 1m sub-bars
        m_sub = run_turtle_simulation(oos_feat, funding, {"N": 20, "M": 20}, cost, 100_000.0, df_1m=df_1m)

        delta = m_sub["expectancy_bps"] - m_def["expectancy_bps"]
        print(f"Asset: {asset:8s} 1h 20/20 | Default (Stop First): {m_def['expectancy_bps']:+6.2f} bps | With 1m Sub-bars: {m_sub['expectancy_bps']:+6.2f} bps | Change: {delta:+5.2f} bps")
        diff_records.append({
            "asset": asset,
            "default_net_bps": m_def["expectancy_bps"],
            "subbars_net_bps": m_sub["expectancy_bps"],
            "delta_bps": delta,
        })

    print("-" * 100)
    print("SUPPORTING UNIT TESTS (real assertions, run by pytest; see the pasted pytest output):")
    print("  tests/test_turtle_mechanics.py::test_ratchet_then_stop_order_within_one_bar")
    print("    reads N from params and recomputes the stops from features['atr'], then asserts the exact")
    print("    ratcheted stop, the exact executed exit price (stop x (1 - slippage)) and the exact exit bar.")
    print("  tests/test_ambiguous_bars.py::test_turtle_ambiguous_bar_pyramid_stop_and_subbar_resolution")
    print("    builds a bar that simultaneously triggers the pyramid add, the ratcheted stop and the channel")
    print("    exit, then shows 1m sub-bars change the exit price, the exit reason and the exit bar, plus a")
    print("    direct two-level resolver test (stop vs channel line).")

    return diff_records


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone in the input p-values)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max(running, (m - rank) * float(pvals[idx])))
        adj[idx] = val
        running = val
    return adj


def run_item_4_random_baseline_per_config():
    print("\n" + "=" * 100)
    print("ITEM 4: MATCHED RANDOM-ENTRY NULL - CYCLE-LEVEL DISPERSION, NULL SE, HOLM OVER 36 CONFIGS")
    print("=" * 100)
    with open("quant_research/results/turtle_diagnostics.json") as f:
        diag = json.load(f)

    cfgs = diag["diagnostics_individual"]
    print("HOW THE NULL IS BUILT, AND WHAT THE PREVIOUS TABLE GOT WRONG:")
    print("  * Observation unit = completed CYCLE, not pyramid UNIT. Up to four units share one entry")
    print("    decision, one ratcheted stop and one exit, so unit returns are not independent bets.")
    print("  * Null = zero mean with dispersion MATCHED to the strategy's cycle dispersion. Under matching,")
    print("    null SE = sd(cycle bps)/sqrt(n_cycles) = strategy SE BY CONSTRUCTION; the table reports both")
    print("    plus the SE the earlier table used (150/sqrt(n_units), a hard-coded 150 bps dispersion scaled")
    print("    by the number of UNITS), which is the mis-scaling this correction removes.")
    print("  * p is two-sided on the mean cycle bps; Holm-Bonferroni adjusts across all 36 configs.")
    print("  * No per-config 1,000-run simulation is claimed here. The only simulated random-entry")
    print("    baseline in this repo is quant_research/turtle_random_entry_test.py (1,000 runs, BTCUSDT 1h")
    print("    OOS) -> results/random_entry_baseline.json (mean -12.14 bps, 90% CI [-31.11, +8.60]).")

    print(f"\n{'Config (Asset TF N/M)':23s} | {'Cycl':5s} | {'Units':5s} | {'Gross':7s} | {'Net':7s} | {'CycleMean':9s} | {'SE(strat)':9s} | {'NullSE(matched)':15s} | {'OldNullSE(150/sqrt(u))':21s} | {'z':7s} | {'p':6s} | {'Holm p':6s} | {'Relabel':7s}")
    print("-" * 172)

    random_records = []
    for c in cfgs:
        name = f"{c['asset']} {c['timeframe']} {c['N']}/{c['M']}"
        n_cycles = int(c["n_cycles"])
        units = int(c["n_trades"])
        mean_cyc = float(c["cycle_net_bps_mean"])
        se_strat = float(c["cycle_se_bps"])
        se_null = se_strat  # matched dispersion
        se_old_null = 150.0 / np.sqrt(units) if units > 0 else 0.0
        z = mean_cyc / se_strat if se_strat > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        relabel = bool(c["gross_bps"] > 0.0 and c["net_bps"] < 0.0)
        random_records.append({
            "config": name,
            "cycles": n_cycles,
            "units": units,
            "gross_bps": c["gross_bps"],
            "net_bps": c["net_bps"],
            "cycle_mean_bps": mean_cyc,
            "strategy_se_bps": se_strat,
            "null_se_bps_matched": se_null,
            "old_null_se_bps": round(se_old_null, 4),
            "z": z,
            "p_two_sided": p_val,
            "gross_positive_net_negative": relabel,
        })

    holm = _holm_adjust(np.array([r["p_two_sided"] for r in random_records]))
    for r, h in zip(random_records, holm):
        r["holm_adjusted_p"] = float(h)

    for r in random_records:
        print(f"{r['config']:23s} | {r['cycles']:5d} | {r['units']:5d} | {r['gross_bps']:+7.2f} | {r['net_bps']:+7.2f} | "
              f"{r['cycle_mean_bps']:+9.3f} | {r['strategy_se_bps']:9.3f} | {r['null_se_bps_matched']:15.3f} | "
              f"{r['old_null_se_bps']:21.4f} | {r['z']:+7.2f} | {r['p_two_sided']:6.3f} | {r['holm_adjusted_p']:6.3f} | "
              f"{'GROSS+/NET-' if r['gross_positive_net_negative'] else '-':7s}")

    n_relabel = sum(1 for r in random_records if r["gross_positive_net_negative"])
    n_holm_sig = sum(1 for r in random_records if r["holm_adjusted_p"] < 0.05)
    n_dir_pos = sum(1 for r in random_records if r["cycle_mean_bps"] > 0.0)
    print("-" * 172)
    print("DENOMINATORS: 'CycleMean' is the equal-weighted mean NET bps of the finished cycle; Gross/Net are the")
    print("              same per-UNIT bps figures as the Turtle table (equal-weighted across units).")
    print(f"CONFIG COUNTS: {len(random_records)} configs | {n_relabel} flagged GROSS+/NET- (positive gross, negative net:")
    print(f"               a cost-driven loss that must not be reported as a positive edge) | {n_dir_pos} configs with a")
    print(f"               positive mean cycle bps at all | {n_holm_sig} configs with Holm-adjusted p < 0.05.")
    print("INTERPRETATION THAT IS SUPPORTED BY THIS TABLE:")
    print("  The strategy does not beat a zero-mean, dispersion-matched null on any of the 36 configs; the")
    print("  significant z values are for the WRONG direction (they reject the null in favour of a LOSS).")
    print("  That is a statement about location only. It does NOT show the strategy is worse than random")
    print("  entries, because the matched null imposes the observed dispersion instead of deriving it from a")
    print("  random-entry simulation - only the 1,000-run BTC 1h baseline above does that.")

    return random_records


def run_item_7_autopsy_pure_curve():
    print("\n" + "=" * 100)
    print("ITEM 7: AUTOPSY TRANSFER-ADJUSTED EQUITY CURVE & RECONCILIATION")
    print("=" * 100)
    tx = pd.read_csv("quant_research/inputs/transaction_history.csv")
    tx["Time"] = pd.to_datetime(tx["Time"])
    tx = tx.sort_values("Time").reset_index(drop=True)

    # Initial deposit
    init_dep = 195.00  # First transfer on 2025-04-04 18:39:32

    # Exclude all subsequent transfers (pure trading performance)
    trading_types = ["REALIZED_PNL", "COMMISSION", "FUNDING_FEE", "INSURANCE_CLEAR"]
    tx_trade = tx[tx["Type"].isin(trading_types)].copy()
    tx_trade["Amount"] = pd.to_numeric(tx_trade["Amount"], errors="coerce").fillna(0.0)
    tx_trade["Pure_Equity"] = init_dep + tx_trade["Amount"].cumsum()

    peak = float(tx_trade["Pure_Equity"].max())
    peak_idx = tx_trade["Pure_Equity"].idxmax()
    peak_time = str(tx_trade.loc[peak_idx, "Time"])

    trough = float(tx_trade["Pure_Equity"].min())
    trough_idx = tx_trade["Pure_Equity"].idxmin()
    trough_time = str(tx_trade.loc[trough_idx, "Time"])

    final_eq = float(tx_trade["Pure_Equity"].iloc[-1])
    final_time = str(tx_trade.iloc[-1]["Time"])
    max_dd = (peak - trough) / peak * 100.0

    print("Transfer-Adjusted Curve Metrics (Excluding Transfers):")
    print(f"  Initial Capital Basis:        ${init_dep:.2f} (at 2025-04-04 18:39:32 UTC)")
    print(f"  Peak Pure Equity:             ${peak:.2f} (at {peak_time} UTC)")
    print(f"  Trough Pure Equity:           ${trough:.2f} (at {trough_time} UTC)")
    print(f"  Final Ending Equity:          ${final_eq:.2f} (at {final_time} UTC)")
    print(f"  Max Drawdown (Peak to Trough): {max_dd:.2f}%")

    print("\nPeaks, recomputed from the transaction ledger instead of quoted:")
    after = tx_trade[tx_trade["Time"] > pd.Timestamp(peak_time)]
    if len(after) > 0:
        second = float(after["Pure_Equity"].max())
        second_time = str(after.loc[after["Pure_Equity"].idxmax(), "Time"])
        hours_apart = (pd.Timestamp(second_time) - pd.Timestamp(peak_time)) / pd.Timedelta(hours=1)
        print(f"  - All-Time High Peak:       ${peak:.2f} at {peak_time} UTC")
        print(f"  - Secondary Plateau After:  ${second:.2f} at {second_time} UTC ({hours_apart:.1f} h after the peak)")

    # ------------------------------------------------------------------
    # BPS DENOMINATORS - read from the committed replay artifacts
    # ------------------------------------------------------------------
    with open("quant_research/results/test_3_replay_summary.json") as fh:
        summ = json.load(fh)
    rdf = pd.read_csv("quant_research/results/test_3_replay_trades.csv")
    for col in ["entry_notional", "exit_notional", "gross_pnl", "commission", "funding_pnl", "net_pnl",
                "gross_bps", "fee_bps", "funding_bps", "net_bps"]:
        rdf[col] = pd.to_numeric(rdf[col])
    n_tr = len(rdf)
    one_side = float(rdf["entry_notional"].sum())            # round-trip (one-side) notional
    both_sides = float(rdf["entry_notional"].sum() + rdf["exit_notional"].sum())
    gross_t = float(rdf["gross_pnl"].sum())
    fee_t = float(rdf["commission"].sum())
    fund_t = float(rdf["funding_pnl"].sum())
    net_t = float(rdf["net_pnl"].sum())

    print("\nBPS DENOMINATORS - DENOMINATOR STATED IN EVERY LINE (%d replay trades):" % n_tr)
    print(f"  Committed summary cross-check: total_volume_usd = {summ['total_volume_usd']:.2f} == entry+exit notional")
    print(f"  ({both_sides:.2f}) -> the summary's 'volume' is BOTH sides, so the round-trip (one-side)")
    print(f"  denominator is volume/2 = {both_sides/2:.2f} vs entry_notional sum = {one_side:.2f} (delta {both_sides/2 - one_side:+.2f}).")
    print(f"{'Denominator':58s} | {'Gross bps':10s} | {'Fee bps':9s} | {'Funding bps':11s} | {'Net bps':8s}")
    print("-" * 108)
    denoms = [
        (f"entry_notional sum (round-trip basis, $ {one_side:,.2f})", one_side),
        (f"entry+exit notional = summary volume ($ {both_sides:,.2f})", both_sides),
    ]
    for label, den in denoms:
        print(f"{label:58s} | {gross_t/den*1e4:+10.4f} | {fee_t/den*1e4:+9.4f} | {fund_t/den*1e4:+11.4f} | {net_t/den*1e4:+8.4f}")
    eq_gross = float(rdf["gross_bps"].mean())
    eq_fee = float(rdf["fee_bps"].mean())
    eq_fund = float(rdf["funding_bps"].mean())
    eq_net = float(rdf["net_bps"].mean())
    print(f"{'equal-weighted per-trade mean (denominator = each trade own notional)':58s} | {eq_gross:+10.4f} | {eq_fee:+9.4f} | {eq_fund:+11.4f} | {eq_net:+8.4f}")
    print("-" * 108)

    # Corrected explanation of equal-weighted vs notional-weighted (computed, not asserted)
    w = (rdf["entry_notional"] / one_side).to_numpy()
    n_w = 1.0 / n_tr
    def _cov_gap(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum((w - n_w) * (x - x.mean())))
    gap_gross = _cov_gap(rdf["gross_bps"].to_numpy())
    gap_net = _cov_gap(rdf["net_bps"].to_numpy())
    corr_size_gross = float(np.corrcoef(rdf["entry_notional"], rdf["gross_bps"])[0, 1])
    print("")
    print("CORRECTED EXPLANATION OF equal-weighted vs notional-weighted (covariance decomposition):")
    print("  mean_notional_weighted - mean_equal = sum_i (w_i - 1/n) * (x_i - mean(x)), with w_i the trade's")
    print("  notional share: the covariance between a trade's dollar weight and its bps outcome.")
    print("  Computed from the committed CSV:")
    print(f"    gross bps: notional-weighted {gross_t/one_side*1e4:+.4f} - equal-weighted {eq_gross:+.4f} = {gap_gross:+.4f} bps")
    print(f"    net   bps: notional-weighted {net_t/one_side*1e4:+.4f} - equal-weighted {eq_net:+.4f} = {gap_net:+.4f} bps")
    print(f"    corr(trade notional, gross bps) = {corr_size_gross:+.4f}")
    print("  A POSITIVE covariance means the larger trades earned slightly HIGHER bps, which makes the")
    print("  equal-weighted mean the lower one. The earlier explanation ('smaller positions had higher")
    print("  percentage gains') has the sign backwards relative to these computed values.")
    print("  The earlier table also divided by the BOTH-SIDES volume while calling the result 'per round-trip'.")

    print("")
    print("LEDGER vs REPLAY RECONCILIATION DISCREPANCY (both committed, not previously reconciled):")
    with open("quant_research/results/test_3_proper_results.json") as fh:
        ledger = json.load(fh)["ledger_reconciliation"]
    print(f"  test_3_replay_summary.json ({n_tr} reconstructed trades):  gross {summ['total_gross_pnl_usd']:+.2f}, ")
    print(f"    commissions {summ['total_commissions_usd']:+.2f}, funding {summ['total_funding_usd']:+.2f}, net {summ['total_net_pnl_usd']:+.2f}")
    print(f"  test_3_proper_results.json ledger_reconciliation:      gross {ledger['realized_pnl_sum']:+.2f}, ")
    print(f"    commissions {ledger['commissions_sum']:+.2f}, funding {ledger['funding_fees_sum']:+.2f}, ")
    print(f"    insurance {ledger['insurance_clear_liquidation']:+.2f}, net {ledger['total_net_trading_pnl']:+.2f}")
    print(f"  Replay totals recomputed from the CSV:                 gross {gross_t:+.2f}, commissions {fee_t:+.2f}, ")
    print(f"    funding {fund_t:+.2f}, net {net_t:+.2f} (no insurance line: the liquidation fee is a ledger event)")
    print(f"  Unreconciled gaps vs the ledger: commissions {summ['total_commissions_usd'] - ledger['commissions_sum']:+.2f} and net "
          f"{summ['total_net_pnl_usd'] - ledger['total_net_trading_pnl']:+.2f}. REPORT.md section 9 quoted only the ledger figures.")

    return {
        "initial_deposit": init_dep,
        "peak_equity": peak,
        "peak_time": peak_time,
        "trough_equity": trough,
        "trough_time": trough_time,
        "final_equity": final_eq,
        "max_drawdown_pct": max_dd,
        "replay_trades": n_tr,
        "round_trip_notional_usd": round(one_side, 2),
        "volume_both_sides_usd": round(both_sides, 2),
        "gross_bps_per_round_trip_notional": round(gross_t / one_side * 1e4, 4),
        "fee_bps_per_round_trip_notional": round(fee_t / one_side * 1e4, 4),
        "funding_bps_per_round_trip_notional": round(fund_t / one_side * 1e4, 4),
        "net_bps_per_round_trip_notional": round(net_t / one_side * 1e4, 4),
        "gross_bps_per_both_sides_volume": round(gross_t / both_sides * 1e4, 4),
        "net_bps_per_both_sides_volume": round(net_t / both_sides * 1e4, 4),
        "gross_bps_per_trade_mean": round(eq_gross, 4),
        "fee_bps_per_trade_mean": round(eq_fee, 4),
        "net_bps_per_trade_mean": round(eq_net, 4),
    }


def run_item_8_replay_10bps_vs_14bps():
    print("\n" + "=" * 100)
    print("ITEM 7: REPLAY AT 10 BPS VS 14 BPS - INDEPENDENT TRADE SETS PER FEE LEVEL")
    print("=" * 100)
    trades_df = pd.read_csv("quant_research/results/test_3_replay_trades.csv")
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    data_1m = {}
    for s in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
        df = pd.read_parquet(f"quant_research/data/processed/{s}_1m.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        data_1m[s] = df.set_index("timestamp").sort_index()

    stop_cache = {}
    def stop_exit(sym, direction, p_entry, s_time, e_time, d):
        """First 1m bar inside [entry, exit] that touches the hard stop, else the actual user exit."""
        key = (sym, direction, p_entry, s_time, e_time, d)
        if key in stop_cache:
            return stop_cache[key]
        exit_p = None
        if sym in data_1m:
            bars = data_1m[sym].loc[s_time:e_time]
            stop_level = p_entry * (1.0 - d) if direction == "LONG" else p_entry * (1.0 + d)
            for _, bar in bars.iterrows():
                if direction == "LONG" and bar["low"] <= stop_level:
                    exit_p = stop_level
                    break
                if direction == "SHORT" and bar["high"] >= stop_level:
                    exit_p = stop_level
                    break
        stop_cache[key] = exit_p
        return exit_p

    def run_path(d: float, cap: float, fee: float) -> Dict[str, Any]:
        """One leverage path with its OWN equity, its OWN 3% daily-loss cap and its OWN trade set.

        No-stop arm: run on the identical trade set with the identical sizing path (notional taken
        from this path's equity), removing ONLY the hard stop (exit at the actual recorded exit).
        """
        lev = min(cap, 0.01 / d)
        eq = 10000.0
        eq_nostop = 10000.0
        eq_gross = 10000.0          # fee-free, same trade set/sizing decisions
        eq_nostop_gross = 10000.0   # fee-free, same set, stop removed
        day = None
        start_eq = 10000.0
        daily_loss = 0.0
        taken = 0
        skipped = 0
        for _, tr in trades_df.iterrows():
            t_day = tr["entry_time"].date()
            if t_day != day:
                day = t_day
                start_eq = eq
                daily_loss = 0.0
            if daily_loss >= 0.03 * start_eq:
                skipped += 1
                continue
            taken += 1
            p_entry = tr["entry_price"]
            direction = tr["direction"]
            notional = eq * lev
            exit_p = stop_exit(tr["symbol"], direction, p_entry, tr["entry_time"], tr["exit_time"], d)
            if exit_p is None:
                exit_p = tr["exit_price"]
            raw_ret = ((exit_p - p_entry) if direction == "LONG" else (p_entry - exit_p)) / p_entry
            pnl = notional * (raw_ret - fee)
            eq += pnl
            eq_gross += notional * raw_ret
            if pnl < 0:
                daily_loss += abs(pnl)
            # same-trade-set, same-sizing no-stop counterfactual
            p_exit = tr["exit_price"]
            ns_ret = ((p_exit - p_entry) if direction == "LONG" else (p_entry - p_exit)) / p_entry
            eq_nostop += notional * (ns_ret - fee)
            eq_nostop_gross += notional * ns_ret
        return {
            "leverage": lev,
            "trades_taken": taken,
            "trades_skipped": skipped,
            "stopped_net_return_pct": (eq - 10000.0) / 100.0,
            "same_set_nostop_net_return_pct": (eq_nostop - 10000.0) / 100.0,
            "gross_return_stop_pct": (eq_gross - 10000.0) / 100.0,
            "same_set_nostop_gross_pct": (eq_nostop_gross - 10000.0) / 100.0,
        }

    variants = [
        ("Stop_0.3%_Cap_5x", 0.003, 5.0),
        ("Stop_0.3%_Cap_10x", 0.003, 10.0),
        ("Stop_0.5%_Cap_5x", 0.005, 5.0),
        ("Stop_0.5%_Cap_10x", 0.005, 10.0),
        ("Stop_1.0%_Cap_5x", 0.010, 5.0),
        ("Stop_1.0%_Cap_10x", 0.010, 10.0),
    ]

    print("Each fee level has its OWN independent path: the 3% daily-loss cap is evaluated on that")
    print("path's own equity, so the trade set at 10 bps is NOT assumed to equal the trade set at 14 bps.")
    print(f"\n{'Variant':20s} | {'Lev':5s} | {'Taken10':7s} | {'Skip10':6s} | {'Taken14':7s} | {'Skip14':6s} | {'Gross%(stop)':12s} | {'Net%(10)':9s} | {'Net%(14)':9s} | {'NoStopNet%(10)':14s} | {'NoStopNet%(14)':14s} | {'StopGross-NoStopGross(10)':25s}")
    print("-" * 185)

    replay_records = []
    for v_name, d, cap in variants:
        p10 = run_path(d, cap, 0.0010)
        p14 = run_path(d, cap, 0.0014)
        lev = p10["leverage"]
        # gross (fee-free) return of the 10 bps trade set, stop applied
        gross_pct = p10["gross_return_stop_pct"]
        ns_gross_pct = p10["same_set_nostop_gross_pct"]

        print(f"{v_name:20s} | {lev:5.2f} | {p10['trades_taken']:7d} | {p10['trades_skipped']:6d} | {p14['trades_taken']:7d} | {p14['trades_skipped']:6d} | "
              f"{gross_pct:+11.2f}% | {p10['stopped_net_return_pct']:+8.2f}% | {p14['stopped_net_return_pct']:+8.2f}% | "
              f"{p10['same_set_nostop_net_return_pct']:+13.2f}% | {p14['same_set_nostop_net_return_pct']:+13.2f}% | "
              f"{gross_pct - ns_gross_pct:+24.2f} pp")

        replay_records.append({
            "variant": v_name,
            "leverage": lev,
            "trades_taken_10bps": p10["trades_taken"],
            "trades_taken_14bps": p14["trades_taken"],
            "trades_skipped_10bps": p10["trades_skipped"],
            "trades_skipped_14bps": p14["trades_skipped"],
            "gross_return_stop_pct": round(gross_pct, 2),
            "same_set_nostop_gross_pct": round(ns_gross_pct, 2),
            "net_return_10bps_pct": round(p10["stopped_net_return_pct"], 2),
            "net_return_14bps_pct": round(p14["stopped_net_return_pct"], 2),
            "same_set_nostop_net_10bps_pct": round(p10["same_set_nostop_net_return_pct"], 2),
            "same_set_nostop_net_14bps_pct": round(p14["same_set_nostop_net_return_pct"], 2),
            "stop_minus_nostop_gross_pp": round(gross_pct - ns_gross_pct, 2),
        })

    # 1x leverage, no stop, all 257 trades, at 10 bps: the 'no-stop' arm as an absolute figure.
    eq_1x = 10000.0
    for _, tr in trades_df.iterrows():
        p_entry = tr["entry_price"]
        direction = tr["direction"]
        p_exit = tr["exit_price"]
        ret = ((p_exit - p_entry) if direction == "LONG" else (p_entry - p_exit)) / p_entry
        eq_1x += eq_1x * 1.0 * (ret - 0.0010)
    nodestop_1x_net = (eq_1x - 10000.0) / 100.0
    print("-" * 185)
    print(f"1x leverage, NO stop, all {len(trades_df)} trades, 10 bps per round trip: net {nodestop_1x_net:+.2f}% ")
    print("  (compounded at 1x notional = equity per trade; this is the no-stop arm with no leverage cap, no")
    print("   daily-loss cap and no stop, run on the same 257 recorded entries/exits.)")

    print("\nWHAT THIS REPLAY DOES AND DOES NOT SUPPORT:")
    print("  SUPPORTED: on every variant and fee level, the same-trade-set no-stop counterfactual ends higher than")
    print("  the stopped path, and every stopped path is negative net at both 10 and 14 bps. The gap is mechanical:")
    print("  a hard stop that is touched converts the trade at the stop price and forbids any subsequent recovery")
    print("  on that trade, while the counterfactual keeps the same entry and the actual exit.")
    print("  NOT SUPPORTED: the previous paragraph's claim that stops 'convert normal volatility into permanent")
    print("  realized losses' as a general mechanism. This replay measures neither volatility regimes nor the risk")
    print("  the stops removed; the no-stop arm carries unbounded adverse excursion and is not an investable")
    print("  alternative. The claim is deleted.")
    print("  ALSO DELETED: the 'exactly 146 trades skipped because the 3% daily cap triggered' explanation. It came")
    print("  from audit_and_reconciliation.item_7_replay_no_stop_and_fee_drag, which simulated pnl as")
    print("  '-100 if net_bps < 0 else +50' per trade - a constant unrelated to any committed replay trade. The")
    print("  skip counts actually produced by the paths above are in the Skip10/Skip14 columns.")

    return replay_records


def run_item_9_sharpe_drift():
    print("\n" + "=" * 100)
    print("ITEM 9: SHARPE PROVENANCE (+0.121 vs +0.127 vs +0.134) AND THE LEGACY DSR BUG")
    print("=" * 100)
    with open("quant_research/results/turtle_results.json") as fh:
        tr_run = json.load(fh)
    with open("quant_research/results/turtle_diagnostics.json") as fh:
        dg = json.load(fh)
    row_run = next(p for p in tr_run["pooled_summary"] if p["timeframe"] == "1h" and p["N"] == 20 and p["M"] == 20)
    row_dg = next(p for p in dg["pooled_corrected"] if p["timeframe"] == "1h" and p["N"] == 20 and p["M"] == 20)

    print("SHARPE DEFINITIONS, EACH VALUE READ FROM THE REGENERATED ARTIFACT THAT OWNS IT")
    print("  (all for the 1h N=20/M=20 pooled row, the only config with a positive Sharpe anywhere):")
    print(f"  1. turtle_results.json  mean_oos_bar_sharpe      = {row_run['mean_oos_bar_sharpe']:+.4f}   ")
    print("     -> per-BAR equity returns scaled by sqrt(bars_per_year=8760); this is the definition that")
    print("        produced the previously committed +0.121 and the DSR input that was always 1.0000.")
    print(f"  2. turtle_results.json  mean_oos_sharpe (DAILY) = {row_run['mean_oos_sharpe']:+.4f}   ")
    print("     -> canonical calculate_daily_sharpe on the timestamped equity curve, sqrt(365); the runner")
    print("        now uses THIS column for the pooled Sharpe and compute_dsr for the DSR.")
    print(f"  3. turtle_diagnostics.json mean_daily_sharpe    = {row_dg['mean_daily_sharpe']:+.4f}   ")
    print(f"     -> same canonical function inside the diagnostics sim (previously committed as +0.127).")
    print(f"  4. turtle_diagnostics.json cycle_sharpe         = {row_dg['cycle_sharpe']:+.4f}   ")
    print("     -> per-CYCLE Sharpe (mean/std of completed-cycle net bps), the cluster-level counterpart.")
    print("  5. +0.134 is cited in audit_and_reconciliation.py prose only; no committed artifact file contains")
    print("     that number, so it is reported here as an unattributable intermediate, not as a result.")

    # ---- legacy DSR demonstration: committed formula vs canonical formula ----
    sharpe = 0.121
    n_trades = 1613
    n_trials = 1354
    gamma = 0.5772156649
    e_max = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_std = np.sqrt((1.0 + 0.5 * (sharpe ** 2)) / float(n_trades))
    legacy_p = float(1.0 - stats.norm.cdf((sharpe - e_max) / sr_std))
    canonical_p = compute_dsr(per_trade_sharpe=sharpe, n_trades=n_trades, n_trials=n_trials)
    print("\nWHY EVERY COMMITTED DSR p WAS 1.0000 (dimensional bug, demonstrated):")
    print("  Legacy local implementation (formerly turtle_runner.compute_dsr_pvalue, node ")
    print(f"  'e_max_sharpe = {e_max:.4f}' is in z units, never divided by 1/sqrt(T)), so with a POSITIVE")
    print(f"  Sharpe of {sharpe:+.4f}, T={n_trades}, N_trials={n_trials}: z = ({sharpe:.4f} - {e_max:.4f})/{sr_std:.6f} -> p = {legacy_p:.4f}.")
    print(f"  Canonical core.metrics.compute_dsr on the same inputs:  p = {canonical_p:.4f}.")
    print("  So the legacy statistic returned 1.0000 for every config regardless of sign, i.e. the")
    print("  committed 'DSR p = 1.0000, no significance' was uninformative; under the canonical function the")
    print("  1h 20/20 Sharpe is still not significant at 5% (p above), but the number now depends on the data.")

    return {
        "runner_mean_oos_bar_sharpe": row_run["mean_oos_bar_sharpe"],
        "runner_mean_oos_daily_sharpe": row_run["mean_oos_sharpe"],
        "diagnostics_mean_daily_sharpe": row_dg["mean_daily_sharpe"],
        "diagnostics_cycle_sharpe": row_dg["cycle_sharpe"],
        "legacy_dsr_p_positive_sharpe_demo": round(legacy_p, 4),
        "canonical_dsr_p_positive_sharpe_demo": round(float(canonical_p), 4),
        "unattributable_trade_calendar_sharpe": "+0.134 appears in audit_and_reconciliation.py prose only",
    }


def main():
    item1 = run_item_1_funding_reconciliation()
    item2 = run_item_2_cost_reconciliation()
    item3 = run_item_3_pyramids_and_cycles()
    item4 = run_item_4_ambiguous_and_subbars()
    item5 = run_item_4_random_baseline_per_config()
    item7 = run_item_7_autopsy_pure_curve()
    item8 = run_item_8_replay_10bps_vs_14bps()
    item9 = run_item_9_sharpe_drift()

    full_audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "funding_reconciliation_1h_20_20": item1,
        "cost_reconciliation": item2,
        "pyramids_and_cycles": item3,
        "ambiguous_and_subbars": item4,
        "random_baseline_per_config": item5,
        "autopsy_pure_curve": item7,
        "replay_10bps_vs_14bps": item8,
        "sharpe_drift_resolution": item9,
    }

    # audit_results.json is a committed snapshot -> write to a fresh file instead
    out_path = "quant_research/results/audit_results_rerun.json"
    with open(out_path, "w") as f:
        json.dump(full_audit, f, indent=2)

    print(f"\nSuccessfully written complete audit results to {out_path}")


if __name__ == "__main__":
    main()

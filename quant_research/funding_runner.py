"""
Amendment E runner: PREREG section 12, committed BEFORE running (92340e7).

6 registered configurations = 3 assets x U1 (taker) + 3 assets x U2 (maker), a
genuinely NEW signal family: the settled funding rate itself. No price condition,
channel, oscillator or daily filter enters any rule.

  U2 "funding-sign maker" : at each settlement T with r_T != 0, target side is the
                            crowd's side (LONG while r_T > 0, SHORT while r_T < 0;
                            the mirror of the committed IS probe). A LIMIT order at
                            the open of the first 1h bar stamped strictly after T,
                            moved 5 bps passively (D.2 offset/lifetime IMPORTED
                            verbatim, not re-chosen: K = 3 closed bars, strict-
                            through fill, 2 bps maker fee, zero entry slippage).
  U1  taker yardstick     : identical rules, entry at that open as a taker
                            (5 bps fee + 2 bps slippage). U2 - U1 isolates the
                            maker conversion at identical signals.

Exits (both arms): (a) FLIP - the first settlement with r opposite in sign to the
open position closes it at the open of the first 1h bar stamped strictly after
that settlement, which simultaneously re-arms the opposite side at the same open;
(b) 2.0 x ATR(55, 1h) intrabar stop from the entry price - 1m sub-bar resolution
on the fill bar (stop tested only AT OR AFTER the fill timestamp), committed
stop-first pessimistic fallback on later bars (A.1); a stop-out is followed by
flat until the next settlement, which re-arms unconditionally (r != 0 always
defines a target side). There is no channel exit of any kind.

Funding accrual: a position held strictly before the bar carrying settlement T
pays dir x (-r_T) (long -r_T, short +r_T), engine conventions; a position entered
at bar open never pays THAT bar's settlement - the rate is never known before it
accrues. Priority within a bar: accrual -> scheduled open exit -> intrabar stop ->
settlement scheduling; a stop-out on a settlement bar outranks its scheduled flip
exit (pessimistic), leaving flat until the next settlement.

Cost schedules: U1 14 bps RT base / 18 bps stress; U2 9 / 11 bps (D.2 schedule).

1 unit strictly; 1%-risk sizing on 2.0 x ATR(55, 1h); 2.0x leverage cap; $10
minimum notional; equity <= $5 ruins the run. DSR at N_trials = 1,450 + 6 =
1,456. The ETH 2026-04-01+ hold-out is untouched unless a pooled verdict clears
ALL six section-7.5 hurdles.

Writes NEW files; nothing committed is overwritten:
  quant_research/results/amendment_e_results.json
"""
import os
import sys
import json
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel, QuantEngine, EngineTrade
from quant_research.core.metrics import compute_dsr
from quant_research.strategies.turtle_strategy import prepare_turtle_features
from quant_research.turtle_runner import slice_features, per_trade_sharpe
from quant_research.candidate_runner import BARS_PER_YEAR, WINDOWS
from quant_research.maker_runner import (_m1_arrays, _fill_scan, _close_trade,
                                         MAKER_FEE_PCT, EXIT_FEE_PCT,
                                         EXIT_SLIP_BASE_PCT,
                                         EXIT_SLIP_STRESS_PCT,
                                         ENTRY_OFFSET_BPS, ORDER_LIFETIME_BARS,
                                         THROUGH_MARGIN_BPS,
                                         THROUGH_MARGIN_BPS_STRESS)

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"
RESULTS_FILE = "quant_research/results/amendment_e_results.json"

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
ARMS = [("U1_taker", "taker"), ("U2_maker", "maker")]
ATR_N = 55                    # PREREG E.2: ATR(55, 1h), verbatim 7.2 stop distance
WARMUP = 55
N_TRIALS_DSR = 1450 + 6       # PREREG E.2

ENGINE = QuantEngine()        # funding helper + metric aggregation only


def _friction_engine(stress: bool) -> QuantEngine:
    """U1 trade-PnL engine: taker fee both sides, 2/4 bps slippage."""
    cm = CostModel(taker_fee_pct=EXIT_FEE_PCT,
                   slippage_pct=EXIT_SLIP_STRESS_PCT if stress else EXIT_SLIP_BASE_PCT,
                   apply_funding=True, fill_on_next_open=True)
    return QuantEngine(cm)


def _atr_array(asset: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1h.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    feats = prepare_turtle_features(df, N=ATR_N, M=ATR_N)
    return feats, df


def _funding_series(asset: str) -> pd.Series:
    f = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    return f.set_index("timestamp")["funding_rate"].sort_index()


def run_funding_simulation(feats: Dict[str, Any], funding: pd.Series,
                           m1: Optional[Dict[str, Any]], mech: str,
                           stress: bool, initial_capital: float = 100_000.0,
                           risk_percent: float = 1.0):
    """mech is 'taker' or 'maker' (the ARMS mechanism, not the label)."""
    open_arr, high_arr = feats["open"], feats["high"]
    low_arr, close_arr = feats["low"], feats["close"]
    atr_arr = feats["atr"]
    timestamps = feats["timestamps"]
    n_bars = len(close_arr)
    ts_ns = pd.DatetimeIndex(timestamps).values.astype("int64")

    f_df = funding.to_frame("funding_rate")
    f_df["t_round"] = f_df.index.round("1min")
    f_map = f_df.groupby("t_round")["funding_rate"].sum()
    funding_arr = f_map.reindex(timestamps, fill_value=0.0).to_numpy(dtype=np.float64)

    eng = _friction_engine(stress)   # U1 conventions; U2 uses _close_trade
    equity = initial_capital
    pos: Optional[Dict[str, Any]] = None
    pending: Optional[Dict[str, Any]] = None
    funding_accrued = 0.0
    trades: List[EngineTrade] = []
    equity_curve = [equity]

    # scheduled open-price actions decided on a settlement bar, executed one bar later
    sched_flip: int = -1            # bar index whose OPEN performs the flip exit+re-arm
    sched_entry: int = -1           # bar index whose OPEN performs a fresh entry

    n_settle = 0
    n_stops = 0
    n_maker_cancel = 0
    n_signal = 0

    for i in range(n_bars - 1):
        ts = timestamps[i]
        next_ts = timestamps[i + 1]
        next_open = open_arr[i + 1]

        if equity <= 5.0:
            equity_curve.append(equity)
            continue

        # 1. funding accrual for a position held strictly before this bar's open
        if pos is not None and funding_arr[i] != 0.0 and pos["entry_time"] < ts:
            r = funding_arr[i]
            payment = ENGINE.funding_payment(pos["side"], pos["qty"], close_arr[i], r)
            equity += payment
            funding_accrued += payment

        # 2. scheduled action bar: exit at THIS bar's open, then re-arm
        if i == sched_flip and pos is not None:
            exit_price = open_arr[i]
            exit_ts = ts
            if mech == "maker":
                trade, net, _ = _close_trade(pos, exit_price, exit_ts,
                                             "Funding-Sign Flip", funding_accrued, stress)
            else:
                _e, _x, gross, cost, net, bps = eng.calculate_trade_pnl(
                    entry_price=pos["entry_price"], exit_price=exit_price,
                    quantity=pos["qty"],
                    direction="long" if pos["side"] == 1 else "short",
                    funding_pnl=funding_accrued)
                trade = EngineTrade(
                    entry_time=pos["entry_time"], exit_time=exit_ts,
                    direction="long" if pos["side"] == 1 else "short",
                    entry_price=pos["entry_price"], executed_entry_price=_e,
                    exit_price=exit_price, executed_exit_price=_x,
                    quantity=pos["qty"], gross_pnl=gross,
                    fee_and_slip_cost=cost, funding_pnl=funding_accrued,
                    net_pnl=net, return_bps=bps, exit_reason="Funding-Sign Flip",
                    unit_number=1)
            equity += net
            trades.append(trade)
            pos = None
            funding_accrued = 0.0
            # the opposite side re-arms AT THE SAME OPEN (pro-side of r_flip)
            r_flip = funding_arr[i - 1]     # the settlement that scheduled this flip
            n_signal += 1
            stop_dist = 2.0 * atr_arr[i - 1]   # last CLOSED bar before this open
            ref = open_arr[i]
            new_side = 1 if r_flip > 0 else -1
            risk_usd = equity * (risk_percent / 100.0)
            qty = (risk_usd / stop_dist) if stop_dist > 0 else 0.0
            qty = min(qty, (equity * 2.0) / ref if ref > 0 else 0.0)  # 2x leverage cap
            if qty > 0.0 and qty * ref >= 10.0:
                if mech == "maker":
                    limit = ref * (1.0 - ENTRY_OFFSET_BPS * 1e-4) if new_side == 1 \
                        else ref * (1.0 + ENTRY_OFFSET_BPS * 1e-4)
                    pending = {"side": new_side, "limit": limit,
                               "placed_bar": i - 1,
                               "deadline_bar": i - 1 + 1 + ORDER_LIFETIME_BARS,
                               "qty": qty,
                               "stop": limit - stop_dist if new_side == 1
                                       else limit + stop_dist}
                else:
                    pos = {"side": new_side, "entry_price": ref, "entry_time": ts,
                           "qty": qty,
                           "stop": ref - stop_dist if new_side == 1
                                   else ref + stop_dist}
                    funding_accrued = 0.0
            sched_flip = -1
        elif i == sched_entry and pos is None and pending is None:
            r_start = funding_arr[i - 1]
            stop_dist = 2.0 * atr_arr[i - 1]   # last CLOSED bar before this open
            ref = open_arr[i]
            new_side = 1 if r_start > 0 else -1
            n_signal += 1
            risk_usd = equity * (risk_percent / 100.0)
            qty = (risk_usd / stop_dist) if stop_dist > 0 else 0.0
            qty = min(qty, (equity * 2.0) / ref if ref > 0 else 0.0)
            if qty > 0.0 and qty * ref >= 10.0:
                if mech == "maker":
                    limit = ref * (1.0 - ENTRY_OFFSET_BPS * 1e-4) if new_side == 1 \
                        else ref * (1.0 + ENTRY_OFFSET_BPS * 1e-4)
                    pending = {"side": new_side, "limit": limit,
                               "placed_bar": i - 1,
                               "deadline_bar": i - 1 + 1 + ORDER_LIFETIME_BARS,
                               "qty": qty,
                               "stop": limit - stop_dist if new_side == 1
                                       else limit + stop_dist}
                else:
                    pos = {"side": new_side, "entry_price": ref, "entry_time": ts,
                           "qty": qty,
                           "stop": ref - stop_dist if new_side == 1
                                   else ref + stop_dist}
                    funding_accrued = 0.0
            sched_entry = -1

        # 3. pending maker order: cancel past deadline, else fill on this bar's tape
        fill_ts = None
        if pending is not None:
            if i >= pending["deadline_bar"]:
                pending = None
                n_maker_cancel += 1
            elif i > pending["placed_bar"]:   # live from the OPEN it was armed at
                fill_ts = _fill_scan(m1, ts_ns[i], ts_ns[i + 1], pending["side"],
                                     pending["limit"],
                                     THROUGH_MARGIN_BPS_STRESS if stress
                                     else THROUGH_MARGIN_BPS)
                if fill_ts is not None:
                    pos = {"side": pending["side"], "entry_price": pending["limit"],
                           "entry_time": pd.Timestamp(fill_ts, tz="UTC"),
                           "qty": pending["qty"], "stop": pending["stop"]}
                    funding_accrued = 0.0   # opened mid-bar: no settlement this bar
                    pending = None
            # (placed_bar == settlement bar index; i == placed_bar+1 is its first live bar)

        # 4. intrabar stop for an open position (stop-first, pessimistic; no channel exit)
        if pos is not None:
            side = pos["side"]
            if fill_ts is not None:
                lo = int(np.searchsorted(m1["ts"], fill_ts, side="left"))
                hi = int(np.searchsorted(m1["ts"], ts_ns[i + 1], side="left"))
                if side == 1:
                    hit = np.flatnonzero(m1["low"][lo:hi] <= pos["stop"])
                else:
                    hit = np.flatnonzero(m1["high"][lo:hi] >= pos["stop"])
            else:
                stop_hit = low_arr[i] <= pos["stop"] if side == 1 \
                    else high_arr[i] >= pos["stop"]
                hit = np.arange(1, dtype=int) if stop_hit else np.arange(0, dtype=int)
            if len(hit):
                if fill_ts is not None:
                    stop_px, stop_ts = pos["stop"], \
                        pd.Timestamp(m1["ts"][lo + hit[0]], tz="UTC")
                else:
                    stop_px, stop_ts = pos["stop"], ts
                if mech == "maker":
                    trade, net, _ = _close_trade(pos, stop_px, stop_ts,
                                                 "2N Stop Loss", funding_accrued, stress)
                else:
                    _e, _x, gross, cost, net, bps = eng.calculate_trade_pnl(
                        entry_price=pos["entry_price"], exit_price=stop_px,
                        quantity=pos["qty"],
                        direction="long" if pos["side"] == 1 else "short",
                        funding_pnl=funding_accrued)
                    trade = EngineTrade(
                        entry_time=pos["entry_time"], exit_time=stop_ts,
                        direction="long" if pos["side"] == 1 else "short",
                        entry_price=pos["entry_price"], executed_entry_price=_e,
                        exit_price=stop_px, executed_exit_price=_x,
                        quantity=pos["qty"], gross_pnl=gross,
                        fee_and_slip_cost=cost, funding_pnl=funding_accrued,
                        net_pnl=net, return_bps=bps, exit_reason="2N Stop Loss",
                        unit_number=1)
                equity += net
                trades.append(trade)
                pos = None
                funding_accrued = 0.0
                n_stops += 1

        # 5. settlement scheduling: actions execute at the NEXT bar's open
        if i >= WARMUP and not np.isnan(atr_arr[i]) and funding_arr[i] != 0.0:
            n_settle += 1
            r = funding_arr[i]
            if pos is not None:
                pos_side_needed = -1 if r < 0 else 1   # pro-side for r's sign
                if pos_side_needed != pos["side"]:
                    sched_flip = i + 1
                # same sign: hold; stop handled in 4
            elif pending is None and sched_flip <= i:
                sched_entry = i + 1

        equity_curve.append(equity)

    eq_series = pd.Series(np.asarray(equity_curve, dtype=np.float64),
                          index=pd.DatetimeIndex(timestamps[: len(equity_curve)]))
    metrics = ENGINE.calculate_metrics(trades, eq_series, initial_capital,
                                       bars_per_year=BARS_PER_YEAR["1h"])
    metrics.update({"n_settlements": n_settle, "n_stops": n_stops,
                    "n_maker_cancels": n_maker_cancel, "n_signals": n_signal})
    return metrics, trades


def evaluate_job(job):
    arm, mech, asset = job
    feats, _df = _atr_array(asset)
    funding = _funding_series(asset)
    m1 = _m1_arrays(asset) if mech == "maker" else None

    m_is_base, trades_is = run_funding_simulation(slice_features(feats, *WINDOWS["is"]),
                                                  funding, m1, mech, stress=False)
    m_is_stress, _ = run_funding_simulation(slice_features(feats, *WINDOWS["is"]),
                                            funding, m1, mech, stress=True)
    m_oos_base, trades_oos = run_funding_simulation(slice_features(feats, *WINDOWS["oos"]),
                                                    funding, m1, mech, stress=False)
    m_oos_stress, _ = run_funding_simulation(slice_features(feats, *WINDOWS["oos"]),
                                             funding, m1, mech, stress=True)
    m_stitched, _ = run_funding_simulation(slice_features(feats, *WINDOWS["stitched"]),
                                           funding, m1, mech, stress=False)

    def _gross(ts_):
        return (sum(t.gross_pnl for t in ts_ if t.gross_pnl > 0),
                -sum(t.gross_pnl for t in ts_ if t.gross_pnl < 0))

    is_win, is_loss = _gross(trades_is)
    oos_win, oos_loss = _gross(trades_oos)

    return {
        "arm": arm, "asset": asset,
        "is_trades": m_is_base["total_trades"], "is_exp_bps": m_is_base["expectancy_bps"],
        "is_return_pct": m_is_base["total_return_pct"], "is_pf": m_is_base["profit_factor"],
        "is_stress_exp_bps": m_is_stress["expectancy_bps"],
        "is_gross_win": is_win, "is_gross_loss": is_loss,
        "is_per_trade_sharpe": round(per_trade_sharpe(trades_is), 6),
        "oos_trades": m_oos_base["total_trades"], "oos_exp_bps": m_oos_base["expectancy_bps"],
        "oos_return_pct": m_oos_base["total_return_pct"], "oos_pf": m_oos_base["profit_factor"],
        "oos_stress_exp_bps": m_oos_stress["expectancy_bps"],
        "oos_gross_win": oos_win, "oos_gross_loss": oos_loss,
        "oos_per_trade_sharpe": round(per_trade_sharpe(trades_oos), 6),
        "oos_settlements": m_oos_base["n_settlements"], "oos_stops": m_oos_base["n_stops"],
        "oos_maker_cancels": m_oos_base["n_maker_cancels"],
        "stitched_max_dd": m_stitched["max_drawdown_pct"],
        "stitched_return_pct": m_stitched["total_return_pct"],
    }


def run_and_write(out_path: str = RESULTS_FILE, log_trials: bool = True):
    print("=" * 100)
    print("AMENDMENT E: FUNDING-SIGN PRO-SIDE POSITIONING (crowd side, no price signal)")
    print(f"6 configurations | DSR N_trials = {N_TRIALS_DSR} | ETH hold-out NOT touched")
    print("=" * 100, flush=True)

    jobs = [(arm, mech, asset) for arm, mech in ARMS for asset in ASSETS]
    t0 = time.time()
    rows = []
    for job in jobs:
        res = evaluate_job(job)
        rows.append(res)
        print(f"[{res['arm']:8s} {res['asset']:7s}] "
              f"IS {res['is_trades']:>3} tr {res['is_exp_bps']:+8.2f} bps | "
              f"OOS {res['oos_trades']:>3} tr {res['oos_exp_bps']:+8.2f} bps | "
              f"stress {res['oos_stress_exp_bps']:+8.2f} | stops {res['oos_stops']:>3}",
              flush=True)
    print(f"\n{len(rows)} backtests in {time.time() - t0:.1f}s", flush=True)

    pooled, pmap = [], {}
    for arm, _mech in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        is_tr = sum(r["is_trades"] for r in g)
        oos_tr = sum(r["oos_trades"] for r in g)
        exp_is = sum(r["is_exp_bps"] * r["is_trades"] for r in g) / is_tr if is_tr else 0.0
        exp_oos = sum(r["oos_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
        exp_stress = sum(r["oos_stress_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
        trade_sharpe = float(np.mean([r["oos_per_trade_sharpe"] for r in g])) if g else 0.0
        dsr_p = compute_dsr(per_trade_sharpe=trade_sharpe, n_trades=oos_tr,
                            n_trials=N_TRIALS_DSR)
        pooled_pf = (sum(r["oos_gross_win"] for r in g) / sum(r["oos_gross_loss"] for r in g)
                     if sum(r["oos_gross_loss"] for r in g) > 0 else 0.0)
        worst_dd = max(r["stitched_max_dd"] for r in g)
        hurdles = {
            "net_expectancy_ge_5bps": bool(exp_oos >= 5.0),
            "stress_expectancy_gt_0bps": bool(exp_stress > 0.0),
            "pooled_oos_trades_ge_150": bool(oos_tr >= 150),
            "dsr_p_lt_0_05": bool(dsr_p < 0.05),
            "stitched_max_dd_lt_30pct": bool(worst_dd < 30.0),
            "pooled_oos_pf_ge_1_25": bool(pooled_pf >= 1.25),
        }
        verdict = "PASS" if all(hurdles.values()) else "FAIL"
        rec = {
            "arm": arm,
            "pooled_is_trades": is_tr, "pooled_is_exp_bps": exp_is,
            "pooled_oos_trades": oos_tr, "pooled_oos_exp_bps": exp_oos,
            "pooled_oos_stress_exp_bps": exp_stress,
            "mean_oos_per_trade_sharpe": trade_sharpe, "dsr_p_value": dsr_p,
            "pooled_oos_profit_factor": pooled_pf,
            "worst_asset_stitched_max_dd_pct": worst_dd,
            "hurdles": hurdles, "verdict": verdict,
            "failed_hurdles": [k for k, ok in hurdles.items() if not ok],
        }
        pooled.append(rec)
        pmap[arm] = rec

    survivors = [p for p in pooled if p["verdict"] == "PASS"]
    payload = {
        "registered_by": "quant_research/PREREG.md section 12 (Amendment E)",
        "runner": "quant_research/funding_runner.py",
        "evidence_basis": "quant_research/results/funding_carry_evidence.json (IS only, c3ae105)",
        "signal": "settled funding rate sign: pro-side (crowd side); no price condition",
        "atr_stop": "2.0 x ATR(55, 1h), stop-first pessimistic fallback (A.1)",
        "cost_schedule": {"u1_base_rt_bps": 14.0, "u1_stress_rt_bps": 18.0,
                          "u2_base_rt_bps": 9.0, "u2_stress_rt_bps": 11.0},
        "maker_entry_offset_bps": ENTRY_OFFSET_BPS,
        "maker_order_lifetime_bars": ORDER_LIFETIME_BARS,
        "n_trials_dsr": N_TRIALS_DSR,
        "windows": WINDOWS,
        "holdout_touched": False,
        "rows": rows,
        "pooled": pooled,
        "n_survivors": len(survivors),
        "survivors": survivors,
        "verdict": "SURVIVORS_AWAIT_HOLDOUT" if survivors else "NO_EDGE_FUNDINGSIGN_VARIANT",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")

    if log_trials:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trial_rows = [{
            "trial_id": f"{r['arm']}-{r['asset']}",
            "timestamp_utc": now, "test_name": "AmendmentE_FundingSign",
            "asset": r["asset"], "timeframe": "1h",
            "strategy_name": f"{r['arm']}_funding_sign_pro",
            "params_json": json.dumps({"ATR_N": ATR_N, "signal": "funding_sign_pro",
                                       "arm": r["arm"], "asset": r["asset"], "tf": "1h"}),
            "is_return_pct": r["is_return_pct"], "is_expectancy_bps": r["is_exp_bps"],
            "oos_return_pct": r["oos_return_pct"], "oos_expectancy_bps": r["oos_exp_bps"],
            "status": "EVALUATED",
        } for r in rows]
        pd.DataFrame(trial_rows).to_csv(TRIALS_FILE, mode="a",
                                        header=not os.path.exists(TRIALS_FILE),
                                        index=False)
        print(f"logged {len(trial_rows)} trials to {TRIALS_FILE}")
    return payload


if __name__ == "__main__":
    run_and_write()

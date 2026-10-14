"""
Amendment D runner: PREREG section 11, committed BEFORE running (99966b1).

24 registered configurations = 12 x R1 + 12 x R0, all on the section-7 skeleton
restricted to the 1h timeframe with the daily-trend filter ON in BOTH arms:

  R1  "maker 1h" : Donchian entry signal -> LIMIT order at open[t+1] moved
                   5 bps passively; fills only when the 1m tape trades STRICTLY
                   THROUGH the limit inside the next K = 3 closed 1h bars; fill
                   price = limit; 2 bps maker fee, zero entry slippage. Exits
                   unchanged and taker.
  R0  taker yardstick: the identical skeleton with the section-7.3 taker entry
                   at open[t+1] (5 bps fee + 2 bps slippage), run by the
                   committed engine. R1 - R0 isolates the maker conversion.

Cost schedules (PREREG D.2): base = 2 bps maker entry + 0 slippage, exit
5 + 2 = 7 bps taker => 9 bps RT.  Stress = entry fill additionally requires the
1m tape to trade through the limit by a further 1 bps, exits 5 + 4 => 11 bps RT.

Conventions inherited verbatim from section 7.2/7.3 and the committed engine:
1%-risk sizing on 2.0 x ATR(signal bar), 2.0x leverage cap, 1 unit (no
pyramiding, passed EXPLICITLY as max_units=1 after the Amendment C defect),
funding at exact empirical settlements paid by longs / received by shorts,
2.0 x ATR(N) intrabar stop with the stop-first pessimistic fallback, close-based
M-bar channel exit filling at the next open.

Fill-bar exit convention (documented here because the PREREG is silent on it):
on the bar in which a pending order fills, the intrabar stop is tested only on
1m sub-bars AT OR AFTER the fill timestamp (the position did not exist before
the fill); the close-based channel exit is tested normally (the bar close is
after the fill). Ambiguous bars on later days use the committed stop-first
pessimistic fallback, no 1m wiring, identical to section A.1.

DSR at N_trials = 1,426 + 24 = 1,450. The ETH 2026-04-01+ hold-out is untouched
unless a configuration clears ALL six section-7.5 hurdles.

Writes NEW files; nothing committed is overwritten:
  quant_research/results/amendment_d_results.json
"""
import os
import sys
import json
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.engine import CostModel, QuantEngine, EngineTrade
from quant_research.core.metrics import compute_dsr
from quant_research.strategies.turtle_strategy import prepare_turtle_features, run_turtle_simulation
from quant_research.turtle_runner import slice_features, per_trade_sharpe
from quant_research.candidate_runner import (apply_daily_mask, daily_trend_map,
                                             BARS_PER_YEAR, WINDOWS)

DATA_DIR = "quant_research/data/processed"
TRIALS_FILE = "quant_research/trials.csv"
RESULTS_FILE = "quant_research/results/amendment_d_results.json"

ASSETS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
CHANNELS = [(20, 10), (20, 20), (55, 10), (55, 20)]
ARMS = [("R1_maker_1h", "maker"), ("R0_taker_1h", "taker")]

ENTRY_OFFSET_BPS = 5           # PREREG D.2: FIXED, anchored to the measured +5.14 bps cross
ORDER_LIFETIME_BARS = 3        # PREREG D.2: K = 3 closed 1h bars
THROUGH_MARGIN_BPS = 0         # base fill rule: strictly through (low < limit)
THROUGH_MARGIN_BPS_STRESS = 1  # stress fill rule: through by >= 1 extra bp

MAKER_FEE_PCT = 0.02           # Binance USDT-M VIP0 maker leg (declared input, D.1.3)
EXIT_FEE_PCT = 0.05            # taker leg confirmed by the ledger
EXIT_SLIP_BASE_PCT = 0.02
EXIT_SLIP_STRESS_PCT = 0.04

N_TRIALS_DSR = 1426 + 24       # PREREG D.2

ENGINE = QuantEngine()         # direction-aware funding helper only


@dataclass
class PendingOrder:
    side: int               # +1 long / -1 short
    limit: float
    placed_bar: int         # signal bar index i (order live from bar i+1)
    deadline_bar: int       # exclusive: cancelled before bar i+1+K
    qty: float
    stop: float


def _m1_arrays(asset: str) -> Dict[str, Any]:
    m1 = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1m.parquet"))
    m1["timestamp"] = pd.to_datetime(m1["timestamp"], utc=True)
    m1 = m1.set_index("timestamp").sort_index()
    return {
        "ts": m1.index.values.astype("int64"),
        "low": m1["low"].to_numpy(dtype=np.float64),
        "high": m1["high"].to_numpy(dtype=np.float64),
    }


def _fill_scan(m1: Dict[str, Any], t0: np.int64, t1: np.int64, side: int,
               limit: float, through_margin_bps: float) -> Optional[np.int64]:
    """First 1m row in [t0, t1) that trades STRICTLY through the limit."""
    lo = int(np.searchsorted(m1["ts"], t0, side="left"))
    hi = int(np.searchsorted(m1["ts"], t1, side="left"))
    if hi <= lo:
        return None
    if through_margin_bps > 0:
        bound = limit * (1.0 - through_margin_bps * 1e-4) if side == 1 else \
            limit * (1.0 + through_margin_bps * 1e-4)
    else:
        bound = limit
    if side == 1:
        arr = m1["low"][lo:hi]
        hit = np.flatnonzero(arr < bound)      # strict: a touch does not fill
    else:
        arr = m1["high"][lo:hi]
        hit = np.flatnonzero(arr > bound)
    return int(m1["ts"][lo + hit[0]]) if len(hit) else None


def _close_trade(pos: Dict[str, Any], exit_price: float, exit_ts, reason: str,
                 funding_accrued: float, stress: bool) -> Tuple[EngineTrade, float, float]:
    """Trade PnL at engine financing conventions, maker entry / taker exit."""
    side = pos["side"]
    maker_fee = MAKER_FEE_PCT / 100.0
    total_exit_friction = (EXIT_FEE_PCT + (EXIT_SLIP_STRESS_PCT if stress else EXIT_SLIP_BASE_PCT)) / 100.0
    if side == 1:
        exec_entry = pos["entry_price"] * (1.0 + maker_fee)
        exec_exit = exit_price * (1.0 - total_exit_friction)
        exec_pnl = pos["qty"] * (exec_exit - exec_entry)
        gross = pos["qty"] * (exit_price - pos["entry_price"])
    else:
        exec_entry = pos["entry_price"] * (1.0 - maker_fee)
        exec_exit = exit_price * (1.0 + total_exit_friction)
        exec_pnl = pos["qty"] * (exec_entry - exec_exit)
        gross = pos["qty"] * (pos["entry_price"] - exit_price)
    friction = gross - exec_pnl
    net = exec_pnl + funding_accrued
    notional = pos["qty"] * pos["entry_price"]
    bps = (net / notional * 10_000.0) if notional > 0 else 0.0
    trade = EngineTrade(
        entry_time=pos["entry_time"], exit_time=exit_ts,
        direction="long" if side == 1 else "short",
        entry_price=pos["entry_price"], executed_entry_price=exec_entry,
        exit_price=exit_price, executed_exit_price=exec_exit,
        quantity=pos["qty"], gross_pnl=gross, fee_and_slip_cost=friction,
        funding_pnl=funding_accrued, net_pnl=net, return_bps=bps,
        exit_reason=reason, unit_number=1)
    return trade, net, bps


def run_maker_simulation(features: Dict[str, Any], funding_series: Optional[pd.Series],
                         m1: Dict[str, Any], params: Dict[str, Any], stress: bool,
                         initial_capital: float = 100_000.0, risk_percent: float = 1.0) -> Tuple[Dict, List]:
    N, M = int(params["N"]), int(params["M"])
    warmup = max(N, M, 30)
    open_arr, high_arr = features["open"], features["high"]
    low_arr, close_arr = features["low"], features["close"]
    entry_high_arr, entry_low_arr = features["entry_high"], features["entry_low"]
    exit_high_arr, exit_low_arr = features["exit_high"], features["exit_low"]
    atr_arr = features["atr"]
    timestamps = features["timestamps"]
    n_bars = len(close_arr)
    ts_ns = pd.DatetimeIndex(timestamps).values.astype("int64")

    if funding_series is not None:
        f_df = funding_series.to_frame("funding_rate")
        f_df["t_round"] = f_df.index.round("1min")
        f_map = f_df.groupby("t_round")["funding_rate"].sum()
        funding_arr = f_map.reindex(timestamps, fill_value=0.0).to_numpy(dtype=np.float64)
    else:
        funding_arr = np.zeros(n_bars, dtype=np.float64)

    equity = initial_capital
    pos: Optional[Dict[str, Any]] = None
    pending: Optional[PendingOrder] = None
    funding_accrued = 0.0
    trades: List[EngineTrade] = []
    equity_curve = [equity]

    n_signals = 0
    n_fills = 0
    n_cancelled = 0

    for i in range(n_bars - 1):
        ts = timestamps[i]
        next_ts = timestamps[i + 1]
        next_open = open_arr[i + 1]

        if equity <= 5.0:
            equity_curve.append(equity)
            continue

        # 1. funding for a position held at the START of this bar
        if pos is not None and funding_arr[i] != 0.0:
            payment = ENGINE.funding_payment(pos["side"], pos["qty"], close_arr[i], funding_arr[i])
            equity += payment
            funding_accrued += payment

        # 2. pending order: cancel past its deadline, else fill on this bar's tape
        fill_ns: Optional[int] = None
        if pending is not None:
            if i >= pending.deadline_bar:
                pending = None
                n_cancelled += 1
            elif i > pending.placed_bar:
                fill_ns = _fill_scan(m1, ts_ns[i], ts_ns[i + 1], pending.side,
                                     pending.limit, THROUGH_MARGIN_BPS_STRESS if stress
                                     else THROUGH_MARGIN_BPS)
                if fill_ns is not None:
                    pos = {"side": pending.side, "entry_price": pending.limit,
                           "entry_time": pd.Timestamp(fill_ns, tz="UTC"),
                           "qty": pending.qty, "stop": pending.stop}
                    funding_accrued = 0.0   # opened mid-bar: no settlement this bar
                    pending = None
                    n_fills += 1

        # 3. exits for an open position
        if pos is not None:
            side = pos["side"]
            stop = pos["stop"]
            if fill_ns is not None:
                # fill bar: stop may only be hit at/after the fill; close-based
                # channel exit is tested normally (close is after the fill).
                lo = int(np.searchsorted(m1["ts"], fill_ns, side="left"))
                hi = int(np.searchsorted(m1["ts"], ts_ns[i + 1], side="left"))
                if side == 1:
                    hit = np.flatnonzero(m1["low"][lo:hi] <= stop)
                else:
                    hit = np.flatnonzero(m1["high"][lo:hi] >= stop)
                stop_on_fill_bar = bool(len(hit))
                channel_hit = close_arr[i] < exit_low_arr[i] if side == 1 else \
                    close_arr[i] > exit_high_arr[i]
                if stop_on_fill_bar and channel_hit:
                    exit_triggered, exit_price, exit_ts, reason = \
                        True, stop, ts, "2N Stop Loss"
                elif stop_on_fill_bar:
                    exit_triggered, exit_price, exit_ts, reason = \
                        True, stop, pd.Timestamp(m1["ts"][lo + hit[0]], tz="UTC"), "2N Stop Loss"
                elif channel_hit:
                    exit_triggered, exit_price, exit_ts, reason = \
                        True, next_open, next_ts, f"{M}-Bar Donchian Exit"
                else:
                    exit_triggered = False
                    exit_price, exit_ts, reason = None, None, None
            else:
                stop_hit = low_arr[i] <= stop if side == 1 else high_arr[i] >= stop
                channel_hit = close_arr[i] < exit_low_arr[i] if side == 1 else \
                    close_arr[i] > exit_high_arr[i]
                if stop_hit or channel_hit:
                    # pessimistic stop-first fallback on ambiguous bars (A.1)
                    if stop_hit:
                        exit_triggered, exit_price, exit_ts, reason = \
                            True, stop, ts, "2N Stop Loss"
                    else:
                        exit_triggered, exit_price, exit_ts, reason = \
                            True, next_open, next_ts, f"{M}-Bar Donchian Exit"
                else:
                    exit_triggered = False
                    exit_price, exit_ts, reason = None, None, None

            if exit_triggered:
                trade, net, _bps = _close_trade(pos, exit_price, exit_ts, reason,
                                                funding_accrued, stress)
                equity += net
                trades.append(trade)
                pos = None
                funding_accrued = 0.0

        # 4. new signal -> pending limit order (only when flat and nothing working)
        if pos is None and pending is None and i >= warmup and not np.isnan(atr_arr[i]) \
                and not np.isnan(entry_high_arr[i]):
            long_sig = close_arr[i] > entry_high_arr[i]
            short_sig = close_arr[i] < entry_low_arr[i]
            if long_sig or short_sig:
                n_signals += 1
                stop_dist = 2.0 * atr_arr[i]
                u = 1 if long_sig else -1
                limit = next_open * (1.0 - ENTRY_OFFSET_BPS * 1e-4) if u == 1 else \
                    next_open * (1.0 + ENTRY_OFFSET_BPS * 1e-4)
                risk_usd = equity * (risk_percent / 100.0)
                qty = (risk_usd / stop_dist) if stop_dist > 0 else 0.0
                qty = min(qty, (equity * 2.0) / limit if limit > 0 else 0.0)  # 2x leverage cap
                if qty * limit < 10.0:
                    qty = 0.0
                if qty > 0.0:
                    pending = PendingOrder(
                        side=u, limit=limit, placed_bar=i,
                        deadline_bar=i + 1 + ORDER_LIFETIME_BARS, qty=qty,
                        stop=limit - stop_dist if u == 1 else limit + stop_dist)

        equity_curve.append(equity)

    eq_series = pd.Series(np.asarray(equity_curve, dtype=np.float64),
                          index=pd.DatetimeIndex(timestamps[: len(equity_curve)]))
    metrics = ENGINE.calculate_metrics(trades, eq_series, initial_capital,
                                       bars_per_year=BARS_PER_YEAR["1h"])
    metrics["n_signals"] = n_signals
    metrics["n_fills"] = n_fills
    metrics["n_cancelled"] = n_cancelled
    return metrics, trades


def evaluate_job(job):
    arm, mechanism, asset, N, M = job
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_1h.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    fdf = pd.read_parquet(os.path.join(DATA_DIR, f"{asset}_funding.parquet"))
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

    features = apply_daily_mask(prepare_turtle_features(df, N=N, M=M), daily_trend_map(asset))
    params = {"N": N, "M": M}
    m1 = _m1_arrays(asset) if mechanism == "maker" else None

    def _sim(feat, stress, want_trades=False):
        if mechanism == "taker":
            cm = CostModel(taker_fee_pct=EXIT_FEE_PCT,
                           slippage_pct=EXIT_SLIP_STRESS_PCT if stress else EXIT_SLIP_BASE_PCT,
                           apply_funding=True, fill_on_next_open=True)
            return run_turtle_simulation(feat, funding, params, cm,
                                         bars_per_year=BARS_PER_YEAR["1h"],
                                         return_trades=want_trades, max_units=1)
        out = run_maker_simulation(feat, funding, m1, params, stress=stress)
        return out[0] if not want_trades else out

    m_is_base, trades_is = _sim(slice_features(features, *WINDOWS["is"]), False, True)
    m_is_stress = _sim(slice_features(features, *WINDOWS["is"]), True)
    m_oos_base, trades_oos = _sim(slice_features(features, *WINDOWS["oos"]), False, True)
    m_oos_stress = _sim(slice_features(features, *WINDOWS["oos"]), True)
    m_stitched = _sim(slice_features(features, *WINDOWS["stitched"]), False)

    def _gross(ts_):
        return (sum(t.gross_pnl for t in ts_ if t.gross_pnl > 0),
                -sum(t.gross_pnl for t in ts_ if t.gross_pnl < 0))

    is_win, is_loss = _gross(trades_is)
    oos_win, oos_loss = _gross(trades_oos)

    return {
        "arm": arm, "mechanism": mechanism, "asset": asset, "N": N, "M": M,
        "is_trades": m_is_base["total_trades"], "is_exp_bps": m_is_base["expectancy_bps"],
        "is_return_pct": m_is_base["total_return_pct"], "is_pf": m_is_base["profit_factor"],
        "is_max_dd": m_is_base["max_drawdown_pct"],
        "is_stress_exp_bps": m_is_stress["expectancy_bps"],
        "is_gross_win": is_win, "is_gross_loss": is_loss,
        "is_per_trade_sharpe": round(per_trade_sharpe(trades_is), 6),
        "oos_trades": m_oos_base["total_trades"], "oos_exp_bps": m_oos_base["expectancy_bps"],
        "oos_return_pct": m_oos_base["total_return_pct"], "oos_pf": m_oos_base["profit_factor"],
        "oos_max_dd": m_oos_base["max_drawdown_pct"],
        "oos_stress_exp_bps": m_oos_stress["expectancy_bps"],
        "oos_gross_win": oos_win, "oos_gross_loss": oos_loss,
        "oos_per_trade_sharpe": round(per_trade_sharpe(trades_oos), 6),
        "oos_signals": m_oos_base.get("n_signals") if mechanism == "maker" else None,
        "oos_fills": m_oos_base.get("n_fills") if mechanism == "maker" else None,
        "stitched_max_dd": m_stitched["max_drawdown_pct"],
        "stitched_return_pct": m_stitched["total_return_pct"],
    }


def run_and_write(out_path: str = RESULTS_FILE, log_trials: bool = True):
    print("=" * 100)
    print("AMENDMENT D: MAKER VS TAKER ENTRY (1h Donchian + daily filter, 1 unit)")
    print(f"24 configurations | DSR N_trials = {N_TRIALS_DSR} | ETH hold-out NOT touched")
    print("=" * 100, flush=True)

    jobs = []
    for arm, mech in ARMS:
        for asset in ASSETS:
            for N, M in CHANNELS:
                jobs.append((arm, mech, asset, N, M))

    t0 = time.time()
    rows = []
    for job in jobs:
        res = evaluate_job(job)
        rows.append(res)
        sig = res["oos_signals"]
        if sig:
            rate = f"{100.0 * res['oos_fills'] / sig:5.1f}%"
        else:
            rate = "  n/a"
        print(f"[{res['arm']} {res['asset']} N={res['N']:>2} M={res['M']}] "
              f"IS {res['is_trades']:>3} tr {res['is_exp_bps']:+8.2f} bps | "
              f"OOS {res['oos_trades']:>3}/{sig} sig fill {rate} "
              f"{res['oos_exp_bps']:+8.2f} bps", flush=True)
    print(f"\n{len(rows)} backtests in {time.time() - t0:.1f}s", flush=True)

    pooled, attr = [], []
    pmap = {}
    for arm, _mech in ARMS:
        for N, M in CHANNELS:
            g = [r for r in rows if r["arm"] == arm and r["N"] == N and r["M"] == M]
            is_tr = sum(r["is_trades"] for r in g)
            oos_tr = sum(r["oos_trades"] for r in g)
            exp_is = sum(r["is_exp_bps"] * r["is_trades"] for r in g) / is_tr if is_tr else 0.0
            exp_oos = sum(r["oos_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
            exp_stress = sum(r["oos_stress_exp_bps"] * r["oos_trades"] for r in g) / oos_tr if oos_tr else 0.0
            trade_sharpe = float(np.mean([r["oos_per_trade_sharpe"] for r in g])) if g else 0.0
            dsr_p = compute_dsr(per_trade_sharpe=trade_sharpe, n_trades=oos_tr, n_trials=N_TRIALS_DSR)
            pooled_pf = (sum(r["oos_gross_win"] for r in g) / sum(r["oos_gross_loss"] for r in g)
                         if sum(r["oos_gross_loss"] for r in g) > 0 else 0.0)
            worst_dd = max(r["stitched_max_dd"] for r in g)
            sig_total = sum(r["oos_signals"] or 0 for r in g)
            fill_total = sum(r["oos_fills"] or r["oos_trades"] for r in g)
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
                "arm": arm, "N": N, "M": M,
                "pooled_is_trades": is_tr, "pooled_is_exp_bps": exp_is,
                "pooled_oos_trades": oos_tr, "pooled_oos_exp_bps": exp_oos,
                "pooled_oos_stress_exp_bps": exp_stress,
                "mean_oos_per_trade_sharpe": trade_sharpe, "dsr_p_value": dsr_p,
                "pooled_oos_profit_factor": pooled_pf,
                "worst_asset_stitched_max_dd_pct": worst_dd,
                "pooled_oos_signals": sig_total, "pooled_oos_fill_rate":
                    (fill_total / sig_total) if sig_total else None,
                "hurdles": hurdles, "verdict": verdict,
                "failed_hurdles": [k for k, ok in hurdles.items() if not ok],
            }
            pooled.append(rec)
            pmap[(arm, N, M)] = rec

    for N, M in CHANNELS:
        r1, r0 = pmap[("R1_maker_1h", N, M)], pmap[("R0_taker_1h", N, M)]
        attr.append({
            "N": N, "M": M,
            "r0_taker_oos_exp_bps": r0["pooled_oos_exp_bps"],
            "r1_maker_oos_exp_bps": r1["pooled_oos_exp_bps"],
            "maker_delta_bps": r1["pooled_oos_exp_bps"] - r0["pooled_oos_exp_bps"],
            "r0_taker_oos_trades": r0["pooled_oos_trades"],
            "r1_maker_oos_trades": r1["pooled_oos_trades"],
            "r1_fill_rate": r1["pooled_oos_fill_rate"],
        })
    attr.sort(key=lambda r: -r["maker_delta_bps"])

    survivors = [p for p in pooled if p["verdict"] == "PASS"]
    payload = {
        "registered_by": "quant_research/PREREG.md section 11 (Amendment D)",
        "runner": "quant_research/maker_runner.py",
        "entry_offset_bps": ENTRY_OFFSET_BPS,
        "order_lifetime_bars": ORDER_LIFETIME_BARS,
        "fill_rule": "1m trades strictly through the limit; stress requires 1 bps of extra through",
        "cost_schedule": {"base_rt_bps": 9.0, "stress_rt_bps": 11.0},
        "maker_fee_pct": MAKER_FEE_PCT,
        "n_trials_dsr": N_TRIALS_DSR,
        "windows": WINDOWS,
        "holdout_touched": False,
        "rows": rows,
        "pooled": pooled,
        "attribution_maker_vs_taker": attr,
        "n_survivors": len(survivors),
        "survivors": survivors,
        "verdict": "SURVIVORS_AWAIT_HOLDOUT" if survivors else "NO_EDGE_MAKER_VARIANT",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")

    if log_trials:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trial_rows = [{
            "trial_id": f"{r['arm']}-{r['asset']}-N{r['N']}-M{r['M']}",
            "timestamp_utc": now, "test_name": "AmendmentD_MakerVsTaker",
            "asset": r["asset"], "timeframe": "1h",
            "strategy_name": f"{r['arm']}_N{r['N']}_M{r['M']}",
            "params_json": json.dumps({"N": r["N"], "M": r["M"], "max_units": 1,
                                       "daily_filter": True, "mechanism": r["mechanism"],
                                       "asset": r["asset"], "tf": "1h"}),
            "is_return_pct": r["is_return_pct"], "is_expectancy_bps": r["is_exp_bps"],
            "oos_return_pct": r["oos_return_pct"], "oos_expectancy_bps": r["oos_exp_bps"],
            "status": "EVALUATED",
        } for r in rows]
        pd.DataFrame(trial_rows).to_csv(TRIALS_FILE, mode="a",
                                        header=not os.path.exists(TRIALS_FILE), index=False)
        print(f"logged {len(trial_rows)} trials to {TRIALS_FILE}")
    return payload


if __name__ == "__main__":
    run_and_write()

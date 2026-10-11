"""
Turtle Strategy Diagnostic Engine & Re-Evaluation Suite.
Implements:
1. Daily Equity Sharpe Ratio (sqrt(365)).
2. DSR using per-trade Sharpe ratio and T = n_trades.
3. % of entries where 2x cap binds on unit 1 (labels >25% as "capped fixed-leverage Donchian").
4. Realized risk per unit and pyramid addition distribution (1, 2, 3, 4 units).
5. Ambiguous bar counts and pessimistic exit ordering.
6. Block-bootstrap 95% CI for net expectancy (unit-level, as originally committed).
7. % of calendar months positive.
8. Sensitivity test: +/-20% parameter perturbation.
9. CYCLE-LEVEL (cluster) inference: per-cycle net bps, per-cycle / per-UTC-day
   clustered standard errors, cycle-block and day-cluster bootstrap CIs, and
   DSR / t-tests that treat the completed CYCLE (not the pyramid unit) as the
   independent observation, clustered by exit UTC day.

NOTE ON THE RANDOM-ENTRY BASELINE: the only simulated random-entry baseline in
this repo is quant_research/turtle_random_entry_test.py (1,000 runs, BTCUSDT 1h
OOS), whose output is quant_research/results/random_entry_baseline.json. The
placeholder that used to live here (rng.choice of five hard-coded bps values)
was deleted during the audit: it was never called by any runner, and drawing
five constants is not a simulation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

from quant_research.core.engine import CostModel, QuantEngine, assert_no_same_bar_pyramiding
from quant_research.core.metrics import calculate_daily_sharpe, compute_dsr, block_bootstrap_ci
from quant_research.strategies.turtle_strategy import prepare_turtle_features

DATA_DIR = "quant_research/data/processed"


def slice_features(features: Dict[str, Any], start_str: str, end_str: str) -> Dict[str, Any]:
    ts = features["timestamps"]
    mask = (ts >= pd.Timestamp(start_str, tz="UTC")) & (ts <= pd.Timestamp(end_str, tz="UTC"))
    return {
        "open": features["open"][mask],
        "high": features["high"][mask],
        "low": features["low"][mask],
        "close": features["close"][mask],
        "entry_high": features["entry_high"][mask],
        "entry_low": features["entry_low"][mask],
        "exit_high": features["exit_high"][mask],
        "exit_low": features["exit_low"][mask],
        "atr": features["atr"][mask],
        "timestamps": ts[mask],
    }


def run_turtle_diagnostic_sim(
    features: Dict[str, Any],
    funding_series: Optional[pd.Series],
    N: int,
    M: int,
    cost_model: CostModel,
    initial_capital: float = 100_000.0,
    risk_percent: float = 1.0,
) -> Dict[str, Any]:
    engine = QuantEngine(cost_model)

    open_arr = features["open"]
    high_arr = features["high"]
    low_arr = features["low"]
    close_arr = features["close"]
    entry_high_arr = features["entry_high"]
    entry_low_arr = features["entry_low"]
    exit_high_arr = features["exit_high"]
    exit_low_arr = features["exit_low"]
    atr_arr = features["atr"]
    timestamps = features["timestamps"]
    n_bars = len(close_arr)

    warmup = max(N, M, 30)

    # Pre-align funding
    if funding_series is not None and cost_model.apply_funding:
        f_df = funding_series.to_frame("funding_rate")
        f_df["t_round"] = f_df.index.round("1min")
        f_map = f_df.groupby("t_round")["funding_rate"].sum()
        funding_arr = f_map.reindex(timestamps, fill_value=0.0).to_numpy(dtype=np.float64)
    else:
        funding_arr = np.zeros(n_bars, dtype=np.float64)

    equity = initial_capital
    units = []
    direction = 0
    active_stop = 0.0
    funding_accrued = 0.0
    last_action_bar = -1

    completed_trades = []
    completed_cycles = []   # one record per completed position CYCLE (1-4 units)
    equity_curve = [equity]
    equity_dates = [timestamps[0]]

    cap_bound_count = 0
    total_entries_count = 0
    pyramid_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
    # AMBIGUOUS BAR definition (this counter, e.g. 655 on SOL 5m N=20/M=10):
    # one bar counts once iff a position is OPEN at the start of that bar AND
    # BOTH intrabar conditions hold on the same bar:
    #   (a) the protective 2N stop lies inside the bar's range
    #        long: low <= active_stop      short: high >= active_stop
    #   (b) the M-bar Donchian channel-exit fires on that bar's CLOSE
    #        long: close < exit_low[i]     short: close > exit_high[i]
    # The two are mutually exclusive under the pessimistic ordering (stop is
    # checked first and wins), so this counts bars whose outcome depends on the
    # ordering rule rather than on the price path alone.  Bars where only one
    # condition holds are NOT counted, and the counter is per BAR, not per unit
    # (a 4-unit cycle that exits on an ambiguous bar still adds exactly 1).
    ambiguous_bars_count = 0

    for i in range(n_bars - 1):
        if equity <= 5.0:
            equity_curve.append(equity)
            equity_dates.append(timestamps[i])
            continue

        ts = timestamps[i]
        next_open = open_arr[i + 1]
        next_ts = timestamps[i + 1]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        atr = atr_arr[i]

        if i < warmup or np.isnan(atr) or np.isnan(entry_high_arr[i]):
            equity_curve.append(equity)
            equity_dates.append(timestamps[i])
            continue

        # 1. Accrue funding
        if direction != 0 and len(units) > 0:
            f_rate = funding_arr[i]
            if f_rate != 0.0:
                tot_pos = sum(u["quantity"] for u in units)
                payment = -tot_pos * close * f_rate
                equity += payment
                funding_accrued += payment

        # 2. Check Exits (Pessimistic Order: Stop Loss intrabar FIRST, Channel Exit SECOND)
        exit_triggered = False
        exit_price = 0.0
        exit_reason = ""

        if direction != 0 and len(units) > 0:
            stop_hit = (low <= active_stop) if direction == 1 else (high >= active_stop)
            channel_hit = (close < exit_low_arr[i]) if direction == 1 else (close > exit_high_arr[i])

            if stop_hit and channel_hit:
                ambiguous_bars_count += 1

            # Pessimistic execution: stop loss hits first
            if stop_hit:
                exit_triggered = True
                exit_price = active_stop
                exit_reason = "Stop Loss"
            elif channel_hit:
                exit_triggered = True
                exit_price = next_open
                exit_reason = "Channel Exit"

            if exit_triggered:
                unit_funding = funding_accrued / len(units) if len(units) > 0 else 0.0
                dir_str = "long" if direction == 1 else "short"
                max_units_in_cycle = len(units)
                pyramid_distribution[max_units_in_cycle] += 1

                cycle_unit_bps = []
                for u in units:
                    exec_entry, exec_exit, gross, cost, net, bps = engine.calculate_trade_pnl(
                        entry_price=u["entry_price"],
                        exit_price=exit_price,
                        quantity=u["quantity"],
                        direction=dir_str,
                        funding_pnl=unit_funding,
                    )
                    equity += net
                    notional = u["quantity"] * u["entry_price"]
                    gross_bps = (gross / notional * 10000.0) if notional > 0 else 0.0
                    fee_bps = (cost / notional * 10000.0) if notional > 0 else 0.0
                    cycle_unit_bps.append(bps)

                    completed_trades.append({
                        "entry_time": u["entry_time"],
                        "exit_time": ts if exit_reason == "Stop Loss" else next_ts,
                        "direction": dir_str,
                        "gross_pnl": gross,
                        "fee_cost": cost,
                        "net_pnl": net,
                        "gross_bps": gross_bps,
                        "fee_bps": fee_bps,
                        "net_bps": bps,
                        "realized_risk_usd": abs(net) if net < 0 else 0.0,
                        "realized_risk_pct": (abs(net) / u["equity_at_entry"] * 100.0) if net < 0 else 0.0,
                        "unit_num": u["unit_num"],
                        "exit_reason": exit_reason,
                    })

                # One cycle observation = the equal-weighted mean net bps of its
                # units, stamped with the cycle's EXIT bar (UTC day is the cluster
                # label used for the day-clustered inference below).
                cycle_exit_ts = ts if exit_reason == "Stop Loss" else next_ts
                completed_cycles.append({
                    "cycle": len(completed_cycles) + 1,
                    "exit_time": cycle_exit_ts,
                    "exit_day": str(pd.Timestamp(cycle_exit_ts).date()),
                    "units_in_cycle": len(units),
                    "net_bps": float(np.mean(cycle_unit_bps)) if cycle_unit_bps else 0.0,
                })

                units = []
                direction = 0
                active_stop = 0.0
                funding_accrued = 0.0

        # 3. Check Entries & Pyramids
        if not exit_triggered:
            risk_usd = equity * (risk_percent / 100.0)
            stop_dist = 2.0 * atr
            desired_qty = (risk_usd / stop_dist) if stop_dist > 0 else 0.0

            cur_notional = sum(u["quantity"] * next_open for u in units)
            max_add_notional = max(0.0, (equity * 2.0) - cur_notional)
            max_unit_qty = max_add_notional / next_open if next_open > 0 else 0.0

            is_cap_bound = (desired_qty > max_unit_qty)
            unit_qty = min(desired_qty, max_unit_qty)
            viable_order = (unit_qty * next_open) >= 10.0

            # Initial Entry
            if direction == 0 and viable_order and i > last_action_bar:
                long_sig = close > entry_high_arr[i]
                short_sig = close < entry_low_arr[i]

                if long_sig or short_sig:
                    total_entries_count += 1
                    if is_cap_bound:
                        cap_bound_count += 1

                    direction = 1 if long_sig else -1
                    active_stop = next_open - stop_dist if long_sig else next_open + stop_dist
                    units.append({
                        "unit_num": 1,
                        "entry_price": next_open,
                        "entry_time": next_ts,
                        "quantity": unit_qty,
                        "equity_at_entry": equity,
                    })
                    last_action_bar = i + 1

            # Pyramid Addition
            elif direction != 0 and len(units) < 4 and viable_order and i > last_action_bar:
                last_price = units[-1]["entry_price"]
                pyramid_step = 0.5 * atr
                add_sig = (close >= last_price + pyramid_step) if direction == 1 else (close <= last_price - pyramid_step)

                if add_sig:
                    units.append({
                        "unit_num": len(units) + 1,
                        "entry_price": next_open,
                        "entry_time": next_ts,
                        "quantity": unit_qty,
                        "equity_at_entry": equity,
                    })
                    active_stop = next_open - stop_dist if direction == 1 else next_open + stop_dist
                    last_action_bar = i + 1

        equity_curve.append(equity)
        equity_dates.append(timestamps[i])

    # Convert equity curve to series
    eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))
    daily_sr = calculate_daily_sharpe(eq_series)

    # ---------------- CYCLE-LEVEL (CLUSTER) INFERENCE ----------------
    # The pyramid UNIT is not an independent observation: up to 4 units share one
    # entry decision, one ratcheted stop and one exit, so their returns are
    # mechanically correlated. The independent observation is the completed
    # CYCLE, and cycles are further clustered by UTC exit day (a day of hostile
    # conditions can hit several cycles).
    cycle_bps = np.array([c["net_bps"] for c in completed_cycles], dtype=float)
    cycle_dates = [c["exit_day"] for c in completed_cycles]
    n_cycles = int(cycle_bps.size)

    if n_cycles >= 2:
        cycle_mean_bps = float(cycle_bps.mean())
        cycle_sd_bps = float(cycle_bps.std(ddof=1))
        cycle_se_bps = cycle_sd_bps / float(np.sqrt(n_cycles))

        cyc_df = pd.DataFrame({"day": cycle_dates, "bps": cycle_bps})
        day_grp = cyc_df.groupby("day")["bps"].agg(["count", "sum"])
        day_n = day_grp["count"].to_numpy(dtype=float)
        day_sums = day_grp["sum"].to_numpy(dtype=float)

        # CR0 cluster-robust SE of the mean with UTC days as clusters:
        #   Var(mean) = sum_d (T_d - n_d * mean)^2 / n_cycles^2
        resid_sums = day_sums - day_n * cycle_mean_bps
        day_se_bps = float(np.sqrt(np.sum(resid_sums ** 2) / (n_cycles ** 2)))
        # Design-effect implied effective sample size (used inside the DSR).
        n_eff_day = int(round(n_cycles * (cycle_se_bps / day_se_bps) ** 2)) if day_se_bps > 0 else n_cycles
        n_eff_day = max(2, min(n_cycles, n_eff_day))

        # Day-cluster bootstrap: resample whole UTC days with replacement.
        rng_c = np.random.default_rng(42)
        n_days = day_n.size
        draw = rng_c.integers(0, n_days, size=(2000, n_days))
        num = day_sums[draw].sum(axis=1)
        den = day_n[draw].sum(axis=1)
        boot_day = num / np.where(den == 0.0, np.nan, den)
        boot_day = boot_day[~np.isnan(boot_day)]
        day_ci_low = float(np.percentile(boot_day, 2.5))
        day_ci_high = float(np.percentile(boot_day, 97.5))

        # Cycle-block bootstrap (block = 10 consecutive completed cycles).
        _, cyc_ci_low, cyc_ci_high = block_bootstrap_ci(cycle_bps, block_size=10, n_bootstraps=2000)

        # Per-cycle Sharpe (mean/std of the CYCLE series) and DSR at T = n_cycles.
        cycle_sr = (cycle_mean_bps / cycle_sd_bps) if cycle_sd_bps > 1e-12 else 0.0
        cycle_dsr_p = compute_dsr(per_trade_sharpe=cycle_sr, n_trades=n_cycles, n_trials=1354)
        cycle_dsr_p_clustered = compute_dsr(per_trade_sharpe=cycle_sr, n_trades=n_eff_day, n_trials=1354)

        # Two-sided t-tests on the mean cycle bps (iid vs day-clustered SE).
        def _two_sided_p(t: float) -> float:
            return float(2.0 * (1.0 - stats.norm.cdf(abs(t))))

        t_iid = cycle_mean_bps / cycle_se_bps if cycle_se_bps > 0 else 0.0
        t_day = cycle_mean_bps / day_se_bps if day_se_bps > 0 else 0.0
        p_cycle_iid = _two_sided_p(t_iid)
        p_day_clustered = _two_sided_p(t_day)
    else:
        cycle_mean_bps = 0.0
        cycle_sd_bps = 0.0
        cycle_se_bps = 0.0
        day_se_bps = 0.0
        n_eff_day = 0
        day_ci_low = day_ci_high = 0.0
        cyc_ci_low = cyc_ci_high = 0.0
        cycle_sr = 0.0
        cycle_dsr_p = 1.0
        cycle_dsr_p_clustered = 1.0
        t_iid = t_day = 0.0
        p_cycle_iid = p_day_clustered = 1.0

    # Trade stats
    n_trades = len(completed_trades)
    if n_trades > 0:
        trades_df = pd.DataFrame(completed_trades)
        gross_bps_mean = float(trades_df["gross_bps"].mean())
        net_bps_mean = float(trades_df["net_bps"].mean())
        fee_bps_mean = float(trades_df["fee_bps"].mean())
        realized_risks = trades_df[trades_df["realized_risk_pct"] > 0]["realized_risk_pct"]
        avg_realized_risk_pct = float(realized_risks.mean()) if len(realized_risks) > 0 else 0.0

        wins = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
        losses = abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())
        profit_factor = (wins / losses) if losses > 0 else 0.0

        # Per-trade Sharpe ratio
        net_bps_arr = trades_df["net_bps"].to_numpy()
        trade_sr = float(np.mean(net_bps_arr) / np.std(net_bps_arr)) if np.std(net_bps_arr) > 1e-12 else 0.0

        # Block-bootstrap 95% CI
        _, net_ci_low, net_ci_high = block_bootstrap_ci(net_bps_arr, block_size=10, n_bootstraps=2000)

        # Monthly positive percentage
        monthly_eq = eq_series.resample("ME").last().pct_change().dropna()
        months_pos_pct = float((monthly_eq > 0).mean() * 100.0) if len(monthly_eq) > 0 else 0.0

    else:
        gross_bps_mean = 0.0
        net_bps_mean = 0.0
        fee_bps_mean = 0.0
        avg_realized_risk_pct = 0.0
        profit_factor = 0.0
        trade_sr = 0.0
        net_ci_low = 0.0
        net_ci_high = 0.0
        months_pos_pct = 0.0

    cap_bound_pct = (cap_bound_count / total_entries_count * 100.0) if total_entries_count > 0 else 0.0
    total_return_pct = ((equity - initial_capital) / initial_capital) * 100.0

    # Max Drawdown
    peaks = eq_series.cummax()
    dds = (peaks - eq_series) / peaks * 100.0
    max_dd_pct = float(dds.max())

    return {
        "n_trades": n_trades,
        "total_entries": total_entries_count,
        "cap_bound_pct": round(cap_bound_pct, 2),
        "is_capped_fixed_leverage": cap_bound_pct > 25.0,
        "avg_realized_risk_pct": round(avg_realized_risk_pct, 2),
        "pyramid_distribution": pyramid_distribution,
        "ambiguous_bars_count": ambiguous_bars_count,
        "gross_bps": round(gross_bps_mean, 2),
        "fee_bps": round(fee_bps_mean, 2),
        "net_bps": round(net_bps_mean, 2),
        "net_ci_95": [round(net_ci_low, 2), round(net_ci_high, 2)],
        # ---- cycle-level (cluster) inference ----
        "n_cycles": n_cycles,
        "cycle_net_bps_mean": round(cycle_mean_bps, 4),
        "cycle_net_bps_sd": round(cycle_sd_bps, 4),
        "cycle_se_bps": round(cycle_se_bps, 4),
        "day_clustered_se_bps": round(day_se_bps, 4),
        "n_eff_day_clustered": int(n_eff_day),
        "cycle_ci_95": [round(cyc_ci_low, 2), round(cyc_ci_high, 2)],
        "day_cluster_ci_95": [round(day_ci_low, 2), round(day_ci_high, 2)],
        "cycle_sharpe": round(cycle_sr, 4),
        "cycle_dsr_p_value": round(cycle_dsr_p, 4),
        "cycle_dsr_p_value_day_clustered": round(cycle_dsr_p_clustered, 4),
        "cycle_t_stat_iid": round(t_iid, 4),
        "cycle_t_stat_day_clustered": round(t_day, 4),
        "cycle_p_value_iid": round(p_cycle_iid, 4),
        "cycle_p_value_day_clustered": round(p_day_clustered, 4),
        "cycle_dates": cycle_dates,
        "cycle_net_bps": [round(float(b), 3) for b in cycle_bps],
        "profit_factor": round(profit_factor, 2),
        "daily_sharpe": round(daily_sr, 3),
        "per_trade_sharpe": round(trade_sr, 4),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "final_equity": round(equity, 2),
        "months_positive_pct": round(months_pos_pct, 2),
    }


def run_full_turtle_diagnostics():
    print("=" * 100)
    print("TURTLE FULL DIAGNOSTICS & RE-EVALUATION SUITE (36 CONFIGS)")
    print("=" * 100)

    assets = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
    timeframes = ["5m", "15m", "1h"]
    configs = [
        (20, 10, "TURTLE_20_10"),
        (20, 20, "TURTLE_20_20"),
        (55, 10, "TURTLE_55_10"),
        (55, 20, "TURTLE_55_20"),
    ]

    base_costs = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True)
    all_diagnostics = []

    for asset in assets:
        for tf in timeframes:
            kline_path = os.path.join(DATA_DIR, f"{asset}_{tf}.parquet")
            fund_path = os.path.join(DATA_DIR, f"{asset}_funding.parquet")
            df = pd.read_parquet(kline_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()

            funding = None
            if os.path.exists(fund_path):
                fdf = pd.read_parquet(fund_path)
                fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
                funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

            for N, M, cid in configs:
                feat = prepare_turtle_features(df, N=N, M=M)
                # OOS slice
                oos_feat = slice_features(feat, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
                diag = run_turtle_diagnostic_sim(oos_feat, funding, N, M, base_costs)

                # +/- 20% perturbation test
                n_up, m_up = int(round(N * 1.2)), int(round(M * 1.2))
                feat_up = prepare_turtle_features(df, N=n_up, M=m_up)
                oos_feat_up = slice_features(feat_up, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
                diag_up = run_turtle_diagnostic_sim(oos_feat_up, funding, n_up, m_up, base_costs)

                n_dn, m_dn = int(round(N * 0.8)), int(round(M * 0.8))
                feat_dn = prepare_turtle_features(df, N=n_dn, M=m_dn)
                oos_feat_dn = slice_features(feat_dn, "2025-04-01 00:00:00", "2026-03-31 23:59:59")
                diag_dn = run_turtle_diagnostic_sim(oos_feat_dn, funding, n_dn, m_dn, base_costs)

                diag["asset"] = asset
                diag["timeframe"] = tf
                diag["N"] = N
                diag["M"] = M
                diag["cid"] = cid
                diag["perturbation_plus20_net_bps"] = diag_up["net_bps"]
                diag["perturbation_minus20_net_bps"] = diag_dn["net_bps"]

                all_diagnostics.append(diag)
                label = "CAPPED FIXED-LEVERAGE DONCHIAN" if diag["is_capped_fixed_leverage"] else "Turtle Vol-Sized"
                print(f"[{asset:8} {tf:3} N={N:2} M={M:2}] Net: {diag['net_bps']:+6.2f} bps | Cycles: {diag['n_cycles']:5d} | Daily SR: {diag['daily_sharpe']:+6.3f} | Cap Binds: {diag['cap_bound_pct']:5.1f}% ({label}) | Pyramids: {diag['pyramid_distribution']} | Ambiguous: {diag['ambiguous_bars_count']}")

    # Re-evaluate Pooled with Daily Sharpe & Per-Trade DSR
    print("\n" + "=" * 100)
    print("CORRECTED POOLED RESULTS (DAILY SHARPE & PER-TRADE DSR)")
    print("=" * 100)

    pooled_corrected = []
    for tf in timeframes:
        for N, M, cid in configs:
            matching = [d for d in all_diagnostics if d["timeframe"] == tf and d["N"] == N and d["M"] == M]
            tot_trades = sum(d["n_trades"] for d in matching)
            pooled_net_bps = (sum(d["net_bps"] * d["n_trades"] for d in matching) / tot_trades) if tot_trades > 0 else 0.0
            pooled_gross_bps = (sum(d["gross_bps"] * d["n_trades"] for d in matching) / tot_trades) if tot_trades > 0 else 0.0
            mean_daily_sr = float(np.mean([d["daily_sharpe"] for d in matching]))
            mean_trade_sr = float(np.mean([d["per_trade_sharpe"] for d in matching]))
            avg_cap_bound = float(np.mean([d["cap_bound_pct"] for d in matching]))
            is_capped = avg_cap_bound > 25.0

            # DSR with per-trade Sharpe and T = tot_trades against N_trials = 1,354
            dsr_p = compute_dsr(per_trade_sharpe=mean_trade_sr, n_trades=tot_trades, n_trials=1354)

            # ---- pooled CYCLE-level (cluster) inference across the 3 assets ----
            pooled_cycle_bps = (np.concatenate([np.asarray(d["cycle_net_bps"], dtype=float) for d in matching])
                                if matching else np.zeros(0))
            pooled_cycle_days = [day for d in matching for day in d["cycle_dates"]]
            n_cycles = int(pooled_cycle_bps.size)

            if n_cycles >= 2:
                cyc_mean = float(pooled_cycle_bps.mean())
                cyc_sd = float(pooled_cycle_bps.std(ddof=1))
                cyc_se = cyc_sd / float(np.sqrt(n_cycles))
                cdf = pd.DataFrame({"day": pooled_cycle_days, "bps": pooled_cycle_bps})
                grp = cdf.groupby("day")["bps"].agg(["count", "sum"])
                g_n = grp["count"].to_numpy(dtype=float)
                g_s = grp["sum"].to_numpy(dtype=float)
                cyc_day_se = float(np.sqrt(np.sum((g_s - g_n * cyc_mean) ** 2) / (n_cycles ** 2)))
                if cyc_day_se > 0:
                    cyc_n_eff = int(round(n_cycles * (cyc_se / cyc_day_se) ** 2))
                else:
                    cyc_n_eff = n_cycles
                cyc_n_eff = max(2, min(n_cycles, cyc_n_eff))
                cyc_sharpe = cyc_mean / cyc_sd if cyc_sd > 1e-12 else 0.0
                _, cyc_ci_lo, cyc_ci_hi = block_bootstrap_ci(pooled_cycle_bps, block_size=10, n_bootstraps=2000)
                rng_p = np.random.default_rng(42)
                draw = rng_p.integers(0, g_n.size, size=(2000, g_n.size))
                num = g_s[draw].sum(axis=1)
                den = g_n[draw].sum(axis=1)
                boot_d = num / np.where(den == 0.0, np.nan, den)
                boot_d = boot_d[~np.isnan(boot_d)]
                cyc_day_ci = [float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))]
                cyc_dsr_p = compute_dsr(per_trade_sharpe=cyc_sharpe, n_trades=n_cycles, n_trials=1354)
                cyc_dsr_p_clustered = compute_dsr(per_trade_sharpe=cyc_sharpe, n_trades=cyc_n_eff, n_trials=1354)
                cyc_t_iid = cyc_mean / cyc_se if cyc_se > 0 else 0.0
                cyc_t_day = cyc_mean / cyc_day_se if cyc_day_se > 0 else 0.0
                cyc_p_iid = float(2.0 * (1.0 - stats.norm.cdf(abs(cyc_t_iid))))
                cyc_p_day = float(2.0 * (1.0 - stats.norm.cdf(abs(cyc_t_day))))
            else:
                cyc_mean = cyc_sd = cyc_se = cyc_day_se = cyc_sharpe = 0.0
                cyc_n_eff = 0
                cyc_ci_lo = cyc_ci_hi = 0.0
                cyc_day_ci = [0.0, 0.0]
                cyc_dsr_p = cyc_dsr_p_clustered = 1.0
                cyc_t_iid = cyc_t_day = cyc_p_iid = cyc_p_day = 0.0

            pooled_corrected.append({
                "timeframe": tf,
                "N": N,
                "M": M,
                "cid": cid,
                "total_trades": tot_trades,
                "pooled_gross_bps": round(pooled_gross_bps, 2),
                "pooled_net_bps": round(pooled_net_bps, 2),
                "mean_daily_sharpe": round(mean_daily_sr, 3),
                "mean_per_trade_sharpe": round(mean_trade_sr, 4),
                "dsr_p_value": round(dsr_p, 4),
                "avg_cap_bound_pct": round(avg_cap_bound, 2),
                "classification": "Capped Fixed-Leverage Donchian" if is_capped else "Volatility-Sized Turtle",
                "verdict": "FAIL",
                # ---- pooled cycle-level (cluster) inference ----
                "pooled_cycles": n_cycles,
                "cycle_net_bps_mean": round(cyc_mean, 4),
                "cycle_net_bps_sd": round(cyc_sd, 4),
                "cycle_se_bps": round(cyc_se, 4),
                "day_clustered_se_bps": round(cyc_day_se, 4),
                "n_eff_day_clustered": int(cyc_n_eff),
                "cycle_sharpe": round(cyc_sharpe, 4),
                "cycle_ci_95": [round(cyc_ci_lo, 2), round(cyc_ci_hi, 2)],
                "day_cluster_ci_95": [round(cyc_day_ci[0], 2), round(cyc_day_ci[1], 2)],
                "cycle_dsr_p_value": round(cyc_dsr_p, 4),
                "cycle_dsr_p_value_day_clustered": round(cyc_dsr_p_clustered, 4),
                "cycle_t_stat_iid": round(cyc_t_iid, 4),
                "cycle_t_stat_day_clustered": round(cyc_t_day, 4),
                "cycle_p_value_iid": round(cyc_p_iid, 4),
                "cycle_p_value_day_clustered": round(cyc_p_day, 4),
                # gross-positive / net-negative relabeling flag (item: cost-driven loss)
                "gross_positive_net_negative": bool(pooled_gross_bps > 0.0 and pooled_net_bps < 0.0),
            })
            print(f"[{tf:3} N={N:2} M={M:2}] Pooled Trades: {tot_trades:5} | Pooled Cycles: {n_cycles:5} | Gross: {pooled_gross_bps:+6.2f} bps | Net: {pooled_net_bps:+6.2f} bps | Daily SR: {mean_daily_sr:+6.3f} | DSR p: {dsr_p:.4f} | Cap Binds: {avg_cap_bound:5.1f}% -> {pooled_corrected[-1]['classification']}")
            print(f"            cycle(bps): mean {cyc_mean:+7.3f} | se(iid,cycle) {cyc_se:6.3f} | se(clustered by UTC day) {cyc_day_se:6.3f} | n_eff {cyc_n_eff:5d} | CI(cycle-block) [{cyc_ci_lo:+7.2f},{cyc_ci_hi:+7.2f}] | CI(day-cluster) [{cyc_day_ci[0]:+7.2f},{cyc_day_ci[1]:+7.2f}] | cycle Sharpe {cyc_sharpe:+6.3f} | DSR p {cyc_dsr_p:.4f} (clustered {cyc_dsr_p_clustered:.4f})")

    # Save to json
    out_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pooled_corrected": pooled_corrected,
        "diagnostics_individual": all_diagnostics,
    }
    out_path = "quant_research/results/turtle_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved full diagnostics to {out_path}")
    print("=" * 100)
    return out_payload


if __name__ == "__main__":
    run_full_turtle_diagnostics()

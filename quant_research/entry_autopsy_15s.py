"""
Entry autopsy on 15-second bars rebuilt from the Binance aggTrades tape.

`entry_autopsy.py` can only see the single 1m bar that was open when a fill
printed, so the intra-minute path is bracketed but not observed. This script
re-runs the same autopsy on 15s bars built by `build_15s_bars.py` from the
trade-by-trade tape, which makes part of that path visible.

Inputs
------
  quant_research/data/processed/<SYM>_15s.parquet       (built by build_15s_bars.py)
  quant_research/results/aggtrades_15s_provenance.json  (its provenance record)
  quant_research/data/processed/<SYM>_1m.parquet        (daily context only)
  quant_research/inputs/trade_history.csv
  quant_research/inputs/transaction_history.csv

Output
------
  quant_research/results/entry_autopsy_15s.json

Everything else - the entry definition (open/add), the clock pinning (UTC+3
primary, UTC+0 sensitivity), the 13 indicator conditions, the matched
(symbol, side, hour-of-day) random control sets and the Holm correction - is
imported unchanged from `entry_autopsy.py`, so the two artifacts are directly
comparable and differ only in the bar size.

Conventions
-----------
  * A bucket stamped T covers [T, T + 15s) and closes at T + 15s. The as-of
    bucket for an entry at time `when` is `when.floor('15s') - 15s`: the last
    bucket that had CLOSED strictly before the entry.
  * `b` is the bucket containing the print and `m0` is the first bucket of the
    print's minute, so the print sits in one of `m0 .. m0+3`.
  * FORWARD REFERENCES ARE MINUTE-ANCHORED, exactly like the 1m autopsy, so the
    two blocks are directly comparable rather than merely similar:
        actual_fill_price    : ref = fill price,     end = close[m0 + 4h - 1]
        open_of_fill_minute  : ref = open[m0],       end = close[m0 + 4h - 1]
        close_of_fill_minute : ref = close[m0 + 3],  end = close[m0 + 4h + 3]
        next_minute_open     : ref = open[m0 + 4],   end = close[m0 + 4h + 3]
    For h = 1 that is the close of the print's own minute for the first two and
    the close of the following minute for the last two - the same convention as
    `close[a + h - 1]` / `close[a + h]` in entry_autopsy.py.
  * TWO MEASURES EXIST ONLY HERE, because only 15s bars can make them:
        intra_15s_adverse_print : side * (fill / open[b] - 1) * 1e4, i.e. how far
            the print sat into its own 15s bucket against the trader.
        post_print_path_bps     : side * (close[b + k] / fill - 1) * 1e4 for
            k = 0..3. The key says `print_bucket_start + 15s(k+1)`, i.e. the
            elapsed time since the actual print is between 0 and 15 seconds LESS
            than the key, because the print is somewhere inside bucket b.
  * WARMUP is 300 bars, the same NUMBER OF BARS as the 1m run (75 minutes here
    versus 5 hours there), so both blocks use the same warmup discipline.
  * Daily context is resampled from the 1m close series for the same reason the
    1m run uses it: a daily EMA20 seeded at 2025-04-04 would be seed noise for
    the first weeks. This also makes the three daily conditions numerically
    identical between the two blocks.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.indicators import compute_indicators  # noqa: E402
from quant_research.entry_autopsy import (  # noqa: E402
    CTRL_PAD_DAYS, HORIZONS, INDICATORS, N_BOOT, N_SETS, PRIMARY_TZ_OFFSET,
    SENSITIVITY_TZ_OFFSET, SEED, boot_ci, build_entries, compare, load_fills,
    holm, timezone_evidence,
)

DATA_DIR = "quant_research/data/processed"
PROVENANCE = "quant_research/results/aggtrades_15s_provenance.json"
OUT_JSON = "quant_research/results/entry_autopsy_15s.json"
BAR = "15s"
WARMUP = 300
BPM = 4                      # 15s buckets per minute
LEAD = 3                     # last bucket of a minute is m0 + 3
TAIL = 3                     # ... so the following minute ends at m0 + 4 + 3


def load_features(sym):
    df = pd.read_parquet(f"{DATA_DIR}/{sym}_15s.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(df["timestamp"])
    contiguous = bool(np.all(np.diff((ts - ts[0]).total_seconds()) == 15.0))

    ind = compute_indicators(df[["open", "high", "low", "close", "volume"]])
    close = ind["close"].to_numpy(dtype=np.float64)
    bb_span = (ind["bb_upper"] - ind["bb_lower"]).replace(0, np.nan)
    feats = {
        "bb_pctb": ((close - ind["bb_lower"]) / bb_span).to_numpy(dtype=np.float64),
        "rsi6": ind["rsi6"].to_numpy(dtype=np.float64),
        "stoch_k": ind["stoch_k"].to_numpy(dtype=np.float64),
        "stoch_d": ind["stoch_d"].to_numpy(dtype=np.float64),
        "macd_hist_bps": (ind["macd_hist"] / ind["close"] * 1e4).to_numpy(dtype=np.float64),
        "ema20_dist_bps": ((ind["close"] / ind["ema20"] - 1.0) * 1e4).to_numpy(dtype=np.float64),
        "ema50_dist_bps": ((ind["close"] / ind["ema50"] - 1.0) * 1e4).to_numpy(dtype=np.float64),
        "ema80_dist_bps": ((ind["close"] / ind["ema80"] - 1.0) * 1e4).to_numpy(dtype=np.float64),
        "ema200_dist_bps": ((ind["close"] / ind["ema200"] - 1.0) * 1e4).to_numpy(dtype=np.float64),
        "adx14": ind["adx14"].to_numpy(dtype=np.float64),
    }

    # Daily context from the 1m series (see the module docstring).
    daily = pd.read_parquet(f"{DATA_DIR}/{sym}_1m.parquet").set_index("timestamp")["close"]
    daily = daily.sort_index().resample("1D").last().dropna()
    dema20 = daily.ewm(span=20, adjust=False).mean()
    d_trend = np.where(daily > dema20, 1.0, -1.0)
    d_dist = ((daily / dema20 - 1.0) * 1e4).to_numpy()
    d_c2c = np.where(daily.diff() > 0, 1.0, -1.0)
    full_days = pd.date_range(daily.index.min() - pd.Timedelta(days=1),
                              daily.index.max(), freq="1D")
    prev_day = ts.normalize() - pd.Timedelta(days=1)
    rows = pd.Index(full_days).get_indexer(prev_day)
    valid = rows >= 0
    rows = np.where(valid, rows, 0)
    trend_d = pd.Series(d_trend, index=daily.index).reindex(full_days).to_numpy()
    dist_d = pd.Series(d_dist, index=daily.index).reindex(full_days).to_numpy()
    c2c_d = pd.Series(d_c2c, index=daily.index).reindex(full_days).to_numpy()

    # Contiguity guards: indicators may not be read across a hole in the tape, and the
    # forward windows may not cross one either. `run_len` counts bars in the contiguous
    # run ending at i; `run_len_fwd` counts bars in the run starting at i.
    ts_np = ts.asi8
    step_ok = np.diff(ts_np) == 15 * 10 ** 9
    run_len = np.ones(len(ts), dtype=np.int64)
    for i in range(1, len(ts)):
        run_len[i] = run_len[i - 1] + 1 if step_ok[i - 1] else 1
    run_len_fwd = np.ones(len(ts), dtype=np.int64)
    for i in range(len(ts) - 2, -1, -1):
        run_len_fwd[i] = run_len_fwd[i + 1] + 1 if step_ok[i] else 1
    n_gap = int((~step_ok).sum())

    return {
        "sym": sym, "ts": ts,
        # position of each 15s bucket inside its minute (0..3), for minute anchors
        "sec_bucket": (ts.second // 15).to_numpy(dtype=np.int64),
        "open": df["open"].to_numpy(dtype=np.float64), "close": close,
        "feat": feats, "ts0": ts[0], "contiguous": contiguous, "n": len(ts),
        "n_gap_boundaries": n_gap, "run_len": run_len, "run_len_fwd": run_len_fwd,
        "d_trend_bar": np.where(valid, trend_d[rows], np.nan),
        "d_dist_bar": np.where(valid, dist_d[rows], np.nan),
        "d_c2c_bar": np.where(valid, c2c_d[rows], np.nan),
    }


def decision_index(fx, when):
    """Index of the last 15s bucket that closed strictly before `when`.

    searchsorted rather than offset arithmetic: the 15s file may contain days that were
    never downloaded, so array index != position on the global 15s grid. A bucket that is
    not present (or lies before the data) returns None and the entry is dropped, which is
    the honest outcome - no indicator may be computed across a hole in the tape.
    """
    bar = when.floor(BAR) - pd.Timedelta(seconds=15)
    j = int(np.searchsorted(fx["ts"], bar, side="right")) - 1
    if j < 0 or fx["ts"][j] != bar:
        return None
    return j


def minute_start(fx, bucket_idx):
    """First 15s bucket of the minute containing bucket index (or array of indices)."""
    idx = np.asarray(bucket_idx)
    return idx - fx["sec_bucket"][idx]


def evaluate(tz_offset, entries, features, seed):
    shift = pd.Timedelta(hours=tz_offset)
    print(f"\n--- 15s bars, CSV clock = UTC{tz_offset:+d}h ---")
    blocks = {}
    for unit in ("open", "add"):
        ue = entries[entries["kind"] == unit].copy()
        ue["utc_time"] = (ue["time"] - shift).dt.tz_localize("UTC")
        ue["hour"] = ue["utc_time"].dt.hour
        ue = ue[ue["symbol"].isin(features)]

        rows = []
        for _, e in ue.iterrows():
            fx = features[e["symbol"]]
            j = decision_index(fx, e["utc_time"])
            need = j + 1 + BPM * max(HORIZONS) + TAIL if j is not None else None
            if j is None or j < WARMUP or need >= fx["n"]:
                continue
            # no indicator and no forward window may span a hole in the tape
            if fx["run_len"][j] < WARMUP + 1 or fx["run_len_fwd"][j + 1] < need - j:
                continue
            rows.append((e, fx, j))
        n = len(rows)
        if n == 0:
            continue

        feat = np.empty((n, len(INDICATORS)))
        side = np.empty(n)
        fill = np.empty(n)
        b_idx = np.zeros(n, dtype=int)
        m0_idx = np.zeros(n, dtype=int)
        syms, stratum = [], []
        for i, (e, fx, j) in enumerate(rows):
            b = j + 1
            for col, name in enumerate(INDICATORS[:10]):
                feat[i, col] = fx["feat"][name][j]
            feat[i, 10] = fx["d_trend_bar"][j]
            feat[i, 11] = fx["d_dist_bar"][j]
            feat[i, 12] = fx["d_c2c_bar"][j]
            side[i] = float(e["direction"])
            fill[i] = float(e["price"])
            b_idx[i] = b
            m0_idx[i] = int(minute_start(fx, b))
            syms.append(e["symbol"])
            stratum.append((e["symbol"], int(e["hour"])))

        # minute-anchored forward endpoints, per row - the 15s analogue of the 1m run.
        # `tail` is the bucket offset from m0 + 4h at which the forward window closes:
        #  -1 = the close of the h-th minute, +3 = the close of the following minute.
        fx_of = [r[1] for r in rows]

        def close_at(m0_of, h, tail):
            want = m0_of + BPM * h + tail
            return np.array([fx_of[i]["close"][want[i]] for i in range(len(m0_of))])

        ref_real = {
            "actual_fill_price": (fill, -1),
            "open_of_fill_minute": (
                np.array([fx_of[i]["open"][m0_idx[i]] for i in range(n)]), -1),
            "close_of_fill_minute": (
                np.array([fx_of[i]["close"][m0_idx[i] + LEAD] for i in range(n)]), TAIL),
            "next_minute_open": (
                np.array([fx_of[i]["open"][m0_idx[i] + LEAD + 1] for i in range(n)]), TAIL),
        }
        fwd_real = {}
        for name, (ref, tail) in ref_real.items():
            m = np.empty((n, len(HORIZONS)))
            for k, h in enumerate(HORIZONS):
                m[:, k] = side * (close_at(m0_idx, h, tail) / ref - 1.0) * 1e4
            fwd_real[name] = m

        # ---- matched control pools on 15s bars ----
        pools = {}
        for sym, fx in features.items():
            win = ue[ue["symbol"] == sym]["utc_time"]
            if win.empty:
                continue
            lo = win.min() - pd.Timedelta(days=CTRL_PAD_DAYS)
            hi = win.max() + pd.Timedelta(days=CTRL_PAD_DAYS)
            hours = fx["ts"].hour.to_numpy()
            idx = np.arange(fx["n"])
            ok = (fx["ts"] >= lo) & (fx["ts"] <= hi)
            ok &= idx >= WARMUP
            ok &= idx + 1 + BPM * max(HORIZONS) + TAIL < fx["n"]
            ok &= fx["run_len"][idx] >= WARMUP + 1
            ok &= fx["run_len_fwd"][np.minimum(idx + 1, fx["n"] - 1)] >= BPM * max(HORIZONS) + TAIL + 1
            for hh in range(24):
                sel = np.where(ok & (hours == hh))[0]
                if sel.size:
                    pools[(sym, hh)] = sel

        rng = np.random.default_rng(seed)
        sums_feat = np.zeros((N_SETS, len(INDICATORS)))
        sums_fwd = {name: np.zeros((N_SETS, len(HORIZONS))) for name in ref_real
                    if name != "actual_fill_price"}
        n_drawn = 0
        for key in sorted({s for s in stratum if s in pools}):
            idxs = np.array([i for i in range(n) if stratum[i] == key])
            pool = pools[key]
            draws = rng.choice(pool, size=(N_SETS, len(idxs)), replace=True)
            fx = features[key[0]]
            a = draws + 1
            m0 = minute_start(fx, a)
            for col in range(10):
                sums_feat[:, col] += fx["feat"][INDICATORS[col]][draws].sum(axis=1)
            for col in range(10, 13):
                bar_vals = (fx["d_trend_bar"], fx["d_dist_bar"], fx["d_c2c_bar"])[col - 10]
                sums_feat[:, col] += bar_vals[draws].sum(axis=1)
            ctrl_ref = {
                "open_of_fill_minute": (fx["open"][m0], -1),
                "close_of_fill_minute": (fx["close"][m0 + LEAD], TAIL),
                "next_minute_open": (fx["open"][m0 + LEAD + 1], TAIL),
            }
            for name, (ref, tail) in ctrl_ref.items():
                for k, h in enumerate(HORIZONS):
                    end = fx["close"][m0 + BPM * h + tail]
                    sums_fwd[name][:, k] += (end / ref - 1.0).sum(axis=1) * 1e4
            n_drawn += len(idxs)
        if n_drawn == 0:
            continue
        null_feat = sums_feat / n_drawn
        null_fwd = {k: v / n_drawn for k, v in sums_fwd.items()}

        ind_rows = []
        for col, name in enumerate(INDICATORS):
            ind_rows.append(compare(name, feat[:, col], null_feat[:, col],
                                    np.random.default_rng(seed + 1)))
        for r, h in zip(ind_rows, holm([r["p_two_sided_empirical"] for r in ind_rows])):
            r["p_holm"] = float(h)
            r["significant_holm_05"] = bool(h < 0.05)

        fwd_blocks = {}
        for name in ref_real:
            if name == "actual_fill_price":
                fwd_blocks[name] = {
                    "horizons_bps": {f"{h}m": float(np.nanmean(fwd_real[name][:, k]))
                                     for k, h in enumerate(HORIZONS)},
                    "horizons_bps_ci95": {
                        f"{h}m": boot_ci(fwd_real[name][:, k], rng) for k, h in enumerate(HORIZONS)},
                }
                continue
            blk = {"horizons": []}
            for k, h in enumerate(HORIZONS):
                blk["horizons"].append(compare(f"forward_{h}m", fwd_real[name][:, k],
                                               null_fwd[name][:, k],
                                               np.random.default_rng(seed + 2), unit="bps"))
            for r, hh in zip(blk["horizons"], holm([r["p_two_sided_empirical"]
                                                    for r in blk["horizons"]])):
                r["p_holm"] = float(hh)
                r["significant_holm_05"] = bool(hh < 0.05)
            blk["any_significant_holm_05"] = bool(any(r["significant_holm_05"]
                                                      for r in blk["horizons"]))
            fwd_blocks[name] = blk

        # ---- the two observations 1m bars could not make ----
        exec15 = side * (fill / np.array([fx_of[i]["open"][b_idx[i]] for i in range(n)]) - 1.0) * 1e4
        exec60 = side * (fill / np.array([fx_of[i]["open"][m0_idx[i]] for i in range(n)]) - 1.0) * 1e4
        path = side[:, None] * (np.array(
            [[fx_of[i]["close"][b_idx[i] + k] for k in range(BPM)] for i in range(n)]) / fill[:, None] - 1.0) * 1e4
        post_print = {f"print_bucket_start+{15 * (k + 1)}s": float(np.nanmean(path[:, k]))
                      for k in range(BPM)}
        post_print_ci = {f"print_bucket_start+{15 * (k + 1)}s": boot_ci(path[:, k], rng)
                         for k in range(BPM)}

        per_sym = {}
        for sym in sorted(set(syms)):
            m = np.array([s == sym for s in syms])
            per_sym[sym] = {
                "n_entries": int(m.sum()),
                "from_fill_bps": {f"{h}m": float(np.nanmean(fwd_real["actual_fill_price"][m, k]))
                                  for k, h in enumerate(HORIZONS)},
                "from_open_of_fill_minute_bps": {
                    f"{h}m": float(np.nanmean(fwd_real["open_of_fill_minute"][m, k]))
                    for k, h in enumerate(HORIZONS)},
                "intra_15s_adverse_print_bps": float(np.nanmean(exec15[m])),
                "intra_minute_adverse_print_bps": float(np.nanmean(exec60[m])),
            }

        blocks[unit] = {
            "n_entries_used": int(n),
            "n_entries_dropped": int(len(ue) - n),
            "null_sets": N_SETS,
            "indicators": ind_rows,
            "indicator_family_size": len(ind_rows),
            "indicator_holm_any_significant": bool(any(r["significant_holm_05"] for r in ind_rows)),
            "forward_by_reference": fwd_blocks,
            "execution": {
                "definition": "side * (actual fill price / reference - 1) * 1e4; positive means "
                              "the print was WORSE than the reference in the direction of the trade",
                "against_15s_bucket_open_bps": float(np.nanmean(exec15)),
                "against_15s_bucket_open_bps_ci95": boot_ci(exec15, rng),
                "against_fill_minute_open_bps": float(np.nanmean(exec60)),
                "against_fill_minute_open_bps_ci95": boot_ci(exec60, rng),
                "post_print_path_bps": post_print,
                "post_print_path_bps_ci95": post_print_ci,
            },
            "per_symbol": per_sym,
            "control_pool_bars": int(sum(p.size for p in pools.values())),
            "control_pool_strata": len(pools),
        }

        print(f"\n  unit={unit}  entries_used={n}  pool_bars={blocks[unit]['control_pool_bars']}")
        print(f"  print vs its 15s bucket open : {exec15.mean():+.3f} bps {boot_ci(exec15, rng)}")
        print(f"  print vs its minute open     : {exec60.mean():+.3f} bps {boot_ci(exec60, rng)}")
        print("  post-print path              : " + ", ".join(
            f"{k} {v:+.3f}" for k, v in post_print.items()))
        print("  indicator                       real       null      diff     95% CI diff      p_emp   p_holm  sig")
        for r in ind_rows:
            print("  {:28s} {:>9.4f} {:>9.4f} {:>9.4f}  [{:>8.4f},{:>8.4f}]  {:>7.4f} {:>7.4f}  {}".format(
                r["metric"], r["real_mean"], r["null_mean_of_set_means"], r["diff_real_minus_null"],
                r["diff_ci95"][0], r["diff_ci95"][1], r["p_two_sided_empirical"], r["p_holm"],
                "YES" if r["significant_holm_05"] else "no"))
        print("\n  forward moves (bps, before fees)")
        for name, blk in fwd_blocks.items():
            if name == "actual_fill_price":
                for k, h in enumerate(HORIZONS):
                    print("  {:28s} {:>9.3f}".format(f"actual_fill_price {h}m",
                                                     float(np.nanmean(fwd_real[name][:, k]))))
                continue
            for r in blk["horizons"]:
                print("  {:28s} {:>9.3f} {:>9.3f} {:>9.3f}  [{:>8.3f},{:>8.3f}]  {:>7.4f} {:>7.4f}  {}".format(
                    f"{name} {r['metric'].split('_')[1]}", r["real_mean"], r["null_mean_of_set_means"],
                    r["diff_real_minus_null"], r["diff_ci95"][0], r["diff_ci95"][1],
                    r["p_two_sided_empirical"], r["p_holm"],
                    "YES" if r["significant_holm_05"] else "no"))
    return blocks


def _nan_check(results):
    """Guarantee the control pools never produced a NaN null mean (silent NaN loss)."""
    bad = []
    for label, blk_outer in results.items():
        for unit, blk in blk_outer["units"].items():
            for r in blk["indicators"]:
                if not np.isfinite(r["null_mean_of_set_means"]):
                    bad.append(f"{label}/{unit}/indicator/{r['metric']}")
            for name, ref in blk["forward_by_reference"].items():
                for r in ref.get("horizons", []):
                    if not np.isfinite(r["null_mean_of_set_means"]):
                        bad.append(f"{label}/{unit}/forward/{name}/{r['metric']}")
    return {"non_finite_null_means": bad, "ok": len(bad) == 0,
            "note": "a non-finite null mean means a drawn control bar carried NaN; the pools "
                    "are built so this cannot happen"}


def main():
    fills = load_fills()
    entries, cycles, per_symbol = build_entries(fills)
    tz_evidence = timezone_evidence()

    used = sorted(s for s in set(entries["symbol"])
                  if os.path.exists(f"{DATA_DIR}/{s}_15s.parquet"))
    missing = sorted(s for s in set(entries["symbol"]) if s not in used)
    print("=" * 112)
    print("ENTRY AUTOPSY ON 15-SECOND BARS (aggTrades) vs 1000 matched random timestamp sets")
    print("=" * 112)
    print(f"symbols with 15s bars : {used}")
    print(f"symbols WITHOUT       : {missing}  (entries excluded: "
          f"{int(entries['symbol'].isin(missing).sum())})")

    features = {}
    for s in used:
        features[s] = load_features(s)
        fx = features[s]
        print(f"  {s}: {fx['n']} 15s bars {fx['ts'][0]} -> {fx['ts'][-1]} contiguous={fx['contiguous']}")

    results = {}
    for label, tz in (("primary_csv_clock_utc_plus_3", PRIMARY_TZ_OFFSET),
                      ("sensitivity_csv_clock_utc", SENSITIVITY_TZ_OFFSET)):
        results[label] = {"csv_clock_offset_hours": tz,
                          "units": evaluate(tz, entries, features, SEED + tz)}

    provenance = {}
    if os.path.exists(PROVENANCE):
        with open(PROVENANCE) as fh:
            provenance = json.load(fh)

    prim = results["primary_csv_clock_utc_plus_3"]["units"]
    nan = _nan_check(results)
    print("\nnan check:", nan)
    payload = {
        "script": "quant_research/entry_autopsy_15s.py",
        "bar_builder": "quant_research/build_15s_bars.py",
        "inputs": {
            "bars": [f"{DATA_DIR}/{s}_15s.parquet" for s in used],
            "bar_provenance": PROVENANCE,
            "fills": "quant_research/inputs/trade_history.csv",
            "ledger": "quant_research/inputs/transaction_history.csv",
            "indicators": "quant_research/core/indicators.py::compute_indicators",
            "shared_method": "quant_research/entry_autopsy.py (imported: entry definition, "
                             "clock pinning, 13 indicators, matched controls, Holm)",
        },
        "bar_provenance": {s: provenance[s] for s in used if s in provenance},
        "timezone_evidence": tz_evidence,
        "check_no_nan_in_null": nan,
        "definitions": {
            "bar": "15s bucket stamped T covers [T, T + 15s); open/high/low/close are "
                   "first/max/min/last traded price, volume is summed base quantity",
            "as_of": "last 15s bucket stamped floor(entry_utc,'15s') - 15s, i.e. the last bucket "
                     "that had closed strictly before the entry",
            "print_bucket": "b = as-of index + 1, the bucket the print landed in",
            "print_minute_first_bucket": "m0 = the first 15s bucket of the minute containing b",
            "forward_references": {
                "actual_fill_price": "ref = fill price, end = close[m0 + 4h - 1]",
                "open_of_fill_minute": "ref = open[m0], end = close[m0 + 4h - 1]",
                "close_of_fill_minute": "ref = close[m0 + 3], end = close[m0 + 4h + 3]",
                "next_minute_open": "ref = open[m0 + 4], end = close[m0 + 4h + 3]",
                "why": "minute-anchored so the four references are the same tradeable instants "
                       "as the 1m autopsy's, making the two blocks directly comparable",
            },
            "warmup_bars": WARMUP,
            "indicator_metrics": INDICATORS,
            "intra_15s_adverse_print": "side * (fill / open[b] - 1) * 1e4 - the part of the "
                                       "adverse print that happened INSIDE the 15s bucket the "
                                       "trade printed in; the 1m run could only measure the "
                                       "whole minute",
            "post_print_path_bps": "side * (close[b + k] / fill - 1) * 1e4 for k = 0..3; the key "
                                   "`print_bucket_start+15s(k+1)` is the close of the (k+1)-th "
                                   "bucket after the print's own bucket, so elapsed time since "
                                   "the actual print is 0-15s LESS than the key says",
            "control": "same as entry_autopsy.py, drawn from 15s bars",
            "multiple_testing": "Holm within family (13 indicators; 4 horizons per reference)",
        },
        "results": results,
        "headline": {
            "primary_block": "primary_csv_clock_utc_plus_3 / open",
            "indicator_family_any_significant_holm_05":
                prim["open"]["indicator_holm_any_significant"],
            "post_print_path_bps": prim["open"]["execution"]["post_print_path_bps"],
            "intra_15s_adverse_print_bps":
                prim["open"]["execution"]["against_15s_bucket_open_bps"],
            "intra_15s_adverse_print_bps_ci95":
                prim["open"]["execution"]["against_15s_bucket_open_bps_ci95"],
            "intra_minute_adverse_print_bps":
                prim["open"]["execution"]["against_fill_minute_open_bps"],
            "from_actual_fill_price_bps":
                prim["open"]["forward_by_reference"]["actual_fill_price"]["horizons_bps"],
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

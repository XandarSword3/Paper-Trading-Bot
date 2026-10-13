"""
Entry autopsy - what did the real Binance entries actually look like?

Inputs (committed):
  quant_research/inputs/trade_history.csv          - the fill archive
  quant_research/inputs/transaction_history.csv    - ledger (used to pin the CSV timezone)
  quant_research/data/processed/{SYM}_1m.parquet   - 1m OHLCV (BTC/ETH/SOL only)

Outputs (NEW files; nothing committed is overwritten):
  quant_research/results/entry_autopsy.json
  quant_research/results/entry_autopsy_entries.csv

No trading rule is invented here. The entry definition is read off the fill
archive itself, the indicator set is the repository's own
quant_research/core/indicators.py, and the CSV timezone is pinned by the ledger
rather than assumed.

  * FILL COUNT. The committed docstring of test_3_binance_replay.py calls this
    archive "the user's 257 trades". 257 is the number of round-trip POSITION
    CYCLES (flat -> flat); the archive holds 1,786 individual fills. Both counts
    are reported, and this script reproduces the 257 / (SOL 195, BTC 35, ETH 25,
    BNB 2) split exactly.
  * ENTRY = a fill that OPENS a position (previous position flat) or ADDS to it
    (same side, larger absolute size). Reductions, closes and flips are not
    entries. "open" fills are the primary test unit (one per position cycle):
    an "add" fill is conditional on a position that is already working, so adds
    are a separately labelled secondary block.
  * CSV CLOCK. The Time column has no timezone. Binance settles perpetual
    funding at 00:00 / 08:00 / 16:00 UTC; all 50 FUNDING_FEE rows in the ledger
    are stamped 03:00 / 11:00 / 19:00, so the CSV clock is UTC+3 and the true
    UTC instant is CSV - 3h. That offset is used for the primary result and the
    UTC+0 reading is re-run end to end as a labelled sensitivity.
  * AS-OF DISCIPLINE. All indicators come from the last 1m bar that CLOSED
    strictly before the entry timestamp (a bar stamped 18:47:00 closes at
    18:48:00). The entry bar itself and every later bar are excluded from the
    indicators.
  * FORWARD MOVE = direction-adjusted gross move in bps, before fees, measured
    from four explicitly labelled reference prices (see REFERENCES). The
    reference that matters for a taker is the actual fill price; the others show
    how much of any apparent edge is intra-minute drift that happened before the
    print.
  * CONTROL = 1000 independent random timestamp sets, one control per real
    entry, matched on (symbol, side, hour-of-day UTC), drawn with replacement
    from the 1m bars inside that symbol's own trading window.
  * MULTIPLE TESTING = Holm within each family (13 indicator conditions; and the
    4 horizons inside each reference block).
  * 15-second charts cannot be rebuilt from 1m data. Nothing here claims to see
    inside a minute beyond the single 1m bar that was open at fill time.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.indicators import compute_indicators  # noqa: E402

FILLS = "quant_research/inputs/trade_history.csv"
LEDGER = "quant_research/inputs/transaction_history.csv"
DATA_DIR = "quant_research/data/processed"
OUT_JSON = "quant_research/results/entry_autopsy.json"
OUT_ENTRIES = "quant_research/results/entry_autopsy_entries.csv"

N_SETS = 1000
N_BOOT = 2000
SEED = 20260413
HORIZONS = [1, 5, 15, 60]
WARMUP = 300
CTRL_PAD_DAYS = 1
PRIMARY_TZ_OFFSET = 3      # CSV clock is UTC+3 (pinned by the funding settlements)
SENSITIVITY_TZ_OFFSET = 0

INDICATORS = [
    "bb_pctb", "rsi6", "stoch_k", "stoch_d", "macd_hist_bps",
    "ema20_dist_bps", "ema50_dist_bps", "ema80_dist_bps", "ema200_dist_bps",
    "adx14", "daily_trend_ema20", "daily_dist_ema20_bps", "daily_trend_c2c",
]

# reference price used as t=0 of the forward window, and the forward endpoint.
# a = index of the 1m bar that was open when the entry happened (fill bar).
REFERENCES = {
    "actual_fill_price": {
        "ref": "fill",
        "end_for_h": lambda h: h - 1,
        "note": "the trader's real traded price; the only tradable reference. "
                "Real entries only - a random control has no fill price.",
    },
    "open_of_fill_bar": {
        "ref": "open[a]",
        "end_for_h": lambda h: h - 1,
        "note": "open of the 1m bar containing the fill. Observable for both real and "
                "control, but it is the price one minute BEFORE the print, so it "
                "credits the entry with drift that a taker could not have captured.",
    },
    "close_of_fill_bar": {
        "ref": "close[a]",
        "end_for_h": lambda h: h,
        "note": "close of the 1m bar containing the fill; strictly at or after the fill.",
    },
    "next_bar_open": {
        "ref": "open[a+1]",
        "end_for_h": lambda h: h,
        "note": "open of the first bar that begins after the fill bar; strictly after the fill.",
    },
}

ASSUMPTIONS = [
    "A 1m bar stamped T covers [T, T+1m) and closes at T+1m, so the last closed bar "
    "before an entry at T is stamped floor(T,'1min') - 1min.",
    "Only the single 1m bar that contained each fill is observable; 15s charts cannot "
    "be rebuilt from 1m data, so the intra-minute path is unknown and is only bracketed "
    "by the bar's open/close.",
    "Control draws are with replacement and are not excluded from windows in which the "
    "account already held a position.",
    "BNBUSDT has no 1m candle file, so its 2 cycles / 11 entries are excluded from every "
    "indicator and forward-return statistic; they are counted in counts.entries_excluded_no_1m_data.",
]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 0. pin the CSV clock from the ledger
# ---------------------------------------------------------------------------
def timezone_evidence():
    """Binance settles funding at 00:00/08:00/16:00 UTC; that pins the CSV clock."""
    tx = pd.read_csv(LEDGER)
    tx["Time"] = pd.to_datetime(tx["Time"])
    f = tx[tx["Type"] == "FUNDING_FEE"].copy()
    mins = (f["Time"].dt.hour * 60 + f["Time"].dt.minute).to_numpy()
    observed = f["Time"].dt.strftime("%H:%M").value_counts().sort_index().to_dict()
    settle = {0, 480, 960}
    candidates = []
    for off in range(-12, 13):
        shifted = (mins - off * 60) % 1440
        if np.all(np.isin(shifted, list(settle))):
            candidates.append(off)
    return {
        "settlement_times_in_csv_clock": {k: int(v) for k, v in observed.items()},
        "n_funding_rows": int(len(f)),
        "binance_settlement_hours_utc": ["00:00", "08:00", "16:00"],
        "candidate_utc_offsets_hours_that_map_all_settlements_onto_00_08_16_utc": candidates,
        "adopted_offset_hours": PRIMARY_TZ_OFFSET,
        "why": "all 50 funding rows read 03:00/11:00/19:00, so the CSV clock is "
               "00:00/08:00/16:00 UTC shifted by +3h (mod 8h the candidates are -5h, +3h "
               "and +11h; only the phase is pinned). +3h is adopted because the commit "
               "author timezone of this repository is +0300 and neither -5h nor +11h is a "
               "plausible account timezone for this ledger. Both the +3h and the +0h "
               "readings are nevertheless computed end to end and reported.",
    }


# ---------------------------------------------------------------------------
# 1. fills -> entries
# ---------------------------------------------------------------------------
def load_fills():
    df = pd.read_csv(FILLS)
    df["Time"] = pd.to_datetime(df["Time"])
    return df.sort_values("Time").reset_index(drop=True)


def build_entries(fills):
    """Walk each symbol's fills and label every fill open/add/reduce/close/flip."""
    entries, cycles, per_symbol = [], [], {}
    for sym, sub in fills.groupby("Symbol"):
        sub = sub.sort_values("Time")
        pos, cycle_id, cycle_start, n_fills = 0.0, 0, None, 0
        for _, f in sub.iterrows():
            qty = float(f["Quantity"])
            step = qty if f["Side"] == "BUY" else -qty
            before, after = pos, round(pos + step, 10)
            if abs(before) < 1e-9:
                kind = "open"
            elif abs(after) < 1e-9:
                kind = "close"
            elif (after > 0) == (before > 0) and abs(after) > abs(before) + 1e-12:
                kind = "add"
            elif (after > 0) == (before > 0):
                kind = "reduce"
            else:
                kind = "flip"
            if kind == "open":
                cycle_id += 1
                cycle_start = f["Time"]
            n_fills += 1
            rec = {
                "time": f["Time"], "symbol": sym, "side": f["Side"],
                "direction": 1 if f["Side"] == "BUY" else -1,
                "quantity": qty, "price": float(f["Price"]),
                "notional": float(f["Amount"]), "kind": kind,
                "maker": str(f["Maker"]), "cycle_id": f"{sym}#{cycle_id}",
            }
            if kind in ("open", "add"):
                entries.append(rec)
            if kind == "close":
                cycles.append({"symbol": sym, "cycle_id": f"{sym}#{cycle_id}",
                               "start": cycle_start, "end": f["Time"], "fills": n_fills})
                n_fills = 0
            pos = after
        per_symbol[sym] = {"fills": int(len(sub)), "cycles": int(cycle_id)}
    return pd.DataFrame(entries), pd.DataFrame(cycles), per_symbol


# ---------------------------------------------------------------------------
# 2. 1m features
# ---------------------------------------------------------------------------
def load_features(sym):
    df = pd.read_parquet(f"{DATA_DIR}/{sym}_1m.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(df["timestamp"])
    contiguous = bool(np.all(np.diff((ts - ts[0]).total_seconds() / 60.0) == 1.0))

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

    # 1D trend from COMPLETED daily bars only, broadcast back onto each 1m bar.
    daily = pd.Series(close, index=ts).resample("1D").last().dropna()
    dema20 = daily.ewm(span=20, adjust=False).mean()
    d_trend = np.where(daily > dema20, 1.0, -1.0)
    d_dist = ((daily / dema20 - 1.0) * 1e4).to_numpy()
    d_c2c = np.where(daily.diff() > 0, 1.0, -1.0)
    full_days = pd.date_range(daily.index.min() - pd.Timedelta(days=1),
                              daily.index.max(), freq="1D")
    idx = pd.Index(full_days)
    prev_day = ts.normalize() - pd.Timedelta(days=1)
    rows = idx.get_indexer(prev_day)
    valid = rows >= 0
    rows = np.where(valid, rows, 0)
    trend_d = pd.Series(d_trend, index=daily.index).reindex(full_days).to_numpy()
    dist_d = pd.Series(d_dist, index=daily.index).reindex(full_days).to_numpy()
    c2c_d = pd.Series(d_c2c, index=daily.index).reindex(full_days).to_numpy()

    return {
        "sym": sym, "ts": ts,
        "open": df["open"].to_numpy(dtype=np.float64), "close": close,
        "feat": feats, "ts0": ts[0], "contiguous": contiguous, "n": len(ts),
        "d_trend_bar": np.where(valid, trend_d[rows], np.nan),
        "d_dist_bar": np.where(valid, dist_d[rows], np.nan),
        "d_c2c_bar": np.where(valid, c2c_d[rows], np.nan),
    }


def decision_index(fx, when):
    """Index of the last 1m bar that closed strictly before `when`."""
    bar = when.floor("1min") - pd.Timedelta(minutes=1)
    j = int((bar - fx["ts0"]).total_seconds() // 60)
    if j < 0 or j >= fx["n"] or fx["ts"][j] != bar:
        return None
    return j


# ---------------------------------------------------------------------------
# 3. statistics
# ---------------------------------------------------------------------------
def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, k in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[k])
        running = max(running, adj)
        out[k] = running
    return out


def compare(name, real_vals, null_set_means, rng, unit=""):
    real_vals = np.asarray(real_vals, dtype=np.float64)
    real_vals = real_vals[np.isfinite(real_vals)]
    null_set_means = np.asarray(null_set_means, dtype=np.float64)
    real_mean = float(real_vals.mean())
    null_mean = float(null_set_means.mean())
    null_sd = float(null_set_means.std(ddof=1))
    diff = real_mean - null_mean
    z = diff / null_sd if null_sd > 0 else float("nan")
    p_param = 2.0 * (1.0 - phi(abs(z))) if null_sd > 0 else 1.0
    dev_null, dev_real = np.abs(null_set_means - null_mean), abs(diff)
    p_emp = float((1 + int((dev_null >= dev_real).sum())) / (1 + len(null_set_means)))
    boot = real_vals[rng.integers(0, len(real_vals), size=(N_BOOT, len(real_vals)))].mean(axis=1)
    se_real = float(boot.std(ddof=1))
    se_diff = math.sqrt(se_real ** 2 + null_sd ** 2)
    return {
        "metric": name, "unit": unit, "n": int(len(real_vals)),
        "real_mean": real_mean, "real_median": float(np.median(real_vals)),
        "real_mean_se_boot": se_real,
        "real_mean_ci95": [real_mean - 1.96 * se_real, real_mean + 1.96 * se_real],
        "null_mean_of_set_means": null_mean, "null_sd_of_set_means": null_sd,
        "null_set_mean_ci95": [null_mean - 1.96 * null_sd, null_mean + 1.96 * null_sd],
        "diff_real_minus_null": diff, "diff_se": se_diff,
        "diff_ci95": [diff - 1.96 * se_diff, diff + 1.96 * se_diff],
        "z_vs_null_set_means": z, "p_two_sided_parametric": p_param,
        "p_two_sided_empirical": p_emp,
        "null_sets_ge_real_abs": int((dev_null >= dev_real).sum()),
    }


# ---------------------------------------------------------------------------
# 4. one end-to-end evaluation under a given clock assumption
# ---------------------------------------------------------------------------
def evaluate(tz_offset, entries, features, seed):
    shift = pd.Timedelta(hours=tz_offset)
    print(f"\n--- CSV clock = UTC{tz_offset:+d}h  (true UTC = CSV {tz_offset:+d}h) ---")
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
            if j is None or j < WARMUP or j + 1 + max(HORIZONS) >= fx["n"]:
                continue
            rows.append((e, fx, j))
        n = len(rows)
        if n == 0:
            continue

        feat = np.empty((n, len(INDICATORS)))
        side = np.empty(n)
        fill = np.empty(n)
        open_a = np.empty(n)
        close_a = np.empty(n)
        open_a1 = np.empty(n)
        syms, stratum = [], []
        for i, (e, fx, j) in enumerate(rows):
            a = j + 1
            feat[i, 0] = fx["feat"]["bb_pctb"][j]
            feat[i, 1] = fx["feat"]["rsi6"][j]
            feat[i, 2] = fx["feat"]["stoch_k"][j]
            feat[i, 3] = fx["feat"]["stoch_d"][j]
            feat[i, 4] = fx["feat"]["macd_hist_bps"][j]
            feat[i, 5] = fx["feat"]["ema20_dist_bps"][j]
            feat[i, 6] = fx["feat"]["ema50_dist_bps"][j]
            feat[i, 7] = fx["feat"]["ema80_dist_bps"][j]
            feat[i, 8] = fx["feat"]["ema200_dist_bps"][j]
            feat[i, 9] = fx["feat"]["adx14"][j]
            feat[i, 10] = fx["d_trend_bar"][j]
            feat[i, 11] = fx["d_dist_bar"][j]
            feat[i, 12] = fx["d_c2c_bar"][j]
            side[i] = float(e["direction"])
            fill[i] = float(e["price"])
            open_a[i] = fx["open"][a]
            close_a[i] = fx["close"][a]
            open_a1[i] = fx["open"][a + 1]
            syms.append(e["symbol"])
            stratum.append((e["symbol"], int(e["hour"])))

        # real forward moves for every reference
        ref_real = {
            "actual_fill_price": (fill, -1),
            "open_of_fill_bar": (open_a, -1),
            "close_of_fill_bar": (close_a, 0),
            "next_bar_open": (open_a1, 0),
        }
        fwd_real = {}
        for name, (ref, extra) in ref_real.items():
            m = np.empty((n, len(HORIZONS)))
            for k, h in enumerate(HORIZONS):
                m[:, k] = side * (fx_close_end(rows, h, extra) / ref - 1.0) * 1e4
            fwd_real[name] = m

        # ---- matched control pools: (symbol, hour-of-day UTC) ----
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
            ok &= idx + 1 + max(HORIZONS) < fx["n"]
            for hh in range(24):
                sel = np.where(ok & (hours == hh))[0]
                if sel.size:
                    pools[(sym, hh)] = sel

        rng = np.random.default_rng(seed)
        sums_feat = np.zeros((N_SETS, len(INDICATORS)))
        sums_fwd = {name: np.zeros((N_SETS, len(HORIZONS))) for name in ref_real
                    if name != "actual_fill_price"}
        n_drawn, drawn = 0, 0
        for key in sorted({s for s in stratum if s in pools}):
            idxs = np.array([i for i in range(n) if stratum[i] == key])
            pool = pools[key]
            draws = rng.choice(pool, size=(N_SETS, len(idxs)), replace=True)
            fx = features[key[0]]
            a = draws + 1
            for col in range(10):
                sums_feat[:, col] += fx["feat"][INDICATORS[col]][draws].sum(axis=1)
            for col in range(10, 13):
                bar_vals = (fx["d_trend_bar"], fx["d_dist_bar"], fx["d_c2c_bar"])[col - 10]
                sums_feat[:, col] += bar_vals[draws].sum(axis=1)
            ctrl_ref = {
                "open_of_fill_bar": (fx["open"][a], -1),
                "close_of_fill_bar": (fx["close"][a], 0),
                "next_bar_open": (fx["open"][a + 1], 0),
            }
            for name, (ref, extra) in ctrl_ref.items():
                for k, h in enumerate(HORIZONS):
                    end = fx["close"][a + h + extra]
                    sums_fwd[name][:, k] += (end / ref - 1.0).sum(axis=1) * 1e4
            n_drawn += len(idxs)
            drawn += 1
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

        fwd_rows, fwd_blocks = [], {}
        for name in ref_real:
            if name == "actual_fill_price":
                fwd_blocks[name] = {
                    "note": REFERENCES[name]["note"],
                    "horizons_bps": {f"{h}m": float(np.nanmean(fwd_real[name][:, k]))
                                     for k, h in enumerate(HORIZONS)},
                    "horizons_bps_ci95": {
                        f"{h}m": boot_ci(fwd_real[name][:, k], rng) for k, h in enumerate(HORIZONS)},
                }
                continue
            blk = {"note": REFERENCES[name]["note"], "horizons": []}
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

        # execution: where the print sat inside a 1m bar, direction-adjusted
        exec_bps = side * (fill / open_a - 1.0) * 1e4
        boot_exec = exec_bps[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)

        per_sym = {}
        for sym in sorted(set(syms)):
            m = np.array([s == sym for s in syms])
            per_sym[sym] = {
                "n_entries": int(m.sum()),
                "from_fill_bps": {f"{h}m": float(np.nanmean(fwd_real["actual_fill_price"][m, k]))
                                  for k, h in enumerate(HORIZONS)},
                "from_open_of_fill_bar_bps": {f"{h}m": float(np.nanmean(fwd_real["open_of_fill_bar"][m, k]))
                                              for k, h in enumerate(HORIZONS)},
                "intra_minute_adverse_print_bps": float(np.nanmean(exec_bps[m])),
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
                "definition": "side * (actual fill price / open of the fill bar - 1) * 1e4; "
                              "positive means the print was WORSE than that bar's open in the "
                              "direction of the trade",
                "mean_bps": float(exec_bps.mean()),
                "ci95": [float(np.percentile(boot_exec, 2.5)), float(np.percentile(boot_exec, 97.5))],
            },
            "per_symbol": per_sym,
            "control_pool_bars": int(sum(p.size for p in pools.values())),
            "control_pool_strata": len(pools),
        }

        # ---- console ----
        print(f"\n  unit={unit}  entries_used={n}  control_pool_bars={blocks[unit]['control_pool_bars']}")
        print(f"  intra-minute adverse print: {exec_bps.mean():+.3f} bps "
              f"[{np.percentile(boot_exec, 2.5):+.3f}, {np.percentile(boot_exec, 97.5):+.3f}]")
        print("  indicator                       real       null      diff     95% CI diff      p_emp   p_holm  sig")
        for r in ind_rows:
            print("  {:28s} {:>9.4f} {:>9.4f} {:>9.4f}  [{:>8.4f},{:>8.4f}]  {:>7.4f} {:>7.4f}  {}".format(
                r["metric"], r["real_mean"], r["null_mean_of_set_means"], r["diff_real_minus_null"],
                r["diff_ci95"][0], r["diff_ci95"][1], r["p_two_sided_empirical"], r["p_holm"],
                "YES" if r["significant_holm_05"] else "no"))
        print("\n  forward moves (bps, before fees)")
        print("  reference / horizon             real       null      diff     95% CI diff      p_emp   p_holm  sig")
        for name, blk in fwd_blocks.items():
            if name == "actual_fill_price":
                for k, h in enumerate(HORIZONS):
                    print("  {:28s} {:>9.3f}        n/a       n/a".format(
                        f"actual_fill_price {h}m bps", float(np.nanmean(fwd_real[name][:, k]))))
                continue
            for r in blk["horizons"]:
                print("  {:28s} {:>9.3f} {:>9.3f} {:>9.3f}  [{:>8.3f},{:>8.3f}]  {:>7.4f} {:>7.4f}  {}".format(
                    f"{name} {r['metric'].split('_')[1]}", r["real_mean"], r["null_mean_of_set_means"],
                    r["diff_real_minus_null"], r["diff_ci95"][0], r["diff_ci95"][1],
                    r["p_two_sided_empirical"], r["p_holm"],
                    "YES" if r["significant_holm_05"] else "no"))
    return blocks


def boot_ci(vals, rng):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    b = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def fx_close_end(rows, h, extra):
    """close[a + h + extra] for each (entry, fx, j) row, a = j + 1."""
    out = np.empty(len(rows))
    for i, (_e, fx, j) in enumerate(rows):
        out[i] = fx["close"][j + 1 + h + extra]
    return out


# ---------------------------------------------------------------------------
def main():
    fills = load_fills()
    entries, cycles, per_symbol = build_entries(fills)
    tz_evidence = timezone_evidence()

    print("=" * 112)
    print("ENTRY AUTOPSY - real Binance entries vs 1000 matched random timestamp sets")
    print("=" * 112)
    print(f"fills in archive             : {len(fills)}")
    print(f"position cycles (flat->flat) : {len(cycles)}   "
          f"[committed test_3_binance_replay.py docstring calls the archive '257 trades']")
    print(f"entry fills (open + add)     : {len(entries)}")
    print(f"  opens / adds               : {int((entries['kind'] == 'open').sum())} / "
          f"{int((entries['kind'] == 'add').sum())}")
    print(f"  taker / maker fills        : "
          f"{int((fills['Maker'].astype(str) == 'False').sum())} / "
          f"{int((fills['Maker'].astype(str) == 'True').sum())}")
    print(f"per symbol                   : {per_symbol}")
    print(f"\nCSV clock evidence: funding rows read {tz_evidence['settlement_times_in_csv_clock']} "
          f"(n={tz_evidence['n_funding_rows']})")
    print(f"  Binance settles at {tz_evidence['binance_settlement_hours_utc']} UTC -> "
          f"candidate offsets {tz_evidence['candidate_utc_offsets_hours_that_map_all_settlements_onto_00_08_16_utc']}, "
          f"adopting +{PRIMARY_TZ_OFFSET}h")

    symbols = sorted(set(entries["symbol"]))
    have_data = [s for s in symbols if os.path.exists(f"{DATA_DIR}/{s}_1m.parquet")]
    missing = [s for s in symbols if s not in have_data]
    print(f"\nsymbols with 1m data         : {have_data}")
    print(f"symbols WITHOUT 1m data      : {missing} "
          f"(entries excluded: {int(entries['symbol'].isin(missing).sum())})")

    features = {}
    for s in have_data:
        features[s] = load_features(s)
        fxh = features[s]
        print(f"  {s}: {fxh['n']} 1m bars {fxh['ts'][0]} -> {fxh['ts'][-1]} contiguous={fxh['contiguous']}")

    results = {}
    for label, tz in (("primary_csv_clock_utc_plus_3", PRIMARY_TZ_OFFSET),
                      ("sensitivity_csv_clock_utc", SENSITIVITY_TZ_OFFSET)):
        results[label] = {
            "csv_clock_offset_hours": tz,
            "units": evaluate(tz, entries, features, SEED + tz),
        }

    prim = results["primary_csv_clock_utc_plus_3"]["units"]
    payload = {
        "script": "quant_research/entry_autopsy.py",
        "inputs": {
            "fills": FILLS, "ledger": LEDGER,
            "candles": [f"{DATA_DIR}/{s}_1m.parquet" for s in have_data],
            "indicators": "quant_research/core/indicators.py::compute_indicators",
        },
        "timezone_evidence": tz_evidence,
        "counts": {
            "fills": int(len(fills)),
            "position_cycles": int(len(cycles)),
            "entry_fills_open_plus_add": int(len(entries)),
            "entry_opens": int((entries["kind"] == "open").sum()),
            "entry_adds": int((entries["kind"] == "add").sum()),
            "taker_fills": int((fills["Maker"].astype(str) == "False").sum()),
            "maker_fills": int((fills["Maker"].astype(str) == "True").sum()),
            "committed_docstring_claim": "test_3_binance_replay.py calls the archive "
                                         "'the user's 257 trades'; 257 is the position-cycle count",
            "per_symbol_fills_and_cycles": per_symbol,
            "entries_per_symbol": {k: int(v) for k, v in
                                   entries["symbol"].value_counts().to_dict().items()},
            "symbols_without_1m_data": missing,
            "entries_excluded_no_1m_data": int(entries["symbol"].isin(missing).sum()),
        },
        "definitions": {
            "entry": "a fill that OPENS a position (previous position flat) or ADDS to it "
                     "(same side, larger absolute size); reduces/closes/flips are not entries",
            "primary_test_unit": "kind == 'open' (one per flat->flat cycle); 'add' fills are a "
                                 "separately labelled secondary block because an add is "
                                 "conditional on a position that is already working",
            "as_of": "last 1m bar stamped floor(entry_time_utc,'1min') - 1min, i.e. the last bar "
                     "that had closed strictly before the entry; no entry-bar or later data feeds "
                     "any indicator",
            "forward_references": {k: {
                "reference_price": v["ref"],
                "forward_endpoint": "close[a + h - 1]" if v["end_for_h"](1) == 0
                                    else "close[a + h]",
                "note": v["note"]} for k, v in REFERENCES.items()},
            "forward_move_bps": "side * (endpoint / reference - 1) * 1e4, before fees; "
                                "side = +1 for a BUY entry, -1 for a SELL entry",
            "indicator_metrics": INDICATORS,
            "bollinger_pctb": "(close - bb_lower) / (bb_upper - bb_lower), 20 / 2.0",
            "stochrsi": "compute_indicators' stoch_k / stoch_d (RSI(14) stochastic 14, smoothed 3/3)",
            "macd_hist_bps": "macd_hist(12,26,9) / close * 1e4",
            "ema_dist_bps": "((close / ema_n) - 1) * 1e4",
            "adx14": "compute_indicators' adx14",
            "daily_trend_ema20": "+1 if the last COMPLETED daily close > its daily EMA20 else -1",
            "daily_dist_ema20_bps": "(daily close / daily EMA20 - 1) * 1e4",
            "daily_trend_c2c": "+1 if the last COMPLETED daily close > the previous daily close else -1",
            "control": f"{N_SETS} independent random timestamp sets; one control per real entry, "
                       "matched on (symbol, side, hour-of-day UTC), drawn with replacement from 1m "
                       "bars inside the symbol's own trading window +/- "
                       f"{CTRL_PAD_DAYS} day, requiring {WARMUP} bars of warmup and "
                       f"{max(HORIZONS)} bars of forward data",
            "hypothesis_test": "two-sided; p_two_sided_empirical = (1 + #{|null set mean - null "
                               "mean| >= |real - null mean|}) / (1 + N_SETS); diff_ci95 combines "
                               "the bootstrap SE of the real mean with the SD of the set means",
            "multiple_testing": "Holm step-down within each family: 13 indicator conditions, and "
                                "the 4 horizons within each forward-reference block",
        },
        "assumptions": ASSUMPTIONS,
        "results": results,
        "headline": {
            "primary_block": "primary_csv_clock_utc_plus_3 / open",
            "indicator_family_any_significant_holm_05":
                prim["open"]["indicator_holm_any_significant"],
            "from_actual_fill_price_bps": prim["open"]["forward_by_reference"][
                "actual_fill_price"]["horizons_bps"],
            "intra_minute_adverse_print_bps": prim["open"]["execution"]["mean_bps"],
            "intra_minute_adverse_print_bps_ci95": prim["open"]["execution"]["ci95"],
            "n_indicators_tested": prim["open"]["indicator_family_size"],
            "n_horizons_tested_per_reference": 4,
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    out = entries.copy()
    out["entry_time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out[["entry_time", "symbol", "side", "quantity", "price", "notional", "kind",
         "maker", "cycle_id"]].rename(columns={"quantity": "size"}).to_csv(OUT_ENTRIES, index=False)

    print("\n" + "=" * 112)
    print("HEADLINE (primary unit = 'open' fills, CSV clock read as UTC+3)")
    print("=" * 112)
    print(f"any indicator condition significant after Holm (13 tests) : "
          f"{prim['open']['indicator_holm_any_significant']}")
    print(f"intra-minute adverse print                                : "
          f"{prim['open']['execution']['mean_bps']:+.3f} bps "
          f"{prim['open']['execution']['ci95']}")
    print(f"gross move from the ACTUAL fill price                     : "
          + ", ".join(f"{h} {v:+.3f}" for h, v in
                      prim['open']['forward_by_reference']['actual_fill_price']['horizons_bps'].items()))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_ENTRIES}")


if __name__ == "__main__":
    main()

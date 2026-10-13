"""
Rebuild 15-second candles from Binance USDT-M futures aggTrades.

Why this exists
---------------
`entry_autopsy.py` can only see the single 1m bar that was open when a fill
printed, so it can bracket the intra-minute path but cannot observe it. The
aggTrades archive is the trade-by-trade tape, so 15s bars rebuilt from it make
that path visible. This script produces those bars; `entry_autopsy_15s.py`
consumes them.

Inputs
------
Daily archives from the Binance public data bucket:

    https://data.binance.vision/data/futures/um/daily/aggTrades/<SYM>/<SYM>-aggTrades-YYYY-MM-DD.zip

Each zip holds one CSV with columns

    agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker

where `transact_time` is a Unix epoch in MILLISECONDS UTC and `price`/`quantity`
are the aggregated trade price and base-asset size.

    export AGG_DIR=quant_research/data/raw/aggtrades   # default; gitignored
    SYM=SOLUSDT python quant_research/build_15s_bars.py

Outputs
-------
    quant_research/data/processed/<SYM>_15s.parquet        (gitignored, like the 1m files)
    quant_research/results/aggtrades_15s_provenance.json   (committed)

The provenance JSON records exactly which daily archives were consumed, how many
trades and 15s buckets each contributed, and a sha256 of the resulting bar file,
so every downstream number is traceable to a named upstream file.

Conventions
-----------
  * A bucket stamped T covers [T, T + 15s).
  * open / high / low / close are the first / max / min / last traded price in the
    bucket; volume is the summed base-asset quantity; n_trades is the number of
    aggregated trades; taker_buy_volume sums quantity for is_buyer_maker == False.
  * A 15s bucket in which nothing traded would break the fixed 15s grid the
    indicators assume. Such buckets are carried forward at the previous close
    with volume 0 and counted in `filled_gap_buckets` so the amount of invention
    is visible rather than hidden. All buckets are UTC; no CSV-clock offset is
    applied here.
"""
import json
import os
import sys
import zipfile
import hashlib

import numpy as np
import pandas as pd

AGG_DIR = os.environ.get("AGG_DIR", "quant_research/data/raw/aggtrades")
OUT_DIR = "quant_research/data/processed"
PROVENANCE = "quant_research/results/aggtrades_15s_provenance.json"
BAR = "15s"


def _read_zip(path):
    """One daily aggTrades zip -> DataFrame[ts, price, quantity, is_buyer_maker]."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(fh, usecols=["price", "quantity", "transact_time",
                                          "is_buyer_maker"])
    t = df["transact_time"].to_numpy(dtype=np.int64)
    # Binance has shipped both millisecond and microsecond epochs; detect by magnitude.
    unit = "us" if t.max() > 10 ** 15 else "ms"
    df["ts"] = pd.to_datetime(t, unit=unit, utc=True)
    return df[["ts", "price", "quantity", "is_buyer_maker"]], unit


def _to_bars(df):
    """Aggregate one day of trades onto the fixed 15s grid."""
    g = df.set_index("ts")
    bars = pd.DataFrame({
        "open": g["price"].resample(BAR).first(),
        "high": g["price"].resample(BAR).max(),
        "low": g["price"].resample(BAR).min(),
        "close": g["price"].resample(BAR).last(),
        "volume": g["quantity"].resample(BAR).sum(),
        "n_trades": g["price"].resample(BAR).count(),
        "taker_buy_volume": g.loc[~g["is_buyer_maker"], "quantity"].resample(BAR).sum(),
    })
    empty = bars["close"].isna()
    n_empty = int(empty.sum())
    if n_empty:
        # carry the previous close forward so the index stays a fixed 15s grid
        for col in ("open", "high", "low", "close"):
            bars.loc[empty, col] = bars["close"].ffill().loc[empty]
        bars["volume"] = bars["volume"].fillna(0.0)
        bars["taker_buy_volume"] = bars["taker_buy_volume"].fillna(0.0)
        bars["n_trades"] = bars["n_trades"].fillna(0).astype(np.int64)
    bars = bars.reset_index().rename(columns={"ts": "timestamp"})
    return bars, n_empty


def verify_against_1m(sym, bars):
    """Independent check of the rebuild: aggregate the 15s bars to 1m and compare with the

    repository's Binance 1m kline file for the same minutes.
    """
    path = f"{OUT_DIR}/{sym}_1m.parquet"
    if not os.path.exists(path):
        return {"checked": False, "reason": f"{path} not present"}
    m1 = pd.read_parquet(path)[["timestamp", "open", "high", "low", "close", "volume"]]
    agg = (bars.set_index("timestamp")
           .resample("1min")
           .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
           .dropna(subset=["close"]).reset_index())
    j = m1.merge(agg, on="timestamp", suffixes=("_1m", "_15s"))
    j = j.sort_values("timestamp").reset_index(drop=True)
    out = {"checked": True, "reference": path, "n_minutes_compared": int(len(j))}

    # Volume attribution: Binance builds its klines from the raw trade stream, while this
    # rebuild uses aggTrade rows, each of which carries a single timestamp for a block of
    # trades. Volume therefore shifts between ADJACENT minutes in equal-and-opposite pairs,
    # which leaves prices untouched. Count that pattern rather than blur it into a mean.
    dv = (j["volume_15s"] - j["volume_1m"]).to_numpy()
    n_diff = int((np.abs(dv) > 1e-9).sum())
    moving = (np.abs(dv) > 1e-9)
    n_pairs = int(((np.abs(dv[:-1] + dv[1:]) <= 1e-9)
                   & (moving[:-1] | moving[1:])).sum())
    out["volume_attribution"] = {
        "minutes_with_any_difference": n_diff,
        "adjacent_minute_pairs_that_exactly_offset": n_pairs,
        "note": "a pair means minute i carries +x and minute i+1 carries -x, so the day's "
                "total volume is unchanged and only the minute assignment of a block of "
                "trades differs; the repo's klines come from the raw trade stream while "
                "this rebuild uses aggTrade rows (one timestamp per aggregated block)",
    }
    for col in ("open", "high", "low", "close", "volume"):
        d = (j[f"{col}_15s"] - j[f"{col}_1m"]).abs()
        denom = j[f"{col}_1m"].abs().replace(0.0, np.nan)
        out[col] = {
            "n_exact": int((d <= 0).sum()),
            "n_within_1e-9_rel": int(((d / denom) <= 1e-9).sum()),
            "max_abs_diff": float(d.max()),
            "max_rel_diff": float((d / denom).max()),
            "mean_abs_diff": float(d.mean()),
        }
    return out


def build(sym):
    files = sorted(f for f in os.listdir(AGG_DIR)
                   if f.startswith(f"{sym}-aggTrades-") and f.endswith(".zip"))
    if not files:
        raise SystemExit(f"no aggTrades zips for {sym} in {AGG_DIR}")
    parts, per_day, units = [], [], set()
    for f in files:
        day = f[len(f"{sym}-aggTrades-"):-len(".zip")]  # YYYY-MM-DD
        df, unit = _read_zip(os.path.join(AGG_DIR, f))
        units.add(unit)
        bars, n_empty = _to_bars(df)
        parts.append(bars)
        per_day.append({
            "file": f, "date_utc": day, "n_trades": int(len(df)),
            "n_buckets": int(len(bars)), "gap_buckets_filled": n_empty,
            "first_ts": str(bars["timestamp"].iloc[0]),
            "last_ts": str(bars["timestamp"].iloc[-1]),
            "bytes": int(os.path.getsize(os.path.join(AGG_DIR, f))),
            "sha256": hashlib.sha256(open(os.path.join(AGG_DIR, f), "rb").read()).hexdigest(),
        })
        print(f"  {f}: {len(df)} trades -> {len(bars)} 15s buckets "
              f"({n_empty} gaps filled)")
    out = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp")
    out = out.sort_values("timestamp").reset_index(drop=True)
    # fixed-grid check, and an explicit record of any hole left by days that were not
    # downloaded (the autopsy refuses to read an indicator across one of these)
    dt = out["timestamp"].diff().dropna().dt.total_seconds()
    gap_idx = np.where(dt.to_numpy() != 15.0)[0]  # dt.iloc[i] spans rows i -> i+1
    gaps = [{"after_bucket": str(out["timestamp"].iloc[i]),
             "missing_seconds": float(dt.iloc[i]) - 15.0,
             "missing_buckets": int(round(dt.iloc[i] / 15.0 - 1))} for i in gap_idx]
    dt = dt.unique()
    path = f"{OUT_DIR}/{sym}_15s.parquet"
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {
        "symbol": sym,
        "bar": BAR,
        "source_url_pattern": "https://data.binance.vision/data/futures/um/daily/"
                              "aggTrades/{sym}/{sym}-aggTrades-YYYY-MM-DD.zip",
        "source_dir": AGG_DIR,
        "epoch_unit_in_source": sorted(units),
        "n_source_files": len(files),
        "date_range_utc": [per_day[0]["date_utc"], per_day[-1]["date_utc"]],
        "n_rows": int(len(out)),
        "n_trades_total": int(sum(d["n_trades"] for d in per_day)),
        "gap_buckets_filled": int(sum(d["gap_buckets_filled"] for d in per_day)),
        "first_ts": str(out["timestamp"].iloc[0]),
        "last_ts": str(out["timestamp"].iloc[-1]),
        "grid_step_seconds": sorted(float(x) for x in dt),
        "n_gap_boundaries": len(gaps),
        "contiguous": len(gaps) == 0,
        "gaps": gaps,
        "bars_path": path,
        "bars_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "cross_check_15s_aggregated_to_1m_vs_binance_1m_klines": verify_against_1m(sym, out),
        "per_day": per_day,
    }
    return out, meta


def main():
    syms = os.environ.get("SYM", "SOLUSDT BTCUSDT ETHUSDT").split()
    existing = {}
    if os.path.exists(PROVENANCE):
        with open(PROVENANCE) as fh:
            existing = json.load(fh)
    for sym in syms:
        print(f"=== {sym} ===")
        bars, meta = build(sym)
        existing[sym] = meta
        print(f"  {sym}: {meta['n_rows']} bars {meta['first_ts']} -> {meta['last_ts']} "
              f"trades={meta['n_trades_total']} grid={meta['grid_step_seconds']}")
    os.makedirs(os.path.dirname(PROVENANCE), exist_ok=True)
    with open(PROVENANCE, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"wrote {PROVENANCE}")


if __name__ == "__main__":
    main()

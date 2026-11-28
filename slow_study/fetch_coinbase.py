"""Fetch OHLCV from Coinbase Exchange public API (no key).

WHY NOT ONLY OKX: verified that OKX /market/candles serves at most the most
recent 1440 bars per bar size (daily -> back to ~Dec 2022 only; 4h -> ~240
days). Coinbase /products/{id}/candles serves full history back to listing,
paginated 350 bars/request. Spec allows "OKX or Coinbase".

Coinbase candle row format: [time, low, high, open, close, volume].
Granularity seconds: 86400 (daily), 14400 (4h).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.exchange.coinbase.com"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# Coinbase product ids (USD pairs). BNB-USD/TRX-USD availability verified at
# runtime; missing products are logged, not fabricated.
PRODUCTS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "BNB": "BNB-USD", "DOGE": "DOGE-USD", "ADA": "ADA-USD", "AVAX": "AVAX-USD",
    "LINK": "LINK-USD", "DOT": "DOT-USD", "LTC": "LTC-USD", "TRX": "TRX-USD",
}
TFS = {"1D": 86400, "4H": 14400}
START = "2019-01-01T00:00:00Z"
PAGE = 350
SLEEP = 0.15


def get_candles(product, granularity, start, end):
    r = requests.get(f"{BASE}/products/{product}/candles", params={
        "granularity": granularity, "start": start, "end": end,
    }, timeout=20, headers={"User-Agent": "slow-study/1.0"})
    if r.status_code != 200:
        raise RuntimeError(f"{product} {granularity}: HTTP {r.status_code} {r.text[:120]}")
    return r.json()


def fetch(product, granularity, start=START):
    """Paginate backwards from now to `start`. Coinbase allows at most ~300
    candles per request, so each call covers a bounded window."""
    rows = []
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    window = 300 * granularity  # seconds covered per request
    while end_ts > start_ts:
        win_start = max(start_ts, end_ts - window)
        s = pd.Timestamp(win_start, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        e = pd.Timestamp(end_ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            data = get_candles(product, granularity, s, e)
        except Exception as ex:
            print(f"  page fail {product} {granularity} {s}..{e}: {ex}")
            break
        if not data:
            break
        rows.extend(data)
        oldest = min(int(r[0]) for r in data)
        if oldest <= start_ts or oldest >= end_ts:
            # no progress or reached start
            if oldest <= start_ts:
                break
            break
        end_ts = oldest
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "vol"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    df = df[df["ts"] >= start_ts]
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    # drop the still-forming candle (its interval hasn't completed)
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    if len(df) and (now_ts - int(df["ts"].iloc[-1])) < granularity:
        df = df.iloc[:-1].reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[["dt", "open", "high", "low", "close", "vol"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=list(PRODUCTS))
    args = ap.parse_args()
    manifest = {}
    for sym in args.symbols:
        product = PRODUCTS[sym]
        for tf_key, gran in TFS.items():
            path = OUT / f"{product}_{tf_key}.csv"
            if path.exists() and not args.force:
                print(f"skip {path.name}")
                continue
            try:
                df = fetch(product, gran)
            except Exception as e:
                print(f"FAIL {product} {tf_key}: {e}")
                continue
            if df.empty:
                print(f"EMPTY {product} {tf_key}")
                continue
            df.to_csv(path, index=False)
            manifest[path.name] = {"rows": len(df), "first": str(df["dt"].iloc[0]),
                                   "last": str(df["dt"].iloc[-1])}
            print(f"OK {path.name}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
    mpath = OUT / "MANIFEST_coinbase.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()

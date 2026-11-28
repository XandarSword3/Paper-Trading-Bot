"""Fetch daily + 4h OHLCV and funding-rate history from OKX public REST (no API key).

Saves CSVs to slow_study/data/. Resumable: skips files that already exist unless
--force is passed. No fabricated data: every row comes from the OKX response.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.okx.com"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# BTC, ETH + ~10 of today's most liquid OKX spot majors.
# SURVIVORSHIP NOTE: this is a CURRENT top-liquidity universe, not point-in-time.
SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX",
]
TIMEFRAMES = {"1D": "1D", "4H": "4H"}  # OKX bar codes are uppercase
START_MS = 1546300800000  # 2019-01-01 UTC
PAGE = 100
SLEEP = 0.12  # stay well under OKX public rate limits


def get(path, params, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=20)
            if r.status_code == 200:
                body = r.json()
                if body.get("code") == "0":
                    return body.get("data", [])
                # OKX error; back off
                time.sleep(1.5 * (attempt + 1))
            else:
                time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OKX request failed after retries: {path} {params}")


def fetch_candles(inst_id, bar, start_ms=START_MS):
    """Return DataFrame of candles from start_ms to now, oldest-first."""
    rows = []
    after = None
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": PAGE}
        if after is not None:
            params["after"] = after
        data = get("/api/v5/market/candles", params)
        if not data:
            break
        rows.extend(data)
        oldest = min(int(r[0]) for r in data)
        if oldest <= start_ms or len(data) < PAGE:
            break
        after = oldest
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_ccy_quote", "confirm"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    df = df[df["ts"] >= start_ms]
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["dt", "open", "high", "low", "close", "vol"]]


def fetch_funding(inst_id_swap, start_ms=START_MS):
    """Funding-rate history for a perpetual swap instId (e.g. BTC-USDT-SWAP)."""
    rows = []
    after = None
    while True:
        params = {"instId": inst_id_swap, "limit": PAGE}
        if after is not None:
            params["after"] = after
        try:
            data = get("/api/v5/public/funding-rate-history", params)
        except RuntimeError:
            break
        if not data:
            break
        rows.extend(data)
        oldest = min(int(r["fundingTime"]) for r in data)
        if oldest <= start_ms or len(data) < PAGE:
            break
        after = oldest
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
    df = df[df["fundingTime"] >= start_ms]
    df = df.sort_values("fundingTime").drop_duplicates("fundingTime").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df[["dt", "fundingRate"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    args = ap.parse_args()

    manifest = {}
    for sym in args.symbols:
        inst = f"{sym}-USDT"
        for bar_key, bar in TIMEFRAMES.items():
            path = OUT / f"{inst}_{bar_key}.csv"
            if path.exists() and not args.force:
                print(f"skip {path.name} (exists)")
                continue
            try:
                df = fetch_candles(inst, bar)
            except Exception as e:
                print(f"FAIL candles {inst} {bar}: {e}")
                continue
            if df.empty:
                print(f"EMPTY candles {inst} {bar}")
                continue
            df.to_csv(path, index=False)
            manifest[path.name] = {
                "rows": len(df), "first": str(df["dt"].iloc[0]), "last": str(df["dt"].iloc[-1]),
            }
            print(f"OK {path.name}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")

        # funding history on the USDT-margined perpetual
        fpath = OUT / f"{inst}-SWAP_funding.csv"
        if fpath.exists() and not args.force:
            print(f"skip {fpath.name} (exists)")
            continue
        try:
            fdf = fetch_funding(f"{inst}-SWAP")
        except Exception as e:
            print(f"FAIL funding {inst}: {e}")
            continue
        if fdf.empty:
            print(f"EMPTY funding {inst}")
            continue
        fdf.to_csv(fpath, index=False)
        manifest[fpath.name] = {
            "rows": len(fdf), "first": str(fdf["dt"].iloc[0]), "last": str(fdf["dt"].iloc[-1]),
        }
        print(f"OK {fpath.name}: {len(fdf)} rows {fdf['dt'].iloc[0]} -> {fdf['dt'].iloc[-1]}")

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()

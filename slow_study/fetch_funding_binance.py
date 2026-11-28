"""Fetch deep BTCUSDT perpetual funding history from Binance public REST.

WHY: OKX /public/funding-rate-history only serves ~the last 3 months publicly
(verified: 'after'/'before' pagination returns empty beyond that window).
Binance fapi is reachable from this machine (it is only blocked in the user's
CI). This file is used ONLY for the S5 carry benchmark; the substitution is
documented in README/REPORT as a data-source caveat.
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
START_MS = 1577836800000  # 2020-01-01 (BTCUSDT perp funding starts ~2019-09)


def main():
    rows = []
    start = START_MS
    while True:
        r = requests.get(URL, params={
            "symbol": "BTCUSDT", "startTime": start, "limit": 1000,
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last = int(data[-1]["fundingTime"])
        if len(data) < 1000:
            break
        start = last + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
    df = df.sort_values("fundingTime").drop_duplicates("fundingTime")
    df["dt"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df[["dt", "fundingRate"]]
    path = OUT / "funding_BTCUSDT_binance.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path.name}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")


if __name__ == "__main__":
    main()

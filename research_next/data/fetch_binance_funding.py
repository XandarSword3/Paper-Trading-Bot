"""Fetch deep perpetual funding-rate history from Binance USD-M futures (public).

OKX public funding history only serves ~3 months; Binance fapi serves full
history from listing. Used for Strategy A (multi-asset funding arb).

Data-source caveat: funding is Binance, spot leg is OKX USDT spot. Funding
rates across major venues are highly correlated but NOT identical; this is a
documented approximation (same caveat class as S5 v1/v2).
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
START_MS = 1577836800000  # 2020-01-01

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX"]


def fetch(sym):
    inst = f"{sym}USDT"
    rows = []
    start = START_MS
    while True:
        r = requests.get(URL, params={"symbol": inst, "startTime": start, "limit": 1000}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last = int(data[-1]["fundingTime"])
        if len(data) < 1000:
            break
        start = last + 1
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["fundingTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
    df = df.sort_values("fundingTime").drop_duplicates("fundingTime")
    df["dt"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df[["dt", "fundingRate"]]


def main():
    for sym in SYMBOLS:
        path = OUT / f"funding_{sym}USDT_binance.csv"
        if path.exists():
            print(f"skip {path.name}")
            continue
        try:
            df = fetch(sym)
        except Exception as e:
            print(f"FAIL {sym}: {e}")
            continue
        if df is None or df.empty:
            print(f"EMPTY {sym}")
            continue
        df.to_csv(path, index=False)
        print(f"OK {path.name}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
    print("done")


if __name__ == "__main__":
    main()

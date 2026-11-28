"""Fetch Bybit public market data for venue-consistent S5 v3 backtest.

Endpoints (all public, no auth):
- /v5/market/funding/history      -> deep funding history (8h settlements)
- /v5/market/mark-price-kline     -> mark price klines (linear perp)
- /v5/market/kline?category=spot  -> spot BTCUSDT klines
- /v5/market/risk-limit           -> maintenance margin tiers

Outputs to slow_study/data/bybit/.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "bybit"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://api.bybit.com"
START_MS = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime.now(timezone.utc).timestamp() * 1000)


def get(path, params, retries=4):
    for i in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=20)
            j = r.json()
            if j.get("retCode") == 0:
                return j["result"]
            if j.get("retCode") == 10006:  # rate limit
                time.sleep(1.5 * (i + 1))
                continue
            raise RuntimeError(f"{path} retCode={j.get('retCode')} {j.get('retMsg')}")
        except requests.RequestException:
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{path} failed after retries")


def fetch_funding():
    rows = []
    end = END_MS
    while True:
        res = get("/v5/market/funding/history", {
            "category": "linear", "symbol": "BTCUSDT",
            "startTime": START_MS, "endTime": end, "limit": 200})
        data = res.get("list", [])
        if not data:
            break
        rows.extend(data)
        oldest = min(int(x["fundingRateTimestamp"]) for x in data)
        if oldest <= START_MS or len(data) < 200:
            break
        end = oldest - 1
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df["fundingRateTimestamp"] = pd.to_numeric(df["fundingRateTimestamp"]).astype("int64")
    df["dt"] = pd.to_datetime(df["fundingRateTimestamp"], unit="ms", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt")[["dt", "fundingRate"]]
    df.to_csv(OUT / "funding_BTCUSDT_bybit_full.csv", index=False)
    print(f"funding: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")


def fetch_klines(path, category, symbol, interval, out_name, start=START_MS):
    rows = []
    end = END_MS
    while True:
        res = get(path, {"category": category, "symbol": symbol,
                         "interval": interval, "start": start, "end": end,
                         "limit": 1000})
        data = res.get("list", [])
        if not data:
            break
        rows.extend(data)
        oldest = min(int(x[0]) for x in data)
        if oldest <= start or len(data) < 1000:
            break
        end = oldest - 1
        time.sleep(0.25)
    ncol = len(data[0])
    cols = ["ts", "open", "high", "low", "close", "vol", "turnover"][:ncol]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"]).astype("int64")
    df = df.sort_values("ts").drop_duplicates("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["dt", "open", "high", "low", "close"] + (["vol"] if "vol" in df.columns else [])]
    df.to_csv(OUT / out_name, index=False)
    print(f"{out_name}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")


def interval_ms(iv):
    m = {"1": 60000, "3": 180000, "5": 300000, "15": 900000, "30": 1800000,
         "60": 3600000, "120": 7200000, "240": 14400000, "360": 21600000,
         "720": 43200000, "D": 86400000, "W": 604800000, "M": 2592000000}
    return m[iv]


def fetch_risk_limit():
    res = get("/v5/market/risk-limit", {"category": "linear", "symbol": "BTCUSDT"})
    (OUT / "risk_limit_BTCUSDT.json").write_text(json.dumps(res, indent=2))
    print("risk-limit tiers saved:", len(res.get("list", [])))


def main():
    fetch_funding()
    fetch_risk_limit()
    fetch_klines("/v5/market/mark-price-kline", "linear", "BTCUSDT", "60",
                 "mark_BTCUSDT_1H.csv")
    fetch_klines("/v5/market/kline", "spot", "BTCUSDT", "D",
                 "spot_BTCUSDT_1D.csv")
    fetch_klines("/v5/market/kline", "linear", "BTCUSDT", "D",
                 "perp_BTCUSDT_1D.csv")
    print("done")


if __name__ == "__main__":
    main()

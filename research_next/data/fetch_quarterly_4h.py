"""Fetch 4H klines for Binance USD-M quarterly futures (alignment fix).

OKX spot 1D candles are stamped 16:00 UTC (candle covers 16:00->16:00 UTC).
Binance futures daily candles are 00:00->24:00 UTC. Using daily futures
closes against 16:00 spot stamps mixes an 8h look-ahead window.

4H futures candles (ts = 00/04/08/12/16/20 UTC) align exactly: the candle
opening at d-4h closes at d, matching the spot candle close at d=16:00 UTC.
"""
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "quarterly_4h"
OUT.mkdir(exist_ok=True)
URL = "https://fapi.binance.com/fapi/v1/klines"

BASES = ["BTC", "ETH"]
START_YEAR, END_YEAR = 2020, 2026


def last_friday(year, month):
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    while last.weekday() != 4:
        last -= timedelta(days=1)
    return last


def contract_expiries():
    return sorted(last_friday(y, m) for y in range(START_YEAR, END_YEAR + 1)
                  for m in (3, 6, 9, 12))


def fetch_contract(symbol, start_ms, end_ms):
    rows = []
    start = start_ms
    while start < end_ms:
        r = requests.get(URL, params={
            "symbol": symbol, "interval": "4h",
            "startTime": start, "endTime": end_ms, "limit": 1500,
        }, timeout=20)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        if len(data) < 1500:
            break
        start = last_open + 4 * 3600000
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "vol", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    df = df.sort_values("ts").drop_duplicates("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["dt", "open", "high", "low", "close"]]


def main():
    exps = contract_expiries()
    for base in BASES:
        for exp in exps:
            sym = f"{base}USDT_{exp.strftime('%y%m%d')}"
            path = OUT / f"{sym}_4H.csv"
            if path.exists():
                continue
            start_ms = int(datetime(exp.year, exp.month, exp.day, tzinfo=timezone.utc).timestamp() * 1000) - 400 * 86400000
            end_ms = int(datetime(exp.year, exp.month, exp.day, tzinfo=timezone.utc).timestamp() * 1000) + 86400000
            try:
                df = fetch_contract(sym, start_ms, end_ms)
            except Exception as e:
                print(f"FAIL {sym}: {e}")
                continue
            if df is None or df.empty:
                print(f"EMPTY {sym}")
                continue
            df = df[df["dt"] <= pd.Timestamp(exp, tz="UTC") + pd.Timedelta(days=1)]
            df.to_csv(path, index=False)
            print(f"OK {sym}: {len(df)} rows {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
            time.sleep(0.2)
    print("done")


if __name__ == "__main__":
    main()

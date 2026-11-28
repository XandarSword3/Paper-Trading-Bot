"""Fetch daily klines for Binance USD-M quarterly delivery futures.

Contract naming: {BASE}USDT_YYMMDD where YYMMDD is the expiry date
(last Friday of Mar/Jun/Sep/Dec). Expired contracts remain queryable via
/fapi/v1/klines even though they no longer appear in exchangeInfo.

Used for Strategy B (cash-and-carry basis). Spot leg = OKX USDT spot 1D
(already in slow_study/data). Basis = futures/spot - 1, annualized by days
to expiry.
"""
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "quarterly"
OUT.mkdir(exist_ok=True)
URL = "https://fapi.binance.com/fapi/v1/klines"

BASES = ["BTC", "ETH"]
START_YEAR, END_YEAR = 2020, 2026


def last_friday(year, month):
    d = date(year, month, 1)
    # last day of month
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    while last.weekday() != 4:  # Friday = 4
        last -= timedelta(days=1)
    return last


def contract_expiries():
    exps = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in (3, 6, 9, 12):
            exps.append(last_friday(y, m))
    return sorted(exps)


def fetch_contract(symbol, start_ms, end_ms):
    rows = []
    start = start_ms
    while start < end_ms:
        r = requests.get(URL, params={
            "symbol": symbol, "interval": "1d",
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
        start = last_open + 86400000
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
            path = OUT / f"{sym}_1D.csv"
            if path.exists():
                continue
            # contract trades from ~listing (up to ~1y before expiry) to expiry
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
            # keep only rows up to expiry day
            df = df[df["dt"] <= pd.Timestamp(exp, tz="UTC") + pd.Timedelta(days=1)]
            df.to_csv(path, index=False)
            print(f"OK {sym}: {len(df)} rows {df['dt'].iloc[0].date()} -> {df['dt'].iloc[-1].date()}")
            time.sleep(0.2)
    print("done")


if __name__ == "__main__":
    main()

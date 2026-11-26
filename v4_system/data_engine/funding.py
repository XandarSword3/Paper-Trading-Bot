"""Funding-rate history.

GET /fapi/v1/fundingRate — full history per symbol, paginated by
startTime/endTime (limit 1000). Funding settles every 8h on most USDⓈ-M
perps (some are 4h — the `fundingIntervalHours` field in exchangeInfo is
the authority; we record the observed interval per symbol).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient


def _ms(dt: str | datetime) -> int:
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_funding_history(symbol: str,
                          start: str | datetime = "2019-09-01",
                          end: str | datetime | None = None,
                          client: BinanceClient | None = None) -> pd.DataFrame:
    client = client or BinanceClient.futures()
    start_ms = _ms(start)
    end_ms = _ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = client.get("/fapi/v1/fundingRate", params={
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["fundingTime"])
        if len(batch) < 1000 or last <= cursor:
            break
        cursor = last + 1

    if not rows:
        raise ValueError(f"no funding history for {symbol}")

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.set_index("fundingTime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["fundingRate"]]

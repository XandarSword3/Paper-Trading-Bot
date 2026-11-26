"""Aggregate-trade (aggTrades) fetcher.

GET /fapi/v1/aggTrades — each record is a merged trade with price, qty,
isBuyerMaker (→ aggressor side), timestamp. Full history for BTCUSDT is
hundreds of millions of rows; Phase 1 implements bounded-window fetches
(for microstructure research windows) and documents that bulk historical
download is deferred to a dedicated job.

Pagination: when a window returns `limit` rows, continue from
fromId = last aggTradeId + 1 (fromId pagination is the reliable cursor;
startTime alone can silently truncate dense windows).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient

AGGTRADE_COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id",
                    "last_trade_id", "timestamp", "is_buyer_maker"]


def _ms(dt: str | datetime) -> int:
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_agg_trades(symbol: str,
                     start: str | datetime,
                     end: str | datetime,
                     max_rows: int = 200_000,
                     client: BinanceClient | None = None) -> pd.DataFrame:
    """Fetch aggTrades in [start, end]. Stops at max_rows to bound memory."""
    client = client or BinanceClient.futures()
    start_ms, end_ms = _ms(start), _ms(end)

    rows = []
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    while len(rows) < max_rows:
        batch = client.get("/fapi/v1/aggTrades", params=params)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        last_id = int(batch[-1]["a"])
        params = {"symbol": symbol, "fromId": last_id + 1, "limit": 1000}
        # Safety: stop if we've paged past end_time
        if int(batch[-1]["T"]) >= end_ms:
            break

    if not rows:
        raise ValueError(f"no aggTrades for {symbol} in window")

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "a": "agg_trade_id", "p": "price", "q": "quantity",
        "f": "first_trade_id", "l": "last_trade_id",
        "T": "timestamp", "m": "is_buyer_maker",
    })[AGGTRADE_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    # Signed aggressor volume: buyer-initiated (m=False) is +, seller -.
    df["aggressor_qty"] = df["quantity"].where(~df["is_buyer_maker"], -df["quantity"])
    df = df.set_index("timestamp").sort_index()
    return df

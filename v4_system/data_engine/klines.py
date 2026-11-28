"""OHLCV kline fetchers.

Covers three Binance kline families:
  - USDⓈ-M perpetual klines      GET /fapi/v1/klines
  - spot klines                  GET /api/v3/klines
  - mark-price / index-price     GET /fapi/v1/markPriceKlines, /fapi/v1/indexPriceKlines

All return UTC-indexed DataFrames preserving open time (the bar's start),
which is the point-in-time-correct timestamp for backtesting.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote"]

# Binance kline limit is 1500 for futures, 1000 for spot.
FAPI_LIMIT = 1500
SPOT_LIMIT = 1000

_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


def _ms(dt: str | datetime) -> int:
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_klines(raw: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["ignore"])
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _paginate(client: BinanceClient, path: str, symbol: str, interval: str,
              start: str | datetime, end: str | datetime | None,
              limit: int, key_param: str = "symbol") -> pd.DataFrame:
    """Page forward through a kline endpoint until `end` or exhaustion.

    key_param is 'symbol' for klines/markPriceKlines but 'pair' for
    indexPriceKlines (Binance quirk — index endpoint rejects 'symbol').
    """
    interval_ms = _INTERVAL_MS[interval]
    start_ms = _ms(start)
    end_ms = _ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)

    frames = []
    cursor = start_ms
    n_batches = 0
    while cursor < end_ms:
        batch = client.get(path, params={
            key_param: symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": limit,
        })
        if not batch:
            break
        frames.append(_parse_klines(batch))
        n_batches += 1
        if n_batches % 100 == 0:
            print(f"    … {symbol} {interval} {path}: {n_batches} batches, "
                  f"through {pd.to_datetime(batch[-1][0], unit='ms', utc=True)}")
        last_open = int(batch[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:  # no progress guard
            break
        cursor = next_cursor
        if len(batch) < limit:
            break
    if not frames:
        raise ValueError(f"no klines returned for {symbol} {interval} {path}")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_perp_klines(symbol: str, interval: str,
                      start: str | datetime = "2019-09-01",
                      end: str | datetime | None = None,
                      client: BinanceClient | None = None) -> pd.DataFrame:
    """USDⓈ-M perpetual klines. BTCUSDT perp listed 2019-09."""
    client = client or BinanceClient.futures()
    return _paginate(client, "/fapi/v1/klines", symbol, interval, start, end, FAPI_LIMIT)


def fetch_spot_klines(symbol: str, interval: str,
                      start: str | datetime = "2017-08-17",
                      end: str | datetime | None = None,
                      client: BinanceClient | None = None) -> pd.DataFrame:
    """Spot klines. BTCUSDT spot listed 2017-08."""
    client = client or BinanceClient.spot()
    return _paginate(client, "/api/v3/klines", symbol, interval, start, end, SPOT_LIMIT)


def fetch_mark_price_klines(symbol: str, interval: str,
                            start: str | datetime = "2019-09-01",
                            end: str | datetime | None = None,
                            client: BinanceClient | None = None) -> pd.DataFrame:
    """Mark-price klines (the price liquidations trigger against)."""
    client = client or BinanceClient.futures()
    return _paginate(client, "/fapi/v1/markPriceKlines", symbol, interval, start, end, FAPI_LIMIT)


def fetch_index_price_klines(symbol: str, interval: str,
                             start: str | datetime = "2019-09-01",
                             end: str | datetime | None = None,
                             client: BinanceClient | None = None) -> pd.DataFrame:
    """Index-price klines (composite spot index the perp tracks)."""
    client = client or BinanceClient.futures()
    return _paginate(client, "/fapi/v1/indexPriceKlines", symbol, interval,
                     start, end, FAPI_LIMIT, key_param="pair")

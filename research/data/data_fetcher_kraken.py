"""
Kraken historical OHLC fetcher — Phase 4 of VALIDATION_REMEDIATION_PLAN.md.

Both live bots (github_bot.py, github_bot_v4.py) already read from Kraken's
public API instead of Binance (STRATEGY_VERSION_AUDIT.md: chosen because
Binance isn't reliably reachable from Lebanon), but only ever pull the last
~720 recent candles for live trading — nothing pulls Kraken's own historical
range for validation. This is that: the "BTC on a different data source" leg
of Phase 4, so the strategy can be checked against the exact price history
its own live bots actually trade against, not just Binance's.

Kraken's public /OHLC endpoint returns at most ~720 candles per call and
takes a `since` (unix seconds) cursor for pagination — there's no `start`/
`end` pair like Binance's klines endpoint, so this fetcher pages forward
from since=start_date until either the API stops returning new candles or
end_date is reached.
"""

import os
import time
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from data_fetcher import _update_manifest, DATA_DIR  # see data_fetcher.py for why this is
                                                       # computed there rather than imported
                                                       # from research/strategies/config.py

KRAKEN_API = "https://api.kraken.com/0/public/OHLC"

# Kraken only accepts these interval values (minutes).
INTERVAL_MINUTES = {"1h": 60, "4h": 240, "1d": 1440}

# Kraken's asset-pair naming doesn't match the exchange's ticker directly
# (BTC is "XBT", and pairs get an X/Z prefix depending on asset class).
# This is the one pair Phase 4 needs; add more here if a later phase does.
PAIR_ALIASES = {"BTCUSD": "XXBTZUSD"}


class KrakenDataFetcher:
    """Fetches historical OHLC data from Kraken's public API."""

    MAX_CONSECUTIVE_FAILURES = 5  # fail loudly instead of retrying forever

    def __init__(self, pair: str = "BTCUSD"):
        self.pair = pair
        self.kraken_pair = PAIR_ALIASES.get(pair, pair)

    def fetch_ohlc(
        self,
        timeframe: str = "4h",
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if timeframe not in INTERVAL_MINUTES:
            raise ValueError(f"Unsupported timeframe for Kraken fetch: {timeframe}")
        interval = INTERVAL_MINUTES[timeframe]

        end_ts = (
            int(datetime.now(timezone.utc).timestamp())
            if end_date is None
            else int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        )
        since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

        all_rows = []
        consecutive_failures = 0
        approx_total_calls = max(1, (end_ts - since) // (720 * interval * 60) + 1)

        print(f"Fetching Kraken {self.pair} ({self.kraken_pair}) {timeframe} data "
              f"from {start_date} to {end_date or 'now'}...")

        with tqdm(total=approx_total_calls, desc="Downloading (Kraken)") as pbar:
            while since < end_ts:
                try:
                    resp = requests.get(
                        KRAKEN_API,
                        params={"pair": self.kraken_pair, "interval": interval, "since": since},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    payload = resp.json()

                    if payload.get("error"):
                        raise RuntimeError(f"Kraken API error: {payload['error']}")

                    result = payload["result"]
                    result_key = next(k for k in result.keys() if k != "last")
                    rows = result[result_key]

                    if not rows:
                        break

                    all_rows.extend(rows)
                    last_candle_time = int(rows[-1][0])
                    next_since = result.get("last", last_candle_time)
                    if int(next_since) <= since:
                        # Guard against a non-advancing cursor looping forever.
                        break
                    since = int(next_since)

                    consecutive_failures = 0
                    pbar.update(1)
                    time.sleep(1.0)  # Kraken's public API is rate-limited more tightly than Binance's

                except (requests.exceptions.RequestException, RuntimeError, KeyError) as e:
                    consecutive_failures += 1
                    print(f"Error fetching Kraken data ({consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}")
                    if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        raise ConnectionError(
                            f"Kraken fetch failed {consecutive_failures} times in a row "
                            f"({KRAKEN_API}) — giving up instead of retrying forever. "
                            f"Last error: {e}"
                        ) from e
                    time.sleep(min(2 ** consecutive_failures, 30))
                    continue

        if not all_rows:
            raise ValueError("No data fetched from Kraken")

        df = pd.DataFrame(all_rows, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        _update_manifest(
            filename, df, parquet_path,
            source="Kraken api.kraken.com/0/public/OHLC",
            symbol=self.kraken_pair,
        )
        print(f"Data saved to:\n  {csv_path}\n  {parquet_path}")
        return csv_path

    def load_data(self, filename: str) -> pd.DataFrame:
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"No data file found: {filename}")


def download_kraken_data(
    pair: str = "BTCUSD",
    timeframe: str = "4h",
    start_date: str = "2017-01-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download or load Kraken OHLCV data. Mirrors data_fetcher.download_binance_data's
    cache-then-fetch shape, without the staleness auto-refresh (Phase 4 only
    needs a static historical snapshot for validation, not a live-updating one —
    the live bots already handle their own recent-candle fetch separately).
    """
    filename = f"kraken_{pair}_{timeframe}"
    fetcher = KrakenDataFetcher(pair=pair)
    parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")

    if os.path.exists(parquet_path) and not force_refresh:
        print(f"Loading existing data from {parquet_path}")
        return fetcher.load_data(filename)

    df = fetcher.fetch_ohlc(timeframe=timeframe, start_date=start_date, end_date=end_date)
    fetcher.save_data(df, filename)
    return df


if __name__ == "__main__":
    df = download_kraken_data(pair="BTCUSD", timeframe="4h", start_date="2020-01-01")
    print(f"\nData shape: {df.shape}")
    print(f"\nSample data:\n{df.tail()}")

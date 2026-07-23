"""
Funding Rate Fetcher - Downloads BTCUSDT perpetual funding rate history
from Binance Futures (fapi.binance.com). Only reachable from environments
with real internet access (e.g. GitHub Actions) - not from this repo's
usual sandboxed research environment, which is why this is meant to run
via .github/workflows/fetch-funding-data.yml rather than locally.

Mirrors data_fetcher.py's BinanceDataFetcher pagination/retry/manifest
conventions so funding-rate provenance is tracked the same way OHLCV is.
"""
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_DIR
from data_fetcher import _update_manifest

MANIFEST_PATH = os.path.join(DATA_DIR, "MANIFEST.json")


class BinanceFundingRateFetcher:
    """Fetches historical BTCUSDT perpetual funding rate from Binance Futures public API."""

    BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol

    def fetch_funding_history(
        self,
        start_date: str = "2019-09-01",  # BTCUSDT perp funding began ~2019-09-08 on Binance
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        all_rows = []
        current_ts = start_ts
        limit = 1000  # Binance max per request
        eight_hours_ms = 8 * 3600 * 1000
        total_batches = max(1, (end_ts - start_ts) // (limit * eight_hours_ms) + 1)

        print(f"Fetching {self.symbol} funding rate history from {start_date} to {end_date}...")

        consecutive_failures = 0
        with tqdm(total=total_batches, desc="Downloading funding rates") as pbar:
            while current_ts < end_ts:
                params = {
                    "symbol": self.symbol,
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": limit,
                }
                try:
                    response = requests.get(self.BASE_URL, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    all_rows.extend(data)
                    current_ts = data[-1]["fundingTime"] + 1
                    pbar.update(1)
                    consecutive_failures = 0
                    time.sleep(0.15)  # Binance funding-rate endpoint rate limit courtesy

                except requests.exceptions.RequestException as e:
                    consecutive_failures += 1
                    print(f"Error fetching funding rate ({consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}")
                    if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        raise ConnectionError(
                            f"Binance funding-rate fetch failed {consecutive_failures} times in a row "
                            f"({self.BASE_URL}) - giving up instead of retrying forever. Last error: {e}"
                        ) from e
                    time.sleep(min(2 ** consecutive_failures, 30))
                    continue

        if not all_rows:
            raise ValueError("No funding rate data fetched from Binance")

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["mark_price"] = df.get("markPrice", pd.Series(dtype=float)).astype(float) if "markPrice" in df else float("nan")
        df = df.set_index("timestamp")[["funding_rate", "mark_price"]]
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        print(f"Downloaded {len(df)} funding settlements from {df.index[0]} to {df.index[-1]}")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = "BTCUSDT_funding_rate") -> str:
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")

        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        _update_manifest(
            filename, df, parquet_path,
            source="Binance fapi.binance.com/fapi/v1/fundingRate",
            symbol=self.symbol,
        )
        print(f"Data saved to:\n  {csv_path}\n  {parquet_path}")
        return csv_path


def download_funding_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2019-09-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    filename = f"{symbol}_funding_rate"
    csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
    if os.path.exists(csv_path) and not force_refresh:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    fetcher = BinanceFundingRateFetcher(symbol=symbol)
    df = fetcher.fetch_funding_history(start_date=start_date)
    fetcher.save_data(df, filename=filename)
    return df


if __name__ == "__main__":
    download_funding_data(force_refresh=True)

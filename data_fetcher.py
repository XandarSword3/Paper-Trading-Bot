"""
Data Fetcher - Downloads BTC historical data from Binance
Supports 1H and 4H timeframes from 2017 to present
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import requests
from tqdm import tqdm

from config import DATA_DIR, DEFAULT_BACKTEST


class BinanceDataFetcher:
    """Fetches historical BTCUSDT data from Binance public API"""
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    INTERVALS = {
        "1h": 3600 * 1000,
        "4h": 4 * 3600 * 1000,
    }
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
    
    def fetch_klines(
        self,
        interval: str = "4h",
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical klines/candlestick data from Binance.
        
        Args:
            interval: Timeframe ('1h' or '4h')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        limit = 1000  # Binance max per request
        
        # Calculate total batches for progress bar
        interval_ms = self.INTERVALS.get(interval, 4 * 3600 * 1000)
        total_batches = max(1, (end_ts - start_ts) // (limit * interval_ms) + 1)
        
        print(f"Fetching {self.symbol} {interval} data from {start_date} to {end_date}...")
        
        with tqdm(total=total_batches, desc="Downloading") as pbar:
            while current_ts < end_ts:
                params = {
                    "symbol": self.symbol,
                    "interval": interval,
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": limit
                }
                
                try:
                    response = requests.get(self.BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                    
                    all_data.extend(data)
                    current_ts = data[-1][0] + 1  # Next timestamp after last candle
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(1)
                    continue
        
        if not all_data:
            raise ValueError("No data fetched from Binance")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Process columns
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]
        
        # Remove duplicates if any
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        
        print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV and Parquet"""
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        
        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        
        print(f"Data saved to:\n  {csv_path}\n  {parquet_path}")
        return csv_path
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from file (prefers Parquet for speed)"""
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df
        else:
            raise FileNotFoundError(f"No data file found: {filename}")


class KrakenDataFetcher:
    """
    Fetches historical BTC/USD data from Kraken's public OHLC API.

    Binance blocks access from US-hosted infrastructure (HTTP 451, "restricted
    location" per their ToS) -- this includes GitHub Actions runners, which are
    Azure/US-hosted. Kraken has no such restriction and is reachable from GH
    Actions runners (confirmed via diag-network.yml probe).

    Kraken's OHLC endpoint caps each response at 720 candles, but supports a
    `since` cursor to walk forward through history: request from an old
    timestamp, take the last returned candle's time as the next `since`, and
    repeat until reaching the present. This assembles arbitrarily long history
    at the cost of one HTTP request per ~720 candles.

    Kraken doesn't offer a USDT pair; XBTUSD (BTC/USD) is used as a close proxy
    since USDT is designed to track USD 1:1. Output is saved under the same
    BTCUSDT_{timeframe} filename so it's a drop-in replacement for every
    downstream script that calls download_btc_data().
    """

    BASE_URL = "https://api.kraken.com/0/public/OHLC"
    PAIR = "XBTUSD"
    INTERVALS = {"1h": 60, "4h": 240}  # Kraken interval = minutes

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol  # kept for interface parity with BinanceDataFetcher

    def fetch_klines(
        self,
        interval: str = "4h",
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"Unsupported interval for Kraken: {interval}")

        minutes = self.INTERVALS[interval]
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = (
            int(datetime.now().timestamp())
            if end_date is None
            else int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        )

        all_rows = []
        seen_sinces = set()
        print(f"Fetching {self.PAIR} {interval} data from Kraken, {start_date} to "
              f"{end_date or 'now'}...")

        with tqdm(desc="Downloading (Kraken, ~720 candles/request)") as pbar:
            while since < end_ts:
                if since in seen_sinces:
                    break  # safety net against infinite loop
                seen_sinces.add(since)

                resp = requests.get(
                    self.BASE_URL,
                    params={"pair": self.PAIR, "interval": minutes, "since": since},
                    timeout=20,
                )
                resp.raise_for_status()
                payload = resp.json()

                if payload.get("error"):
                    raise RuntimeError(f"Kraken API error: {payload['error']}")

                result = payload["result"]
                pair_key = next(k for k in result.keys() if k != "last")
                rows = result[pair_key]

                if not rows:
                    break

                all_rows.extend(rows)
                new_since = int(rows[-1][0])
                if new_since <= since:
                    break
                since = new_since + 1
                pbar.update(len(rows))
                time.sleep(1.0)  # polite rate limiting for Kraken public API

        if not all_rows:
            raise ValueError("No data fetched from Kraken")

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]} (source: Kraken XBTUSD)")
        return df

    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
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


def download_btc_data(
    timeframe: str = "4h",
    start_date: str = "2017-01-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Main function to download or load BTC data.

    Args:
        timeframe: '1h' or '4h'
        start_date: Start date
        end_date: End date (default: today)
        force_refresh: Force re-download even if file exists
        source: 'auto' (try Binance, fall back to Kraken), 'binance', or 'kraken'

    Returns:
        DataFrame with OHLCV data
    """
    filename = f"BTCUSDT_{timeframe}"

    def get_fetcher(name: str):
        return BinanceDataFetcher() if name == "binance" else KrakenDataFetcher()

    def fetch_with_fallback(tf, s_date, e_date):
        if source == "binance":
            return get_fetcher("binance").fetch_klines(tf, s_date, e_date)
        if source == "kraken":
            return get_fetcher("kraken").fetch_klines(tf, s_date, e_date)
        # auto: try Binance first (works if run from an unrestricted region),
        # fall back to Kraken on any failure (e.g. HTTP 451 geo-block).
        try:
            return get_fetcher("binance").fetch_klines(tf, s_date, e_date)
        except Exception as e:
            print(f"   Binance fetch failed ({e}); falling back to Kraken...")
            return get_fetcher("kraken").fetch_klines(tf, s_date, e_date)

    fetcher = get_fetcher("kraken" if source in ("auto", "kraken") else "binance")

    # Check if data exists
    parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")

    if os.path.exists(parquet_path) and not force_refresh:
        print(f"Loading existing data from {parquet_path}")
        df = fetcher.load_data(filename)

        # Check if we need to update
        last_date = df.index[-1]
        today = pd.Timestamp.now()
        staleness_days = (today - last_date).days

        if staleness_days > 7:
            print(f"\n⚠️  WARNING: Data is {staleness_days} days old (last candle: {last_date.strftime('%Y-%m-%d')})")
            print(f"   Attempting auto-refresh...")

        if staleness_days > 1:
            try:
                new_start = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
                new_df = fetch_with_fallback(timeframe, new_start, end_date)
                df = pd.concat([df, new_df])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
                fetcher.save_data(df, filename)
                print(f"   ✅ Data updated to {df.index[-1].strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"   ❌ Auto-refresh failed: {e}")
                if staleness_days > 7:
                    print(f"   ⚠️  Proceeding with STALE data. Results may not reflect current market.")
                    print(f"   Run with force_refresh=True to force full re-download.")

        return df

    # Fresh download
    df = fetch_with_fallback(timeframe, start_date, end_date)
    fetcher.save_data(df, filename)

    return df


if __name__ == "__main__":
    # Download 4H data when run directly
    df = download_btc_data(
        timeframe="4h",
        start_date="2017-01-01",
        force_refresh=False,
        source="auto",
    )
    print(f"\nData shape: {df.shape}")
    print(f"\nSample data:\n{df.tail()}")

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


def download_btc_data(
    timeframe: str = "4h",
    start_date: str = "2017-01-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function to download or load BTC data.
    
    Args:
        timeframe: '1h' or '4h'
        start_date: Start date
        end_date: End date (default: today)
        force_refresh: Force re-download even if file exists
    
    Returns:
        DataFrame with OHLCV data
    """
    filename = f"BTCUSDT_{timeframe}"
    fetcher = BinanceDataFetcher()
    
    # Check if data exists
    parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
    
    if os.path.exists(parquet_path) and not force_refresh:
        print(f"Loading existing data from {parquet_path}")
        df = fetcher.load_data(filename)
        
        # Check if we need to update
        last_date = df.index[-1]
        today = pd.Timestamp.now()
        
        if (today - last_date).days > 1:
            print(f"Data is outdated (last: {last_date}). Updating...")
            new_start = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
            new_df = fetcher.fetch_klines(timeframe, new_start, end_date)
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
            fetcher.save_data(df, filename)
        
        return df
    
    # Fresh download
    df = fetcher.fetch_klines(timeframe, start_date, end_date)
    fetcher.save_data(df, filename)
    
    return df


if __name__ == "__main__":
    # Download 4H data when run directly
    df = download_btc_data(
        timeframe="4h",
        start_date="2017-01-01",
        force_refresh=False
    )
    print(f"\nData shape: {df.shape}")
    print(f"\nSample data:\n{df.tail()}")

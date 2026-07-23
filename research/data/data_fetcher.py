"""
Data Fetcher - Downloads BTC historical data from Binance
Supports 1H and 4H timeframes from 2017 to present
"""

import os
import time
import hashlib
import json
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
from tqdm import tqdm

from config import DATA_DIR, DEFAULT_BACKTEST

MANIFEST_PATH = os.path.join(DATA_DIR, "MANIFEST.json")


def _update_manifest(
    filename: str,
    df: pd.DataFrame,
    parquet_path: str,
    source: str = "Binance api.binance.com/api/v3/klines",
    symbol: str = "BTCUSDT",
) -> None:
    """
    Record dataset provenance — Phase 1 of the remediation plan asks for a
    canonical, versioned dataset instead of silent re-fetches. This doesn't
    stop re-fetching, but it makes every snapshot on disk identifiable: exact
    row count, date range, fetch time, and a content hash, so it's possible
    to tell whether two runs actually used the same data.

    source/symbol are parameters (not hardcoded to Binance/BTCUSDT) so Phase
    4's cross-market/cross-source fetches (data_fetcher_kraken.py, other
    Binance symbols) get accurate provenance too.
    """
    with open(parquet_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    manifest[filename] = {
        "source": source,
        "symbol": symbol,
        "rows": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": file_hash,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


class BinanceDataFetcher:
    """Fetches historical BTCUSDT data from Binance public API"""
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    INTERVALS = {
        "1h": 3600 * 1000,
        "4h": 4 * 3600 * 1000,
    }
    MAX_CONSECUTIVE_FAILURES = 5  # give up loudly instead of retrying forever
    
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
        
        consecutive_failures = 0
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
                    response = requests.get(self.BASE_URL, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                    
                    all_data.extend(data)
                    current_ts = data[-1][0] + 1  # Next timestamp after last candle
                    pbar.update(1)
                    consecutive_failures = 0
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except requests.exceptions.RequestException as e:
                    consecutive_failures += 1
                    print(f"Error fetching data ({consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}")
                    if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        raise ConnectionError(
                            f"Binance fetch failed {consecutive_failures} times in a row "
                            f"({self.BASE_URL}) — giving up instead of retrying forever. "
                            f"Last error: {e}"
                        ) from e
                    time.sleep(min(2 ** consecutive_failures, 30))  # capped exponential backoff
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
        _update_manifest(
            filename, df, parquet_path,
            source="Binance api.binance.com/api/v3/klines",
            symbol=self.symbol,
        )

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


class CryptoDataDownloadFetcher:
    """
    Downloads a full-history OHLCV CSV from cryptodatadownload.com.

    Kraken's public OHLC endpoint (see data_fetcher_kraken.py) ignores the
    `since` cursor and always returns only its most recent ~720 candles —
    confirmed live (since_probe.txt): requesting since=2017, since=2020, and
    no `since` at all returned the identical latest window. It's a live-data
    endpoint, not a historical-backfill one, so it can't provide 2017-2025
    depth at any pagination speed.

    cryptodatadownload.com instead hosts a single prebuilt CSV per exchange/
    pair/granularity spanning that exchange's full history — one HTTP GET,
    no pagination, no rate limit. Bitstamp is used here (BTC/USD, live since
    2011) since its file already covers 2017 onward and is kept current
    (confirmed live via source_probe.txt: top row was the current day).

    Only 1h/1d/minute granularities are published per exchange (no native 4h
    file), so 4h is produced by resampling the 1h data.
    """

    BASE_URL = "https://www.cryptodatadownload.com/cdd/{exchange}_{pair}_{gran}.csv"

    def __init__(self, exchange: str = "Bitstamp", pair: str = "BTCUSD"):
        self.exchange = exchange
        self.pair = pair

    def fetch(
        self,
        timeframe: str = "4h",
        start_date: str = "2017-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        gran = "1h" if timeframe in ("1h", "4h") else timeframe
        url = self.BASE_URL.format(exchange=self.exchange, pair=self.pair, gran=gran)

        print(f"Fetching {self.exchange} {self.pair} {gran} full-history CSV from {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        # CDD's files have a banner URL on line 1, then the real CSV header.
        lines = resp.text.splitlines()
        csv_text = "\n".join(lines[1:]) if lines and lines[0].strip().lower().startswith("http") else resp.text

        df = pd.read_csv(StringIO(csv_text))
        df["timestamp"] = pd.to_datetime(df["unix"], unit="s")
        df.set_index("timestamp", inplace=True)

        volume_col = "Volume BTC" if "Volume BTC" in df.columns else "Volume"
        df = df.rename(columns={volume_col: "volume"})
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if end_date else df.index.max()
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if df.empty:
            raise ValueError(
                f"CryptoDataDownload returned data but none fell within "
                f"{start_date} .. {end_date or 'now'} (file range: "
                f"{df.index.min() if len(df) else 'n/a'})"
            )

        if timeframe == "4h":
            df = df.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

        print(f"Downloaded {len(df)} {timeframe} candles from {df.index[0]} to {df.index[-1]} "
              f"(source: CryptoDataDownload, {self.exchange} {self.pair})")
        return df


def download_binance_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    start_date: str = "2017-01-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download or load Binance OHLCV data for any symbol.

    Generalizes what used to be BTC-only logic in download_btc_data() — Phase
    4 of the remediation plan (cross-market validation) needs the exact same
    fetch/cache/staleness-refresh behavior for other assets (e.g. ETHUSDT),
    not a second parallel implementation. download_btc_data() below is now a
    thin wrapper over this for backward compatibility.

    Args:
        symbol: Binance symbol, e.g. 'BTCUSDT', 'ETHUSDT'
        timeframe: '1h' or '4h'
        start_date: Start date
        end_date: End date (default: today)
        force_refresh: Force re-download even if file exists

    Returns:
        DataFrame with OHLCV data
    """
    filename = f"{symbol}_{timeframe}"
    fetcher = BinanceDataFetcher(symbol=symbol)

    parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")

    if os.path.exists(parquet_path) and not force_refresh:
        print(f"Loading existing data from {parquet_path}")
        df = fetcher.load_data(filename)

        last_date = df.index[-1]
        today = pd.Timestamp.now()
        staleness_days = (today - last_date).days

        if staleness_days > 7:
            print(f"\n⚠️  WARNING: Data is {staleness_days} days old (last candle: {last_date.strftime('%Y-%m-%d')})")
            print(f"   Attempting auto-refresh from Binance...")

        if staleness_days > 1:
            try:
                new_start = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
                new_df = fetcher.fetch_klines(timeframe, new_start, end_date)
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
    df = fetcher.fetch_klines(timeframe, start_date, end_date)
    fetcher.save_data(df, filename)

    return df


def download_btc_data(
    timeframe: str = "4h",
    start_date: str = "2017-01-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function to download or load BTC data. Thin wrapper over
    download_binance_data(symbol='BTCUSDT', ...) — kept as its own function
    since it's the primary strategy's data path and imported by name all
    over the repo (strategy.py, walk_forward_test.py, robustness_test.py,
    main.py, etc.).

    Binance is not reliably reachable from Lebanon or from GitHub Actions'
    US-hosted runners (HTTP 451, confirmed via diag-network.yml) — so on
    failure this transparently falls back to a full-history Bitstamp CSV via
    CryptoDataDownload.com as a USD-pegged proxy for BTCUSDT. (Kraken's own
    OHLC endpoint was tried first, but it only ever serves its most recent
    ~720 candles regardless of the `since` parameter — confirmed live via
    since_probe.txt — so it can't backfill 2017-2025 history at any speed.
    data_fetcher_kraken.py is still used elsewhere for Phase 4's cross-market
    validation, which only needs Kraken's recent window.) The fallback result
    is cached under the same BTCUSDT_{timeframe} filename regardless of
    source, since every caller imports download_btc_data() by name and
    doesn't care which exchange it came from — MANIFEST.json records the true
    source for whichever snapshot ends up on disk.

    Args:
        timeframe: '1h' or '4h'
        start_date: Start date
        end_date: End date (default: today)
        force_refresh: Force re-download even if file exists

    Returns:
        DataFrame with OHLCV data
    """
    try:
        return download_binance_data(
            symbol="BTCUSDT",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )
    except Exception as e:
        print(f"\n⚠️  Binance unavailable ({e})")
        print(f"   Falling back to CryptoDataDownload (Bitstamp BTCUSD) as a USD-pegged proxy for BTCUSDT...")

        filename = f"BTCUSDT_{timeframe}"
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")

        cdd_fetcher = CryptoDataDownloadFetcher(exchange="Bitstamp", pair="BTCUSD")
        df = cdd_fetcher.fetch(timeframe=timeframe, start_date=start_date, end_date=end_date)

        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        _update_manifest(
            filename, df, parquet_path,
            source="CryptoDataDownload.com Bitstamp_BTCUSD CSV (proxy — Binance unreachable)",
            symbol="BTCUSD",
        )
        print(f"Data saved to:\n  {csv_path}\n  {parquet_path}")

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

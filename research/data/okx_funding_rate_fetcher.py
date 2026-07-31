"""BTC-USDT-SWAP funding rate fetcher — OKX public API.

Binance (fapi.binance.com) returns HTTP 451 to GitHub-hosted runner IPs
(regulatory geo-block) and Bybit's public API is blocked at the CloudFront
level by country. OKX's public funding-rate-history endpoint returned 200
with real data when probed from the same runner IP, so this is the
data source that is actually reachable from CI.

OKX's public endpoint paginates backwards via the `before` cursor
(fundingTime of the oldest row seen so far) and — per OKX docs — only
serves a limited trailing window of history through this public,
no-auth endpoint (deeper history requires an authenticated business
endpoint this project doesn't have keys for). This fetcher pages back
as far as the public endpoint allows and reports exactly how far that
was, rather than assuming it reaches any particular start date.
"""
import os
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# DATA_DIR is computed in data_fetcher.py from that module's own file location —
# see data_fetcher.py for why config.py's DATA_DIR is unreliable across different
# cwd/PYTHONPATH states. _update_manifest has a no-op fallback so this fetcher can
# still run standalone if data_fetcher.py somehow isn't importable.
try:
    from data_fetcher import DATA_DIR, _update_manifest
except ImportError:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    os.makedirs(DATA_DIR, exist_ok=True)
    def _update_manifest(*args, **kwargs):
        pass


class OKXFundingRateFetcher:
    """Fetches historical BTC-USDT-SWAP funding rate from OKX's public API."""

    BASE_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, inst_id: str = "BTC-USDT-SWAP"):
        self.inst_id = inst_id

    def fetch_funding_history(self, max_pages: int = 200) -> pd.DataFrame:
        all_rows = []
        before_cursor: Optional[str] = None
        consecutive_failures = 0

        print(f"Fetching {self.inst_id} funding rate history from OKX (public endpoint, "
              f"limited trailing window)...")

        with tqdm(total=max_pages, desc="Downloading funding rates (OKX)") as pbar:
            for _ in range(max_pages):
                params = {"instId": self.inst_id, "limit": 100}
                if before_cursor:
                    params["before"] = before_cursor

                try:
                    response = requests.get(self.BASE_URL, params=params, timeout=30)
                    response.raise_for_status()
                    payload = response.json()

                    if payload.get("code") != "0":
                        raise ValueError(f"OKX API error: {payload.get('msg')}")

                    data = payload.get("data", [])
                    if not data:
                        break

                    all_rows.extend(data)
                    # OKX returns newest-first; page backwards using the oldest
                    # fundingTime seen so far as the next `before` cursor.
                    oldest_ts = min(int(row["fundingTime"]) for row in data)
                    if before_cursor is not None and str(oldest_ts) == before_cursor:
                        break  # no forward progress, stop rather than loop forever
                    before_cursor = str(oldest_ts)

                    pbar.update(1)
                    consecutive_failures = 0
                    time.sleep(0.15)

                    if len(data) < 100:
                        break  # short page = reached the end of available history

                except requests.exceptions.RequestException as e:
                    consecutive_failures += 1
                    print(f"Error fetching funding rate ({consecutive_failures}/"
                          f"{self.MAX_CONSECUTIVE_FAILURES}): {e}")
                    if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        raise ConnectionError(
                            f"OKX funding-rate fetch failed {consecutive_failures} times "
                            f"in a row ({self.BASE_URL}) - giving up. Last error: {e}"
                        ) from e
                    time.sleep(min(2 ** consecutive_failures, 30))
                    continue

        if not all_rows:
            raise ValueError("No funding rate data fetched from OKX")

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        n_bad = df["funding_rate"].isna().sum()
        if n_bad:
            print(f"WARNING: {n_bad} settlement(s) had unparseable fundingRate — dropping them")
            df = df.dropna(subset=["funding_rate"])
        df["mark_price"] = float("nan")  # OKX funding-rate-history doesn't include mark price
        df = df.set_index("timestamp")[["funding_rate", "mark_price"]]
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        print(f"Downloaded {len(df)} funding settlements from {df.index[0]} to {df.index[-1]} "
              f"(OKX public endpoint's available trailing window)")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = "BTCUSDT_funding_rate") -> str:
        csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
        parquet_path = os.path.join(DATA_DIR, f"{filename}.parquet")

        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        _update_manifest(
            filename, df, parquet_path,
            source="OKX www.okx.com/api/v5/public/funding-rate-history "
                   "(Binance fapi returns 451, Bybit CloudFront-blocked by country, "
                   "from GitHub-hosted runner IPs)",
            symbol=self.inst_id,
        )
        print(f"Data saved to:\n  {csv_path}\n  {parquet_path}")
        return csv_path


def download_funding_data(
    inst_id: str = "BTC-USDT-SWAP",
    force_refresh: bool = False,
    filename: str = "BTCUSDT_funding_rate",
) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
    if os.path.exists(csv_path) and not force_refresh:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    fetcher = OKXFundingRateFetcher(inst_id=inst_id)
    df = fetcher.fetch_funding_history()
    fetcher.save_data(df, filename=filename)
    return df


if __name__ == "__main__":
    download_funding_data(force_refresh=True)

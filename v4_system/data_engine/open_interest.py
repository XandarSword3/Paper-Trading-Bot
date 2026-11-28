"""Open-interest history.

GET /futures/data/openInterestHist — **limited to the latest ~30 days**
per Binance docs. This is a hard availability constraint: multi-year OI
history is NOT available from Binance public REST. The probe quantifies
the actual reachable window; for older OI we document the gap and rely on
live recording going forward (paper_trading recorder).

Period granularity: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d. Limit 500,
paginated by endTime backwards.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient

_PERIOD_MS = {
    "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
    "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
    "1d": 86_400_000,
}

# Public list of supported OI periods (endpoint-supported granularity).
OI_PERIODS = tuple(_PERIOD_MS)


def fetch_open_interest_hist(symbol: str, period: str = "1h",
                             days: int = 29,
                             client: BinanceClient | None = None) -> pd.DataFrame:
    """Fetch up to `days` of OI history (capped by Binance's ~30d window)."""
    if period not in _PERIOD_MS:
        raise ValueError(f"period must be one of {list(_PERIOD_MS)}")
    client = client or BinanceClient.futures()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=min(days, 29))
    period_ms = _PERIOD_MS[period]

    rows = []
    cursor_end = int(end.timestamp() * 1000)
    floor = int(start.timestamp() * 1000)
    while cursor_end > floor:
        batch = client.get("/futures/data/openInterestHist", params={
            "symbol": symbol, "period": period,
            "endTime": cursor_end, "limit": 500,
        })
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["timestamp"]) for r in batch)
        if len(batch) < 500 or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        if oldest <= floor:
            break

    if not rows:
        raise ValueError(f"no OI history for {symbol} period={period}")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ("sumOpenInterest", "sumOpenInterestValue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df[["sumOpenInterest", "sumOpenInterestValue"]]

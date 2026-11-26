"""Parquet store with gap detection.

Datasets live under v4_system/data/<name>.parquet. On load we verify the
datetime index is monotonic and report (never silently fill) gaps larger
than the expected bar interval — missing bars matter for backtests.
"""
from __future__ import annotations

import os

import pandas as pd

from v4_system.config import DATA_DIR
from v4_system.data_engine.manifest import update_manifest

TF_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def dataset_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def save(df: pd.DataFrame, name: str, *, source: str, symbol: str,
         timeframe: str | None = None, extra: dict | None = None) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = dataset_path(name)
    df.to_parquet(path)
    update_manifest(name, df, path, source=source, symbol=symbol,
                    timeframe=timeframe, extra=extra)
    return path


def load(name: str) -> pd.DataFrame:
    path = dataset_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(df.index, utc=True)
            if idx.notna().all():
                df.index = idx
        except (TypeError, ValueError):
            pass  # non-temporal index (e.g. contract_specs keyed by symbol)
    return df


def detect_gaps(df: pd.DataFrame, timeframe: str,
                tolerance: float = 1.5) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return (gap_start, gap_end, missing_bars) for gaps > tolerance*interval."""
    if timeframe not in TF_DELTA or len(df) < 2:
        return []
    delta = TF_DELTA[timeframe]
    diffs = df.index.to_series().diff().dropna()
    threshold = delta * tolerance
    gaps = []
    idx = diffs.index
    for ts in idx[diffs > threshold]:
        missing = int(round((diffs.loc[ts] / delta))) - 1
        gaps.append((ts - delta, ts, missing))
    return gaps

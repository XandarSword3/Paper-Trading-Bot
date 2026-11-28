"""Parquet store with gap detection, atomic writes, and PIT loading.

Datasets live under v4_system/data/<name>.parquet. On load we verify the
datetime index is monotonic and report (never silently fill) gaps larger
than the expected bar interval — missing bars matter for backtests.

Writes are atomic: data is written to a temporary file in the SAME
directory, validated by reading it back, and only then moved over the
destination with os.replace (atomic on Windows for same-volume moves).
A failed write or validation leaves the previous valid dataset intact,
and temporary files are never mistaken for datasets (they carry a
``.tmp-`` suffix and are never matched by ``*.parquet`` globs or by
``load``).

Loading is point-in-time aware: ``load(name, as_of=..., strict=...)``
applies the unified cutoff from v4_system.data_engine.pit. ``as_of=None``
is the explicit legacy opt-out (no cutoff); there is no hidden wall-clock
default. ``load_many`` applies ONE cutoff to every dataset of a decision.
"""
from __future__ import annotations

import itertools
import json
import os

import pandas as pd

from v4_system.config import DATA_DIR, MANIFEST_PATH
from v4_system.data_engine import pit
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


_tmp_seq = itertools.count()


def dataset_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _manifest_timeframe(name: str) -> str | None:
    """Timeframe recorded in the manifest for ``name``, if any."""
    if not os.path.exists(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            entry = json.load(f).get(name, {})
        tf = entry.get("timeframe")
        return tf if isinstance(tf, str) else None
    except Exception:
        return None


def _dataset_kind(name: str, timeframe: str | None) -> str:
    """Classify a dataset as 'candle' or 'series' for PIT cutoff rules.

    Candle closure applies only to actual kline datasets. Funding, open
    interest and aggregate trades are event/period series: their
    timestamps are observation times, not candle open times, so candle
    closure must NOT be applied to them even when their manifest
    timeframe happens to look like a candle interval (e.g. OI saved as
    "1h").
    """
    if timeframe not in pit.CANDLE_DELTA:
        return "series"
    lowered = name.lower()
    if "_funding" in lowered or "_oi_" in lowered or "_agg" in lowered:
        return "series"
    return "candle"


def _timeframe_for(name: str) -> str | None:
    tf = _manifest_timeframe(name)
    if tf:
        return tf
    # Fallback: trailing timeframe token in the dataset name.
    tail = name.rsplit("_", 1)[-1]
    return tail if tail in TF_DELTA else None


def save(df: pd.DataFrame, name: str, *, source: str, symbol: str,
         timeframe: str | None = None, extra: dict | None = None,
         available_at=None) -> str:
    """Persist ``df`` atomically and update the manifest.

    ``available_at`` (optional): scalar timestamp or Series aligned with
    ``df`` recording when each observation became available to this
    system. When omitted, no availability is recorded (unknown -> NaT),
    which strict PIT loading will reject. Historical bulk downloads
    should leave this unset.
    """
    if available_at is not None:
        df = df.copy()
        df["available_at"] = available_at
    os.makedirs(DATA_DIR, exist_ok=True)
    path = dataset_path(name)
    tmp = f"{path}.tmp-{os.getpid()}-{next(_tmp_seq)}"
    try:
        df.to_parquet(tmp)
        # Validate the completed temporary output before replacing.
        check = pd.read_parquet(tmp)
        if len(check) != len(df) or list(check.columns) != list(df.columns):
            raise ValueError(
                f"parquet validation failed for {name}: "
                f"rows {len(check)} != {len(df)} or columns differ")
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    # Atomic replace: the previous valid dataset survives any failure
    # above this point.
    os.replace(tmp, path)
    update_manifest(name, df, path, source=source, symbol=symbol,
                    timeframe=timeframe, extra=extra)
    return path


def load(name: str, as_of=None, *, strict: bool = True) -> pd.DataFrame:
    """Load a dataset, optionally applying the unified PIT cutoff.

    ``as_of=None`` is the explicit legacy opt-out (no cutoff, no
    availability filtering) — backward compatibility for existing
    callers. When ``as_of`` is given it must be timezone-aware; the same
    cutoff rules (future events, unfinished candles, availability) are
    applied via ``pit.apply_cutoff``. ``strict=True`` rejects observations
    with unknown ``available_at``.
    """
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
    if as_of is not None:
        tf = _timeframe_for(name)
        kind = _dataset_kind(name, tf)
        df = pit.apply_cutoff(df, as_of,
                              timeframe=tf if kind == "candle" else None,
                              strict=strict)
    return df


def load_many(names, as_of, *, strict: bool = True) -> dict:
    """Load several datasets under ONE common ``as_of`` cutoff.

    This is the entry point for decisions/backtests: every dataset is
    filtered by the same decision timestamp, so a decision never combines
    datasets from different effective cutoffs.
    """
    return {n: load(n, as_of, strict=strict) for n in names}


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

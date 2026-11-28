"""Point-in-time (PIT) correctness utilities for the data engine.

Three rules are enforced here, consistently for every dataset:

1. **Unfinished candles are never usable.** A candle opening at ``t`` with
   interval ``Δ`` closes at ``t + Δ`` (UTC). It may only be used by a
   decision at ``as_of`` when ``t + Δ <= as_of``. A decision at 10:30 must
   never see a 1h candle that ends at 11:00.

2. **``available_at`` distinguishes event time from availability.** The
   dataframe index (or ``event_time`` column) is when the market event
   occurred. ``available_at`` is when the observation actually became
   available to THIS system. Policy:
     - Live-recorded observations carry their real receipt timestamp
       (e.g. the recorder's ``_recv_ts``).
     - Bulk-downloaded historical observations have UNKNOWN availability
       and are stored with ``available_at = NaT`` (null). We never
       fabricate availability from candle close time or fetch time.
     - Strict PIT mode REJECTS unknown availability. Non-strict mode
       (backward compatibility) keeps unknown-availability rows but still
       rejects rows whose availability is KNOWN to be after ``as_of``.

3. **One unified ``as_of`` cutoff.** ``apply_cutoff`` is the single place
   where a decision timestamp filters a dataset. ``store.load_many``
   applies the SAME ``as_of`` to every dataset used in one decision, so a
   decision never mixes datasets from different effective cutoffs.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

# Candle timeframes supported by the data engine (v4_system.config.TIMEFRAMES).
CANDLE_TFS = ("1m", "5m", "15m", "1h", "4h", "1d")

CANDLE_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def normalize_as_of(as_of) -> pd.Timestamp:
    """Return ``as_of`` as a UTC-aware Timestamp.

    Naive timestamps are rejected: a decision time without a timezone is
    ambiguous and must never silently default to local or UTC time.
    """
    if as_of is None:
        raise ValueError("as_of is required (no hidden wall-clock default)")
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        raise ValueError(f"as_of must be timezone-aware, got naive {as_of!r}")
    return ts.tz_convert("UTC")


def candle_close_time(open_time, timeframe: str) -> pd.Timestamp:
    """UTC close time of the candle opening at ``open_time``."""
    if timeframe not in CANDLE_DELTA:
        raise ValueError(f"not a candle timeframe: {timeframe!r}")
    return pd.Timestamp(open_time) + CANDLE_DELTA[timeframe]


def last_closed_open(as_of, timeframe: str) -> pd.Timestamp:
    """Open time of the most recent candle that is CLOSED at ``as_of``.

    A candle opening at ``t`` is closed iff ``t + Δ <= as_of``. All Binance
    kline open times are aligned to the interval since the epoch, so the
    last closed open is ``floor((as_of - Δ) / Δ) * Δ``.
    """
    as_of = normalize_as_of(as_of)
    delta = CANDLE_DELTA[timeframe]
    return (as_of - delta).floor(delta)


def drop_unfinished_candles(df: pd.DataFrame, timeframe: str,
                            as_of) -> pd.DataFrame:
    """Keep only candles whose close time is <= ``as_of``.

    The index must be the candle OPEN time (UTC). Works on any as_of;
    the result is identical for every row regardless of strict mode —
    an unfinished candle is never usable, period.
    """
    as_of = normalize_as_of(as_of)
    delta = CANDLE_DELTA[timeframe]
    idx = pd.DatetimeIndex(df.index)
    closed = idx + delta <= as_of
    return df[closed]


def _event_mask(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Rows whose event time is not in the future relative to ``as_of``."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index <= as_of
    if "event_time" in df.columns:
        ev = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        return ev.fillna(pd.Timestamp.min.tz_localize("UTC")) <= as_of
    # No temporal information at all: nothing can be proven point-in-time.
    return pd.Series(False, index=df.index)


def apply_cutoff(df: pd.DataFrame, as_of, *, timeframe: str | None = None,
                 strict: bool = True) -> pd.DataFrame:
    """Apply the unified PIT cutoff to one dataset.

    Parameters
    ----------
    df : DataFrame indexed by event time (UTC) for time series, or with an
        ``event_time`` column / ``available_at`` column as applicable.
    as_of : timezone-aware decision timestamp (UTC). Required — there is
        no wall-clock default.
    timeframe : candle timeframe (e.g. "1h") when ``df`` holds candles;
        unfinished candles are dropped. ``None`` for event series
        (funding, OI, aggTrades, order-book observations).
    strict : when True, rows with UNKNOWN ``available_at`` are rejected.
        When False (backward compatibility), unknown-availability rows are
        kept, but rows with KNOWN availability after ``as_of`` are always
        rejected, as are future event timestamps and unfinished candles.

    Returns a new DataFrame; the input is never mutated.
    """
    as_of = normalize_as_of(as_of)
    out = df.copy()

    # 1. Future event timestamps are never usable.
    out = out[_event_mask(out, as_of)]

    # 2. Unfinished candles are never usable.
    if timeframe in CANDLE_DELTA:
        out = drop_unfinished_candles(out, timeframe, as_of)

    # 3. Availability cutoff.
    if "available_at" in out.columns:
        av = pd.to_datetime(out["available_at"], utc=True, errors="coerce")
        known_ok = av.notna() & (av <= as_of)
        if strict:
            out = out[known_ok.to_numpy()]
        else:
            out = out[(known_ok | av.isna()).to_numpy()]
    elif strict:
        # No availability information at all -> unknown -> reject everything
        # in strict mode. Non-strict keeps the rows (legacy behavior).
        out = out.iloc[0:0]

    return out


def live_record_available_at(rec: dict):
    """Availability timestamp of a live-recorded observation.

    Live recorder rows carry ``_recv_ts`` (ISO-8601 UTC) — the actual
    receipt time. Returns a tz-aware Timestamp, or None when the record
    has no receipt timestamp (unknown availability).
    """
    raw = rec.get("_recv_ts") or rec.get("available_at")
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        return None
    return ts.tz_convert("UTC")

"""Thin adapter: V4 data engine -> existing QuantResearch implementations.

Purpose
-------
Make the stored V4 datasets consumable by the already-audited research
code (``quant_research/core/*``, ``quant_research/strategies/*``,
``quant_research/candidate_runner.py``) WITHOUT duplicating indicators,
backtesting or statistics.

PIT policy (explicit, never hidden)
-----------------------------------
Historical bulk downloads have UNKNOWN availability: the store records no
``available_at`` for them. This adapter therefore loads with
``strict=False`` — event-time filtering and unfinished-candle exclusion
still apply (a decision at ``as_of`` never sees a candle that had not
closed, nor any event timestamped after ``as_of``), but rows with unknown
availability are kept. Every frame returned carries the attribute
``PIT_LABEL`` = "EVENT-TIME PIT — HISTORICAL AVAILABILITY UNKNOWN" so
reports cannot silently claim strict point-in-time validity.

Live-recorded data (``_recv_ts``) is NOT served by this adapter; it has
real availability timestamps and should be ingested into the store first.

Schema mapping
--------------
V4 candle datasets are indexed by candle OPEN time (UTC) with columns
``open, high, low, close, volume, close_time, quote_volume, trades``.
QuantResearch expects a DataFrame indexed by timestamp (UTC) with at
least ``open, high, low, close, volume`` — the V4 frame already satisfies
this; the adapter only guarantees the index dtype and column presence.

Funding datasets are indexed by settlement time with column
``fundingRate``; the adapter renames to ``funding_rate`` (the name used
by ``quant_research`` runners and ``QuantEngine.funding_payment``).

Known limitations (documented, not worked around)
-------------------------------------------------
* OI datasets cover only ~30 days (exchange cap) — excluded from
  historical features by default (``include_oi=False``).
* Margin-tier snapshots are current-only; historical effective dates are
  unknown — the adapter exposes them for reference but they must NOT be
  used for historical liquidation claims.
* ``daily_trend_map`` is rebuilt from V4 1m perp data (available from
  2023-01-01). The EMA(20) daily warmup is ~20 trading days, so trend
  values are valid well before the Program D IS/OOS windows; the map is
  NOT identical to the original archive-seeded version and must be
  labelled as such in any report.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from v4_system.data_engine import pit, store

PIT_LABEL = "EVENT-TIME PIT — HISTORICAL AVAILABILITY UNKNOWN"

# V4 dataset naming: <SYMBOL>_<market>_<timeframe> / <SYMBOL>_funding
MARKETS = ("perp", "spot", "mark", "index")
CANDLE_TFS = ("1m", "5m", "15m", "1h", "4h", "1d")

# Columns required by quant_research strategies/engine.
OHLCV_COLS = ("open", "high", "low", "close", "volume")


def _check_as_of(as_of) -> pd.Timestamp:
    """Require an explicit timezone-aware UTC decision timestamp."""
    return pit.normalize_as_of(as_of)


def dataset_name(symbol: str, timeframe: str, market: str = "perp") -> str:
    """V4 store dataset name for a candle series."""
    if market not in MARKETS:
        raise ValueError(f"market must be one of {MARKETS}, got {market!r}")
    if timeframe not in CANDLE_TFS:
        raise ValueError(f"timeframe must be one of {CANDLE_TFS}, got {timeframe!r}")
    return f"{symbol}_{market}_{timeframe}"


def load_klines(symbol: str, timeframe: str, as_of, *,
                market: str = "perp", strict: bool = False) -> pd.DataFrame:
    """Load a V4 candle dataset as a QuantResearch-compatible frame.

    Returns a DataFrame indexed by candle OPEN time (UTC, tz-aware),
    sorted, with OHLCV columns present. ``as_of`` is required and must be
    timezone-aware; it is applied as the single unified cutoff (future
    events and unfinished candles are always excluded). ``strict=False``
    is the default because historical availability is unknown — the
    returned frame carries ``PIT_LABEL``.
    """
    as_of = _check_as_of(as_of)
    name = dataset_name(symbol, timeframe, market)
    df = store.load(name, as_of=as_of, strict=strict)
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset {name} missing OHLCV columns: {missing}")
    df = df.sort_index()
    df.attrs["pit_label"] = PIT_LABEL
    df.attrs["dataset"] = name
    df.attrs["as_of"] = str(as_of)
    df.attrs["strict"] = strict
    return df


def load_funding(symbol: str, as_of, *, strict: bool = False) -> pd.Series:
    """Load historical funding settlements as a ``funding_rate`` Series.

    Index = actual settlement timestamps (UTC). No resampling, no
    interpolation — settlement times are used exactly as recorded.
    """
    as_of = _check_as_of(as_of)
    df = store.load(f"{symbol}_funding", as_of=as_of, strict=strict)
    if "fundingRate" not in df.columns:
        raise ValueError(f"{symbol}_funding missing 'fundingRate' column")
    s = df["fundingRate"].astype(float).sort_index()
    s.attrs["pit_label"] = PIT_LABEL
    s.attrs["dataset"] = f"{symbol}_funding"
    s.attrs["as_of"] = str(as_of)
    return s.rename("funding_rate")


def load_universe(symbols, timeframe: str, as_of, *,
                  market: str = "perp", strict: bool = False) -> dict:
    """Load klines + funding for several symbols under ONE ``as_of``.

    Returns ``{"klines": {sym: df}, "funding": {sym: series}}``. All
    datasets share the identical cutoff, so a decision never mixes
    effective times.
    """
    as_of = _check_as_of(as_of)
    return {
        "klines": {s: load_klines(s, timeframe, as_of, market=market,
                                  strict=strict) for s in symbols},
        "funding": {s: load_funding(s, as_of, strict=strict) for s in symbols},
    }


def load_1m_for_subbars(symbol: str, as_of, *, strict: bool = False) -> pd.DataFrame:
    """Load 1m perp candles for intrabar (sub-bar) exit resolution.

    ``run_turtle_simulation(..., df_1m=...)`` uses this to resolve
    stop-vs-target ordering inside a signal bar. V4 1m perp history
    starts 2023-01-01 — sufficient for the Program D IS/OOS windows but
    NOT for earlier history.
    """
    return load_klines(symbol, "1m", as_of, market="perp", strict=strict)


def daily_trend_map(symbol: str, as_of, *, ema_span: int = 20,
                    strict: bool = False) -> pd.Series:
    """Daily EMA trend filter built from V4 1m perp data.

    Delegates the trend VALUES to the single audited implementation
    ``quant_research.candidate_runner.daily_trend_from_daily_close`` —
    there is exactly one EMA/sign construction in the codebase. The V4
    adapter only supplies the daily close series (last close of each UTC
    day resampled from the 1m perp series, identical resampling to
    ``candidate_runner.daily_trend_map``).

    Seeded at the V4 archive start (2023-01-01), not the original
    quant_research archive start — values from ~2023-02 onward are valid;
    label reports accordingly. Warmup (~``ema_span`` trading days) yields
    unreliable early values.
    """
    from quant_research.candidate_runner import daily_trend_from_daily_close
    as_of = _check_as_of(as_of)
    m1 = load_klines(symbol, "1m", as_of, market="perp", strict=strict)
    daily = m1["close"].resample("1D").last().dropna()
    trend = daily_trend_from_daily_close(daily, span=ema_span)
    trend.attrs["pit_label"] = PIT_LABEL
    trend.attrs["source"] = f"{symbol}_perp_1m resampled 1D, EMA{ema_span}"
    return trend


def apply_daily_mask(features: dict, trend: pd.Series) -> dict:
    """The audited daily-trend entry gate, verbatim from
    ``quant_research.candidate_runner.apply_daily_mask``. Re-exported so
    Phase 2 runners use the exact tested function — no second copy."""
    from quant_research.candidate_runner import apply_daily_mask as _impl
    return _impl(features, trend)


def load_turtle_bundle(symbol: str, timeframe: str, as_of, *,
                       strict: bool = False,
                       include_1m_subbars: bool = True,
                       include_daily_trend: bool = True) -> dict:
    """Load everything ``run_turtle_simulation`` needs for one symbol/TF.

    Returns ``{"df", "funding", "df_1m", "daily_trend"}``:
      - ``df``        : signal-timeframe klines (OHLCV, UTC index)
      - ``funding``   : funding_rate Series at actual settlement times
      - ``df_1m``     : 1m perp klines for intrabar stop/level resolution
                        (None when ``include_1m_subbars=False``)
      - ``daily_trend``: EMA20 daily trend map for ``apply_daily_mask``
                        (None when ``include_daily_trend=False``)

    All datasets share the SAME ``as_of`` cutoff. ``df_1m`` covers only
    the V4 1m history (from 2023-01-01); bars outside that range fall
    back to the engine's documented pessimistic stop-first rule — the
    runner MUST report ``ambiguous_bars`` and the sub-bar coverage so
    this fallback is never silent.
    """
    as_of = _check_as_of(as_of)
    bundle = {
        "df": load_klines(symbol, timeframe, as_of, market="perp",
                          strict=strict),
        "funding": load_funding(symbol, as_of, strict=strict),
        "df_1m": (load_1m_for_subbars(symbol, as_of, strict=strict)
                  if include_1m_subbars else None),
        "daily_trend": (daily_trend_map(symbol, as_of, strict=strict)
                        if include_daily_trend else None),
    }
    for k, v in bundle.items():
        if v is not None and hasattr(v, "attrs"):
            v.attrs["pit_label"] = PIT_LABEL
    return bundle


def load_margin_tiers_as_of(symbol: str, as_of):
    """Current margin-tier snapshot visible at ``as_of`` (PIT query).

    WARNING: snapshots are dated observations; historical effective dates
    are UNKNOWN. Suitable for current-risk checks and clearly-labelled
    assumptions — NOT for historical liquidation reconstruction.
    """
    from v4_system.data_engine import margin_tiers as mt
    return mt.margin_tiers_as_of(symbol, pit.normalize_as_of(as_of))


def load_contract_specs(symbol: str):
    """Current contract specification snapshot (tick/step/min notional/
    margin params). Current snapshot only — not historical."""
    df = store.load("contract_specs")
    return df[df["symbol"] == symbol]

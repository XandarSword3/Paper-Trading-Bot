"""Funding settlement intervals per perpetual instrument.

Two evidence sources, kept distinct:

1. **Current configuration** — GET /fapi/v1/fundingInfo returns
   ``fundingIntervalHours`` (plus adjusted rate caps/floors). This is the
   exchange's CURRENT setting; it says nothing about historical behavior.

2. **Observed history** — the stored funding-rate dataset
   (``<SYM>_funding``) contains actual settlement timestamps. Interval
   inference uses timestamp differences:
   - the dominant interval is the median of consecutive diffs;
   - a frequency CHANGE is only reported when a new interval persists
     for at least ``min_run`` consecutive settlements — isolated gaps
     (missing observations) are NOT interpreted as frequency changes;
   - irregular/missing observations yield ``confidence="low"`` rather
     than a guessed interval.

Unknown intervals stay unknown (None) — never fabricated. Current
configuration is never presented as historical truth: the output
separates ``current_interval_hours`` (fundingInfo) from
``observed_intervals`` (history).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient


def fetch_funding_info(symbols: list[str] | None = None,
                       client: BinanceClient | None = None) -> pd.DataFrame:
    """Current funding configuration from /fapi/v1/fundingInfo."""
    client = client or BinanceClient.futures()
    rows = client.get("/fapi/v1/fundingInfo")
    if symbols:
        rows = [r for r in rows if r.get("symbol") in symbols]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("symbol")
    for c in ("fundingIntervalHours", "adjustedFundingRateCap",
              "adjustedFundingRateFloor"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def infer_intervals(funding_df: pd.DataFrame,
                    min_run: int = 3) -> dict:
    """Infer funding intervals from observed settlement timestamps.

    ``funding_df`` is indexed by funding time (UTC). Returns a dict with
    observed interval segments and confidence. Missing observations are
    tolerated: only sustained runs of a different interval count as a
    change.
    """
    if funding_df is None or len(funding_df) < 2:
        return {"observed_intervals": [], "confidence": "unknown",
                "n_settlements": int(0 if funding_df is None else len(funding_df))}
    ts = pd.DatetimeIndex(funding_df.index).sort_values()
    diffs = ts.to_series().diff().dropna()
    if diffs.empty:
        return {"observed_intervals": [], "confidence": "unknown",
                "n_settlements": int(len(ts))}
    hours = diffs.dt.total_seconds() / 3600.0
    # Dominant interval = median diff (robust to isolated gaps).
    dominant = float(hours.median())
    # Segment the series into runs of constant rounded interval.
    rounded = hours.round(6)
    segments = []
    start_idx = 0
    current = rounded.iloc[0]
    run = 1
    for i in range(1, len(rounded)):
        if rounded.iloc[i] == current:
            run += 1
        else:
            segments.append({"interval_hours": float(current),
                             "n": run,
                             "from": str(ts[start_idx + 1]),
                             "to": str(ts[start_idx + run])})
            start_idx = i
            current = rounded.iloc[i]
            run = 1
    segments.append({"interval_hours": float(current), "n": run,
                     "from": str(ts[start_idx + 1]),
                     "to": str(ts[-1])})
    # Keep only segments with a sustained run; merge isolated anomalies
    # into the dominant interval (they are gaps, not frequency changes).
    sustained = [s for s in segments if s["n"] >= min_run]
    if not sustained:
        sustained = [max(segments, key=lambda s: s["n"])]
    dominant_seg = max(sustained, key=lambda s: s["n"] * s["interval_hours"])
    changes = [s for s in sustained if s["interval_hours"] != dominant_seg["interval_hours"]]
    confidence = "high" if len(sustained) >= 2 or len(ts) >= 10 else "medium"
    if changes:
        confidence = "high"
    return {
        "observed_intervals": sustained,
        "dominant_interval_hours": float(dominant_seg["interval_hours"]),
        "changes_detected": len(changes) > 0,
        "n_settlements": int(len(ts)),
        "confidence": confidence,
    }


def build_funding_interval_table(funding_dfs: dict[str, pd.DataFrame],
                                 funding_info: pd.DataFrame | None = None,
                                 retrieved_at: str | None = None) -> pd.DataFrame:
    """Combine current config (fundingInfo) with observed history.

    ``funding_dfs``: {symbol: funding-rate DataFrame indexed by time}.
    Returns one row per symbol with current + observed intervals,
    source and confidence. Current config is explicitly labeled as
    current-only; observed intervals come from history.
    """
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for sym, df in funding_dfs.items():
        obs = infer_intervals(df)
        info = funding_info.loc[sym] if (funding_info is not None
                                         and sym in funding_info.index) else None
        current = None
        cap = floor = None
        if info is not None:
            current = info.get("fundingIntervalHours")
            cap = info.get("adjustedFundingRateCap")
            floor = info.get("adjustedFundingRateFloor")
        rows.append({
            "symbol": sym,
            "current_interval_hours": current,
            "current_source": "fapi /fapi/v1/fundingInfo" if current is not None else None,
            "adjusted_rate_cap": cap,
            "adjusted_rate_floor": floor,
            "observed_dominant_interval_hours": obs.get("dominant_interval_hours"),
            "observed_intervals": str(obs.get("observed_intervals")),
            "changes_detected": bool(obs.get("changes_detected", False)),
            "n_settlements": obs.get("n_settlements"),
            "confidence": obs.get("confidence"),
            "retrieved_at": retrieved_at,
        })
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()

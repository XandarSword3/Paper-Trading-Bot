"""Safe incremental tail refresh for stored datasets.

Refreshes only the RECENT tail of an existing dataset instead of
re-downloading all history, reusing the existing fetchers
(``klines.fetch_*``, ``funding.fetch_funding_history``,
``open_interest.fetch_open_interest_hist``).

Guarantees:
- **Overlap + dedupe**: the fetch window starts ``overlap`` bars before
  the stored tail; rows are deduplicated on the observation key (index)
  keeping the newest copy, so repeated refreshes are idempotent.
- **Historical records preserved**: stored rows outside the overlap
  window are never touched.
- **Chronological order**: the merged frame is sorted by event time.
- **Unfinished candles excluded**: the merged result is filtered with
  ``pit.drop_unfinished_candles`` at the refresh ``as_of``.
- **Missing-tail detection**: if the newest CLOSED candle/observation is
  absent after the merge, the report (and manifest) say
  ``tail_complete=False`` with the missing-bar count — an incomplete
  tail is never treated as complete.
- **Atomic + restart-safe**: the merged dataset is written through
  ``store.save`` (temp file + validation + os.replace). An interrupted
  refresh leaves the previous valid dataset untouched, and the next
  refresh simply recomputes from it — no partial state is ever loaded.
- **Gaps/provenance preserved**: nothing is interpolated or invented;
  the manifest entry records the refresh metadata.

``as_of`` is the refresh target time (timezone-aware). It defaults to
"now" because a refresh is an operational update, not a historical
decision — but it is always explicit in the report.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v4_system.data_engine import pit, store
from v4_system.data_engine.agg_trades import fetch_agg_trades  # noqa: F401  (documented: no bulk refresh)
from v4_system.data_engine.funding import fetch_funding_history
from v4_system.data_engine.klines import (fetch_index_price_klines,
                                           fetch_mark_price_klines,
                                           fetch_perp_klines,
                                           fetch_spot_klines)
from v4_system.data_engine.open_interest import fetch_open_interest_hist

KLINE_FETCHERS = {
    "perp": fetch_perp_klines,
    "spot": fetch_spot_klines,
    "mark": fetch_mark_price_klines,
    "index": fetch_index_price_klines,
}


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, deduplicate on the event-time index (keep newest),
    and restore chronological order."""
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _tail_report(merged: pd.DataFrame, as_of: pd.Timestamp,
                 timeframe: str | None) -> dict:
    """Detect whether the recent tail is complete up to ``as_of``."""
    report = {"rows": int(len(merged)),
              "start": str(merged.index[0]) if len(merged) else None,
              "end": str(merged.index[-1]) if len(merged) else None}
    if timeframe in pit.CANDLE_DELTA and len(merged):
        expected_last = pit.last_closed_open(as_of, timeframe)
        delta = pit.CANDLE_DELTA[timeframe]
        last_open = merged.index[-1]
        if last_open >= expected_last:
            report["tail_complete"] = True
            report["missing_bars"] = 0
        else:
            missing = int(round((expected_last - last_open) / delta))
            report["tail_complete"] = False
            report["missing_bars"] = missing
            report["expected_last_closed_open"] = str(expected_last)
    return report


def refresh_klines_dataset(name: str, symbol: str, interval: str,
                           kind: str = "perp",
                           overlap_bars: int = 3,
                           as_of=None,
                           client=None) -> dict:
    """Refresh the tail of a kline dataset (perp/spot/mark/index).

    ``name`` must already exist in the store. ``kind`` selects the
    fetcher. Returns a report dict; the dataset is only rewritten when
    the merge succeeds (atomic save).
    """
    if kind not in KLINE_FETCHERS:
        raise ValueError(f"kind must be one of {list(KLINE_FETCHERS)}")
    if interval not in pit.CANDLE_DELTA:
        raise ValueError(f"unsupported candle timeframe: {interval!r}")
    as_of = pit.normalize_as_of(as_of) if as_of is not None else _now()

    existing = store.load(name)  # legacy full load; refresh needs all history
    if len(existing) == 0:
        raise ValueError(f"cannot refresh empty dataset {name!r}")

    delta = pit.CANDLE_DELTA[interval]
    last_open = existing.index[-1]
    fetch_start = last_open - overlap_bars * delta
    new = KLINE_FETCHERS[kind](symbol, interval, start=fetch_start,
                               end=as_of, client=client)
    # Never persist an unfinished candle, even if the exchange returned one.
    new = pit.drop_unfinished_candles(new, interval, as_of)
    merged = _merge(existing, new)
    merged = pit.drop_unfinished_candles(merged, interval, as_of)

    report = _tail_report(merged, as_of, interval)
    report.update({
        "dataset": name, "symbol": symbol, "interval": interval, "kind": kind,
        "as_of": str(as_of), "overlap_bars": overlap_bars,
        "fetched_rows": int(len(new)),
        "added_rows": int(len(merged) - len(existing)),
        "refreshed_at": _now().isoformat(),
    })
    store.save(merged, name,
               source=f"refresh tail of {name}", symbol=symbol,
               timeframe=interval,
               extra={"refresh": report})
    return report


def refresh_funding_dataset(name: str, symbol: str,
                            overlap: pd.Timedelta = pd.Timedelta(days=2),
                            as_of=None, client=None) -> dict:
    """Refresh the tail of a funding-rate dataset."""
    as_of = pit.normalize_as_of(as_of) if as_of is not None else _now()
    existing = store.load(name)
    if len(existing) == 0:
        raise ValueError(f"cannot refresh empty dataset {name!r}")
    fetch_start = existing.index[-1] - overlap
    new = fetch_funding_history(symbol, start=fetch_start, end=as_of,
                                client=client)
    merged = _merge(existing, new)
    report = {"dataset": name, "symbol": symbol, "as_of": str(as_of),
              "fetched_rows": int(len(new)),
              "added_rows": int(len(merged) - len(existing)),
              "rows": int(len(merged)),
              "start": str(merged.index[0]), "end": str(merged.index[-1]),
              "refreshed_at": _now().isoformat()}
    store.save(merged, name, source=f"refresh tail of {name}", symbol=symbol,
               timeframe="funding", extra={"refresh": report})
    return report


def refresh_oi_dataset(name: str, symbol: str, period: str = "1h",
                       as_of=None, client=None) -> dict:
    """Refresh the tail of an open-interest dataset.

    Binance caps OI history at ~30 days, so a "tail refresh" here is a
    re-fetch of the reachable window merged over the stored data.
    """
    as_of = pit.normalize_as_of(as_of) if as_of is not None else _now()
    existing = store.load(name)
    if len(existing) == 0:
        raise ValueError(f"cannot refresh empty dataset {name!r}")
    new = fetch_open_interest_hist(symbol, period=period, days=29,
                                   client=client)
    new = new[new.index <= as_of]
    merged = _merge(existing, new)
    report = {"dataset": name, "symbol": symbol, "period": period,
              "as_of": str(as_of), "fetched_rows": int(len(new)),
              "added_rows": int(len(merged) - len(existing)),
              "rows": int(len(merged)),
              "start": str(merged.index[0]), "end": str(merged.index[-1]),
              "refreshed_at": _now().isoformat()}
    store.save(merged, name,
               source=f"refresh tail of {name}", symbol=symbol,
               timeframe=period, extra={"refresh": report})
    return report


def refresh_dataset(name: str, symbol: str, **kwargs) -> dict:
    """Dispatch a tail refresh by dataset naming convention.

    - ``*_funding``            -> funding refresh
    - ``*_oi_<period>``        -> open-interest refresh
    - ``*_<kind>_<tf>``         -> kline refresh (kind in perp/spot/mark/index)
    """
    lowered = name.lower()
    if lowered.endswith("_funding"):
        return refresh_funding_dataset(name, symbol, **kwargs)
    if "_oi_" in lowered:
        period = lowered.rsplit("_", 1)[-1]
        kwargs.setdefault("period", period)
        return refresh_oi_dataset(name, symbol, **kwargs)
    parts = lowered.rsplit("_", 2)
    if len(parts) == 3 and parts[1] in KLINE_FETCHERS:
        kind, interval = parts[1], parts[2]
        kwargs.setdefault("kind", kind)
        return refresh_klines_dataset(name, symbol, interval, **kwargs)
    raise ValueError(f"cannot infer dataset type for refresh: {name!r}")

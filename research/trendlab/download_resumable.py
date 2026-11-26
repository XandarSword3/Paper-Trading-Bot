"""Resumable Binance-vision downloader for trendlab.

Same sources and output layout as ``data.download_all_fast`` (so
``load_panel`` / ``load_funding`` work unchanged), but caches **each monthly
file** as its own parquet before moving on. A dropped connection loses at
most the files in flight; rerunning skips everything already cached.

Months that fail after retries are recorded in ``months/FAILED.json`` and
reported at the end — they are never silently treated as "no data".

Layout::

    data/binance/months/klines/{SYM}_{interval}/{YYYY-MM}.parquet
    data/binance/months/funding/{SYM}/{YYYY-MM}.parquet
    data/binance/klines/{SYM}_{interval}.parquet      (assembled)
    data/binance/funding/{SYM}.parquet               (assembled)

Usage::

    python -m research.trendlab.download_resumable [--interval 1h] [--symbols BTCUSDT ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import data
from .data import (FUNDING_DIR, KLINE_DIR, UNIVERSE, VISION, _fetch,
                   _month_key, _month_range, _normalise_klines, _parse_zip_csv)

logger = logging.getLogger(__name__)

MONTHS_DIR = data.CACHE_DIR / "months"
KLINE_MONTHS = MONTHS_DIR / "klines"
FUNDING_MONTHS = MONTHS_DIR / "funding"
_FAILED_LOCK = threading.Lock()
FAILED_PATH = MONTHS_DIR / "FAILED.json"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_failed() -> dict:
    if FAILED_PATH.exists():
        try:
            return json.loads(FAILED_PATH.read_text())
        except Exception:
            return {}
    return {}


def _record_failure(kind: str, key: str, ym: tuple[int, int], err: str) -> None:
    with _FAILED_LOCK:
        failed = _load_failed()
        failed.setdefault(kind, {})[f"{key}/{_month_key(*ym)}"] = err
        MONTHS_DIR.mkdir(parents=True, exist_ok=True)
        FAILED_PATH.write_text(json.dumps(failed, indent=2))


def _clear_failure(kind: str, key: str, ym: tuple[int, int]) -> None:
    with _FAILED_LOCK:
        failed = _load_failed()
        bucket = failed.get(kind, {})
        bucket.pop(f"{key}/{_month_key(*ym)}", None)
        if bucket:
            failed[kind] = bucket
        else:
            failed.pop(kind, None)
        MONTHS_DIR.mkdir(parents=True, exist_ok=True)
        FAILED_PATH.write_text(json.dumps(failed, indent=2))


def _month_parquet_path(kind_dir: Path, key: str, ym: tuple[int, int]) -> Path:
    return kind_dir / key / f"{_month_key(*ym)}.parquet"


def _fetch_kline_month(symbol: str, interval: str, ym: tuple[int, int]) -> tuple[str, str, str | None]:
    """Returns (symbol, month_key, error_or_None). Writes parquet on success."""
    out = _month_parquet_path(KLINE_MONTHS, f"{symbol}_{interval}", ym)
    if out.exists():
        return symbol, _month_key(*ym), None
    url = (f"{VISION}/monthly/klines/{symbol}/{interval}/"
           f"{symbol}-{interval}-{_month_key(*ym)}.zip")
    payload = _fetch(url)
    if payload is None:
        return symbol, _month_key(*ym), f"fetch failed: {url}"
    try:
        df = _normalise_klines(_parse_zip_csv(payload))
        if df.empty:
            return symbol, _month_key(*ym), "empty after parse"
        _atomic_write_parquet(df, out)
        _clear_failure("klines", f"{symbol}_{interval}", ym)
        return symbol, _month_key(*ym), None
    except Exception as exc:  # noqa: BLE001
        return symbol, _month_key(*ym), f"parse failed: {exc}"


def _fetch_funding_month(symbol: str, ym: tuple[int, int]) -> tuple[str, str, str | None]:
    out = _month_parquet_path(FUNDING_MONTHS, symbol, ym)
    if out.exists():
        return symbol, _month_key(*ym), None
    url = f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{_month_key(*ym)}.zip"
    payload = _fetch(url)
    if payload is None:
        return symbol, _month_key(*ym), f"fetch failed: {url}"
    try:
        raw = _parse_zip_csv(payload).iloc[:, :3].copy()
        raw.columns = ["timestamp", "interval_hours", "funding_rate"]
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna(subset=["timestamp", "funding_rate"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="ms", utc=True)
        df = raw[["timestamp", "funding_rate"]]
        if df.empty:
            return symbol, _month_key(*ym), "empty after parse"
        _atomic_write_parquet(df, out)
        _clear_failure("funding", symbol, ym)
        return symbol, _month_key(*ym), None
    except Exception as exc:  # noqa: BLE001
        return symbol, _month_key(*ym), f"parse failed: {exc}"


def _assemble_klines(symbol: str, interval: str, months: list[tuple[int, int]]) -> Path | None:
    frames = []
    missing = []
    for ym in months:
        p = _month_parquet_path(KLINE_MONTHS, f"{symbol}_{interval}", ym)
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append(_month_key(*ym))
    if not frames:
        return None
    frame = (pd.concat(frames, ignore_index=True)
             .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))
    out = KLINE_DIR / f"{symbol}_{interval}.parquet"
    _atomic_write_parquet(frame, out)
    if missing:
        logger.warning("%s %s assembled with %d missing months: %s",
                       symbol, interval, len(missing), missing[:10])
    return out


def _assemble_funding(symbol: str, months: list[tuple[int, int]]) -> Path | None:
    frames = []
    missing = []
    for ym in months:
        p = _month_parquet_path(FUNDING_MONTHS, symbol, ym)
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append(_month_key(*ym))
    if not frames:
        return None
    frame = (pd.concat(frames, ignore_index=True)
             .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))
    out = FUNDING_DIR / f"{symbol}.parquet"
    _atomic_write_parquet(frame, out)
    if missing:
        logger.warning("%s funding assembled with %d missing months: %s",
                       symbol, len(missing), missing[:10])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--start", default="2019-09")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip downloads; just rebuild combined parquets from cached months")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbols = args.symbols or UNIVERSE
    sy, sm = (int(x) for x in args.start.split("-"))
    now = pd.Timestamp.now(tz="UTC")
    months = _month_range((sy, sm), (now.year, now.month))

    KLINE_MONTHS.mkdir(parents=True, exist_ok=True)
    FUNDING_MONTHS.mkdir(parents=True, exist_ok=True)
    KLINE_DIR.mkdir(parents=True, exist_ok=True)
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    if not args.assemble_only:
        tasks = [("k", s, ym) for s in symbols for ym in months]
        tasks += [("f", s, ym) for s in symbols for ym in months]

        done = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for kind, sym, ym in tasks:
                if kind == "k":
                    futs[pool.submit(_fetch_kline_month, sym, args.interval, ym)] = ("k", sym, ym)
                else:
                    futs[pool.submit(_fetch_funding_month, sym, ym)] = ("f", sym, ym)
            for fut in as_completed(futs):
                kind, sym, ym = futs[fut]
                try:
                    _, mkey, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    err = f"worker exception: {exc}"
                    mkey = _month_key(*ym)
                done += 1
                if err:
                    failed += 1
                    bucket = "klines" if kind == "k" else "funding"
                    key = f"{sym}_{args.interval}" if kind == "k" else sym
                    _record_failure(bucket, key, ym, err)
                if done % 500 == 0:
                    rate = done / max(time.time() - t0, 1)
                    print(f"  {done}/{len(tasks)} months ({rate:.1f}/s), {failed} failed")
        print(f"downloads done: {done} tasks, {failed} failed "
              f"(see {FAILED_PATH} if non-empty)")

    # Assemble combined parquets.
    for sym in symbols:
        kp = _assemble_klines(sym, args.interval, months)
        fp = _assemble_funding(sym, months)
        if kp:
            df = pd.read_parquet(kp)
            print(f"  {sym:12s} klines {len(df):8d} rows  {df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")
        if fp:
            print(f"  {sym:12s} funding {len(pd.read_parquet(fp)):6d} rows")

    failed = _load_failed()
    if failed:
        print(f"\nWARNING: {sum(len(v) for v in failed.values())} failed months recorded in {FAILED_PATH}")
        print("Rerun this script to retry only the missing months.")


if __name__ == "__main__":
    main()

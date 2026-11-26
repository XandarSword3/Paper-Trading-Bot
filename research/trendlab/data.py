"""Binance USDT-M futures data download + panel construction.

Public bulk data (https://data.binance.vision) is used instead of the REST
API because it gives full history without pagination and without rate limits.

Cached layout::

    data/binance/klines/{SYMBOL}_{interval}.parquet
    data/binance/funding/{SYMBOL}.parquet
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import urllib.request

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "binance"
KLINE_DIR = CACHE_DIR / "klines"
FUNDING_DIR = CACHE_DIR / "funding"

VISION = "https://data.binance.vision/data/futures/um"

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# A broad, liquid USDT-M perpetual universe.  Symbols that listed later simply
# start later; the backtester handles ragged panels.
UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "TRXUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "FILUSDT",
    "ICPUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "INJUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "WIFUSDT",
    "TONUSDT",
    "ENAUSDT",
    "AAVEUSDT",
    "UNIUSDT",
    "RENDERUSDT",
    "FETUSDT",
    "ARUSDT",
]


def _fetch(url: str, timeout: float = 45.0, retries: int = 3) -> bytes | None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - 404 means "file does not exist"
            last = exc
            if "404" in str(exc) or "Not Found" in str(exc):
                return None
            time.sleep(0.5 * (attempt + 1))
    logger.debug("fetch failed %s: %s", url, last)
    return None


def _parse_zip_csv(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        raw = pd.read_csv(zf.open(name), header=None)
    # Files after ~2024 include a header row; older files do not.
    if isinstance(raw.iloc[0, 0], str) and not str(raw.iloc[0, 0]).lstrip("-").isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    return raw


def _month_key(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y, m = start
    out: list[tuple[int, int]] = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _normalise_klines(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.iloc[:, : len(KLINE_COLS)].copy()
    raw.columns = KLINE_COLS
    for col in KLINE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["open_time", "open", "high", "low", "close"])
    raw["timestamp"] = pd.to_datetime(raw["open_time"].astype("int64"), unit="ms", utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp")
    out = raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "count",
               "taker_buy_volume", "taker_buy_quote_volume"]].reset_index(drop=True)
    return out


@dataclass
class DownloadResult:
    symbol: str
    interval: str
    rows: int
    start: str
    end: str
    months: int


def download_klines(
    symbol: str,
    interval: str = "1h",
    start: tuple[int, int] = (2019, 9),
    end: tuple[int, int] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> DownloadResult | None:
    """Download all available monthly files and cache a combined parquet."""
    KLINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = KLINE_DIR / f"{symbol}_{interval}.parquet"
    if out_path.exists() and not force:
        frame = pd.read_parquet(out_path)
        return DownloadResult(symbol, interval, len(frame), str(frame.timestamp.iloc[0]),
                              str(frame.timestamp.iloc[-1]), 0)

    if end is None:
        now = datetime.now(timezone.utc)
        end = (now.year, now.month)
    months = _month_range(start, end)

    def fetch_month(ym: tuple[int, int]) -> pd.DataFrame | None:
        url = (
            f"{VISION}/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{_month_key(*ym)}.zip"
        )
        payload = _fetch(url)
        if payload is None:
            return None
        try:
            return _normalise_klines(_parse_zip_csv(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("bad kline file %s: %s", url, exc)
            return None

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_month, ym): ym for ym in months}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        logger.warning("no kline data for %s", symbol)
        return None

    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    frame.to_parquet(out_path, index=False)
    logger.info("cached %s %s: %d rows %s -> %s", symbol, interval, len(frame),
                frame.timestamp.iloc[0], frame.timestamp.iloc[-1])
    return DownloadResult(symbol, interval, len(frame), str(frame.timestamp.iloc[0]),
                          str(frame.timestamp.iloc[-1]), len(months))


def download_funding(
    symbol: str,
    start: tuple[int, int] = (2019, 9),
    end: tuple[int, int] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> pd.DataFrame | None:
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FUNDING_DIR / f"{symbol}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    if end is None:
        now = datetime.now(timezone.utc)
        end = (now.year, now.month)
    months = _month_range(start, end)

    def fetch_month(ym: tuple[int, int]) -> pd.DataFrame | None:
        url = f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{_month_key(*ym)}.zip"
        payload = _fetch(url)
        if payload is None:
            return None
        try:
            raw = _parse_zip_csv(payload)
            raw = raw.iloc[:, :3].copy()
            raw.columns = ["timestamp", "interval_hours", "funding_rate"]
            for col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw = raw.dropna(subset=["timestamp", "funding_rate"])
            raw["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="ms", utc=True)
            return raw[["timestamp", "funding_rate"]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("bad funding file %s: %s", url, exc)
            return None

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_month, ym) for ym in months]
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        return None
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    frame.to_parquet(out_path, index=False)
    return frame


def download_all_fast(
    symbols: list[str] | None = None,
    interval: str = "1h",
    force: bool = False,
    workers: int = 24,
) -> list[DownloadResult]:
    """Globally parallel fetch across (symbol, month) tasks.

    Much faster than :func:`download_all` on a cold cache because the thread
    pool is shared instead of being re-created per symbol.
    """
    symbols = symbols or UNIVERSE
    now = datetime.now(timezone.utc)
    months = _month_range((2019, 9), (now.year, now.month))
    KLINE_DIR.mkdir(parents=True, exist_ok=True)
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)

    todo = [s for s in symbols if force or not (KLINE_DIR / f"{s}_{interval}.parquet").exists()]
    if not todo:
        print("all kline caches present")
    else:
        k_tasks = [(s, ym) for s in todo for ym in months]

        def kline_task(task: tuple[str, tuple[int, int]]):
            sym, ym = task
            url = f"{VISION}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{_month_key(*ym)}.zip"
            payload = _fetch(url)
            if payload is None:
                return sym, None
            try:
                return sym, _normalise_klines(_parse_zip_csv(payload))
            except Exception:  # noqa: BLE001
                return sym, None

        collected: dict[str, list[pd.DataFrame]] = {s: [] for s in todo}
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, df in pool.map(kline_task, k_tasks, chunksize=8):
                done += 1
                if df is not None and not df.empty:
                    collected[sym].append(df)
                if done % 400 == 0:
                    print(f"  klines {done}/{len(k_tasks)}")
        for sym in todo:
            if not collected[sym]:
                print(f"  {sym:12s} no data")
                continue
            frame = (pd.concat(collected[sym], ignore_index=True)
                     .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))
            frame.to_parquet(KLINE_DIR / f"{sym}_{interval}.parquet", index=False)
            print(f"  {sym:12s} {len(frame):8d} rows  {frame.timestamp.iloc[0].date()} -> {frame.timestamp.iloc[-1].date()}")

    # Funding is small; fetch in the same shared pool.
    ftodo = [s for s in symbols if force or not (FUNDING_DIR / f"{s}.parquet").exists()]
    if ftodo:
        f_tasks = [(s, ym) for s in ftodo for ym in months]

        def funding_task(task: tuple[str, tuple[int, int]]):
            sym, ym = task
            url = f"{VISION}/monthly/fundingRate/{sym}/{sym}-fundingRate-{_month_key(*ym)}.zip"
            payload = _fetch(url)
            if payload is None:
                return sym, None
            try:
                raw = _parse_zip_csv(payload).iloc[:, :3].copy()
                raw.columns = ["timestamp", "interval_hours", "funding_rate"]
                for col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
                raw = raw.dropna(subset=["timestamp", "funding_rate"])
                raw["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="ms", utc=True)
                return sym, raw[["timestamp", "funding_rate"]]
            except Exception:  # noqa: BLE001
                return sym, None

        fcol: dict[str, list[pd.DataFrame]] = {s: [] for s in ftodo}
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, df in pool.map(funding_task, f_tasks, chunksize=8):
                done += 1
                if df is not None and not df.empty:
                    fcol[sym].append(df)
                if done % 400 == 0:
                    print(f"  funding {done}/{len(f_tasks)}")
        for sym in ftodo:
            if fcol[sym]:
                frame = (pd.concat(fcol[sym], ignore_index=True)
                         .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))
                frame.to_parquet(FUNDING_DIR / f"{sym}.parquet", index=False)

    results = []
    for sym in symbols:
        p = KLINE_DIR / f"{sym}_{interval}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            results.append(DownloadResult(sym, interval, len(f), str(f.timestamp.iloc[0]),
                                          str(f.timestamp.iloc[-1]), 0))
    return results


def download_all(
    symbols: list[str] | None = None,
    interval: str = "1h",
    force: bool = False,
) -> list[DownloadResult]:
    symbols = symbols or UNIVERSE
    results: list[DownloadResult] = []
    for symbol in symbols:
        try:
            res = download_klines(symbol, interval=interval, force=force)
            if res is not None:
                results.append(res)
                print(f"  {symbol:12s} {res.rows:8d} rows  {res.start[:10]} -> {res.end[:10]}")
            download_funding(symbol, force=force)
        except Exception as exc:  # noqa: BLE001
            logger.warning("download failed for %s: %s", symbol, exc)
    return results


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
FREQ_RULE = {"1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D", "2d": "2D", "1w": "1W"}


def load_symbol(symbol: str, freq: str = "1d", interval: str = "1h") -> pd.DataFrame:
    path = KLINE_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing cache {path}; run download_all() first")
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    rule = FREQ_RULE.get(freq, freq)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "count": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
    return out


def load_panel(
    symbols: list[str] | None = None,
    freq: str = "1d",
    fields: tuple[str, ...] = ("open", "high", "low", "close", "quote_volume"),
) -> dict[str, pd.DataFrame]:
    """Return ``{field: DataFrame(index=time, columns=symbol)}``."""
    symbols = symbols or UNIVERSE
    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            per_symbol[symbol] = load_symbol(symbol, freq=freq)
        except FileNotFoundError:
            continue
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        cols = {sym: df[field] for sym, df in per_symbol.items() if field in df}
        panel = pd.DataFrame(cols).sort_index()
        out[field] = panel
    return out


def load_funding(symbol: str) -> pd.Series:
    path = FUNDING_DIR / f"{symbol}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path).set_index("timestamp").sort_index()
    return df["funding_rate"]


if __name__ == "__main__":  # pragma: no cover - manual helper
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"downloading {len(UNIVERSE)} symbols from Binance vision...")
    download_all_fast()

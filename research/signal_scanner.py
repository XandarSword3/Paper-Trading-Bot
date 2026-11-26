"""Standalone statistical signal scanner using cached candles only."""

from __future__ import annotations

import argparse
from math import erfc, sqrt
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


HORIZON_HOURS: tuple[int, int, int] = (1, 4, 8)
MIN_LIQ_ROWS_FOR_IMBALANCE = 500


def _parse_timeframe_to_minutes(timeframe: str) -> int:
    text = str(timeframe).strip().upper()
    if text.isdigit():
        return max(int(text), 1)

    if text.endswith("H") and text[:-1].replace(".", "", 1).isdigit():
        return max(int(float(text[:-1]) * 60), 1)

    if text.endswith("D") and text[:-1].replace(".", "", 1).isdigit():
        return max(int(float(text[:-1]) * 1440), 1)

    mapping = {
        "D": 1440,
        "W": 10080,
    }
    if text in mapping:
        return mapping[text]

    raise ValueError(f"Unsupported timeframe format: {timeframe}")


def _load_cached_candles(symbol: str, timeframe: str, cache_file: str | None = None) -> tuple[pd.DataFrame, Path]:
    cache_path = Path(cache_file) if cache_file else (config.BASE_DIR / "data" / f"candles_{symbol}_{timeframe}.pkl")
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    frame = pd.read_pickle(cache_path)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Cache missing required columns: {missing}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable rows in cache: {cache_path}")

    return out, cache_path


def _load_feed_cache(path: Path, columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        frame = pd.read_pickle(path)
    except Exception:
        return pd.DataFrame(columns=columns)

    if frame.empty:
        return pd.DataFrame(columns=columns)

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        return pd.DataFrame(columns=columns)

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp"] + numeric_columns)
    out = out[columns].sort_values("timestamp").drop_duplicates(columns, keep="last").reset_index(drop=True)
    return out


def _load_persisted_liquidation_feeds(symbol: str) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    data_dir = config.BASE_DIR / "data"
    paths = {
        "funding": data_dir / f"funding_{symbol}.pkl",
        "oi": data_dir / f"oi_{symbol}.pkl",
        "lsratio": data_dir / f"lsratio_{symbol}.pkl",
        "liquidations": data_dir / f"liquidations_{symbol}.pkl",
    }
    feeds = {
        "funding": _load_feed_cache(paths["funding"], ["timestamp", "funding_rate"], ["funding_rate"]),
        "oi": _load_feed_cache(paths["oi"], ["timestamp", "open_interest", "oi_delta"], ["open_interest", "oi_delta"]),
        "lsratio": _load_feed_cache(paths["lsratio"], ["timestamp", "buy_ratio", "sell_ratio", "ls_ratio"], ["buy_ratio", "sell_ratio", "ls_ratio"]),
        "liquidations": _load_feed_cache(paths["liquidations"], ["timestamp", "side", "size", "price"], ["size", "price"]),
    }
    return feeds, paths


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd_hist(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal


def _bb_percent_b(close: pd.Series, length: int = 20, stdev_mult: float = 2.0) -> pd.Series:
    basis = close.rolling(length, min_periods=length).mean()
    stdev = close.rolling(length, min_periods=length).std(ddof=0)
    upper = basis + (stdev_mult * stdev)
    lower = basis - (stdev_mult * stdev)
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def _volume_zscore(volume: pd.Series, length: int = 20) -> pd.Series:
    mean = volume.rolling(length, min_periods=length).mean()
    stdev = volume.rolling(length, min_periods=length).std(ddof=0).replace(0.0, np.nan)
    return (volume - mean) / stdev


def _atr_percentile_rank(high: pd.Series, low: pd.Series, close: pd.Series, atr_length: int = 14, rank_window: int = 100) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(atr_length, min_periods=atr_length).mean()

    def _last_percentile(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        last = values[-1]
        return float(np.mean(values <= last))

    return atr.rolling(rank_window, min_periods=max(atr_length, 20)).apply(_last_percentile, raw=True)


def _align_asof(candle_ts: pd.Series, feed: pd.DataFrame, value_col: str) -> pd.Series:
    if feed.empty or value_col not in feed.columns:
        return pd.Series(np.nan, index=candle_ts.index)

    left = pd.DataFrame({"timestamp": pd.to_datetime(candle_ts, utc=True, errors="coerce")}).reset_index(names="_idx")
    right = feed[["timestamp", value_col]].copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    right[value_col] = pd.to_numeric(right[value_col], errors="coerce")
    right = right.dropna(subset=["timestamp", value_col]).sort_values("timestamp")
    if right.empty:
        return pd.Series(np.nan, index=candle_ts.index)

    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right,
        on="timestamp",
        direction="backward",
    )
    merged = merged.sort_values("_idx")
    return merged[value_col].reset_index(drop=True)


def _spearman_ic_pvalue(signal_values: pd.Series, target_returns: pd.Series) -> tuple[float, float, int]:
    aligned = pd.concat(
        [
            signal_values.rename("signal"),
            target_returns.rename("target"),
        ],
        axis=1,
    ).dropna()

    n = int(len(aligned))
    if n < 3:
        return 0.0, 1.0, n

    signal_rank = aligned["signal"].rank(method="average")
    target_rank = aligned["target"].rank(method="average")
    if signal_rank.nunique(dropna=True) <= 1 or target_rank.nunique(dropna=True) <= 1:
        return 0.0, 1.0, n

    ic = float(signal_rank.corr(target_rank))
    if not np.isfinite(ic):
        return 0.0, 1.0, n

    bounded = float(np.clip(ic, -0.999999, 0.999999))
    denominator = max(1.0 - (bounded * bounded), 1e-12)
    t_stat = bounded * np.sqrt(max(n - 2, 1) / denominator)
    p_value = float(max(0.0, min(1.0, erfc(abs(float(t_stat)) / sqrt(2.0)))))
    return bounded, p_value, n


def _build_base_signal_frame(candles: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
    open_ = candles["open"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)

    price_range = (high - low).replace(0.0, np.nan)

    signals = pd.DataFrame(index=candles.index)
    signals["RSI(14)"] = _rsi(close, length=14)
    signals["MACD_hist"] = _macd_hist(close)
    signals["BB_%B"] = _bb_percent_b(close, length=20, stdev_mult=2.0)
    signals["volume_zscore"] = _volume_zscore(volume, length=20)
    signals["body_range_ratio"] = (close - open_) / price_range
    signals["ATR_pct_rank"] = _atr_percentile_rank(high, low, close, atr_length=14, rank_window=100)
    signals["hour_of_day"] = ts.dt.hour.astype(float)
    signals["day_of_week"] = ts.dt.dayofweek.astype(float)

    return signals


def _build_liquidation_signal_frame(
    candles: pd.DataFrame,
    symbol: str,
    timeframe_minutes: int,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int], dict[str, Path]]:
    feeds, paths = _load_persisted_liquidation_feeds(symbol)
    notes: dict[str, str] = {}
    counts = {
        "funding_rows": int(len(feeds["funding"])),
        "oi_rows": int(len(feeds["oi"])),
        "lsratio_rows": int(len(feeds["lsratio"])),
        "liquidation_rows": int(len(feeds["liquidations"])),
    }

    ts = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
    signals = pd.DataFrame(index=candles.index)

    funding_aligned = _align_asof(ts, feeds["funding"], "funding_rate")
    if funding_aligned.notna().sum() >= 3:
        bars_30d = max(int(np.ceil((30 * 24 * 60) / max(timeframe_minutes, 1))), 30)
        min_periods = max(20, bars_30d // 3)
        f_mean = funding_aligned.rolling(bars_30d, min_periods=min_periods).mean()
        f_std = funding_aligned.rolling(bars_30d, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
        signals["funding_rate"] = (funding_aligned - f_mean) / f_std
    else:
        signals["funding_rate"] = np.nan
        notes["funding_rate"] = "missing/insufficient persisted funding data"

    oi_aligned = _align_asof(ts, feeds["oi"], "open_interest")
    if oi_aligned.notna().sum() >= 3:
        oi_delta_pct = oi_aligned.pct_change().replace([np.inf, -np.inf], np.nan)
        signals["oi_delta_pct"] = oi_delta_pct
    else:
        signals["oi_delta_pct"] = np.nan
        notes["oi_delta_pct"] = "missing/insufficient persisted OI data"

    ls_aligned = _align_asof(ts, feeds["lsratio"], "ls_ratio")
    if ls_aligned.notna().sum() >= 3:
        signals["ls_ratio"] = ls_aligned.replace([np.inf, -np.inf], np.nan)
    else:
        signals["ls_ratio"] = np.nan
        notes["ls_ratio"] = "missing/insufficient persisted long-short ratio data"

    liq_feed = feeds["liquidations"]
    if len(liq_feed) < MIN_LIQ_ROWS_FOR_IMBALANCE:
        signals["liq_imbalance"] = np.nan
        notes["liq_imbalance"] = (
            f"skipped (requires >= {MIN_LIQ_ROWS_FOR_IMBALANCE} liquidation rows, have {len(liq_feed)})"
        )
    else:
        liq = liq_feed.copy()
        liq["timestamp"] = pd.to_datetime(liq["timestamp"], utc=True, errors="coerce")
        liq["side"] = liq["side"].astype(str).str.lower()
        liq["size"] = pd.to_numeric(liq["size"], errors="coerce")
        liq = liq.dropna(subset=["timestamp", "size"]).set_index("timestamp").sort_index()

        liq["long_liq_size"] = np.where(liq["side"].str.startswith("s"), liq["size"], 0.0)
        liq["short_liq_size"] = np.where(liq["side"].str.startswith("b"), liq["size"], 0.0)

        rule = f"{max(int(timeframe_minutes), 1)}min"
        liq_bar = liq[["long_liq_size", "short_liq_size"]].resample(rule).sum().fillna(0.0)

        candle_index = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
        liq_bar = liq_bar.reindex(candle_index, fill_value=0.0)

        bars_4h = max(int(np.ceil((4 * 60) / max(timeframe_minutes, 1))), 1)
        long_roll = liq_bar["long_liq_size"].rolling(bars_4h, min_periods=1).sum()
        short_roll = liq_bar["short_liq_size"].rolling(bars_4h, min_periods=1).sum()
        signals["liq_imbalance"] = (long_roll - short_roll).reset_index(drop=True)

    signals["funding_x_oi"] = signals["funding_rate"] * signals["oi_delta_pct"]
    if signals["funding_x_oi"].notna().sum() < 3:
        notes["funding_x_oi"] = "missing funding_rate and/or oi_delta_pct inputs"

    return signals, notes, counts, paths


def _scan_signals(
    candles: pd.DataFrame,
    timeframe: str,
    symbol: str,
) -> tuple[pd.DataFrame, dict[int, pd.Series], dict[int, int], list[str], dict[str, int], dict[str, Path]]:
    timeframe_minutes = _parse_timeframe_to_minutes(timeframe)
    bars_per_horizon = {
        h: max(int(np.ceil((h * 60) / timeframe_minutes)), 1)
        for h in HORIZON_HOURS
    }

    close = candles["close"].astype(float)
    future_returns = {
        h: (close.shift(-bars_per_horizon[h]) / close) - 1.0
        for h in HORIZON_HOURS
    }

    base_signals = _build_base_signal_frame(candles)
    liquidation_signals, signal_notes, feed_counts, feed_paths = _build_liquidation_signal_frame(
        candles=candles,
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
    )
    signals = pd.concat([base_signals, liquidation_signals], axis=1)

    rows: list[dict[str, Any]] = []
    skip_messages: list[str] = []

    for signal_name in signals.columns:
        series = signals[signal_name]
        valid_count = int(series.notna().sum())
        if valid_count < 3:
            note = signal_notes.get(signal_name, f"insufficient valid samples ({valid_count})")
            skip_messages.append(f"{signal_name}: {note}")
            rows.append(
                {
                    "signal": signal_name,
                    "ic_1h": np.nan,
                    "ic_4h": np.nan,
                    "ic_8h": np.nan,
                    "p_value": np.nan,
                    "any_ic_gt_0_05": False,
                }
            )
            continue

        ic_map: dict[int, float] = {}
        p_map: dict[int, float] = {}

        for h in HORIZON_HOURS:
            ic, p_value, _ = _spearman_ic_pvalue(series, future_returns[h])
            ic_map[h] = ic
            p_map[h] = p_value

        best_h = max(HORIZON_HOURS, key=lambda h: abs(ic_map[h]))
        rows.append(
            {
                "signal": signal_name,
                "ic_1h": ic_map[1],
                "ic_4h": ic_map[4],
                "ic_8h": ic_map[8],
                "p_value": p_map[best_h],
                "any_ic_gt_0_05": any(ic_map[h] > 0.05 for h in HORIZON_HOURS),
            }
        )

    results = pd.DataFrame(rows)
    return results, future_returns, bars_per_horizon, skip_messages, feed_counts, feed_paths


def _ema_direction_test(candles: pd.DataFrame, future_returns_4h: pd.Series) -> dict[str, Any]:
    close = candles["close"].astype(float)
    ema50 = close.ewm(span=50, adjust=False).mean()
    above_ema = close > ema50
    next_up = future_returns_4h > 0.0

    valid = above_ema.notna() & next_up.notna()
    if int(valid.sum()) == 0:
        return {
            "samples": 0,
            "hit_rate": 0.0,
            "better_than_50pct": False,
            "edge_vs_50pct": -0.5,
        }

    hits = (above_ema[valid] == next_up[valid]).astype(float)
    hit_rate = float(hits.mean())
    return {
        "samples": int(valid.sum()),
        "hit_rate": hit_rate,
        "better_than_50pct": bool(hit_rate > 0.5),
        "edge_vs_50pct": hit_rate - 0.5,
    }


def _format_results_table(results: pd.DataFrame) -> str:
    display = results.copy()
    display["any_ic_gt_0_05"] = display["any_ic_gt_0_05"].map(lambda x: "YES" if bool(x) else "NO")

    def _fmt_metric(value: Any) -> str:
        return "n/a" if pd.isna(value) else f"{float(value):.4f}"

    def _fmt_p(value: Any) -> str:
        return "n/a" if pd.isna(value) else f"{float(value):.4g}"

    return display.to_string(
        index=False,
        formatters={
            "ic_1h": _fmt_metric,
            "ic_4h": _fmt_metric,
            "ic_8h": _fmt_metric,
            "p_value": _fmt_p,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone candle signal scanner (no trading logic).")
    parser.add_argument("--symbol", default=config.SYMBOL, help="Symbol used for cache filename (default: config.SYMBOL)")
    parser.add_argument("--timeframe", default=config.TIMEFRAME, help="Timeframe used for cache filename (default: config.TIMEFRAME)")
    parser.add_argument("--cache-file", default=None, help="Optional explicit cache file path")
    parser.add_argument(
        "--refresh-liquidation-feeds",
        action="store_true",
        help="Refresh persisted funding/OI/long-short caches before scanning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.refresh_liquidation_feeds:
        from data.bybit_client import BybitClient
        from data.liquidation_feed import LiquidationFeed

        client = BybitClient(
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            testnet=config.BYBIT_TESTNET,
            data_testnet=config.BYBIT_DATA_TESTNET,
            category=config.CATEGORY,
        )
        feed = LiquidationFeed(
            symbol=args.symbol,
            testnet=config.BYBIT_TESTNET,
            client=client,
            data_testnet=config.BYBIT_DATA_TESTNET,
        )
        feed.update_all(include_recent_liquidations_snapshot=True)

    candles, cache_path = _load_cached_candles(
        symbol=args.symbol,
        timeframe=args.timeframe,
        cache_file=args.cache_file,
    )

    results, future_returns, bars_per_horizon, skip_messages, feed_counts, feed_paths = _scan_signals(
        candles,
        timeframe=args.timeframe,
        symbol=args.symbol,
    )

    print("=== Signal Scanner (Statistical Only) ===")
    print(f"cache_file: {cache_path}")
    print(f"rows_loaded: {len(candles)}")
    print(
        "horizon_bars: "
        + ", ".join(f"{h}h={bars_per_horizon[h]} bars" for h in HORIZON_HOURS)
    )
    print("feed_cache_paths:")
    print(f"  funding: {feed_paths['funding']}")
    print(f"  oi: {feed_paths['oi']}")
    print(f"  lsratio: {feed_paths['lsratio']}")
    print(f"  liquidations: {feed_paths['liquidations']}")
    print("feed_rows:")
    print(f"  funding={feed_counts['funding_rows']} oi={feed_counts['oi_rows']} lsratio={feed_counts['lsratio_rows']} liquidations={feed_counts['liquidation_rows']}")

    print("\nColumns: signal, IC@1H, IC@4H, IC@8H, p-value (best-|IC| horizon), any IC > 0.05")
    print(_format_results_table(results))

    if skip_messages:
        print("\nSkipped/partial signals:")
        for item in skip_messages:
            print(f"- {item}")

    ema_test = _ema_direction_test(candles, future_returns_4h=future_returns[4])
    print("\n=== EMA(50) Direction Test (next-4H return direction) ===")
    print(f"samples: {ema_test['samples']}")
    print(f"hit_rate: {ema_test['hit_rate']:.4%}")
    print(f"edge_vs_50pct: {ema_test['edge_vs_50pct']:.4%}")
    print(f"better_than_50pct: {'YES' if ema_test['better_than_50pct'] else 'NO'}")


if __name__ == "__main__":
    main()

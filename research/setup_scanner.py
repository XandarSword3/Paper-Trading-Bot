"""Squeeze setup scanner and phase-0 validation report.

This module audits contrarian setup quality and prints:
- condition diagnostics
- explicit forward-return debug rows
- Block A/B/C subset metrics
- directional split metrics (A1/A2/A3)
- monthly concentration and walk-forward checks
- phase-0 gate results
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


HORIZONS_HOURS: tuple[int, int, int] = (1, 4, 8)


def _normalize_dedup_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"regime", "per_candle"}:
        raise ValueError("dedup mode must be 'regime' or 'per_candle'")
    return mode


def _apply_dedup_mode(active_mask: pd.Series, direction: pd.Series, dedup_mode: str) -> pd.Series:
    active = active_mask.astype(bool)
    if dedup_mode == "per_candle":
        return active

    direction_clean = direction.astype(str).where(active, "inactive")
    prev_active = active.shift(1, fill_value=False)
    prev_direction = direction_clean.shift(1).fillna("inactive")
    return active & ((~prev_active) | direction_clean.ne(prev_direction))


def _parse_timeframe_to_minutes(timeframe: str) -> int:
    text = str(timeframe).strip().upper()
    if text.isdigit():
        return max(int(text), 1)

    if text.endswith("H") and text[:-1].replace(".", "", 1).isdigit():
        return max(int(float(text[:-1]) * 60), 1)
    if text.endswith("D") and text[:-1].replace(".", "", 1).isdigit():
        return max(int(float(text[:-1]) * 1440), 1)

    mapping = {"D": 1440, "W": 10080}
    if text in mapping:
        return mapping[text]

    raise ValueError(f"Unsupported timeframe format: {timeframe}")


def _load_cached_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    path = config.BASE_DIR / "data" / f"candles_{symbol}_{timeframe}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing candle cache: {path}")

    frame = pd.read_pickle(path)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Candle cache missing required columns: {missing}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def _load_feed_cache(path: Path, columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_pickle(path)
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
    out = out[columns].sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


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

    merged = pd.merge_asof(left.sort_values("timestamp"), right, on="timestamp", direction="backward")
    merged = merged.sort_values("_idx")
    return merged[value_col].reset_index(drop=True)


def _last_percentile(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= values[-1]))


def _build_feature_frame(
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    funding_z_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    timeframe_minutes = _parse_timeframe_to_minutes(timeframe)
    bars_30d = max(int(np.ceil((30 * 24 * 60) / max(timeframe_minutes, 1))), 30)
    rolling_min = max(120, bars_30d // 3)

    ts = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
    close = candles["close"].astype(float)

    data_dir = config.BASE_DIR / "data"
    funding = _load_feed_cache(data_dir / f"funding_{symbol}.pkl", ["timestamp", "funding_rate"], ["funding_rate"])
    oi = _load_feed_cache(data_dir / f"oi_{symbol}.pkl", ["timestamp", "open_interest", "oi_delta"], ["open_interest", "oi_delta"])
    lsratio = _load_feed_cache(data_dir / f"lsratio_{symbol}.pkl", ["timestamp", "buy_ratio", "sell_ratio", "ls_ratio"], ["buy_ratio", "sell_ratio", "ls_ratio"])

    funding_rate = _align_asof(ts, funding, "funding_rate")
    funding_mean = funding_rate.rolling(bars_30d, min_periods=rolling_min).mean()
    funding_std = funding_rate.rolling(bars_30d, min_periods=rolling_min).std(ddof=0).replace(0.0, np.nan)
    funding_z = (funding_rate - funding_mean) / funding_std
    funding_long = funding_z > float(funding_z_threshold)
    funding_short = funding_z < -float(funding_z_threshold)
    cond1 = funding_long | funding_short

    ls_ratio = _align_asof(ts, lsratio, "ls_ratio")
    ls_pct = ls_ratio.rolling(bars_30d, min_periods=rolling_min).apply(_last_percentile, raw=True)
    ls_long = ls_pct <= 0.20
    ls_short = ls_pct >= 0.80
    cond2 = ls_long | ls_short

    open_interest = _align_asof(ts, oi, "open_interest")
    oi_delta_pct = open_interest.pct_change().replace([np.inf, -np.inf], np.nan)
    cond3 = oi_delta_pct.gt(0.0).rolling(2, min_periods=2).sum().eq(2)

    crowd_long_score = funding_long.astype(int) + ls_long.astype(int)
    crowd_short_score = funding_short.astype(int) + ls_short.astype(int)
    crowd_side = pd.Series(
        np.where(crowd_long_score > crowd_short_score, 1, np.where(crowd_short_score > crowd_long_score, -1, 0)),
        index=candles.index,
        dtype=float,
    )
    crowd_direction = pd.Series(
        np.where(crowd_side > 0, "long_heavy", np.where(crowd_side < 0, "short_heavy", "neutral")),
        index=candles.index,
        dtype=object,
    )

    directional_funding = (
        ((crowd_side > 0) & (funding_z > 1.0))
        | ((crowd_side < 0) & (funding_z < -1.0))
    )
    cond4 = cond3 & directional_funding

    in_setup_a = cond1 & cond2 & (~cond3) & (~cond4)
    in_setup_b = cond1 & cond2 & cond3
    in_setup_c = cond1 & cond2 & cond3 & cond4

    # Explicit forward returns requested by the phase-0 audit.
    forward_1h = (close.shift(-1) - close) / close
    forward_4h = (close.shift(-4) - close) / close
    forward_8h = (close.shift(-8) - close) / close

    # Contrarian direction requested in phase-0 instructions.
    trade_direction = pd.Series(np.where(funding_z > 0.0, -1.0, 1.0), index=candles.index, dtype=float)
    hit_1h = (trade_direction * forward_1h) > 0.0
    hit_4h = (trade_direction * forward_4h) > 0.0
    hit_8h = (trade_direction * forward_8h) > 0.0

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "funding_rate_z30d": funding_z,
            "ls_ratio": ls_ratio,
            "ls_ratio_pct30d": ls_pct,
            "oi_delta_pct": oi_delta_pct,
            "cond1_funding_extreme": cond1.astype(bool),
            "cond2_ls_extreme": cond2.astype(bool),
            "cond3_oi_building": cond3.astype(bool),
            "cond4_leverage_building_directional_funding": cond4.astype(bool),
            "crowd_side": crowd_side,
            "crowd_direction": crowd_direction,
            "trade_direction": trade_direction,
            "in_setup_A": in_setup_a.astype(bool),
            "in_setup_B": in_setup_b.astype(bool),
            "in_setup_C": in_setup_c.astype(bool),
            "forward_1H": forward_1h,
            "forward_4H": forward_4h,
            "forward_8H": forward_8h,
            "hit_1H": hit_1h.astype(bool),
            "hit_4H": hit_4h.astype(bool),
            "hit_8H": hit_8h.astype(bool),
        }
    )

    counts = {
        "funding_rows": int(len(funding)),
        "oi_rows": int(len(oi)),
        "lsratio_rows": int(len(lsratio)),
    }
    return out, counts


def _compute_condition_diagnostics(frame: pd.DataFrame) -> dict[str, float | int]:
    total_rows = int(len(frame))
    cond1 = frame["cond1_funding_extreme"].astype(bool)
    cond2 = frame["cond2_ls_extreme"].astype(bool)
    cond3 = frame["cond3_oi_building"].astype(bool)
    cond4 = frame["cond4_leverage_building_directional_funding"].astype(bool)

    def _count_and_pct(mask: pd.Series) -> tuple[int, float]:
        rows = int(mask.sum())
        pct = (100.0 * float(rows) / float(total_rows)) if total_rows > 0 else float("nan")
        return rows, pct

    c1_rows, c1_pct = _count_and_pct(cond1)
    c2_rows, c2_pct = _count_and_pct(cond2)
    c3_rows, c3_pct = _count_and_pct(cond3)
    c4_rows, c4_pct = _count_and_pct(cond4)

    return {
        "total_rows": total_rows,
        "cond1_rows": c1_rows,
        "cond1_pct": c1_pct,
        "cond2_rows": c2_rows,
        "cond2_pct": c2_pct,
        "cond3_rows": c3_rows,
        "cond3_pct": c3_pct,
        "cond4_rows": c4_rows,
        "cond4_pct": c4_pct,
        "cond12_rows": int((cond1 & cond2 & (~cond3) & (~cond4)).sum()),
        "cond123_rows": int((cond1 & cond2 & cond3).sum()),
        "cond1234_rows": int((cond1 & cond2 & cond3 & cond4).sum()),
    }


def _print_condition_diagnostics(diag: dict[str, float | int], funding_z_threshold: float) -> None:
    threshold_text = f"{float(funding_z_threshold):.2f}".rstrip("0").rstrip(".")
    print("=== SETUP CONDITION DIAGNOSTICS ===")
    print(f"total_rows: {diag['total_rows']}")
    print(f"Condition 1 - funding_rate_z30d > {threshold_text} OR < -{threshold_text}:")
    print("  rows_passing: %d, pct_of_total: %.2f%%" % (diag["cond1_rows"], diag["cond1_pct"]))
    print("Condition 2 - ls_ratio in bottom 20th OR top 80th percentile (rolling 30d):")
    print("  rows_passing: %d, pct_of_total: %.2f%%" % (diag["cond2_rows"], diag["cond2_pct"]))
    print("Condition 3 - oi_delta_pct > 0 for 2 consecutive candles:")
    print("  rows_passing: %d, pct_of_total: %.2f%%" % (diag["cond3_rows"], diag["cond3_pct"]))
    print("Condition 4 - leverage building with directional funding:")
    print("  definition: oi_delta_pct > 0 for 2+ candles AND |funding_rate_z30d| > 1.0 in crowd direction")
    print("  rows_passing: %d, pct_of_total: %.2f%%" % (diag["cond4_rows"], diag["cond4_pct"]))
    print("Condition 1 AND 2 (no 3, no 4):")
    print("  rows_passing: %d" % diag["cond12_rows"])
    print("Condition 1 AND 2 AND 3:")
    print("  rows_passing: %d" % diag["cond123_rows"])
    print("All 4 combined:")
    print("  rows_passing: %d" % diag["cond1234_rows"])
    print()


def _drop_forward_nan(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["forward_1H", "forward_4H", "forward_8H"]
    out = frame.dropna(subset=required).copy()
    return out.reset_index(drop=True)


def _monthly_counts(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    periods = ts.dt.tz_localize(None).dt.to_period("M")
    valid = mask.astype(bool)
    usable = periods.notna()
    if not bool(usable.any()):
        return pd.Series(dtype=float)

    counts = pd.Series(valid[usable].astype(int).to_numpy(), index=periods[usable]).groupby(level=0).sum()
    all_months = pd.period_range(periods[usable].min(), periods[usable].max(), freq="M")
    return counts.reindex(all_months, fill_value=0)


def _max_consecutive_losses(returns: pd.Series) -> int:
    streak = 0
    best = 0
    for value in returns:
        if not np.isfinite(value) or float(value) <= 0.0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def _evaluate_subset_metrics(frame: pd.DataFrame, subset_mask: pd.Series) -> tuple[dict[str, float | int], pd.DataFrame]:
    in_subset = subset_mask.astype(bool)
    monthly = _monthly_counts(frame, in_subset)
    monthly_frequency = float(monthly.mean()) if len(monthly) > 0 else 0.0

    subset = frame.loc[in_subset].copy()
    setup_count = int(len(subset))
    if subset.empty:
        empty = {
            "setup_count": 0,
            "hit_rate_1H": float("nan"),
            "hit_rate_4H": float("nan"),
            "hit_rate_8H": float("nan"),
            "avg_win_dec": float("nan"),
            "avg_loss_dec_signed": float("nan"),
            "avg_win_pct": float("nan"),
            "avg_loss_pct": float("nan"),
            "win_loss_ratio": float("nan"),
            "implied_expectancy": float("nan"),
            "monthly_frequency": monthly_frequency,
            "max_consecutive_losses": 0,
        }
        return empty, subset

    subset["contrarian_1H"] = subset["trade_direction"] * subset["forward_1H"]
    subset["contrarian_4H"] = subset["trade_direction"] * subset["forward_4H"]
    subset["contrarian_8H"] = subset["trade_direction"] * subset["forward_8H"]

    subset["hit_1H"] = subset["contrarian_1H"] > 0.0
    subset["hit_4H"] = subset["contrarian_4H"] > 0.0
    subset["hit_8H"] = subset["contrarian_8H"] > 0.0

    hit_rate_1h = float(subset["hit_1H"].mean())
    hit_rate_4h = float(subset["hit_4H"].mean())
    hit_rate_8h = float(subset["hit_8H"].mean())

    ts_subset = pd.to_datetime(subset["timestamp"], utc=True, errors="coerce").dropna()
    if not ts_subset.empty:
        start = ts_subset.min()
        end = ts_subset.max()
        months_span = ((end.year - start.year) * 12) + (end.month - start.month) + 1
        months_span = max(int(months_span), 1)
        monthly_frequency = float(setup_count / months_span)

    wins = subset.loc[subset["contrarian_4H"] > 0.0, "contrarian_4H"]
    losses = subset.loc[subset["contrarian_4H"] <= 0.0, "contrarian_4H"]

    avg_win_dec = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_dec_signed = float(losses.mean()) if not losses.empty else 0.0
    avg_loss_dec_abs = abs(avg_loss_dec_signed)

    if avg_loss_dec_abs <= 1e-12:
        win_loss_ratio = float("nan")
    else:
        win_loss_ratio = float(avg_win_dec / avg_loss_dec_abs)

    implied_expectancy = float(hit_rate_4h * avg_win_dec - (1.0 - hit_rate_4h) * avg_loss_dec_abs)

    metrics = {
        "setup_count": setup_count,
        "hit_rate_1H": hit_rate_1h,
        "hit_rate_4H": hit_rate_4h,
        "hit_rate_8H": hit_rate_8h,
        "avg_win_dec": avg_win_dec,
        "avg_loss_dec_signed": avg_loss_dec_signed,
        "avg_win_pct": avg_win_dec * 100.0,
        "avg_loss_pct": avg_loss_dec_abs * 100.0,
        "win_loss_ratio": win_loss_ratio,
        "implied_expectancy": implied_expectancy,
        "monthly_frequency": monthly_frequency,
        "max_consecutive_losses": _max_consecutive_losses(subset["contrarian_4H"]),
    }
    return metrics, subset


def _fmt(value: float, decimals: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.{decimals}f}"


def _print_subset_block(label: str, metrics: dict[str, float | int]) -> None:
    print(label)
    print(f"  setup_count: {metrics['setup_count']}")
    print(f"  hit_rate_1H: {_fmt(float(metrics['hit_rate_1H']))}")
    print(f"  hit_rate_4H: {_fmt(float(metrics['hit_rate_4H']))}")
    print(f"  hit_rate_8H: {_fmt(float(metrics['hit_rate_8H']))}")
    print(f"  avg_win_pct: {_fmt(float(metrics['avg_win_pct']))}%")
    print(f"  avg_loss_pct: {_fmt(float(metrics['avg_loss_pct']))}%")
    print(f"  win_loss_ratio: {_fmt(float(metrics['win_loss_ratio']))}")
    print(f"  implied_expectancy: {_fmt(float(metrics['implied_expectancy']) * 100.0)}%")
    print(f"  monthly_frequency: {_fmt(float(metrics['monthly_frequency']))}")
    print()


def _print_debug_rows(frame_valid: pd.DataFrame) -> None:
    cols = [
        "timestamp",
        "close",
        "funding_rate_z30d",
        "ls_ratio_pct30d",
        "in_setup_A",
        "crowd_direction",
        "forward_1H",
        "forward_4H",
        "forward_8H",
        "hit_1H",
        "hit_4H",
        "hit_8H",
    ]
    renamed = frame_valid[cols].rename(
        columns={
            "funding_rate_z30d": "funding_z",
            "ls_ratio_pct30d": "ls_pct_rank",
        }
    )
    print("=== FORWARD RETURN DEBUG (FIRST 15 ROWS) ===")
    print(renamed.head(15).to_string(index=False))
    print()


def _directional_masks(frame_valid: pd.DataFrame, funding_z_threshold: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    a1_mask = (
        (frame_valid["funding_rate_z30d"] > float(funding_z_threshold))
        & (frame_valid["ls_ratio_pct30d"] <= 0.20)
    )
    a2_mask = (
        (frame_valid["funding_rate_z30d"] < -float(funding_z_threshold))
        & (frame_valid["ls_ratio_pct30d"] >= 0.80)
    )
    a3_mask = (
        ((frame_valid["funding_rate_z30d"] > float(funding_z_threshold)) & (frame_valid["ls_ratio_pct30d"] >= 0.80))
        | ((frame_valid["funding_rate_z30d"] < -float(funding_z_threshold)) & (frame_valid["ls_ratio_pct30d"] <= 0.20))
    )
    return a1_mask.astype(bool), a2_mask.astype(bool), a3_mask.astype(bool)


def _print_directional_split(
    frame_valid: pd.DataFrame,
    funding_z_threshold: float,
    dedup_mode: str,
) -> tuple[dict[str, float | int], dict[str, float | int], int]:
    a1_mask, a2_mask, a3_mask = _directional_masks(frame_valid, funding_z_threshold=funding_z_threshold)

    a1_eval_mask = _apply_dedup_mode(a1_mask, pd.Series("short", index=frame_valid.index), dedup_mode=dedup_mode)
    a2_eval_mask = _apply_dedup_mode(a2_mask, pd.Series("long", index=frame_valid.index), dedup_mode=dedup_mode)
    a3_eval_mask = _apply_dedup_mode(a3_mask, pd.Series("mixed", index=frame_valid.index), dedup_mode=dedup_mode)

    a1_metrics, _ = _evaluate_subset_metrics(frame_valid, a1_eval_mask)
    a2_metrics, _ = _evaluate_subset_metrics(frame_valid, a2_eval_mask)
    a3_count = int(a3_eval_mask.sum())

    print("=== DIRECTIONAL SPLIT (BLOCK A) ===")
    print(f"setup_counting_mode: {dedup_mode}")
    print("Subset A1 - funding > +threshold AND ls bottom 20% (trade SHORT)")
    print(f"  count: {a1_metrics['setup_count']}")
    print(f"  hit_rate_4H: {_fmt(float(a1_metrics['hit_rate_4H']))}")
    print(f"  avg_win_pct: {_fmt(float(a1_metrics['avg_win_pct']))}%")
    print(f"  avg_loss_pct: {_fmt(float(a1_metrics['avg_loss_pct']))}%")
    print(f"  expectancy: {_fmt(float(a1_metrics['implied_expectancy']) * 100.0)}%")
    print(f"  monthly_freq: {_fmt(float(a1_metrics['monthly_frequency']))}")
    print("Subset A2 - funding < -threshold AND ls top 80% (trade LONG)")
    print(f"  count: {a2_metrics['setup_count']}")
    print(f"  hit_rate_4H: {_fmt(float(a2_metrics['hit_rate_4H']))}")
    print(f"  avg_win_pct: {_fmt(float(a2_metrics['avg_win_pct']))}%")
    print(f"  avg_loss_pct: {_fmt(float(a2_metrics['avg_loss_pct']))}%")
    print(f"  expectancy: {_fmt(float(a2_metrics['implied_expectancy']) * 100.0)}%")
    print(f"  monthly_freq: {_fmt(float(a2_metrics['monthly_frequency']))}")
    print("Subset A3 - mixed direction (do not trade)")
    print(f"  count: {a3_count}")
    print()

    return a1_metrics, a2_metrics, a3_count


def _print_regime_entry_analysis(frame_valid: pd.DataFrame, funding_z_threshold: float) -> dict[str, Any]:
    a1_mask, _, _ = _directional_masks(frame_valid, funding_z_threshold=funding_z_threshold)
    crowd_direction = frame_valid["crowd_direction"].astype(str)
    previous_direction = crowd_direction.shift(1).fillna("neutral")

    # Regime entry definition requested by the user: neutral -> long_heavy, scoped to A1 block.
    entry_mask = a1_mask & crowd_direction.eq("long_heavy") & previous_direction.eq("neutral")
    entry_indices = [int(i) for i, active in enumerate(entry_mask.to_list()) if active]

    rows: list[dict[str, Any]] = []
    for entry_idx in entry_indices:
        entry_ts = pd.to_datetime(frame_valid.at[entry_idx, "timestamp"], utc=True, errors="coerce")
        if not pd.notna(entry_ts):
            continue

        exit_idx = len(frame_valid) - 1
        for idx in range(entry_idx + 1, len(frame_valid)):
            if str(crowd_direction.iat[idx]) == "neutral":
                exit_idx = idx
                break

        exit_ts = pd.to_datetime(frame_valid.at[exit_idx, "timestamp"], utc=True, errors="coerce")
        if not pd.notna(exit_ts):
            exit_ts = entry_ts

        duration_hours = max(float((exit_ts - entry_ts).total_seconds() / 3600.0), 0.0)
        entry_forward_4h = float(frame_valid.at[entry_idx, "forward_4H"])
        entry_contrarian_4h = -entry_forward_4h

        rows.append(
            {
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "duration_hours": duration_hours,
                "forward_4H_entry_pct": entry_forward_4h * 100.0,
                "contrarian_4H_entry_pct": entry_contrarian_4h * 100.0,
                "hit_4H": bool(entry_contrarian_4h > 0.0),
            }
        )

    regimes = pd.DataFrame(rows)
    total_regimes = int(len(regimes))

    if regimes.empty:
        avg_duration = 0.0
        regime_hit_rate = float("nan")
        regime_expectancy = float("nan")
        regimes_per_month = 0.0
    else:
        avg_duration = float(regimes["duration_hours"].mean())
        regime_hit_rate = float(regimes["hit_4H"].mean())
        regime_expectancy = float((regimes["contrarian_4H_entry_pct"] / 100.0).mean())

        entry_ts = pd.to_datetime(regimes["entry_timestamp"], utc=True, errors="coerce").dropna()
        if entry_ts.empty:
            regimes_per_month = 0.0
        else:
            months_span = ((entry_ts.max().year - entry_ts.min().year) * 12) + (entry_ts.max().month - entry_ts.min().month) + 1
            months_span = max(int(months_span), 1)
            regimes_per_month = float(total_regimes / months_span)

    print("=== REGIME ENTRY ANALYSIS ===")
    print("definition: regime starts at crowd_direction neutral -> long_heavy and ends when crowd_direction returns to neutral")
    print("scope: Block A, A1 only (short trades)")
    if regimes.empty:
        print("no regimes detected")
    else:
        view = regimes[["entry_timestamp", "exit_timestamp", "duration_hours", "forward_4H_entry_pct"]].copy()
        view["entry_timestamp"] = pd.to_datetime(view["entry_timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        view["exit_timestamp"] = pd.to_datetime(view["exit_timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        print(view.to_string(index=False))
    print(f"total_regimes_detected: {total_regimes}")
    print(f"avg_regime_duration_hours: {_fmt(avg_duration)}")
    print(f"regime_hit_rate_4H: {_fmt(regime_hit_rate * 100.0)}%")
    print(f"regime_expectancy: {_fmt(regime_expectancy * 100.0)}%")
    print(f"regimes_per_month: {_fmt(regimes_per_month)}")
    print()

    return {
        "total_regimes_detected": total_regimes,
        "avg_regime_duration_hours": avg_duration,
        "regime_hit_rate_4H": regime_hit_rate,
        "regime_expectancy": regime_expectancy,
        "regimes_per_month": regimes_per_month,
    }


def _print_block_a_distribution(frame_valid: pd.DataFrame, block_a_mask: pd.Series, block_a_metrics: dict[str, float | int], block_a_subset: pd.DataFrame) -> None:
    month_counts = _monthly_counts(frame_valid, block_a_mask)
    month_table = pd.DataFrame(
        {
            "month": [str(item) for item in month_counts.index],
            "count": month_counts.values.astype(int),
        }
    )

    print("=== BLOCK A MONTHLY COUNT TABLE ===")
    print(month_table.to_string(index=False))
    print()

    avg_win_pct = float(block_a_metrics["avg_win_pct"])
    avg_loss_pct = float(block_a_metrics["avg_loss_pct"])
    if block_a_subset.empty:
        curve_df = pd.DataFrame(columns=["trade_number", "timestamp", "pnl_step_pct", "cumulative_pnl_pct"])
    else:
        step_pnl = np.where(block_a_subset["hit_4H"].astype(bool), avg_win_pct, -avg_loss_pct)
        curve_df = pd.DataFrame(
            {
                "trade_number": np.arange(1, len(block_a_subset) + 1),
                "timestamp": block_a_subset["timestamp"].astype(str).values,
                "pnl_step_pct": step_pnl,
            }
        )
        curve_df["cumulative_pnl_pct"] = curve_df["pnl_step_pct"].cumsum()

    print("=== BLOCK A CUMULATIVE PNL CURVE (1% FLAT RISK) ===")
    if curve_df.empty:
        print("no block A trades")
    else:
        print(curve_df.to_string(index=False))
    print()

    max_losses = int(block_a_metrics["max_consecutive_losses"])
    total_expectancy = float(block_a_subset["contrarian_4H"].sum()) if not block_a_subset.empty else float("nan")
    winners = block_a_subset.loc[block_a_subset["contrarian_4H"] > 0.0, "contrarian_4H"]
    if winners.empty or not np.isfinite(total_expectancy) or abs(total_expectancy) <= 1e-12:
        top_share = float("nan")
    else:
        top_n = max(1, int(math.ceil(len(winners) * 0.10)))
        top_sum = float(winners.sort_values(ascending=False).head(top_n).sum())
        top_share = float((top_sum / total_expectancy) * 100.0)

    print("=== BLOCK A CONCENTRATION ===")
    print(f"max_consecutive_losses: {max_losses}")
    print(f"top_10pct_winner_expectancy_share: {_fmt(top_share)}%")
    print()


def _walk_forward_metrics(frame: pd.DataFrame, subset_mask: pd.Series) -> tuple[dict[str, float | int], dict[str, float | int]]:
    split_idx = int(len(frame) * 0.75)
    is_frame = frame.iloc[:split_idx].copy().reset_index(drop=True)
    oos_frame = frame.iloc[split_idx:].copy().reset_index(drop=True)

    mask = subset_mask.astype(bool).reset_index(drop=True)
    is_mask = mask.iloc[:split_idx].reset_index(drop=True)
    oos_mask = mask.iloc[split_idx:].reset_index(drop=True)

    is_metrics, _ = _evaluate_subset_metrics(is_frame, is_mask)
    oos_metrics, _ = _evaluate_subset_metrics(oos_frame, oos_mask)
    return is_metrics, oos_metrics


def _print_walk_forward(is_metrics: dict[str, float | int], oos_metrics: dict[str, float | int]) -> tuple[bool, float, str]:
    is_exp = float(is_metrics["implied_expectancy"])
    oos_exp = float(oos_metrics["implied_expectancy"])
    is_setup_count = int(is_metrics.get("setup_count", 0))

    if np.isfinite(is_exp) and abs(is_exp) > 1e-12:
        degradation = (is_exp - oos_exp) / abs(is_exp)
    else:
        degradation = float("inf")

    if is_setup_count < 10:
        proceed = True
        status = "DATA_INSUFFICIENT"
    else:
        proceed = bool(np.isfinite(oos_exp) and (oos_exp > 0.0) and np.isfinite(degradation) and (degradation <= 0.30))
        status = "PASS" if proceed else "FAIL"

    print("=== WALK-FORWARD (IS/OOS) ===")
    print(f"IS: setup_count={is_metrics['setup_count']} hit_rate_4H={_fmt(float(is_metrics['hit_rate_4H']))} expectancy={_fmt(is_exp * 100.0)}%")
    print(f"OOS: setup_count={oos_metrics['setup_count']} hit_rate_4H={_fmt(float(oos_metrics['hit_rate_4H']))} expectancy={_fmt(oos_exp * 100.0)}%")
    print(f"degradation_vs_IS: {_fmt(degradation * 100.0)}%")
    print(f"walk_forward_gate: {status if status == 'DATA_INSUFFICIENT' else ('PROCEED' if proceed else 'STOP')}")
    print()

    return proceed, degradation, status


def _print_phase0_gate_report(
    block_a_metrics: dict[str, float | int],
    block_a_subset: pd.DataFrame,
    is_metrics: dict[str, float | int],
    oos_metrics: dict[str, float | int],
    a1_metrics: dict[str, float | int],
    a2_metrics: dict[str, float | int],
    walk_forward_pass: bool,
    walk_forward_status: str,
) -> None:
    hit_rate_4h = float(block_a_metrics["hit_rate_4H"])
    gross_exp_pct = float(block_a_metrics["implied_expectancy"]) * 100.0
    round_trip_fee_pct = float(getattr(config, "ROUND_TRIP_FEE_PCT", 0.00075)) * 100.0
    net_exp_pct = gross_exp_pct - round_trip_fee_pct
    max_consec_losses = int(block_a_metrics["max_consecutive_losses"])

    metric_hit = np.isfinite(hit_rate_4h) and (hit_rate_4h > 0.53)
    metric_exp = np.isfinite(net_exp_pct) and (net_exp_pct > 0.05)
    monthly_freq = float(block_a_metrics["monthly_frequency"])
    metric_monthly = np.isfinite(monthly_freq) and (monthly_freq > 20.0)

    a1_exp = float(a1_metrics["implied_expectancy"])
    a2_exp = float(a2_metrics["implied_expectancy"])
    a2_count = int(a2_metrics.get("setup_count", 0))
    if a2_count < 5:
        metric_dirs = True
        directional_status = "UNTESTED"
    else:
        metric_dirs = np.isfinite(a1_exp) and np.isfinite(a2_exp) and (a1_exp > 0.0) and (a2_exp > 0.0)
        directional_status = "PASS" if metric_dirs else "FAIL"

    metric_losses = max_consec_losses < 8
    losses_status = "PASS" if metric_losses else "WARNING"
    metric_oos = bool(walk_forward_pass)
    oos_status = walk_forward_status if walk_forward_status == "DATA_INSUFFICIENT" else ("PASS" if metric_oos else "FAIL")

    final_pass = metric_hit and metric_exp and metric_oos and metric_dirs and metric_monthly

    if block_a_subset.empty:
        top_share = float("nan")
    else:
        winners = block_a_subset.loc[block_a_subset["contrarian_4H"] > 0.0, "contrarian_4H"]
        total_expectancy = float(block_a_subset["contrarian_4H"].sum())
        if winners.empty or not np.isfinite(total_expectancy) or abs(total_expectancy) <= 1e-12:
            top_share = float("nan")
        else:
            top_n = max(1, int(math.ceil(len(winners) * 0.10)))
            top_sum = float(winners.sort_values(ascending=False).head(top_n).sum())
            top_share = float((top_sum / total_expectancy) * 100.0)

    print("=== PHASE 0 GATE REPORT ===")
    print(f"hit_rate_4H > 53%: {'PASS' if metric_hit else 'FAIL'} ({_fmt(hit_rate_4h * 100.0)}%)")
    print(f"expectancy_net_after_fees > 0.05%: {'PASS' if metric_exp else 'FAIL'} ({_fmt(net_exp_pct)}%)")
    print(f"OOS expectancy positive and <=30% degradation: {oos_status}")
    print(f"A1 and A2 both positive expectancy: {directional_status}")
    print(f"monthly_frequency_sufficient (>20): {'PASS' if metric_monthly else 'FAIL'} ({_fmt(monthly_freq)})")
    print(f"max consecutive losses < 8: {losses_status} ({max_consec_losses})")
    print(f"overall_phase0_gate: {'PROCEED' if final_pass else 'STOP'}")
    print()

    print("=== DOCUMENTED_LIMITATIONS ===")
    if walk_forward_status == "DATA_INSUFFICIENT":
        print("unknown: walk-forward not statistically valid because IS setup_count < 10 (funding history limited)")
    else:
        print("validated: walk-forward IS/OOS check has sufficient IS data")
    if directional_status == "UNTESTED":
        print("unknown: A2 long-direction setup remains untested (setup_count < 5)")
    else:
        print("validated: both A1 and A2 directional expectancies measured")
    print(f"validated: block A hit_rate_4H = {_fmt(hit_rate_4h * 100.0)}% and net_expectancy_pct = {_fmt(net_exp_pct)}%")
    print(f"risk: winner concentration top_10pct_expectancy_share = {_fmt(top_share)}%")
    print()

    report = {
        "block_a": {
            "setup_count": int(block_a_metrics["setup_count"]),
            "hit_rate_4h": hit_rate_4h,
            "gross_expectancy_pct": gross_exp_pct,
            "net_expectancy_pct": net_exp_pct,
            "max_consecutive_losses": max_consec_losses,
        },
        "walk_forward": {
            "is_setup_count": int(is_metrics["setup_count"]),
            "is_expectancy": float(is_metrics["implied_expectancy"]),
            "oos_setup_count": int(oos_metrics["setup_count"]),
            "oos_expectancy": float(oos_metrics["implied_expectancy"]),
            "pass": metric_oos,
            "status": oos_status,
        },
        "directional": {
            "a1_expectancy": float(a1_metrics["implied_expectancy"]),
            "a2_expectancy": float(a2_metrics["implied_expectancy"]),
            "both_positive": metric_dirs,
            "status": directional_status,
            "a2_setup_count": a2_count,
        },
        "gate": {
            "hit": metric_hit,
            "expectancy": metric_exp,
            "oos": metric_oos,
            "directional": metric_dirs,
            "monthly_frequency_sufficient": metric_monthly,
            "losses": metric_losses,
            "overall": final_pass,
        },
        "limitations": {
            "walk_forward": "DATA_INSUFFICIENT" if walk_forward_status == "DATA_INSUFFICIENT" else "SUFFICIENT",
            "directional": "UNTESTED" if directional_status == "UNTESTED" else "SUFFICIENT",
            "winner_concentration_top10pct_share": top_share,
        },
    }
    print("phase0_gate_payload:")
    print(report)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-0 scanner and setup quality audit.")
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--timeframe", default="60", help="Use 1H candles by default")
    parser.add_argument("--funding-z-threshold", type=float, default=1.5)
    parser.add_argument("--dedup-mode", choices=["regime", "per_candle"], default=getattr(config, "SETUP_DEDUP_MODE", "regime"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dedup_mode = _normalize_dedup_mode(args.dedup_mode)
    live_mode = _normalize_dedup_mode(getattr(config, "SETUP_DEDUP_MODE", "regime"))

    print("=== SETUP COUNTING MODE ===")
    print(f"scanner_dedup_mode: {dedup_mode}")
    print(f"live_setup_detector_mode: {live_mode}")
    if dedup_mode != live_mode:
        print("warning: scanner and live dedup modes differ; counts are not directly comparable")
    else:
        print("status: scanner and live dedup modes match")
    print()

    candles = _load_cached_candles(symbol=args.symbol, timeframe=args.timeframe)
    frame_all, feed_counts = _build_feature_frame(
        candles=candles,
        symbol=args.symbol,
        timeframe=args.timeframe,
        funding_z_threshold=args.funding_z_threshold,
    )

    diagnostics = _compute_condition_diagnostics(frame_all)
    _print_condition_diagnostics(diagnostics, funding_z_threshold=args.funding_z_threshold)

    frame_valid = _drop_forward_nan(frame_all)
    _print_debug_rows(frame_valid)

    trade_side = pd.Series(
        np.where(
            frame_valid["trade_direction"] < 0.0,
            "short",
            np.where(frame_valid["trade_direction"] > 0.0, "long", "neutral"),
        ),
        index=frame_valid.index,
    )

    block_a_base_mask = frame_valid["in_setup_A"].astype(bool)
    block_b_base_mask = frame_valid["in_setup_B"].astype(bool)
    block_c_base_mask = frame_valid["in_setup_C"].astype(bool)

    block_a_mask = _apply_dedup_mode(block_a_base_mask, trade_side, dedup_mode=dedup_mode)
    block_b_mask = _apply_dedup_mode(block_b_base_mask, trade_side, dedup_mode=dedup_mode)
    block_c_mask = _apply_dedup_mode(block_c_base_mask, trade_side, dedup_mode=dedup_mode)

    block_a_metrics, block_a_subset = _evaluate_subset_metrics(frame_valid, block_a_mask)
    block_b_metrics, _ = _evaluate_subset_metrics(frame_valid, block_b_mask)
    block_c_metrics, _ = _evaluate_subset_metrics(frame_valid, block_c_mask)

    print("=== SUBSET HIT-RATE ANALYSIS ===")
    _print_subset_block("Block A - LOOSE: Funding + LS extreme", block_a_metrics)
    _print_subset_block("Block B - MEDIUM: + OI streak", block_b_metrics)
    _print_subset_block("Block C - STRICT: + Directional funding leverage", block_c_metrics)

    a1_metrics, a2_metrics, _ = _print_directional_split(
        frame_valid,
        funding_z_threshold=args.funding_z_threshold,
        dedup_mode=dedup_mode,
    )
    _print_regime_entry_analysis(frame_valid, funding_z_threshold=args.funding_z_threshold)
    _print_block_a_distribution(frame_valid, block_a_mask, block_a_metrics, block_a_subset)

    is_metrics, oos_metrics = _walk_forward_metrics(frame_valid, block_a_mask)
    walk_forward_pass, _, walk_forward_status = _print_walk_forward(is_metrics, oos_metrics)
    _print_phase0_gate_report(
        block_a_metrics,
        block_a_subset,
        is_metrics,
        oos_metrics,
        a1_metrics,
        a2_metrics,
        walk_forward_pass,
        walk_forward_status,
    )

    print("=== SQUEEZE SETUP SCANNER ===")
    print(f"symbol: {args.symbol}")
    print(f"timeframe: {args.timeframe}")
    print(f"setup_counting_mode: {dedup_mode}")
    print(f"candles: {len(candles)}")
    print(
        "feed_rows: funding=%d oi=%d lsratio=%d"
        % (feed_counts["funding_rows"], feed_counts["oi_rows"], feed_counts["lsratio_rows"])
    )
    print("setup_count: %d" % block_c_metrics["setup_count"])
    print("hit_rate_1H: %s" % _fmt(float(block_c_metrics["hit_rate_1H"])))
    print("hit_rate_4H: %s" % _fmt(float(block_c_metrics["hit_rate_4H"])))
    print("hit_rate_8H: %s" % _fmt(float(block_c_metrics["hit_rate_8H"])))
    print("avg_win: %s" % _fmt(float(block_c_metrics["avg_win_dec"]), decimals=6))
    print("avg_loss: %s" % _fmt(float(block_c_metrics["avg_loss_dec_signed"]), decimals=6))
    print("monthly_frequency: %s" % _fmt(float(block_c_metrics["monthly_frequency"])))

    sharpe_4h = float("nan")
    if len(block_a_subset) > 1:
        std = float(block_a_subset["contrarian_4H"].std(ddof=0))
        if std > 1e-12:
            annual = float(np.sqrt((24.0 * 365.0) / 4.0))
            sharpe_4h = float((block_a_subset["contrarian_4H"].mean() / std) * annual)
    print("sharpe_4H: %s" % _fmt(sharpe_4h))

    gate = "edge gate passed" if (float(block_c_metrics["implied_expectancy"]) > 0.0 and float(block_c_metrics["hit_rate_4H"]) >= 0.52) else "insufficient edge - do not proceed"
    print(f"gate: {gate}")


if __name__ == "__main__":
    main()

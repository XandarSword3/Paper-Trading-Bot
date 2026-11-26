"""Walk-forward validation for Block-A style contrarian setups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import config
from data.bybit_client import BybitClient
from data.feed_manager import FeedManager
from strategy.signal_engine import SignalEngine


@dataclass
class FoldResult:
    fold: int
    is_setup_count: int
    is_hit_rate_4h: float
    is_expectancy: float
    oos_setup_count: int
    oos_hit_rate_4h: float
    oos_expectancy: float


def _metrics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"setup_count": 0, "hit_rate_4h": float("nan"), "expectancy": float("nan")}

    hit = (df["contrarian_4h"] > 0.0).astype(float)
    wins = df.loc[df["contrarian_4h"] > 0.0, "contrarian_4h"]
    losses = df.loc[df["contrarian_4h"] <= 0.0, "contrarian_4h"]

    hit_rate = float(hit.mean()) if len(hit) else float("nan")
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
    expectancy = float(hit_rate * avg_win - (1.0 - hit_rate) * avg_loss) if np.isfinite(hit_rate) else float("nan")
    return {
        "setup_count": int(len(df)),
        "hit_rate_4h": hit_rate,
        "expectancy": expectancy,
    }


def run_walk_forward(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> list[FoldResult]:
    client = BybitClient(
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET,
        testnet=config.BYBIT_TESTNET,
        data_testnet=config.BYBIT_DATA_TESTNET,
        category=config.CATEGORY,
    )
    fm = FeedManager(symbol=symbol, bybit_client=client, timeframe=timeframe)

    candles = fm.get_latest_candles(timeframe, n=max(config.BACKTEST_CANDLES, 1000))
    funding = fm.get_funding_history(n=max(config.BACKTEST_CANDLES, 1000))
    oi = fm.get_oi_history(n=max(config.BACKTEST_CANDLES, 1000))
    ls = fm.get_lsratio_history(n=max(config.BACKTEST_CANDLES, 1000))
    aligned = fm.align_feeds_to_candles(candles, funding, oi, ls)

    if aligned.empty:
        return []

    engine = SignalEngine(fm)
    sig = engine.compute_history(aligned)
    frame = aligned.copy()
    frame["setup_active"] = sig["setup_active"]
    frame["crowd_direction"] = sig["crowd_direction"]

    close = pd.to_numeric(frame["close"], errors="coerce")
    forward_4h = (close.shift(-4) - close) / close
    trade_direction = np.where(frame["crowd_direction"].eq("long_heavy"), -1.0, np.where(frame["crowd_direction"].eq("short_heavy"), 1.0, 0.0))
    frame["contrarian_4h"] = trade_direction * forward_4h
    frame = frame.dropna(subset=["contrarian_4h"]).reset_index(drop=True)

    active = frame["setup_active"].astype(bool) & (trade_direction != 0.0)
    active_frame = frame.loc[active].copy().reset_index(drop=True)
    if active_frame.empty:
        return []

    folds = max(int(getattr(config, "WALK_FORWARD_FOLDS", 4)), 1)
    oos_ratio = float(getattr(config, "WALK_FORWARD_OOS_RATIO", 0.25))
    oos_len = max(int(len(active_frame) * oos_ratio), 1)
    stride = max((len(active_frame) - oos_len) // folds, 1)

    results: list[FoldResult] = []
    for idx in range(folds):
        oos_start = min(idx * stride, max(len(active_frame) - oos_len, 0))
        oos_end = oos_start + oos_len

        oos = active_frame.iloc[oos_start:oos_end].copy()
        is_df = pd.concat([active_frame.iloc[:oos_start], active_frame.iloc[oos_end:]], ignore_index=True)

        is_metrics = _metrics(is_df)
        oos_metrics = _metrics(oos)
        results.append(
            FoldResult(
                fold=idx + 1,
                is_setup_count=int(is_metrics["setup_count"]),
                is_hit_rate_4h=float(is_metrics["hit_rate_4h"]),
                is_expectancy=float(is_metrics["expectancy"]),
                oos_setup_count=int(oos_metrics["setup_count"]),
                oos_hit_rate_4h=float(oos_metrics["hit_rate_4h"]),
                oos_expectancy=float(oos_metrics["expectancy"]),
            )
        )

    return results


def main() -> None:
    rows = run_walk_forward()
    print("=== WALK FORWARD REPORT ===")
    print(f"generated_at: {datetime.now(timezone.utc).isoformat()}")
    if not rows:
        print("no results")
        return

    for row in rows:
        print(
            f"fold={row.fold} "
            f"IS(setups={row.is_setup_count}, hit4h={row.is_hit_rate_4h:.4f}, exp={row.is_expectancy*100:.4f}%) "
            f"OOS(setups={row.oos_setup_count}, hit4h={row.oos_hit_rate_4h:.4f}, exp={row.oos_expectancy*100:.4f}%)"
        )


if __name__ == "__main__":
    main()

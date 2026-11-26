"""Institutional Outlier Hygiene and Robust Estimators.

Detects:
- Stale quotes (timestamp difference > max_stale_seconds).
- Crossed/locked quotes (bid >= ask).
- Zero-size or non-positive quotes (bid_size <= 0, ask_size <= 0, bid <= 0, ask <= 0).
- WebSocket gaps (interval spans a network disconnection).

Computes robust statistics across valid, gap-free episodes:
- mean
- median
- 5% trimmed mean
- 1/99% winsorized mean
- share of total PnL from the top 1% of episodes
- leave-one-coin-out in both directions (drop best coin, drop worst coin)
- drop-top-3 coins
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("omnipulse.outlier_hygiene")


def is_valid_quote(
    bid: float,
    ask: float,
    bid_size: float = 1.0,
    ask_size: float = 1.0,
    quote_ts: Optional[float] = None,
    target_ts: Optional[float] = None,
    max_stale_seconds: float = 5.0,
) -> Tuple[bool, str]:
    """Validate a quote against crossed, zero-size, and stale quote filters.

    Returns (is_valid, reason_str).
    """
    if bid <= 0.0 or ask <= 0.0:
        return False, "NON_POSITIVE_PRICE"

    if bid >= ask:
        return False, f"CROSSED_OR_LOCKED_QUOTE(bid={bid}, ask={ask})"

    if bid_size <= 0.0 or ask_size <= 0.0:
        return False, f"ZERO_SIZE_QUOTE(bid_sz={bid_size}, ask_sz={ask_size})"

    if quote_ts is not None and target_ts is not None:
        diff = abs(quote_ts - target_ts)
        if diff > max_stale_seconds:
            return False, f"STALE_QUOTE(lag={diff:.2f}s > {max_stale_seconds}s)"

    return True, "VALID"


def compute_outlier_hygiene_metrics(
    df: pd.DataFrame,
    ret_col: str = "5m",
    symbol_col: str = "symbol",
) -> Dict[str, Any]:
    """Computes comprehensive robust metrics across clean, non-gap episodes.

    Filters out:
    - observations with has_ws_gap == True
    - observations with has_invalid_quote == True
    - crossed/zero-size detected quotes
    - controls (is_control == True)
    - intra-burst secondary ticks (is_primary_episode == False)

    Metrics returned:
    - mean
    - median
    - 5% trimmed mean
    - 1/99% winsorized mean
    - share of total PnL from top 1% of episodes
    - leave-one-coin-out (drop best coin, drop worst coin)
    - drop-top-3 coins
    """
    if df.empty or ret_col not in df.columns:
        return {"status": "NO_DATA", "count": 0}

    # Strict hygiene filtering
    mask = (
        (df.get("is_primary_episode", True) == True) &
        (~df.get("is_control", False)) &
        (~df.get("has_ws_gap", False)) &
        (~df.get("has_invalid_quote", False))
    )
    clean = df[mask].dropna(subset=[ret_col]).copy()

    # Quote sanity if bid/ask columns present
    if "detected_bid" in clean.columns and "detected_ask" in clean.columns:
        clean = clean[
            (clean["detected_bid"] > 0) &
            (clean["detected_ask"] > 0) &
            (clean["detected_bid"] < clean["detected_ask"])
        ]
    if "detected_spread_bps" in clean.columns:
        clean = clean[clean["detected_spread_bps"] > 0]

    vals = clean[ret_col].to_numpy(dtype=float)
    N = len(vals)
    if N == 0:
        return {"status": "NO_DATA", "count": 0}

    # 1. Mean & Median
    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))

    # 2. 5% Trimmed Mean (removes lowest 5% and highest 5%)
    sorted_vals = np.sort(vals)
    k_trim = int(N * 0.05)
    if k_trim > 0 and (N - 2 * k_trim) > 0:
        trimmed_slice = sorted_vals[k_trim : N - k_trim]
        trimmed_mean = float(np.mean(trimmed_slice))
    else:
        trimmed_mean = mean_val

    # 3. 1/99% Winsorized Mean (clips below 1st and above 99th percentile)
    p1 = float(np.percentile(vals, 1.0))
    p99 = float(np.percentile(vals, 99.0))
    winsorized_vals = np.clip(vals, p1, p99)
    winsorized_mean = float(np.mean(winsorized_vals))

    # 4. Top 1% Share of Total PnL
    k_top = max(1, int(np.ceil(0.01 * N)))
    top_1pct_pnl = float(np.sum(sorted_vals[-k_top:]))
    total_pnl = float(np.sum(vals))
    pos_pnl_sum = float(np.sum(vals[vals > 0])) if np.any(vals > 0) else 0.0

    if total_pnl > 0.0:
        top_1pct_share = (top_1pct_pnl / total_pnl) * 100.0
    elif pos_pnl_sum > 0.0:
        # If total PnL is negative/zero, report share of positive gains
        top_1pct_share = (top_1pct_pnl / pos_pnl_sum) * 100.0
    else:
        top_1pct_share = 0.0

    # 5. Leave-One-Coin-Out (both directions) and Drop-Top-3
    if symbol_col in clean.columns and clean[symbol_col].nunique() >= 2:
        coin_sums = clean.groupby(symbol_col)[ret_col].sum().sort_values(ascending=False)
        best_coin = str(coin_sums.index[0])
        worst_coin = str(coin_sums.index[-1])

        drop_best_df = clean[clean[symbol_col] != best_coin]
        drop_worst_df = clean[clean[symbol_col] != worst_coin]

        drop_best_mean = float(drop_best_df[ret_col].mean()) if len(drop_best_df) > 0 else 0.0
        drop_worst_mean = float(drop_worst_df[ret_col].mean()) if len(drop_worst_df) > 0 else 0.0

        if len(coin_sums) >= 4:
            top_3_coins = [str(c) for c in coin_sums.index[:3]]
            drop_top_3_df = clean[~clean[symbol_col].isin(top_3_coins)]
            drop_top_3_mean = float(drop_top_3_df[ret_col].mean()) if len(drop_top_3_df) > 0 else 0.0
        else:
            top_3_coins = [str(c) for c in coin_sums.index]
            drop_top_3_mean = drop_best_mean
    else:
        best_coin, worst_coin = "N/A", "N/A"
        drop_best_mean = mean_val
        drop_worst_mean = mean_val
        top_3_coins = []
        drop_top_3_mean = mean_val

    return {
        "status": "VALID",
        "count": N,
        "mean_bps": round(mean_val, 2),
        "median_bps": round(median_val, 2),
        "trimmed_mean_5pct_bps": round(trimmed_mean, 2),
        "winsorized_mean_1_99_bps": round(winsorized_mean, 2),
        "top_1pct_pnl_share_pct": round(top_1pct_share, 2),
        "loco_drop_best": {
            "coin": best_coin,
            "mean_net_bps": round(drop_best_mean, 2),
            "delta_bps": round(drop_best_mean - mean_val, 2),
        },
        "loco_drop_worst": {
            "coin": worst_coin,
            "mean_net_bps": round(drop_worst_mean, 2),
            "delta_bps": round(drop_worst_mean - mean_val, 2),
        },
        "drop_top_3": {
            "coins": top_3_coins,
            "mean_net_bps": round(drop_top_3_mean, 2),
            "delta_bps": round(drop_top_3_mean - mean_val, 2),
        },
    }

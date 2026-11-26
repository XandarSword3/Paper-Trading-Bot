"""Authoritative Episode Definition and Chronological Partitioning for OmniPulse Research.

DEFINITIONS & RULES
===================
1. What is an Episode?
   An episode represents a single, independent market surge or flow-anomaly event on a specific
   trading instrument.

2. Independence & Debouncing (5-minute Quiet Window):
   High-frequency volatility and order flow arrive in bursts. A single physical event often spans
   several seconds to minutes. To prevent burst-multiplication (treating 50 consecutive ticks within
   one 30-second spike as 50 independent statistical observations):
   - A signal detected on symbol S at timestamp t is designated as a PRIMARY EPISODE if and only if:
         t - t_last_primary(S) >= EPISODE_GAP_SECONDS (default: 300.0s / 5 minutes)
   - Any signal arriving on symbol S while t - t_last_primary(S) < EPISODE_GAP_SECONDS is designated
     as an INTRA-BURST TICK (Secondary Signal) with `is_primary_episode = False`.

3. Classification Schema:
   - Primary Signal Episode:
         `is_primary_episode == True` AND `is_control == False`
     This is the ONLY record type counted towards sample size N for primary hypothesis testing.
   - Intra-Burst Tick (Secondary Signal):
         `is_primary_episode == False` AND `is_control == False`
     Used strictly for execution micro-structure diagnostics, not primary significance.
   - Matched Control:
         `is_control == True`
     A counterfactual observation generated at an independent or matched timestamp for baseline
     comparison. Controls are NEVER primary signal episodes.

4. Strict Chronological Calendar-Time Partitioning:
   Splits are strictly computed on CALENDAR ELAPSED TIME across the recording interval [T_start, T_end]:
         T_split = T_start + split_pct * (T_end - T_start)
   - Discovery Partition: { e | T_start <= e.timestamp < T_split }
   - Holdout Partition:   { e | T_split <= e.timestamp <= T_end }

   Non-Uniform Episode Distribution Note:
   Because market events cluster in time (volatility clustering), the number of episodes in the
   Discovery partition may differ from exactly 70% of total episodes (e.g., 77% vs 23%). The split
   is strictly by CALENDAR TIME, not by event index.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("omnipulse.episode")

EPISODE_GAP_SECONDS: float = 300.0  # 5 minutes quiet gap between independent episodes


def is_primary_signal_episode(record: Union[Dict[str, Any], Any]) -> bool:
    """Return True if and only if the record represents a genuine primary signal episode.
    
    Strictly excludes:
    - Intra-burst secondary ticks (is_primary_episode == False)
    - Matched controls (is_control == True)
    """
    if isinstance(record, dict):
        is_primary = bool(record.get("is_primary_episode", False))
        is_ctrl = bool(record.get("is_control", False))
    else:
        is_primary = bool(getattr(record, "is_primary_episode", False))
        is_ctrl = bool(getattr(record, "is_control", False))
    return is_primary and not is_ctrl


def split_by_calendar_time(
    df: pd.DataFrame,
    split_pct: float = 0.70,
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological split by calendar elapsed time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing observations with a timestamp column.
    split_pct : float, default 0.70
        Fraction of calendar duration allocated to Discovery (e.g. 0.70 for 70/30).
    time_col : str, default "timestamp"
        Column name holding POSIX epoch seconds or datetime timestamps.
        
    Returns
    -------
    discovery_df : pd.DataFrame
        Observations where timestamp < T_split.
    holdout_df : pd.DataFrame
        Observations where timestamp >= T_split.
    """
    if df.empty:
        return df.copy(), df.copy()

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    t_split = t_min + split_pct * (t_max - t_min)

    discovery_df = df[df[time_col] < t_split].copy().reset_index(drop=True)
    holdout_df = df[df[time_col] >= t_split].copy().reset_index(drop=True)

    logger.info(
        "Strict Calendar Split (%.0f/%.0f): T_min=%.1f, T_split=%.1f, T_max=%.1f | "
        "Discovery N=%d (%.1f%%) | Holdout N=%d (%.1f%%)",
        split_pct * 100, (1 - split_pct) * 100,
        float(t_min), float(t_split), float(t_max),
        len(discovery_df), (len(discovery_df) / len(df) * 100) if len(df) else 0.0,
        len(holdout_df), (len(holdout_df) / len(df) * 100) if len(df) else 0.0,
    )
    return discovery_df, holdout_df

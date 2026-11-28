"""Dated margin-tier (leverage bracket) snapshots.

Source: GET /fapi/v1/leverageBracket (SIGNED — requires API key/secret;
not a public endpoint). Each call returns the CURRENT brackets for a
symbol. We store every retrieval as a dated SNAPSHOT:

- ``snapshot_time`` = when WE retrieved it (observation date).
- ``effective_date`` = when the exchange says the brackets took effect.
  Binance does not provide historical effective dates, so this is
  ALWAYS null — snapshots are observations, not reconstructed history.
- Successive snapshots are appended (deduplicated on
  symbol+bracket+snapshot_time) so future changes can be tracked.

PIT rule: a decision at time T may only use snapshots with
``snapshot_time <= T`` (``margin_tiers_as_of``). Today's brackets are
never claimed to be historically effective.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v4_system.data_engine import store

DATASET = "margin_tiers"

BRACKET_COLUMNS = ["symbol", "bracket", "notional_floor", "notional_cap",
                   "initial_leverage", "max_leverage", "maint_margin_ratio",
                   "cum", "snapshot_time", "effective_date", "source"]


def fetch_leverage_brackets(symbols: list[str],
                            api_key: str, api_secret: str,
                            client=None) -> dict[str, list[dict]]:
    """Fetch current leverage brackets for each symbol (signed endpoint).

    Returns {symbol: [bracket dicts]}. Raises on auth/API failure —
    callers must handle the no-keys case explicitly.
    """
    from v4_system.data_engine.binance_client import BinanceClient
    client = client or BinanceClient.futures()
    out = {}
    for sym in symbols:
        resp = client.get_signed("/fapi/v1/leverageBracket",
                                 {"symbol": sym}, api_key, api_secret)
        # Response: [{"symbol":..., "brackets":[...]}]
        if isinstance(resp, list) and resp:
            out[sym] = resp[0].get("brackets", [])
        else:
            out[sym] = []
    return out


def brackets_to_frame(brackets_by_symbol: dict[str, list[dict]],
                      snapshot_time=None) -> pd.DataFrame:
    """Flatten {symbol: [brackets]} into a snapshot DataFrame."""
    snapshot_time = snapshot_time or datetime.now(timezone.utc)
    snapshot_time = pd.Timestamp(snapshot_time)
    if snapshot_time.tzinfo is None:
        snapshot_time = snapshot_time.tz_localize("UTC")
    else:
        snapshot_time = snapshot_time.tz_convert("UTC")
    rows = []
    for sym, brackets in brackets_by_symbol.items():
        for b in brackets:
            rows.append({
                "symbol": sym,
                "bracket": b.get("bracket"),
                "notional_floor": b.get("notionalFloor"),
                "notional_cap": b.get("notionalCap"),
                "initial_leverage": b.get("initialLeverage"),
                "max_leverage": b.get("maxLeverage"),
                "maint_margin_ratio": b.get("maintMarginRatio"),
                "cum": b.get("cum"),
                "snapshot_time": snapshot_time,
                "effective_date": None,  # unknown — never fabricated
                "source": "fapi /fapi/v1/leverageBracket (signed)",
            })
    df = pd.DataFrame(rows, columns=BRACKET_COLUMNS)
    for c in ("notional_floor", "notional_cap", "maint_margin_ratio", "cum"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("bracket", "initial_leverage", "max_leverage"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_brackets(df: pd.DataFrame) -> list[str]:
    """Validate bracket ordering/ranges per symbol+snapshot.

    Returns a list of human-readable problems (empty = valid).
    """
    problems = []
    keys = ["symbol", "snapshot_time"] if "snapshot_time" in df.columns else ["symbol"]
    for key, g in df.groupby(keys, dropna=False):
        g = g.sort_values("bracket")
        brackets = g["bracket"].tolist()
        if brackets != sorted(brackets) or len(set(brackets)) != len(brackets):
            problems.append(f"{key}: bracket numbers not unique/ordered")
        floors = g["notional_floor"].tolist()
        caps = g["notional_cap"].tolist()
        for i in range(len(g)):
            if floors[i] is not None and caps[i] is not None \
                    and not (floors[i] < caps[i]):
                problems.append(f"{key}: bracket {brackets[i]} floor >= cap")
        for i in range(len(g) - 1):
            if caps[i] is not None and floors[i + 1] is not None \
                    and caps[i] > floors[i + 1]:
                problems.append(f"{key}: overlap between brackets "
                                f"{brackets[i]} and {brackets[i + 1]}")
    return problems


def save_snapshot(df: pd.DataFrame) -> str:
    """Append a snapshot to the margin_tiers dataset (atomic, deduped)."""
    try:
        existing = store.load(DATASET)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=BRACKET_COLUMNS)
    if existing.empty:
        merged = df.copy()
    else:
        merged = pd.concat([existing, df], ignore_index=True)
    dedupe_cols = ["symbol", "bracket", "snapshot_time"]
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    merged = merged.sort_values(["symbol", "snapshot_time", "bracket"])
    return store.save(merged, DATASET,
                      source="fapi /fapi/v1/leverageBracket (signed)",
                      symbol="MULTI", timeframe=None,
                      extra={"note": "dated observed snapshots; effective_date unknown"},
                      available_at=df["snapshot_time"].max())


def margin_tiers_as_of(symbol: str, as_of, dataset: str = DATASET) -> pd.DataFrame:
    """Latest margin-tier snapshot observable at ``as_of``.

    Only snapshots with ``snapshot_time <= as_of`` are eligible — a
    snapshot retrieved AFTER the decision time is never used.
    """
    df = store.load(dataset)
    df = df[df["symbol"] == symbol]
    if df.empty:
        return df
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    eligible = df[df["snapshot_time"] <= as_of]
    if eligible.empty:
        return eligible
    latest = eligible["snapshot_time"].max()
    return eligible[eligible["snapshot_time"] == latest].sort_values("bracket")


def snapshot_dates(dataset: str = DATASET) -> pd.Series:
    """All distinct snapshot retrieval times (observation dates)."""
    df = store.load(dataset)
    return pd.Series(sorted(df["snapshot_time"].unique()))

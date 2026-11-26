"""Dataset provenance manifest.

Every persisted dataset gets an entry recording source endpoint, symbol,
row count, date range, fetch timestamp and a sha256 of the parquet file —
same discipline as research/data/_update_manifest, extended with the
endpoint path so perp vs spot vs futures-data sources are distinguishable.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from v4_system.config import MANIFEST_PATH


def _load() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def update_manifest(
    name: str,
    df: pd.DataFrame,
    parquet_path: str,
    *,
    source: str,
    symbol: str,
    timeframe: str | None = None,
    extra: dict | None = None,
) -> None:
    with open(parquet_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = _load()
    entry = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": file_hash,
    }
    if extra:
        entry.update(extra)
    manifest[name] = entry

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

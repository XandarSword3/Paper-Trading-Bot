"""Diagnostics for H3: is the breakout edge real or a same-bar artifact?

1. Holdout 2025-01-01 .. 2026-10-31 with the frozen primary.
2. One-hour delayed entry on discovery (enter 01:00, exit 01:00 next day).
3. Long/short decomposition and BTC beta.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.new_edge.h03_vol_breakout as h3

RESULTS = h3.RESULTS_DIR / "h03_diagnostics.json"


def delayed_trades(frames: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    rows = []
    for symbol, frame in frames.items():
        sig = h3.symbol_daily_signals(frame, **kwargs)
        opens = frame["open"]
        for date, row in sig.iterrows():
            if not row["long"] and not row["short"]:
                continue
            entry_ts = date + pd.Timedelta(days=1, hours=1)
            exit_ts = date + pd.Timedelta(days=2, hours=1)
            if entry_ts not in opens.index or exit_ts not in opens.index:
                continue
            entry = float(opens.loc[entry_ts])
            exit_price = float(opens.loc[exit_ts])
            if not np.isfinite(entry) or entry <= 0:
                continue
            direction = 1 if row["long"] else -1
            fwd = exit_price / entry - 1.0
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "direction": direction,
                    "fwd": fwd,
                    "net_return": direction * fwd - h3.COST_ROUND_TRIP,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    frames = {
        path.stem.replace("_1h", ""): h3.load_symbol(path)
        for path in sorted(h3.KLINES_DIR.glob("*_1h.parquet"))
    }

    # 1) Holdout with the frozen primary.
    h3.DISCOVERY_START = pd.Timestamp("2025-01-01", tz="UTC")
    h3.DISCOVERY_END = pd.Timestamp("2026-11-01", tz="UTC")
    holdout_daily, holdout_trades = h3.combine(frames, **h3.PRIMARY)
    holdout = {
        **h3.metrics(holdout_daily),
        "signals": int(len(holdout_trades)),
    }

    # 2) One-hour delayed entry on discovery.
    h3.DISCOVERY_START = pd.Timestamp("2020-01-01", tz="UTC")
    h3.DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
    delayed = delayed_trades(frames, **h3.PRIMARY)
    delayed_daily = delayed.groupby("date")["net_return"].mean().sort_index()
    delayed_metrics = {
        **h3.metrics(delayed_daily),
        "signals": int(len(delayed)),
    }

    # 3) Decomposition on the primary discovery book.
    _, trades = h3.combine(frames, **h3.PRIMARY)
    by_direction = trades.groupby("direction")["net_return"].agg(["count", "mean"]).to_dict("index")
    exposure = trades.assign(sign=np.sign(trades["direction"])).groupby("date")["sign"].mean()

    payload = {
        "holdout_2025_2026": holdout,
        "delayed_entry_1h_discovery": delayed_metrics,
        "by_direction": {str(k): v for k, v in by_direction.items()},
        "avg_net_exposure": float(exposure.mean()),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

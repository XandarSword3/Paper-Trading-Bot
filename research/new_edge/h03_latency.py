"""H3 latency test: does the breakout edge survive a one-minute entry delay?

The 1h-delayed entry destroyed the edge, which points to a same-bar/latency
artifact.  This test uses 1-minute opens to enter at 00:01 instead of 00:00,
on the 20 symbols with 1-minute data (2025-11-01 .. 2026-10-31).
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

MINUTE_DIR = h3.WORKSPACE / "Trading Bot" / "liquidation-hunter" / "data" / "binance_1m" / "full"
START = pd.Timestamp("2025-11-01", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")
RESULTS = h3.RESULTS_DIR / "h03_latency.json"


def load_minute_opens(symbol: str) -> pd.Series:
    path = MINUTE_DIR / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=["minute", "open"])
    frame["timestamp"] = pd.to_datetime(frame["minute"].astype("int64"), unit="s", utc=True)
    return frame.drop_duplicates("timestamp").set_index("timestamp")["open"].sort_index()


def main() -> None:
    frames_1h = {
        path.stem.replace("_1h", ""): h3.load_symbol(path)
        for path in sorted(h3.KLINES_DIR.glob("*_1h.parquet"))
    }
    rows = []
    for symbol, opens in ((s, load_minute_opens(s)) for s in sorted(p.stem for p in MINUTE_DIR.glob("*.parquet"))):
        if symbol not in frames_1h:
            continue
        sig = h3.symbol_daily_signals(frames_1h[symbol], **h3.PRIMARY)
        for date, row in sig.iterrows():
            if date < START or date >= END:
                continue
            if not row["long"] and not row["short"]:
                continue
            entry_ts = date + pd.Timedelta(days=1, minutes=1)
            exit_ts = date + pd.Timedelta(days=2, minutes=1)
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
                    "net_return": direction * fwd - h3.COST_ROUND_TRIP,
                }
            )
    trades = pd.DataFrame(rows)
    daily = trades.groupby("date")["net_return"].mean().sort_index()
    payload = {
        "period": [str(START.date()), "2026-10-31"],
        "entry": "00:01 UTC, one minute after the signal close",
        "signals": int(len(trades)),
        "metrics": h3.metrics(daily),
        "by_direction": trades.groupby("direction")["net_return"].agg(["count", "mean"]).to_dict("index"),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

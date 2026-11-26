"""H9 pilot: open-interest crowding, live 3-week sample.

The cached klines stop on 2026-10-31, but Binance's public futures API is
reachable and serves open interest history for the last ~21 days.  This is a
mechanism check, not a strategy test: 21 days and one regime cannot promote
anything.  The point is to see whether OI + price crowding has a gross effect
large enough to justify recording OI going forward.

Hypothesis (pre-registered):
    A sharp rise in open interest together with a sharp price move is new
    leverage entering the book.  When price is up and OI is up, longs are
    crowded and the next 8 hours mean-revert down; when price is down and OI is
    up, shorts are crowded and the next 8 hours mean-revert up.

Protocol:
    * window: whatever the API returns (~2026-11-05 .. 2026-11-26), 1h bars
    * r4 = 4h return, doi4 = 4h OI change, z-scored cross-sectionally
    * event: |z(doi4)| >= 1.5 and |r4| >= 0.5%
    * direction: -sign(r4) * sign(z) (fade the crowded move)
    * hold 8h, 8 bps per side (16 bps round trip)
    * report gross and net per trade, hit rate, t-stat, and bucket means
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.new_edge.h03_vol_breakout import RESULTS_DIR

BASE = "https://fapi.binance.com"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "FILUSDT", "NEARUSDT",
    "UNIUSDT", "AAVEUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
    "SEIUSDT", "INJUSDT", "TIAUSDT", "FETUSDT", "ARUSDT", "ICPUSDT",
    "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "TONUSDT", "ENAUSDT", "RENDERUSDT",
]
COST_ROUND_TRIP = 16.0 / 10_000.0


def get_json(url: str):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                return json.loads(response.read())
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def fetch_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = {}
    ois = {}
    for symbol in SYMBOLS:
        klines = get_json(f"{BASE}/fapi/v1/klines?symbol={symbol}&interval=1h&limit=500")
        if klines:
            frame = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
            frame["ts"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
            closes[symbol] = frame.set_index("ts")["close"].astype(float)
        hist = get_json(f"{BASE}/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=500")
        if hist:
            frame = pd.DataFrame(hist)
            frame["ts"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            ois[symbol] = frame.set_index("ts")["sumOpenInterest"].astype(float)
        time.sleep(0.05)
    return pd.concat(closes, axis=1).sort_index(), pd.concat(ois, axis=1).sort_index()


def stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"n": int(len(values))}
    return {
        "n": int(len(values)),
        "mean_bps": float(values.mean() * 1e4),
        "std_bps": float(values.std(ddof=1) * 1e4),
        "hit_rate": float((values > 0).mean()),
        "t_stat": float(values.mean() / values.std(ddof=1) * np.sqrt(len(values))),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    closes, ois = fetch_panel()
    print(f"closes {closes.shape} {closes.index[0]} .. {closes.index[-1]}")
    print(f"oi {ois.shape} {ois.index[0]} .. {ois.index[-1]}")

    r4 = closes.pct_change(4)
    doi4 = ois.pct_change(4)
    fwd8 = closes.shift(-8) / closes - 1.0

    # Cross-sectional z of OI change at each bar.
    z = doi4.sub(doi4.mean(axis=1), axis=0).div(doi4.std(axis=1).replace(0.0, np.nan), axis=0)

    rows = []
    for ts in closes.index:
        zrow = z.loc[ts].dropna()
        rrow = r4.loc[ts].reindex(zrow.index).dropna()
        frow = fwd8.loc[ts].reindex(rrow.index).dropna()
        common = rrow.index.intersection(frow.index)
        if len(common) < 10:
            continue
        for symbol in common:
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "r4": float(rrow[symbol]),
                    "z_oi": float(zrow[symbol]),
                    "fwd8": float(frow[symbol]),
                }
            )
    data = pd.DataFrame(rows)
    print(f"observations {len(data)}")

    # Bucket means: price direction x OI-change tercile.
    data["oi_bucket"] = pd.qcut(data["z_oi"], 3, labels=["oi_down", "oi_mid", "oi_up"], duplicates="drop")
    data["px_dir"] = np.where(data["r4"] > 0, "up", "down")
    buckets = {}
    for (px, oi), group in data.groupby(["px_dir", "oi_bucket"], observed=True):
        buckets[f"{px}_{oi}"] = stats(group["fwd8"].to_numpy())

    # Event trade: fade the crowded move.
    events = data[(data["z_oi"].abs() >= 1.5) & (data["r4"].abs() >= 0.005)].copy()
    events["direction"] = -np.sign(events["r4"])
    events["gross"] = events["direction"] * events["fwd8"]
    events["net"] = events["gross"] - COST_ROUND_TRIP

    # Same event, but only the "new leverage" side (OI up) and only OI down.
    up = events[events["z_oi"] > 0]
    down = events[events["z_oi"] < 0]

    payload = {
        "hypothesis": "H9 open-interest crowding pilot (21-day live sample)",
        "window": [str(closes.index[0]), str(closes.index[-1])],
        "symbols": int(closes.shape[1]),
        "observations": int(len(data)),
        "bucket_means_fwd8": buckets,
        "event_gross_net": {
            "all": stats(events["net"].to_numpy()),
            "gross": stats(events["gross"].to_numpy()),
            "oi_up": stats(up["net"].to_numpy()),
            "oi_down": stats(down["net"].to_numpy()),
        },
        "event_count": int(len(events)),
    }
    (RESULTS_DIR / "h09_oi_pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

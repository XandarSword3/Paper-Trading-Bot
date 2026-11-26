"""H4 pilot: cross-sectional beta overshoot after BTC shocks.

Hypothesis (pre-registered before running):
    When BTC prints a large 5-minute move, high-beta alts overshoot their
    fair beta-adjusted move.  Over the next 30 minutes the cross-section
    partially reverses.  Short the positive residual outliers, buy the
    negative residual outliers, dollar-neutral.

Protocol:
    * discovery only: 2025-11-01 .. 2026-07-31
    * 5-minute bars from 1-minute Binance USDT perps
    * event: |BTC 5m return| > 3 sigma (1-day rolling)
    * beta: 3-day rolling covariance / BTC variance
    * residual z-score: (alt return - beta * BTC return) / rolling residual sigma
    * long bottom 3, short top 3, equal weight, 0.5 long / 0.5 short
    * enter next 5m open, exit after 6 bars (30 minutes)
    * 8 bps per side; holdout (2026-08-01 .. 2026-10-31) is not loaded
"""

from __future__ import annotations

from dataclasses import dataclass
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
RESULTS_DIR = h3.RESULTS_DIR
DISCOVERY_START = pd.Timestamp("2025-11-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2026-08-01", tz="UTC")
COST_ROUND_TRIP = 16.0 / 10_000.0
PRIMARY = {"event_sigma": 3.0, "hold_bars": 6, "names": 3, "beta_window": 864, "sigma_window": 288}


@dataclass
class Event:
    time: pd.Timestamp
    entry_idx: int
    exit_idx: int
    longs: list[str]
    shorts: list[str]
    gross_return: float
    net_return: float


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = {}
    closes = {}
    for path in sorted(MINUTE_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["minute", "open", "close"])
        frame["timestamp"] = pd.to_datetime(frame["minute"].astype("int64"), unit="s", utc=True)
        frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        symbol = path.stem
        opens[symbol] = frame["open"].resample("5min", label="left", closed="left").first()
        closes[symbol] = frame["close"].resample("5min", label="left", closed="left").last()
        for series in (opens[symbol], closes[symbol]):
            series.index.freq = None
    return pd.concat(opens, axis=1).sort_index(), pd.concat(closes, axis=1).sort_index()


def residual_z(closes: pd.DataFrame, beta_window: int, sigma_window: int) -> tuple[pd.DataFrame, pd.Series]:
    ret = closes.pct_change()
    btc = ret["BTCUSDT"]
    var = btc.rolling(beta_window, min_periods=int(beta_window * 0.7)).var()
    out = {}
    for symbol in ret.columns:
        if symbol == "BTCUSDT":
            continue
        cov = ret[symbol].rolling(beta_window, min_periods=int(beta_window * 0.7)).cov(btc)
        beta = cov / var
        resid = ret[symbol] - beta * btc
        sigma = resid.rolling(beta_window, min_periods=int(beta_window * 0.7)).std()
        out[symbol] = resid / sigma
    return pd.DataFrame(out), btc


def build_events(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    event_sigma: float,
    hold_bars: int,
    names: int,
    beta_window: int,
    sigma_window: int,
) -> list[Event]:
    z, btc = residual_z(closes, beta_window, sigma_window)
    btc_sigma = btc.rolling(sigma_window, min_periods=int(sigma_window * 0.7)).std()
    event_mask = (btc.abs() > event_sigma * btc_sigma).fillna(False)

    index = closes.index
    events: list[Event] = []
    last_exit = -1
    for t in index[event_mask]:
        if t < DISCOVERY_START or t >= DISCOVERY_END:
            continue
        entry_idx = int(index.get_loc(t)) + 1
        exit_idx = entry_idx + hold_bars
        if entry_idx <= last_exit or exit_idx >= len(index):
            continue
        row = z.loc[t].dropna()
        entry_prices = opens.iloc[entry_idx]
        exit_prices = opens.iloc[exit_idx]
        valid = [
            s for s in row.index
            if np.isfinite(entry_prices.get(s, np.nan))
            and np.isfinite(exit_prices.get(s, np.nan))
            and entry_prices[s] > 0
        ]
        if len(valid) < max(10, names * 2):
            continue
        ranked = row[valid].sort_values()
        longs = list(ranked.index[:names])
        shorts = list(ranked.index[-names:])
        long_ret = float((exit_prices[longs] / entry_prices[longs] - 1.0).mean())
        short_ret = float(-(exit_prices[shorts] / entry_prices[shorts] - 1.0).mean())
        gross = 0.5 * long_ret + 0.5 * short_ret
        net = gross - COST_ROUND_TRIP
        events.append(
            Event(
                time=t,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                longs=longs,
                shorts=shorts,
                gross_return=gross,
                net_return=net,
            )
        )
        last_exit = exit_idx
    return events


def daily_series(events: list[Event]) -> pd.Series:
    if not events:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(
        {
            "date": [e.time.normalize() for e in events],
            "net": [e.net_return for e in events],
        }
    )
    return frame.groupby("date")["net"].sum().sort_index()


def event_stats(events: list[Event]) -> dict:
    if not events:
        return {"events": 0, "hit_rate": float("nan"), "avg_event_bps": float("nan")}
    nets = np.array([e.net_return for e in events])
    return {
        "events": int(len(events)),
        "hit_rate": float((nets > 0).mean()),
        "avg_event_bps": float(nets.mean() * 10_000.0),
        "avg_gross_bps": float(np.mean([e.gross_return for e in events]) * 10_000.0),
    }


def placebo(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    events: list[Event],
    actual_sharpe: float,
    *,
    beta_window: int,
    sigma_window: int,
    draws: int = 200,
) -> dict:
    rng = np.random.default_rng(20261126)
    z, _ = residual_z(closes, beta_window, sigma_window)
    sharpes: list[float] = []
    for _ in range(draws):
        sample: list[float] = []
        for event in events:
            row = z.loc[event.time].dropna()
            entry_prices = opens.iloc[event.entry_idx]
            exit_prices = opens.iloc[event.exit_idx]
            valid = [
                s for s in row.index
                if np.isfinite(entry_prices.get(s, np.nan))
                and np.isfinite(exit_prices.get(s, np.nan))
                and entry_prices[s] > 0
            ]
            if len(valid) < 6:
                continue
            picks = list(rng.choice(valid, size=6, replace=False))
            longs, shorts = picks[:3], picks[3:]
            long_ret = float((exit_prices[longs] / entry_prices[longs] - 1.0).mean())
            short_ret = float(-(exit_prices[shorts] / entry_prices[shorts] - 1.0).mean())
            sample.append(0.5 * long_ret + 0.5 * short_ret - COST_ROUND_TRIP)
        if not sample:
            continue
        frame = pd.DataFrame({"date": [e.time.normalize() for e in events[: len(sample)]], "net": sample})
        daily = frame.groupby("date")["net"].sum()
        sharpes.append(h3.metrics(daily)["sharpe"])
    arr = np.array([s for s in sharpes if np.isfinite(s)])
    return {
        "draws": int(len(arr)),
        "placebo_mean": float(arr.mean()) if len(arr) else float("nan"),
        "placebo_p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "actual": actual_sharpe,
        "p_value": float((arr >= actual_sharpe).mean()) if len(arr) and np.isfinite(actual_sharpe) else float("nan"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    opens, closes = load_panel()
    print(f"panel {closes.shape}, {closes.index[0]} .. {closes.index[-1]}")

    events = build_events(opens, closes, **PRIMARY)
    daily = daily_series(events)
    overall = {**h3.metrics(daily), **event_stats(events)}

    sensitivity = []
    for sigma in (2.5, 3.0, 3.5):
        for hold in (3, 6, 12):
            for names in (2, 3, 5):
                cfg = {**PRIMARY, "event_sigma": sigma, "hold_bars": hold, "names": names}
                ev = build_events(opens, closes, **cfg)
                d = daily_series(ev)
                sensitivity.append({"sigma": sigma, "hold": hold, "names": names, **h3.metrics(d), **event_stats(ev)})

    # No-beta control: rank on raw 5m return at the event instead of residual.
    raw_ret = closes.pct_change()
    raw_events: list[Event] = []
    btc = raw_ret["BTCUSDT"]
    btc_sigma = btc.rolling(PRIMARY["sigma_window"], min_periods=200).std()
    mask = (btc.abs() > PRIMARY["event_sigma"] * btc_sigma).fillna(False)
    last_exit = -1
    for t in closes.index[mask]:
        if t < DISCOVERY_START or t >= DISCOVERY_END:
            continue
        entry_idx = int(closes.index.get_loc(t)) + 1
        exit_idx = entry_idx + PRIMARY["hold_bars"]
        if entry_idx <= last_exit or exit_idx >= len(closes):
            continue
        entry_prices = opens.iloc[entry_idx]
        exit_prices = opens.iloc[exit_idx]
        row = raw_ret.loc[t].drop("BTCUSDT").dropna()
        valid = [s for s in row.index if np.isfinite(entry_prices.get(s, np.nan)) and np.isfinite(exit_prices.get(s, np.nan)) and entry_prices[s] > 0]
        if len(valid) < 10:
            continue
        ranked = row[valid].sort_values()
        longs = list(ranked.index[:3])
        shorts = list(ranked.index[-3:])
        long_ret = float((exit_prices[longs] / entry_prices[longs] - 1.0).mean())
        short_ret = float(-(exit_prices[shorts] / entry_prices[shorts] - 1.0).mean())
        gross = 0.5 * long_ret + 0.5 * short_ret
        raw_events.append(Event(t, entry_idx, exit_idx, longs, shorts, gross, gross - COST_ROUND_TRIP))
        last_exit = exit_idx
    control = {**h3.metrics(daily_series(raw_events)), **event_stats(raw_events)}

    null = placebo(
        opens,
        closes,
        events,
        overall["sharpe"],
        beta_window=PRIMARY["beta_window"],
        sigma_window=PRIMARY["sigma_window"],
    )

    payload = {
        "hypothesis": "H4 cross-sectional beta overshoot after BTC shocks",
        "primary": PRIMARY,
        "discovery": [str(DISCOVERY_START.date()), "2026-07-31"],
        "overall": overall,
        "sensitivity": sensitivity,
        "no_beta_control": control,
        "placebo": null,
        "first_events": [
            {
                "time": e.time.isoformat(),
                "longs": e.longs,
                "shorts": e.shorts,
                "net_bps": e.net_return * 10_000.0,
            }
            for e in events[:3]
        ],
    }
    (RESULTS_DIR / "h04_beta_overshoot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "control": control, "placebo": null}, indent=2))


if __name__ == "__main__":
    main()

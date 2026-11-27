"""H10: perp/index premium extreme fade (time-series, 1h).

Pre-registered before running.

Hypothesis:
    The premium of the perpetual over its spot index is a crowding measure.
    When the premium is stretched far above its own recent range the perp is
    rich and should mean-revert down; when it is stretched below it should
    mean-revert up.  This is a *time-series* basis-reversion signal, distinct
    from H2/H6 (cross-sectional funding rank) and from the delta-neutral
    cash-and-carry project in research_next/.

Protocol (frozen):
    * universe: BTCUSDT, ETHUSDT perps on Binance USD-M
      (SOLUSDT is excluded: its index series prints -2000 bps vs mark around
       the 2022-11-09 FTX crash, which is a data artifact, not a tradeable
       dislocation.  Recorded here so the exclusion is not silent.)
    * premium_t = perp_close_t / index_close_t - 1   (primary, directly
      tradeable); mark_close_t / index_close_t - 1 is a secondary panel.
    * z_t = (premium_t - rolling_mean_720h) / rolling_std_720h, min 168 obs
    * event when |z_t| >= 2.0 at the close of bar t
    * enter at the OPEN of bar t+1 (one full bar of delay: no same-bar fill)
    * direction = fade the premium: short when z > 0, long when z < 0
    * hold exactly 8 hours (primary).  24h and 72h are reported as secondary
      diagnostics only and are NOT used to pick the primary.
    * non-overlapping within a symbol: after an event the next event can only
      start once the previous position has exited
    * cost 8 bps per side (16 bps round trip); also reported at 4 bps
    * discovery: 2019-12-23 .. 2024-12-31
    * holdout:   2025-01-01 .. 2026-11-26  (loaded once, never tuned on)

Kill gates (any failure = KILL):
    * discovery events >= 100
    * discovery net Sharpe >= 0.50  (daily Sharpe of the position backtest)
    * holdout net Sharpe >= 0.0
    * max drawdown (discovery+holdout stitched) >= -30%
    * placebo p-value <= 0.05 (shuffle z within symbol, 200 draws)
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

from research.new_edge.h02_funding_reversal import metrics

DATA_DIR = ROOT / "v4_system" / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
Z_WINDOW = 720
MIN_PERIODS = 168
Z_ENTRY = 2.0
PRIMARY_HOLD = 8
SECONDARY_HOLDS = [24, 72]
COST_PER_SIDE = 8.0 / 10_000.0
DISCOVERY_START = pd.Timestamp("2019-12-23", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-27", tz="UTC")
PLACEBO_DRAWS = 200
SEED = 20261127

PREMIUMS = {
    "perp_index": ("perp_1h", "index_1h"),
    "mark_index": ("mark_1h", "index_1h"),
}


def load_panel(symbol: str, num_kind: str, den_kind: str) -> pd.DataFrame:
    num = pd.read_parquet(DATA_DIR / f"{symbol}_{num_kind}.parquet")
    den = pd.read_parquet(DATA_DIR / f"{symbol}_{den_kind}.parquet")
    num = num[["open", "close"]].rename(columns={"open": "num_open", "close": "num_close"})
    den = den[["close"]].rename(columns={"close": "den_close"})
    frame = num.join(den, how="inner").dropna()
    frame["premium"] = frame["num_close"] / frame["den_close"] - 1.0
    return frame


def zscore(premium: pd.Series) -> pd.Series:
    mean = premium.rolling(Z_WINDOW, min_periods=MIN_PERIODS).mean()
    std = premium.rolling(Z_WINDOW, min_periods=MIN_PERIODS).std()
    return (premium - mean) / std.replace(0.0, np.nan)


def event_returns(
    frame: pd.DataFrame, z: pd.Series, hold: int, cost_per_side: float
) -> pd.DataFrame:
    """Non-overlapping fade events.  Signal at close t, enter open t+1."""
    rows = []
    next_free = 0
    opens = frame["num_open"].to_numpy()
    zvals = z.to_numpy()
    stamps = frame.index
    n = len(frame)
    for t in range(n - hold - 1):
        if t < next_free:
            continue
        zv = zvals[t]
        if not np.isfinite(zv) or abs(zv) < Z_ENTRY:
            continue
        entry_i = t + 1
        exit_i = entry_i + hold
        if exit_i >= n:
            break
        entry = opens[entry_i]
        exit_ = opens[exit_i]
        if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0):
            continue
        direction = -1.0 if zv > 0 else 1.0
        gross = direction * (exit_ / entry - 1.0)
        rows.append(
            {
                "signal_time": stamps[t],
                "entry_time": stamps[entry_i],
                "exit_time": stamps[exit_i],
                "z": zv,
                "direction": direction,
                "gross": gross,
                "net": gross - 2.0 * cost_per_side,
            }
        )
        next_free = exit_i
    return pd.DataFrame(rows)


def positions_from_events(frame: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Position held during each open-to-open interval, from the event list."""
    pos = np.zeros(len(frame))
    if events.empty:
        return pos
    index = frame.index
    for entry_time, exit_time, direction in zip(
        events["entry_time"], events["exit_time"], events["direction"]
    ):
        i0 = index.get_loc(entry_time)
        i1 = index.get_loc(exit_time)
        pos[i0:i1] = direction
    return pos


def hourly_pnl(frame: pd.DataFrame, pos: np.ndarray, cost_per_side: float) -> pd.Series:
    """Open-to-open hourly PnL with per-side cost charged on position changes."""
    open_ = frame["num_open"]
    fwd = open_.shift(-1) / open_ - 1.0
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pd.Series(pos * fwd.to_numpy() - turn * cost_per_side, index=frame.index)
    return pnl.iloc[:-1]


def portfolio_daily(per_symbol: dict[str, pd.Series]) -> pd.Series:
    frame = pd.concat(per_symbol, axis=1).sort_index()
    hourly = frame.mean(axis=1, skipna=True).fillna(0.0)
    daily = hourly.resample("1D").sum(min_count=1).fillna(0.0)
    daily.index.freq = None
    return daily


def slice_period(series: pd.Series, start, end) -> pd.Series:
    return series[(series.index >= start) & (series.index < end)]


def event_summary(events: pd.DataFrame, label: str) -> dict:
    if events.empty:
        return {"label": label, "events": 0}
    g = events["gross"]
    n = len(events)
    t_stat = float(g.mean() / (g.std(ddof=1) / np.sqrt(n))) if n > 1 and g.std(ddof=1) > 0 else float("nan")
    return {
        "label": label,
        "events": int(n),
        "avg_gross_bps": float(g.mean() * 1e4),
        "median_gross_bps": float(g.median() * 1e4),
        "avg_net_bps": float(events["net"].mean() * 1e4),
        "hit_rate": float((g > 0).mean()),
        "gross_t": t_stat,
        "long_share": float((events["direction"] > 0).mean()),
    }


def daily_metrics(series: pd.Series) -> dict:
    m = metrics(series)
    return {
        "days": m["days"],
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
    }


def build_book(
    panels: dict[str, pd.DataFrame], hold: int, cost_per_side: float, rng: np.random.Generator | None = None
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Return per-symbol event tables and the daily portfolio PnL series."""
    events_by_symbol = {}
    pnl_by_symbol = {}
    for symbol, frame in panels.items():
        z = zscore(frame["premium"])
        if rng is not None:
            z = pd.Series(rng.permutation(z.to_numpy()), index=z.index)
        ev = event_returns(frame, z, hold, cost_per_side)
        events_by_symbol[symbol] = ev
        pos = positions_from_events(frame, ev)
        pnl_by_symbol[symbol] = hourly_pnl(frame, pos, cost_per_side)
    return events_by_symbol, portfolio_daily(pnl_by_symbol)


def placebo_pvalue(
    panels: dict[str, pd.DataFrame], hold: int, actual_sharpe: float, rng: np.random.Generator
) -> dict:
    sharpes = []
    for _ in range(PLACEBO_DRAWS):
        _, port = build_book(panels, hold, COST_PER_SIDE, rng=rng)
        disc = slice_period(port, DISCOVERY_START, DISCOVERY_END)
        sharpes.append(daily_metrics(disc)["sharpe"])
    sharpes = np.array([s for s in sharpes if np.isfinite(s)])
    if len(sharpes) == 0:
        return {"p_value": float("nan"), "placebo_mean_sharpe": float("nan")}
    return {
        "p_value": float((sharpes >= actual_sharpe).mean()),
        "placebo_mean_sharpe": float(sharpes.mean()),
        "placebo_p95_sharpe": float(np.percentile(sharpes, 95)),
    }


def run_panel(name: str, num_kind: str, den_kind: str, rng: np.random.Generator) -> dict:
    panels = {s: load_panel(s, num_kind, den_kind) for s in SYMBOLS}
    out = {"premium": name, "primary_hold_hours": PRIMARY_HOLD}

    hold_results = {}
    for hold in [PRIMARY_HOLD] + SECONDARY_HOLDS:
        events_by_symbol, port = build_book(panels, hold, COST_PER_SIDE)
        all_events = (
            pd.concat(events_by_symbol.values(), ignore_index=True)
            if any(len(e) for e in events_by_symbol.values())
            else pd.DataFrame(columns=["signal_time", "entry_time", "exit_time", "z", "direction", "gross", "net"])
        )
        disc_events = all_events[
            (all_events["signal_time"] >= DISCOVERY_START)
            & (all_events["signal_time"] < DISCOVERY_END)
        ]
        hold_events = all_events[
            (all_events["signal_time"] >= DISCOVERY_END)
            & (all_events["signal_time"] < HOLDOUT_END)
        ]
        disc = slice_period(port, DISCOVERY_START, DISCOVERY_END)
        hold_s = slice_period(port, DISCOVERY_END, HOLDOUT_END)
        result = {
            "discovery": {**event_summary(disc_events, "discovery"), **daily_metrics(disc)},
            "holdout": {**event_summary(hold_events, "holdout"), **daily_metrics(hold_s)},
            "discovery_sharpe": daily_metrics(disc)["sharpe"],
            "holdout_sharpe": daily_metrics(hold_s)["sharpe"],
            "stitched_max_drawdown": metrics(port)["max_drawdown"],
        }
        if hold == PRIMARY_HOLD:
            result["placebo"] = placebo_pvalue(panels, hold, result["discovery_sharpe"], rng)
            result["events_by_symbol"] = {s: int(len(e)) for s, e in events_by_symbol.items()}
            out["primary"] = result
        else:
            hold_results[f"{hold}h"] = result
    out["secondary"] = hold_results

    events4 = {}
    for symbol, frame in panels.items():
        z = zscore(frame["premium"])
        events4[symbol] = event_returns(frame, z, PRIMARY_HOLD, 4.0 / 10_000.0)
    all4 = pd.concat(events4.values(), ignore_index=True)
    disc4 = all4[
        (all4["signal_time"] >= DISCOVERY_START) & (all4["signal_time"] < DISCOVERY_END)
    ]
    out["primary_at_4bps"] = event_summary(disc4, "discovery_4bps")
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    payload = {
        "hypothesis": "H10 perp/index premium extreme fade, time-series 1h",
        "universe": SYMBOLS,
        "excluded": {"SOLUSDT": "index corrupted around 2022-11-09 FTX crash"},
        "params": {
            "z_window_hours": Z_WINDOW,
            "z_entry": Z_ENTRY,
            "primary_hold_hours": PRIMARY_HOLD,
            "secondary_holds": SECONDARY_HOLDS,
            "cost_per_side_bps": COST_PER_SIDE * 1e4,
        },
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "holdout": ["2025-01-01", "2026-11-26"],
        "results": {},
    }
    for name, (num_kind, den_kind) in PREMIUMS.items():
        print(f"=== {name} ===")
        res = run_panel(name, num_kind, den_kind, rng)
        payload["results"][name] = res
        print(json.dumps(res["primary"], indent=2, default=str))
        print("secondary:", json.dumps(res["secondary"], indent=2, default=str))
        print("at 4bps:", json.dumps(res["primary_at_4bps"], indent=2, default=str))

    primary = payload["results"]["perp_index"]["primary"]
    gates = {
        "events>=100": primary["discovery"].get("events", 0) >= 100,
        "discovery_sharpe>=0.50": primary["discovery_sharpe"] >= 0.50,
        "holdout_sharpe>=0.0": primary["holdout_sharpe"] >= 0.0,
        "max_dd>=-30%": primary["stitched_max_drawdown"] >= -0.30,
        "placebo_p<=0.05": primary["placebo"]["p_value"] <= 0.05,
    }
    payload["gates"] = gates
    payload["verdict"] = "PASS" if all(gates.values()) else "KILL"
    (RESULTS_DIR / "h10_premium_fade.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print("gates:", json.dumps(gates, indent=2))
    print("verdict:", payload["verdict"])


if __name__ == "__main__":
    main()

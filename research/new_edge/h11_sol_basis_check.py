"""H11: held-out-asset check of the H10 72h mark/index premium fade.

Why this exists:
    H10's primary (8h perp/index) failed, but its *secondary* 72h mark/index
    panel was the only variant with positive net expectancy in both discovery
    (+9.3 bps/event) and holdout (+5.5 bps/event).  That was a diagnostic, not
    a pre-registered primary.  Rather than promote it on the same BTC/ETH
    data (which would be tuning on the holdout), this trial freezes the exact
    72h spec and tests it on a different asset, SOLUSDT, which played no part
    in selecting it.

Frozen spec (identical to H10's 72h mark/index panel):
    * premium_t = mark_close_t / index_close_t - 1
    * z_t = (premium_t - rolling_mean_720h) / rolling_std_720h, min 168 obs
    * event when |z_t| >= 2.0 at the close of bar t
    * enter at the OPEN of bar t+1, hold exactly 72 hours, fade the premium
    * non-overlapping, 8 bps per side (16 bps round trip)
    * SOL sample: 2021-04-01 .. 2026-11-26 (liquid period)
    * excluded: 2022-11-07 .. 2022-11-15 (FTX crash; SOL index prints
      -2000 bps vs mark, a data artifact)

Gates (any failure = KILL):
    * events >= 100
    * avg net bps per event > 0
    * net Sharpe >= 0.30 (daily position backtest)
    * max drawdown >= -40%
    * placebo p-value <= 0.05 (shuffle z, 200 draws)
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

import research.new_edge.h10_premium_fade as h10
from research.new_edge.h02_funding_reversal import metrics

SYMBOL = "SOLUSDT"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HOLD = 72
COST_PER_SIDE = 8.0 / 10_000.0
SAMPLE_START = pd.Timestamp("2021-04-01", tz="UTC")
SAMPLE_END = pd.Timestamp("2026-11-27", tz="UTC")
EXCLUDE_START = pd.Timestamp("2022-11-07", tz="UTC")
EXCLUDE_END = pd.Timestamp("2022-11-15", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
PLACEBO_DRAWS = 200
SEED = 20261127


def clean_frame() -> pd.DataFrame:
    frame = h10.load_panel(SYMBOL, "mark_1h", "index_1h")
    frame = frame[(frame.index >= SAMPLE_START) & (frame.index < SAMPLE_END)]
    keep = ~((frame.index >= EXCLUDE_START) & (frame.index < EXCLUDE_END))
    return frame[keep].copy()


def run_events(frame: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    ev = h10.event_returns(frame, z, HOLD, COST_PER_SIDE)
    if ev.empty:
        return ev
    # Drop any event whose holding window spans a removed bar (data gap).
    span_ok = (ev["exit_time"] - ev["entry_time"]) == pd.Timedelta(hours=HOLD)
    return ev[span_ok].reset_index(drop=True)


def book(frame: pd.DataFrame, z: pd.Series) -> pd.Series:
    ev = run_events(frame, z)
    pos = h10.positions_from_events(frame, ev)
    return h10.portfolio_daily({SYMBOL: h10.hourly_pnl(frame, pos, COST_PER_SIDE)})


def placebo(frame: pd.DataFrame, actual: float, rng: np.random.Generator) -> dict:
    z = h10.zscore(frame["premium"])
    sharpes = []
    for _ in range(PLACEBO_DRAWS):
        shuffled = pd.Series(rng.permutation(z.to_numpy()), index=z.index)
        port = book(frame, shuffled)
        sharpes.append(h10.daily_metrics(port)["sharpe"])
    sharpes = np.array([s for s in sharpes if np.isfinite(s)])
    return {
        "p_value": float((sharpes >= actual).mean()) if len(sharpes) else float("nan"),
        "placebo_mean_sharpe": float(sharpes.mean()) if len(sharpes) else float("nan"),
        "placebo_p95_sharpe": float(np.percentile(sharpes, 95)) if len(sharpes) else float("nan"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = clean_frame()
    z = h10.zscore(frame["premium"])
    ev = run_events(frame, z)
    port = book(frame, z)

    disc = h10.slice_period(port, SAMPLE_START, DISCOVERY_END)
    hold = h10.slice_period(port, DISCOVERY_END, SAMPLE_END)
    ev_disc = ev[(ev["signal_time"] >= SAMPLE_START) & (ev["signal_time"] < DISCOVERY_END)]
    ev_hold = ev[(ev["signal_time"] >= DISCOVERY_END) & (ev["signal_time"] < SAMPLE_END)]

    rng = np.random.default_rng(SEED)
    overall_metrics = h10.daily_metrics(port)
    payload = {
        "hypothesis": "H11 held-out-asset check of the H10 72h mark/index premium fade",
        "symbol": SYMBOL,
        "sample": [str(SAMPLE_START.date()), str(SAMPLE_END.date())],
        "excluded": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "hold_hours": HOLD,
        "cost_per_side_bps": COST_PER_SIDE * 1e4,
        "overall": {
            **h10.event_summary(ev, "overall"),
            **overall_metrics,
            "stitched_max_drawdown": metrics(port)["max_drawdown"],
        },
        "discovery": {**h10.event_summary(ev_disc, "discovery"), **h10.daily_metrics(disc)},
        "holdout": {**h10.event_summary(ev_hold, "holdout"), **h10.daily_metrics(hold)},
        "placebo": placebo(frame, overall_metrics["sharpe"], rng),
        "yearly_sharpe": {
            str(y): h10.daily_metrics(port[port.index.year == y])["sharpe"]
            for y in sorted(port.index.year.unique())
        },
    }

    # 4 bps best-case check.
    ev4 = run_events(frame, z)
    if not ev4.empty:
        ev4 = ev4.copy()
        ev4["net"] = ev4["gross"] - 2.0 * (4.0 / 10_000.0)
        payload["at_4bps"] = h10.event_summary(ev4, "overall_4bps")

    o = payload["overall"]
    gates = {
        "events>=100": o.get("events", 0) >= 100,
        "avg_net_bps>0": o.get("avg_net_bps", -1) > 0,
        "sharpe>=0.30": o.get("sharpe", -1) >= 0.30,
        "max_dd>=-40%": o.get("max_drawdown", -1) >= -0.40,
        "placebo_p<=0.05": payload["placebo"]["p_value"] <= 0.05,
    }
    payload["gates"] = gates
    payload["verdict"] = "PASS" if all(gates.values()) else "KILL"

    (RESULTS_DIR / "h11_sol_basis_check.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

"""H12: BTC/ETH/SOL pairs mean-reversion (rolling-beta log spread, 1h).

Pre-registered before running.

Hypothesis:
    BTC, ETH and SOL share a common crypto factor but their relative prices
    periodically dislocate (alt seasons, ETF flows, single-name shocks).  A
    rolling-beta log spread that stretches to 2 standard deviations should
    partly revert.  Unlike the 1-12 bps intraday signals in H1/H4/H9/H10, the
    dislocations here are hundreds of bps, so a 32 bps round trip (two legs)
    can be paid if the reversion is real.

Protocol (frozen):
    * pairs: BTC/ETH, BTC/SOL, ETH/SOL (Binance USD-M perps, 1h)
    * log prices; beta from a 720h rolling OLS of log(A) on log(B),
      min 168h, clipped to [0.2, 3.0]
    * residual_t = log(A)_t - beta_t * log(B)_t
    * z_t = 720h rolling z-score of residual, min 168h
    * event when |z_t| >= 2.0 at the close of bar t
    * enter at the OPEN of bar t+1; fade: short A / long beta*B when z>0
    * exit at the open after |z| <= 0.5, or after 168 hours (7 days),
      whichever comes first
    * non-overlapping within a pair
    * cost 8 bps per side per leg (4 legs round trip = 32 bps)
    * discovery: pair start .. 2024-12-31
    * holdout:   2025-01-01 .. 2026-11-26  (loaded once, never tuned on)

Kill gates (any failure = KILL):
    * discovery events >= 100
    * discovery net Sharpe >= 0.50 (daily Sharpe, sqrt(365))
    * holdout net Sharpe >= 0.0
    * max drawdown (stitched) >= -30%
    * placebo p-value <= 0.05 (shuffle z within pair, 200 draws)
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

PAIRS = [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT"), ("ETHUSDT", "SOLUSDT")]
BETA_WINDOW = 720
Z_WINDOW = 720
MIN_PERIODS = 168
Z_ENTRY = 2.0
Z_EXIT = 0.5
MAX_HOLD = 168
BETA_BOUNDS = (0.2, 3.0)
COST_PER_SIDE = 8.0 / 10_000.0
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-27", tz="UTC")
PLACEBO_DRAWS = 200
SEED = 20261127


def load_ohlc(symbol: str) -> pd.DataFrame:
    frame = pd.read_parquet(DATA_DIR / f"{symbol}_perp_1h.parquet")
    frame = frame[["open", "close"]].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated()].sort_index()


def pair_frame(a: str, b: str) -> pd.DataFrame:
    fa, fb = load_ohlc(a), load_ohlc(b)
    frame = pd.DataFrame(
        {
            "open_a": fa["open"],
            "open_b": fb["open"],
            "close_a": fa["close"],
            "close_b": fb["close"],
        }
    ).dropna()
    return frame


def signal(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    la = np.log(frame["close_a"])
    lb = np.log(frame["close_b"])
    cov = la.rolling(BETA_WINDOW, min_periods=MIN_PERIODS).cov(lb)
    var = lb.rolling(BETA_WINDOW, min_periods=MIN_PERIODS).var()
    beta = (cov / var.replace(0.0, np.nan)).clip(*BETA_BOUNDS)
    resid = la - beta * lb
    mean = resid.rolling(Z_WINDOW, min_periods=MIN_PERIODS).mean()
    std = resid.rolling(Z_WINDOW, min_periods=MIN_PERIODS).std()
    z = (resid - mean) / std.replace(0.0, np.nan)
    return z, beta


def find_events(frame: pd.DataFrame, z: pd.Series, beta: pd.Series) -> pd.DataFrame:
    n = len(frame)
    zz = z.to_numpy()
    bb = beta.to_numpy()
    stamps = frame.index
    rows = []
    i = MIN_PERIODS
    while i < n - 2:
        zv = zz[i]
        if not np.isfinite(zv) or abs(zv) < Z_ENTRY or not np.isfinite(bb[i]):
            i += 1
            continue
        entry_i = i + 1
        last_i = min(entry_i + MAX_HOLD, n - 1)
        exit_i = last_i
        for j in range(entry_i, last_i):
            zj = zz[j]
            if np.isfinite(zj) and abs(zj) <= Z_EXIT:
                exit_i = j + 1
                break
        if exit_i <= entry_i:
            i = entry_i
            continue
        direction = -1.0 if zv > 0 else 1.0  # fade the residual
        rows.append(
            {
                "signal_time": stamps[i],
                "entry_time": stamps[entry_i],
                "exit_time": stamps[exit_i],
                "z": zv,
                "beta": bb[i],
                "direction": direction,
            }
        )
        i = exit_i
    return pd.DataFrame(rows)


def leg_positions(frame: pd.DataFrame, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pos_a = np.zeros(len(frame))
    pos_b = np.zeros(len(frame))
    index = frame.index
    for entry_time, exit_time, direction, b in zip(
        events["entry_time"], events["exit_time"], events["direction"], events["beta"]
    ):
        i0 = index.get_loc(entry_time)
        i1 = index.get_loc(exit_time)
        pos_a[i0:i1] = direction
        pos_b[i0:i1] = -direction * b
    return pos_a, pos_b


def pair_pnl(frame: pd.DataFrame, pos_a: np.ndarray, pos_b: np.ndarray) -> pd.Series:
    fwd_a = (frame["open_a"].shift(-1) / frame["open_a"] - 1.0).to_numpy()
    fwd_b = (frame["open_b"].shift(-1) / frame["open_b"] - 1.0).to_numpy()
    turn_a = np.abs(np.diff(np.concatenate([[0.0], pos_a])))
    turn_b = np.abs(np.diff(np.concatenate([[0.0], pos_b])))
    pnl = pos_a * fwd_a + pos_b * fwd_b - (turn_a + turn_b) * COST_PER_SIDE
    return pd.Series(pnl[:-1], index=frame.index[:-1])


def daily(series: pd.Series) -> pd.Series:
    out = series.resample("1D").sum(min_count=1).fillna(0.0)
    out.index.freq = None
    return out


def build(pairs: list[tuple[str, str]], rng: np.random.Generator | None = None):
    per_pair = {}
    events = {}
    for a, b in pairs:
        frame = pair_frame(a, b)
        z, beta = signal(frame)
        if rng is not None:
            z = pd.Series(rng.permutation(z.to_numpy()), index=z.index)
        ev = find_events(frame, z, beta)
        pos_a, pos_b = leg_positions(frame, ev)
        per_pair[f"{a}/{b}"] = daily(pair_pnl(frame, pos_a, pos_b))
        events[f"{a}/{b}"] = ev
    port = pd.concat(per_pair, axis=1).fillna(0.0).mean(axis=1)
    return events, per_pair, port


def event_stats(ev: pd.DataFrame, frame: pd.DataFrame, label: str) -> dict:
    if ev.empty:
        return {"label": label, "events": 0}
    # Reconstruct gross per event from the legs for reporting.
    index = frame.index
    gross = []
    for entry_time, exit_time, direction, b in zip(
        ev["entry_time"], ev["exit_time"], ev["direction"], ev["beta"]
    ):
        i0 = index.get_loc(entry_time)
        i1 = index.get_loc(exit_time)
        ra = frame["open_a"].iloc[i1] / frame["open_a"].iloc[i0] - 1.0
        rb = frame["open_b"].iloc[i1] / frame["open_b"].iloc[i0] - 1.0
        gross.append(direction * ra - direction * b * rb)
    g = np.array(gross)
    n = len(g)
    t = float(g.mean() / (g.std(ddof=1) / np.sqrt(n))) if n > 1 and g.std(ddof=1) > 0 else float("nan")
    return {
        "label": label,
        "events": int(n),
        "avg_gross_bps": float(g.mean() * 1e4),
        "median_gross_bps": float(np.median(g) * 1e4),
        "avg_net_bps": float((g.mean() - 4.0 * COST_PER_SIDE) * 1e4),
        "hit_rate": float((g > 0).mean()),
        "gross_t": t,
        "avg_hold_hours": float(((ev["exit_time"] - ev["entry_time"]).dt.total_seconds() / 3600).mean()),
        "long_a_share": float((ev["direction"] > 0).mean()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    events, per_pair, port = build(PAIRS)
    frames = {f"{a}/{b}": pair_frame(a, b) for a, b in PAIRS}

    disc = port[(port.index < DISCOVERY_END)]
    hold = port[(port.index >= DISCOVERY_END) & (port.index < HOLDOUT_END)]

    all_events = []
    for name, ev in events.items():
        if ev.empty:
            continue
        frame = frames[name]
        start = frame.index.min()
        ev_disc = ev[(ev["signal_time"] >= start) & (ev["signal_time"] < DISCOVERY_END)]
        ev_hold = ev[(ev["signal_time"] >= DISCOVERY_END) & (ev["signal_time"] < HOLDOUT_END)]
        all_events.append(
            {
                "pair": name,
                "discovery": event_stats(ev_disc, frame, "discovery"),
                "holdout": event_stats(ev_hold, frame, "holdout"),
            }
        )

    # Placebo on the discovery Sharpe.
    actual_disc_sharpe = metrics(disc)["sharpe"]
    sharpes = []
    for _ in range(PLACEBO_DRAWS):
        _, _, shuffled_port = build(PAIRS, rng=rng)
        sd = shuffled_port[shuffled_port.index < DISCOVERY_END]
        sharpes.append(metrics(sd)["sharpe"])
    sharpes = np.array([s for s in sharpes if np.isfinite(s)])

    payload = {
        "hypothesis": "H12 BTC/ETH/SOL pairs mean-reversion, rolling-beta log spread 1h",
        "pairs": [f"{a}/{b}" for a, b in PAIRS],
        "params": {
            "beta_window_hours": BETA_WINDOW,
            "z_window_hours": Z_WINDOW,
            "z_entry": Z_ENTRY,
            "z_exit": Z_EXIT,
            "max_hold_hours": MAX_HOLD,
            "cost_per_side_per_leg_bps": COST_PER_SIDE * 1e4,
            "round_trip_bps": 4 * COST_PER_SIDE * 1e4,
        },
        "discovery": [str(port.index.min().date()), "2024-12-31"],
        "holdout": ["2025-01-01", "2026-11-26"],
        "overall": metrics(port),
        "discovery_metrics": metrics(disc),
        "holdout_metrics": metrics(hold),
        "stitched_max_drawdown": metrics(port)["max_drawdown"],
        "events": all_events,
        "placebo": {
            "p_value": float((sharpes >= actual_disc_sharpe).mean()) if len(sharpes) else float("nan"),
            "placebo_mean_sharpe": float(sharpes.mean()) if len(sharpes) else float("nan"),
            "placebo_p95_sharpe": float(np.percentile(sharpes, 95)) if len(sharpes) else float("nan"),
        },
    }

    disc_events = sum(e["discovery"].get("events", 0) for e in all_events)
    gates = {
        "events>=100": disc_events >= 100,
        "discovery_sharpe>=0.50": metrics(disc)["sharpe"] >= 0.50,
        "holdout_sharpe>=0.0": metrics(hold)["sharpe"] >= 0.0,
        "max_dd>=-30%": metrics(port)["max_drawdown"] >= -0.30,
        "placebo_p<=0.05": payload["placebo"]["p_value"] <= 0.05,
    }
    payload["gates"] = gates
    payload["verdict"] = "PASS" if all(gates.values()) else "KILL"
    (RESULTS_DIR / "h12_pairs.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

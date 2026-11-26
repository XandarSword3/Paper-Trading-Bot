"""H7 pilot: 12-month time-series momentum across the perp basket.

Hypothesis (pre-registered before running):
    Crypto trends persist at the annual horizon.  Coins that are up over the
    past year keep outperforming over the next month; coins that are down keep
    underperforming.  A vol-scaled long/short basket of those signs harvests
    the trend premium with very low turnover, which is the only way a price
    signal survives an 8 bps/side cost model.

Protocol:
    * discovery only: 2020-01-01 .. 2024-12-31
    * 36 Binance USDT perps, daily closes at 00:00 UTC
    * signal = sign of the 365-day return ending 5 days ago
    * rebalance on the first trading day of each month
    * weight = sign / 30-day realized vol, normalised to 1.0 gross, 10% cap
    * 8 bps per side on turnover (primary), 4 bps best case, 16 bps stress
    * holdout (2025-01-01 .. 2026-10-31) is never loaded here
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

from research.new_edge.h02_funding_reversal import (
    DISCOVERY_END,
    DISCOVERY_START,
    RESULTS_DIR,
    beta_to_btc,
    load_prices,
)
from research.new_edge.h03_vol_breakout import metrics
from research.new_edge.h05_cross_sectional_momentum import subperiods, yearly

COST_PER_SIDE = 8.0 / 10_000.0
PRIMARY = {"lookback": 365, "skip": 5, "vol_window": 30, "cap": 0.10}


def month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    out = []
    last = None
    for date in index:
        key = (date.year, date.month)
        if key != last:
            out.append(date)
            last = key
    return out


def build_weights(prices: pd.DataFrame, **cfg) -> pd.DataFrame:
    past = prices.shift(cfg["skip"]) / prices.shift(cfg["lookback"]) - 1.0
    signal = np.sign(past)
    vol = prices.pct_change().rolling(cfg["vol_window"], min_periods=15).std() * np.sqrt(365.0)

    dates = month_starts(prices.index)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for date in dates:
        sig = signal.loc[date]
        v = vol.loc[date]
        valid = sig.notna() & v.notna() & (v > 1e-4)
        if valid.sum() < 5:
            weights.loc[date, :] = 0.0
            continue
        raw = (sig[valid] / v[valid]).clip(-3.0, 3.0)
        raw = raw / raw.abs().sum()
        raw = raw.clip(-cfg["cap"], cfg["cap"])
        raw = raw / raw.abs().sum()
        weights.loc[date, :] = 0.0
        weights.loc[date, raw.index] = raw
    return weights.ffill().fillna(0.0)


def run_book(weights: pd.DataFrame, prices: pd.DataFrame, cost_per_side: float):
    fwd = prices.shift(-1) / prices - 1.0
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_per_side
    net = gross - cost
    mask = (net.index >= DISCOVERY_START) & (net.index < DISCOVERY_END)
    return gross[mask], cost[mask], net[mask], turnover[mask]


def placebo(prices: pd.DataFrame, cfg: dict, cost_per_side: float, actual_sharpe: float, draws: int = 200) -> dict:
    rng = np.random.default_rng(20261126)
    dates = month_starts(prices.index)
    vol = prices.pct_change().rolling(cfg["vol_window"], min_periods=15).std() * np.sqrt(365.0)
    sharpes = []
    for _ in range(draws):
        weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for date in dates:
            v = vol.loc[date]
            valid = v.notna() & (v > 1e-4)
            if valid.sum() < 5:
                weights.loc[date, :] = 0.0
                continue
            signs = pd.Series(rng.choice([-1.0, 1.0], size=int(valid.sum())), index=v[valid].index)
            raw = (signs / v[valid]).clip(-3.0, 3.0)
            raw = raw / raw.abs().sum()
            raw = raw.clip(-cfg["cap"], cfg["cap"])
            raw = raw / raw.abs().sum()
            weights.loc[date, :] = 0.0
            weights.loc[date, raw.index] = raw
        weights = weights.ffill().fillna(0.0)
        _, _, net, _ = run_book(weights, prices, cost_per_side)
        sharpes.append(metrics(net)["sharpe"])
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
    prices = load_prices()
    print(f"prices {prices.shape}")

    weights = build_weights(prices, **PRIMARY)
    gross, cost, net, turnover = run_book(weights, prices, COST_PER_SIDE)

    overall = {
        **metrics(net),
        "gross_sharpe": metrics(gross)["sharpe"],
        "gross_cagr": metrics(gross)["cagr"],
        "avg_cost_bps_per_day": float(cost.mean() * 10_000.0),
        "avg_turnover_per_rebalance": float(turnover[turnover > 0].mean()),
        "rebalances": int((turnover > 0).sum()),
    }
    overall.update(beta_to_btc(net, prices))

    sensitivity = []
    for lookback in (180, 365, 540):
        for vol_window in (20, 30, 60):
            cfg = {**PRIMARY, "lookback": lookback, "vol_window": vol_window}
            w = build_weights(prices, **cfg)
            g, c, n, t = run_book(w, prices, COST_PER_SIDE)
            sensitivity.append(
                {"lookback": lookback, "vol_window": vol_window, **metrics(n), "gross_sharpe": metrics(g)["sharpe"]}
            )

    _, _, net_4bps, _ = run_book(weights, prices, 4.0 / 10_000.0)
    _, _, net_16bps, _ = run_book(weights, prices, 16.0 / 10_000.0)

    # Long/flat variant: same signal, shorts dropped.
    w_long = weights.clip(lower=0.0)
    w_long = w_long.div(w_long.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    _, _, net_long, _ = run_book(w_long, prices, COST_PER_SIDE)

    null = placebo(prices, PRIMARY, COST_PER_SIDE, overall["sharpe"])

    payload = {
        "hypothesis": "H7 12-month time-series momentum, monthly rebalance",
        "primary": PRIMARY,
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "cost_per_side_bps": COST_PER_SIDE * 10_000.0,
        "overall": overall,
        "yearly": yearly(net),
        "subperiods": subperiods(net),
        "sensitivity": sensitivity,
        "cost_ladder": {"4bps": metrics(net_4bps), "8bps": metrics(net), "16bps": metrics(net_16bps)},
        "long_flat_variant": metrics(net_long),
        "placebo": null,
    }
    (RESULTS_DIR / "h07_tsmom.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("overall", "yearly", "subperiods", "cost_ladder", "long_flat_variant", "placebo")}, indent=2))


if __name__ == "__main__":
    main()

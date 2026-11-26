"""H2b pilot: funding-extreme reversal with a lower-turnover holding period.

H2 found a real gross effect (gross Sharpe 1.54) that daily rebalancing destroyed
with ~105% daily turnover (8.4 bps/day of cost).  This pre-registered variant
keeps the same signal and construction but only refreshes the book every three
days, cutting turnover and testing whether the net edge survives.

Primary: hold_days = 3.  Sensitivity: 1, 5, 7.
Discovery only: 2020-01-01 .. 2024-12-31.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.new_edge.h02_funding_reversal import (
    COST_PER_SIDE,
    DISCOVERY_END,
    DISCOVERY_START,
    MIN_NAMES,
    RESULTS_DIR,
    beta_to_btc,
    load_funding,
    load_prices,
    metrics,
    weights_from_rank,
    zscores,
)

HOLD_DAYS = 3
SENSITIVITY_HOLDS = (1, 3, 5, 7)


def hold_weights(weights: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    if hold_days <= 1:
        return weights
    held = weights.copy()
    rows = np.arange(len(weights))
    refresh = (rows % hold_days) == 0
    held.loc[~refresh, :] = np.nan
    return held.ffill().fillna(0.0)


def run_book(
    weights: pd.DataFrame, prices: pd.DataFrame, hold_days: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fwd = prices.shift(-1) / prices - 1.0
    held = hold_weights(weights, hold_days)
    gross = (held * fwd).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    turnover = turnover.reindex(gross.index).fillna(0.0)
    cost = turnover * COST_PER_SIDE
    net = gross - cost
    mask = (net.index >= DISCOVERY_START) & (net.index < DISCOVERY_END)
    return gross[mask], cost[mask], net[mask]


def yearly(series: pd.Series) -> dict:
    out = {}
    for year in range(2020, 2025):
        subset = series[
            (series.index >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & (series.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ]
        out[str(year)] = metrics(subset)
    return out


def placebo_hold(
    score: pd.DataFrame, prices: pd.DataFrame, actual_sharpe: float, hold_days: int, draws: int = 200
) -> dict:
    rng = np.random.default_rng(20261126)
    sharpes: list[float] = []
    for _ in range(draws):
        shuffled = score.copy()
        for date in shuffled.index:
            row = shuffled.loc[date].dropna()
            if len(row) < MIN_NAMES:
                continue
            shuffled.loc[date, row.index] = rng.permutation(row.to_numpy())
        weights, _ = weights_from_rank(shuffled, prices)
        _, _, net = run_book(weights, prices, hold_days)
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
    funding = load_funding()
    prices = load_prices()
    score = zscores(funding)
    weights, spreads = weights_from_rank(score, prices)

    gross, cost, net = run_book(weights, prices, HOLD_DAYS)
    overall = {
        **metrics(net),
        "gross_sharpe": metrics(gross)["sharpe"],
        "gross_cagr": metrics(gross)["cagr"],
        "avg_cost_bps_per_day": float(cost.mean() * 10_000.0),
        "avg_turnover": float(
            hold_weights(weights, HOLD_DAYS).diff().abs().sum(axis=1)[
                (weights.index >= DISCOVERY_START) & (weights.index < DISCOVERY_END)
            ].mean()
        ),
    }
    overall.update(beta_to_btc(net, prices))

    sensitivity = {}
    for hold in SENSITIVITY_HOLDS:
        g, c, n = run_book(weights, prices, hold)
        sensitivity[str(hold)] = {
            **metrics(n),
            "gross_sharpe": metrics(g)["sharpe"],
            "avg_cost_bps_per_day": float(c.mean() * 10_000.0),
        }

    null = placebo_hold(score, prices, overall["sharpe"], HOLD_DAYS)

    payload = {
        "hypothesis": "H2b funding-extreme reversal, low turnover",
        "primary_hold_days": HOLD_DAYS,
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "overall": overall,
        "yearly": yearly(net),
        "sensitivity": sensitivity,
        "placebo": null,
        "first_days": [
            {
                "date": idx.date().isoformat(),
                "gross": float(gross.loc[idx]),
                "cost": float(cost.loc[idx]),
                "net": float(net.loc[idx]),
            }
            for idx in net.index[:3]
        ],
    }
    (RESULTS_DIR / "h02b_funding_reversal_hold.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"overall": overall, "yearly": payload["yearly"], "sensitivity": sensitivity, "placebo": null}, indent=2))


if __name__ == "__main__":
    main()

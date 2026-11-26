"""H6: funding-crowding reversal, now including the funding cash flow.

H2 measured only price PnL and therefore ignored a direct cash flow: a book
that is short the highest-funding coins and long the lowest-funding coins
*collects* the funding differential every 8 hours.  H6 is the same signal and
construction as H2 with that cash flow added.  This is an accounting
correction, not a parameter change, so the H2 ranking stays frozen.

Protocol (unchanged from H2 except the added cash flow):
    * discovery only: 2020-01-01 .. 2024-12-31
    * signal at 00:00 UTC from the trailing 30-day z-score of that day's
      funding print
    * long bottom 20%, short top 20%, equal weight, 1.0x gross, dollar-neutral
    * hold one day, rebalance daily
    * funding cash flow counted only from the 08:00 and 16:00 prints of the
      holding day (the 00:00 print is skipped so nothing depends on a
      same-instant fill)
    * 8 bps per side on turnover
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
    COST_PER_SIDE,
    DISCOVERY_END,
    DISCOVERY_START,
    FUNDING_DIR,
    RESULTS_DIR,
    beta_to_btc,
    load_funding,
    load_prices,
    metrics,
    weights_from_rank,
    zscores,
)
from research.new_edge.h05_cross_sectional_momentum import subperiods, yearly


def load_funding_cash() -> pd.DataFrame:
    """Funding paid by a long on day d: 08:00 + 16:00 prints, per symbol."""
    frames = {}
    for path in sorted(FUNDING_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.drop_duplicates("timestamp")
        frame["day"] = frame["timestamp"].dt.floor("1D")
        frame["hour"] = frame["timestamp"].dt.hour
        cash = (
            frame[frame["hour"].isin([8, 16])]
            .groupby("day")["funding_rate"]
            .sum()
            .sort_index()
        )
        cash.index.freq = None
        frames[path.stem] = cash
    return pd.concat(frames, axis=1).sort_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    funding = load_funding()
    prices = load_prices()
    cash = load_funding_cash()
    print(f"funding {funding.shape}, prices {prices.shape}, cash {cash.shape}")

    score = zscores(funding)
    weights, spreads = weights_from_rank(score, prices)

    fwd = prices.shift(-1) / prices - 1.0
    gross = (weights * fwd).sum(axis=1).reindex(weights.index)

    cash_aligned = cash.reindex(index=weights.index, columns=weights.columns)
    funding_pnl = (
        -(weights * cash_aligned).sum(axis=1, min_count=1)
        .reindex(weights.index)
        .fillna(0.0)
    )

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * COST_PER_SIDE

    price_net = gross - cost
    net = gross + funding_pnl - cost

    mask = (weights.index >= DISCOVERY_START) & (weights.index < DISCOVERY_END)
    gross, funding_pnl, cost, price_net, net = (
        series[mask] for series in (gross, funding_pnl, cost, price_net, net)
    )

    overall = {
        **metrics(net),
        "price_only_net_sharpe": metrics(price_net)["sharpe"],
        "price_only_net_cagr": metrics(price_net)["cagr"],
        "gross_sharpe": metrics(gross)["sharpe"],
        "gross_cagr": metrics(gross)["cagr"],
        "funding_pnl_ann": float(funding_pnl.mean() * 365.0),
        "funding_pnl_bps_per_day": float(funding_pnl.mean() * 10_000.0),
        "funding_pnl_sharpe": metrics(funding_pnl)["sharpe"],
        "avg_cost_bps_per_day": float(cost.mean() * 10_000.0),
        "avg_turnover": float(turnover[mask].mean()),
    }
    overall.update(beta_to_btc(net, prices))

    payload = {
        "hypothesis": "H6 funding-crowding reversal including funding cash flow",
        "primary": {"z_window": 90, "quantile": 0.20, "hold_days": 1, "funding_prints": [8, 16]},
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "cost_per_side_bps": COST_PER_SIDE * 10_000.0,
        "overall": overall,
        "yearly": yearly(net),
        "subperiods": subperiods(net),
    }
    (RESULTS_DIR / "h06_funding_cash.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("overall", "yearly", "subperiods")}, indent=2))


if __name__ == "__main__":
    main()

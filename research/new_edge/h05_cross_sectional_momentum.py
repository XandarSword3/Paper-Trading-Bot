"""H5 pilot: cross-sectional momentum.

Hypothesis (pre-registered before running):
    Crypto perps exhibit cross-sectional momentum.  Coins with the strongest
    trailing 4-week return keep outperforming over the next week, and the
    weakest keep underperforming.  A dollar-neutral long-winners /
    short-losers book harvests that spread with low market beta.

Protocol:
    * discovery only: 2020-01-01 .. 2024-12-31
    * 36 Binance USDT perps, daily closes at 00:00 UTC
    * score = 28-day return ending 1 day ago (skip the most recent day)
    * rebalance every Monday 00:00 UTC, hold 7 days
    * long top 20%, short bottom 20%, equal weight, 0.5 long / 0.5 short
    * 8 bps per side on turnover (primary), 4 bps as best-case sensitivity
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

COST_PER_SIDE = 8.0 / 10_000.0
PRIMARY = {"lookback": 28, "skip": 1, "quantile": 0.20, "rebalance_weekday": 0}
MIN_NAMES = 10


def momentum_score(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def rebalance_dates(index: pd.DatetimeIndex, weekday: int) -> list[pd.Timestamp]:
    return [date for date in index if date.weekday() == weekday]


def weights_at(
    score: pd.DataFrame, prices: pd.DataFrame, dates: list[pd.Timestamp], quantile: float
) -> pd.DataFrame:
    # Build a full row at every rebalance date (including zeros for names that
    # drop out of the book), then carry it forward.  Initialising with NaN and
    # forward-filling only between rebalances avoids the classic bug where a
    # dropped name's zero weight is treated as "no new information" and the old
    # position is silently held forever.
    weights = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for date in dates:
        row = score.loc[date].dropna()
        row = row[prices.loc[date, row.index].notna()]
        if len(row) < MIN_NAMES:
            weights.loc[date, :] = 0.0
            continue
        n = max(1, int(np.floor(len(row) * quantile)))
        ranked = row.sort_values()
        weights.loc[date, :] = 0.0
        weights.loc[date, ranked.index[:n]] = 0.5 / n
        weights.loc[date, ranked.index[-n:]] = -0.5 / n
    return weights.ffill().fillna(0.0)


def run_book(
    weights: pd.DataFrame, prices: pd.DataFrame, cost_per_side: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    fwd = prices.shift(-1) / prices - 1.0
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_per_side
    net = gross - cost
    mask = (net.index >= DISCOVERY_START) & (net.index < DISCOVERY_END)
    return gross[mask], cost[mask], net[mask], turnover[mask]


def legs(weights: pd.DataFrame, prices: pd.DataFrame) -> dict:
    fwd = prices.shift(-1) / prices - 1.0
    long_w = weights.clip(lower=0.0)
    short_w = weights.clip(upper=0.0)
    long_ret = (long_w * fwd).sum(axis=1)
    short_ret = (short_w * fwd).sum(axis=1)
    mask = (long_ret.index >= DISCOVERY_START) & (long_ret.index < DISCOVERY_END)
    return {
        "long_leg_ann": float(long_ret[mask].mean() * 365.0),
        "short_leg_ann": float(short_ret[mask].mean() * 365.0),
        "long_leg_sharpe": metrics(long_ret[mask])["sharpe"],
        "short_leg_sharpe": metrics(short_ret[mask])["sharpe"],
    }


def subperiods(series: pd.Series) -> dict:
    windows = {
        "2020_2021": ("2020-01-01", "2022-01-01"),
        "2022_2023": ("2022-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"),
    }
    out = {}
    for name, (start, end) in windows.items():
        subset = series[
            (series.index >= pd.Timestamp(start, tz="UTC"))
            & (series.index < pd.Timestamp(end, tz="UTC"))
        ]
        out[name] = metrics(subset)
    return out


def yearly(series: pd.Series) -> dict:
    out = {}
    for year in range(2020, 2025):
        subset = series[
            (series.index >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & (series.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ]
        out[str(year)] = metrics(subset)
    return out


def placebo(
    score: pd.DataFrame,
    prices: pd.DataFrame,
    dates: list[pd.Timestamp],
    quantile: float,
    cost_per_side: float,
    actual_sharpe: float,
    draws: int = 200,
) -> dict:
    rng = np.random.default_rng(20261126)
    sharpes: list[float] = []
    for _ in range(draws):
        shuffled = score.copy()
        for date in dates:
            row = shuffled.loc[date].dropna()
            row = row[prices.loc[date, row.index].notna()]
            if len(row) < MIN_NAMES:
                continue
            shuffled.loc[date, row.index] = rng.permutation(row.to_numpy())
        weights = weights_at(shuffled, prices, dates, quantile)
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
    print(f"prices {prices.shape}, {prices.index[0]} .. {prices.index[-1]}")

    score = momentum_score(prices, PRIMARY["lookback"], PRIMARY["skip"])
    dates = rebalance_dates(score.index, PRIMARY["rebalance_weekday"])
    weights = weights_at(score, prices, dates, PRIMARY["quantile"])
    gross, cost, net, turnover = run_book(weights, prices, COST_PER_SIDE)

    overall = {
        **metrics(net),
        "gross_sharpe": metrics(gross)["sharpe"],
        "gross_cagr": metrics(gross)["cagr"],
        "avg_cost_bps_per_day": float(cost.mean() * 10_000.0),
        "avg_daily_turnover": float(turnover.mean()),
        "rebalances": int(len(dates)),
        "avg_names": float(weights.abs().gt(0).sum(axis=1)[weights.index.isin(dates)].mean()),
    }
    overall.update(beta_to_btc(net, prices))
    overall.update(legs(weights, prices))

    sensitivity = []
    for lookback in (14, 28, 56, 90):
        for quantile in (0.20, 0.30):
            sc = momentum_score(prices, lookback, PRIMARY["skip"])
            w = weights_at(sc, prices, dates, quantile)
            g, c, n, t = run_book(w, prices, COST_PER_SIDE)
            sensitivity.append(
                {
                    "lookback": lookback,
                    "quantile": quantile,
                    **metrics(n),
                    "gross_sharpe": metrics(g)["sharpe"],
                    "avg_cost_bps_per_day": float(c.mean() * 10_000.0),
                }
            )

    # Best-case costs on the primary config.
    _, _, net_4bps, _ = run_book(weights, prices, 4.0 / 10_000.0)

    # Leave-one-symbol-out on the primary config.
    loo = {}
    for symbol in prices.columns:
        w = weights.copy()
        w[symbol] = 0.0
        _, _, n, _ = run_book(w, prices, COST_PER_SIDE)
        loo[symbol] = metrics(n)["sharpe"]
    loo_sorted = sorted(loo.items(), key=lambda kv: kv[1])

    null = placebo(
        score, prices, dates, PRIMARY["quantile"], COST_PER_SIDE, overall["sharpe"]
    )

    payload = {
        "hypothesis": "H5 cross-sectional momentum, weekly rebalance",
        "primary": PRIMARY,
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "cost_per_side_bps": COST_PER_SIDE * 10_000.0,
        "overall": overall,
        "yearly": yearly(net),
        "subperiods": subperiods(net),
        "sensitivity": sensitivity,
        "best_case_4bps": metrics(net_4bps),
        "leave_one_symbol_out": {
            "worst": loo_sorted[:5],
            "best": loo_sorted[-3:],
        },
        "placebo": null,
    }
    (RESULTS_DIR / "h05_cross_sectional_momentum.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("overall", "yearly", "subperiods", "best_case_4bps", "placebo")}, indent=2))


if __name__ == "__main__":
    main()

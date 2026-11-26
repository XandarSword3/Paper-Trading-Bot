"""H5 (corrected): cross-sectional momentum, long winners / short losers.

The first H5 implementation sorted the score ascending and bought the bottom
names, which is a *reversal* book.  That book lost (net Sharpe -1.00), and the
mirror -- the hypothesis as actually pre-registered -- is the profitable one.
This file is the corrected test.

Pre-registered hypothesis:
    Coins with the strongest trailing 4-week return keep outperforming over the
    next week, and the weakest keep underperforming.

Protocol:
    * discovery: 2020-01-01 .. 2024-12-31
    * 36 Binance USDT perps, daily closes at 00:00 UTC
    * score = 28-day return ending 1 day ago
    * rebalance every Monday 00:00 UTC, hold 7 days
    * long top 20%, short bottom 20%, dollar-neutral, 1.0 gross
    * primary weighting: equal weight within each leg
    * 8 bps per side on turnover
    * holdout: 2025-01-01 .. 2026-10-31, evaluated once for the primary only
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
    RESULTS_DIR,
    beta_to_btc,
    load_prices,
)
from research.new_edge.h03_vol_breakout import metrics
from research.new_edge.h05_cross_sectional_momentum import (
    COST_PER_SIDE,
    momentum_score,
    rebalance_dates,
)

PRIMARY = {"lookback": 28, "skip": 1, "quantile": 0.20, "weighting": "equal"}
DISCOVERY = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
MIN_NAMES = 10


def vol_30d(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().rolling(30, min_periods=15).std()


def build_weights(
    prices: pd.DataFrame, dates: list[pd.Timestamp], *, lookback: int, skip: int,
    quantile: float, weighting: str,
) -> pd.DataFrame:
    score = momentum_score(prices, lookback, skip)  # high = winner
    vol = vol_30d(prices)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for date in dates:
        row = score.loc[date].dropna()
        row = row[prices.loc[date, row.index].notna()]
        if len(row) < MIN_NAMES:
            weights.loc[date, :] = 0.0
            continue
        n = max(1, int(np.floor(len(row) * quantile)))
        ranked = row.sort_values()
        longs = ranked.index[-n:]   # winners
        shorts = ranked.index[:n]   # losers
        weights.loc[date, :] = 0.0
        if weighting == "equal":
            weights.loc[date, longs] = 0.5 / n
            weights.loc[date, shorts] = -0.5 / n
        else:
            for leg, sign in ((longs, 1.0), (shorts, -1.0)):
                inv = 1.0 / vol.loc[date, leg].replace(0.0, np.nan)
                inv = inv.fillna(inv.mean())
                weights.loc[date, leg] = sign * 0.5 * inv / inv.sum()
    return weights.ffill().fillna(0.0)


def run_book(weights: pd.DataFrame, prices: pd.DataFrame, window: tuple, cost_per_side: float):
    fwd = prices.shift(-1) / prices - 1.0
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * cost_per_side
    net = gross - cost
    mask = (net.index >= window[0]) & (net.index < window[1])
    return gross[mask], cost[mask], net[mask], turnover[mask]


def summarize(gross, cost, net, turnover) -> dict:
    out = {
        **metrics(net),
        "gross_sharpe": metrics(gross)["sharpe"],
        "gross_cagr": metrics(gross)["cagr"],
        "avg_cost_bps_per_day": float(cost.mean() * 10_000.0),
        "avg_daily_turnover": float(turnover.mean()),
    }
    return out


def yearly(series: pd.Series) -> dict:
    return {
        str(y): metrics(series[series.index.year == y]) for y in sorted(set(series.index.year))
    }


def placebo(prices, dates, cfg, window, actual_sharpe, draws=200) -> dict:
    rng = np.random.default_rng(20261126)
    score = momentum_score(prices, cfg["lookback"], cfg["skip"])
    vol = vol_30d(prices)
    sharpes = []
    for _ in range(draws):
        shuffled = score.copy()
        for date in dates:
            row = shuffled.loc[date].dropna()
            row = row[prices.loc[date, row.index].notna()]
            if len(row) < MIN_NAMES:
                continue
            shuffled.loc[date, row.index] = rng.permutation(row.to_numpy())
        # Reuse build_weights logic on the shuffled score by swapping it in.
        weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for date in dates:
            row = shuffled.loc[date].dropna()
            row = row[prices.loc[date, row.index].notna()]
            if len(row) < MIN_NAMES:
                weights.loc[date, :] = 0.0
                continue
            n = max(1, int(np.floor(len(row) * cfg["quantile"])))
            ranked = row.sort_values()
            weights.loc[date, :] = 0.0
            weights.loc[date, ranked.index[-n:]] = 0.5 / n
            weights.loc[date, ranked.index[:n]] = -0.5 / n
        weights = weights.ffill().fillna(0.0)
        _, _, net, _ = run_book(weights, prices, window, COST_PER_SIDE)
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
    dates = rebalance_dates(momentum_score(prices, 28, 1).index, 0)
    print(f"prices {prices.shape}, {len(dates)} rebalances")

    # --- Discovery: primary -------------------------------------------------
    w = build_weights(prices, dates, **PRIMARY)
    g, c, n, t = run_book(w, prices, DISCOVERY, COST_PER_SIDE)
    primary = summarize(g, c, n, t)
    primary.update(beta_to_btc(n, prices))

    # --- Discovery: sensitivity --------------------------------------------
    sensitivity = []
    for lookback in (14, 28, 56, 90):
        for quantile in (0.20, 0.30):
            for weighting in ("equal", "invvol"):
                cfg = {"lookback": lookback, "skip": 1, "quantile": quantile, "weighting": weighting}
                ww = build_weights(prices, dates, **cfg)
                gg, cc, nn, tt = run_book(ww, prices, DISCOVERY, COST_PER_SIDE)
                sensitivity.append({"lookback": lookback, "quantile": quantile, "weighting": weighting, **summarize(gg, cc, nn, tt)})

    # --- Discovery: leave-one-symbol-out on the primary ---------------------
    loo = {}
    for symbol in prices.columns:
        ww = w.copy()
        ww[symbol] = 0.0
        _, _, nn, _ = run_book(ww, prices, DISCOVERY, COST_PER_SIDE)
        loo[symbol] = metrics(nn)["sharpe"]
    loo_sorted = sorted(loo.items(), key=lambda kv: kv[1])

    # --- Discovery: cost ladder --------------------------------------------
    ladder = {}
    for bps in (4.0, 8.0, 16.0):
        _, _, nn, _ = run_book(w, prices, DISCOVERY, bps / 10_000.0)
        ladder[f"{bps:g}bps"] = metrics(nn)

    null = placebo(prices, dates, PRIMARY, DISCOVERY, primary["sharpe"])

    # --- Holdout: primary only, once ---------------------------------------
    gh, ch, nh, th = run_book(w, prices, HOLDOUT, COST_PER_SIDE)
    holdout = summarize(gh, ch, nh, th)
    holdout.update(beta_to_btc(nh, prices))

    payload = {
        "hypothesis": "H5 cross-sectional momentum (corrected sign), weekly rebalance",
        "primary": PRIMARY,
        "discovery": ["2020-01-01", "2024-12-31"],
        "holdout": ["2025-01-01", "2026-10-31"],
        "cost_per_side_bps": COST_PER_SIDE * 10_000.0,
        "discovery_primary": primary,
        "discovery_yearly": yearly(n),
        "discovery_sensitivity": sensitivity,
        "discovery_leave_one_symbol_out": {"worst": loo_sorted[:5], "best": loo_sorted[-3:]},
        "discovery_cost_ladder": ladder,
        "discovery_placebo": null,
        "holdout_primary": holdout,
        "holdout_yearly": yearly(nh),
    }
    (RESULTS_DIR / "h05b_csm_momentum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("discovery_primary", "discovery_yearly", "discovery_cost_ladder", "discovery_placebo", "holdout_primary", "holdout_yearly")}, indent=2))
    print("sensitivity sorted by net sharpe:")
    for row in sorted(sensitivity, key=lambda r: -r["sharpe"])[:8]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("lookback", "quantile", "weighting", "sharpe", "gross_sharpe", "cagr", "max_drawdown")})


if __name__ == "__main__":
    main()

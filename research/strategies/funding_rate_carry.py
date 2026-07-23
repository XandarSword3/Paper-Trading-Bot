"""
BTC Funding-Rate Carry strategy — market-neutral, not directional.

Position: equal-notional long BTC spot + short BTC perpetual futures, held
continuously. No entry/exit signal — this collects the funding payment
that longs periodically pay shorts (or pays it, when funding goes negative)
every 8h settlement, independent of where price goes. A switching variant
(flat when funding turns negative) was tested and rejected: it only
improved returns 16.3%/yr vs 15.0%/yr while requiring 561 position flips
over 4 years, and realistic transaction costs on that many round-trips
would erase the gain. So: static, always-on, no signal logic.

Data: real historical BTCUSDT perpetual funding settlements from Binance
Futures (data/BTCUSDT_funding_rate.csv), fetched by funding_rate_fetcher.py
via .github/workflows/fetch-funding-data.yml.

Uses the same DEFAULT_SPLIT holdout boundary and bars-per-year inference
as the rest of the repo (config.py / metrics_utils.py) so this strategy's
numbers are directly comparable to V1/V4's, and so the same acceptance
check — no settlement at or after holdout_start informs the reported
numbers — applies here too. Unlike V1/V4 there's no parameter search, so
there's no in-sample/out-of-sample optimization split to freeze; the
holdout is still respected purely so nothing here is quietly reported off
data meant to be untouched until final_holdout_validation.py runs.
"""
import argparse
import json
import os

import pandas as pd

from config import DATA_DIR, DEFAULT_SPLIT, RESULTS_DIR
from metrics_utils import infer_bars_per_year


def load_funding_data(path: str = None) -> pd.Series:
    path = path or os.path.join(DATA_DIR, "BTCUSDT_funding_rate.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["funding_rate"].sort_index()


def run_backtest(funding_rate: pd.Series, initial_capital: float = 100_000.0) -> pd.Series:
    """Static short-perp/long-spot hedge: compounds the funding rate paid at
    each settlement onto the deployed notional. Positive funding_rate means
    longs pay shorts, so the short leg (this position) receives it;
    negative means this position pays. No signal, no flips, no sizing
    decisions — this is deliberately the simplest thing that could work."""
    # Real BTCUSDT settlements are nowhere near this large (typical range is
    # roughly ±0.03%/8h, with rare spikes into the low single-digit percent
    # during extreme volatility) — this clip only ever protects against a
    # bad data point (e.g. a fetch/parsing glitch), not a real market move,
    # and keeps a single corrupted row from being able to send the whole
    # compounding curve negative or to zero.
    safe_rate = funding_rate.clip(lower=-0.20, upper=0.20)
    equity = (1.0 + safe_rate).cumprod() * initial_capital
    equity.name = "equity"
    return equity


def compute_stats(equity: pd.Series, initial_capital: float) -> dict:
    if len(equity) < 2:
        return {}

    returns = equity.pct_change().dropna()
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    bars_per_year = infer_bars_per_year(equity.index)
    total_return = (equity.iloc[-1] / initial_capital) - 1
    years = len(equity) / bars_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = (returns.mean() / returns.std() * (bars_per_year ** 0.5)) if returns.std() > 0 else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
    pct_profitable = (returns > 0).mean() * 100 if len(returns) else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": equity.iloc[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "pct_periods_profitable": pct_profitable,
        "num_settlements": len(equity),
        "start": equity.index[0],
        "end": equity.index[-1],
    }


def per_calendar_year_cagr(funding_rate: pd.Series) -> dict:
    out = {}
    for year, group in funding_rate.groupby(funding_rate.index.year):
        eq = run_backtest(group, initial_capital=100_000.0)
        stats = compute_stats(eq, 100_000.0)
        out[str(year)] = round(stats.get("cagr_pct", 0.0), 2) if stats else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--out", default="funding_carry_results.json")
    args = ap.parse_args()

    funding_rate = load_funding_data()

    holdout_start = pd.Timestamp(DEFAULT_SPLIT.holdout_start)
    leaked = len(funding_rate) > 0 and funding_rate.index[-1] >= holdout_start
    usable = funding_rate[funding_rate.index < holdout_start]
    if leaked:
        print(f"NOTE: trimming {len(funding_rate) - len(usable)} settlement(s) at/after "
              f"holdout_start ({holdout_start}) — not used in the numbers below.")
    funding_rate = usable

    equity = run_backtest(funding_rate, args.initial_capital)
    stats = compute_stats(equity, args.initial_capital)
    yearly = per_calendar_year_cagr(funding_rate)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    equity.rename("equity").to_frame().to_csv(
        os.path.join(RESULTS_DIR, "funding_carry_equity.csv"), index_label="timestamp"
    )

    summary = {
        "strategy": "funding_rate_carry",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "acceptance_check_passed": True,  # by construction: holdout already trimmed above
        "holdout_start": str(holdout_start),
        **{k: (str(v) if k in ("start", "end") else v) for k, v in stats.items()},
        "cagr_by_calendar_year": yearly,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {args.out}")
    print(f"CAGR={stats.get('cagr_pct'):.2f}% Sharpe={stats.get('sharpe_ratio'):.2f} "
          f"maxDD={stats.get('max_drawdown_pct'):.2f}% Calmar={stats.get('calmar_ratio'):.2f} "
          f"settlements={stats.get('num_settlements')}")


if __name__ == "__main__":
    main()

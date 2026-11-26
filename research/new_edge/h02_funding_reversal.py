"""H2 pilot: cross-sectional funding-extreme reversal.

Hypothesis (pre-registered before running):
    Perpetual funding is a crowdedness tax.  Symbols in the top funding quintile
    are crowded longs and underperform; symbols in the bottom quintile are
    crowded shorts and outperform, over the next day.  A dollar-neutral
    long-low-funding / short-high-funding book isolates that effect from market
    beta.

Protocol:
    * discovery only: 2020-01-01 .. 2024-12-31
    * signal at 00:00 UTC from that day's funding print
    * rank 30-day funding z-scores cross-sectionally
    * long bottom 20%, short top 20%, equal weight, 1.0x gross, dollar-neutral
    * hold one day, rebalance daily, 8 bps per side on turnover
    * holdout (2025-01-01 .. 2026-10-31) is never loaded here
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
FUNDING_DIR = WORKSPACE / "Trading Bot" / "liquidation-hunter" / "data" / "binance" / "funding"
KLINES_DIR = WORKSPACE / "Trading Bot" / "liquidation-hunter" / "data" / "binance" / "klines"
RESULTS_DIR = HERE / "results"

DISCOVERY_START = pd.Timestamp("2020-01-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")  # exclusive
COST_PER_SIDE = 8.0 / 10_000.0
Z_WINDOW = 90          # 30 days of 8h funding prints
MIN_HISTORY = 60
QUANTILE = 0.20
MIN_NAMES = 10


@dataclass
class Day:
    date: str
    gross_return: float
    cost: float
    net_return: float
    n_long: int
    n_short: int
    turnover: float
    score_spread: float


def load_funding() -> pd.DataFrame:
    frames = {}
    for path in sorted(FUNDING_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.drop_duplicates("timestamp").set_index("timestamp")["funding_rate"]
        # One observation per day at 00:00 UTC.
        daily = frame.resample("1D").first()
        daily.index.freq = None
        frames[path.stem] = daily
    return pd.concat(frames, axis=1).sort_index()


def load_prices() -> pd.DataFrame:
    frames = {}
    for path in sorted(KLINES_DIR.glob("*_1h.parquet")):
        frame = pd.read_parquet(path, columns=["timestamp", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.drop_duplicates("timestamp").set_index("timestamp")["close"]
        # Label each day by its 00:00 boundary: the 23:00 bar is the price at 00:00.
        daily = frame.resample("1D", label="right", closed="right").last()
        daily.index.freq = None
        frames[path.stem.replace("_1h", "")] = daily
    return pd.concat(frames, axis=1).sort_index()


def zscores(funding: pd.DataFrame) -> pd.DataFrame:
    mean = funding.rolling(Z_WINDOW, min_periods=MIN_HISTORY).mean()
    std = funding.rolling(Z_WINDOW, min_periods=MIN_HISTORY).std()
    return (funding - mean) / std.replace(0.0, np.nan)


def weights_from_rank(score: pd.DataFrame, prices: pd.DataFrame, quantile: float = QUANTILE):
    """Return daily dollar-neutral weights and the leg funding spread."""
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    spreads = pd.Series(np.nan, index=score.index)
    for date, row in score.iterrows():
        valid = row.dropna()
        valid = valid[prices.loc[date, valid.index].notna()] if date in prices.index else valid.iloc[0:0]
        if len(valid) < MIN_NAMES:
            continue
        n = max(1, int(np.floor(len(valid) * quantile)))
        ranked = valid.sort_values()
        longs = ranked.index[:n]
        shorts = ranked.index[-n:]
        weights.loc[date, longs] = 0.5 / n
        weights.loc[date, shorts] = -0.5 / n
        spreads.loc[date] = float(ranked.iloc[-n:].mean() - ranked.iloc[:n].mean())
    return weights, spreads


def run_book(weights: pd.DataFrame, prices: pd.DataFrame, spreads: pd.Series) -> tuple[list[Day], pd.Series]:
    fwd = prices.shift(-1) / prices - 1.0
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * COST_PER_SIDE
    net = gross - cost

    days: list[Day] = []
    for date in weights.index:
        if date < DISCOVERY_START or date >= DISCOVERY_END:
            continue
        row = weights.loc[date]
        days.append(
            Day(
                date=date.date().isoformat(),
                gross_return=float(gross.loc[date]) if pd.notna(gross.loc[date]) else 0.0,
                cost=float(cost.loc[date]) if pd.notna(cost.loc[date]) else 0.0,
                net_return=float(net.loc[date]) if pd.notna(net.loc[date]) else 0.0,
                n_long=int((row > 0).sum()),
                n_short=int((row < 0).sum()),
                turnover=float(turnover.loc[date]) if pd.notna(turnover.loc[date]) else 0.0,
                score_spread=float(spreads.loc[date]) if pd.notna(spreads.loc[date]) else float("nan"),
            )
        )
    series = pd.Series(
        {pd.Timestamp(d.date, tz="UTC"): d.net_return for d in days}
    ).sort_index()
    return days, series


def metrics(series: pd.Series) -> dict:
    if series.empty:
        return {"days": 0, "sharpe": float("nan"), "cagr": float("nan"), "max_drawdown": float("nan")}
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).min()
    std = float(series.std(ddof=1))
    sharpe = float(series.mean() / std * np.sqrt(365.0)) if std > 0 else float("nan")
    years = len(series) / 365.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    return {
        "days": int(len(series)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown),
        "hit_rate": float((series > 0).mean()),
        "avg_daily_bps": float(series.mean() * 10_000.0),
    }


def beta_to_btc(series: pd.Series, prices: pd.DataFrame) -> dict:
    btc = prices["BTCUSDT"].reindex(series.index)
    btc_ret = btc.pct_change()
    frame = pd.concat([series.rename("book"), btc_ret.rename("btc")], axis=1).dropna()
    if len(frame) < 30:
        return {"beta": float("nan"), "alpha_annual": float("nan")}
    beta = float(np.cov(frame["book"], frame["btc"])[0, 1] / np.var(frame["btc"], ddof=1))
    alpha = float(frame["book"].mean() - beta * frame["btc"].mean())
    return {"beta": beta, "alpha_annual": alpha * 365.0}


def placebo(score: pd.DataFrame, prices: pd.DataFrame, actual_sharpe: float, draws: int = 200) -> dict:
    rng = np.random.default_rng(20261126)
    sharpes: list[float] = []
    for _ in range(draws):
        shuffled = score.copy()
        for date in shuffled.index:
            row = shuffled.loc[date].dropna()
            if len(row) < MIN_NAMES:
                continue
            shuffled.loc[date, row.index] = rng.permutation(row.to_numpy())
        weights, spreads = weights_from_rank(shuffled, prices)
        _, series = run_book(weights, prices, spreads)
        sharpes.append(metrics(series)["sharpe"])
    arr = np.array([s for s in sharpes if np.isfinite(s)])
    return {
        "draws": int(len(arr)),
        "placebo_mean": float(arr.mean()) if len(arr) else float("nan"),
        "placebo_p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "actual": actual_sharpe,
        "p_value": float((arr >= actual_sharpe).mean()) if len(arr) and np.isfinite(actual_sharpe) else float("nan"),
    }


def momentum_control(prices: pd.DataFrame) -> dict:
    """Same construction, but rank on trailing 30-day return instead of funding."""
    past = prices / prices.shift(30) - 1.0
    weights, spreads = weights_from_rank(past, prices)
    _, series = run_book(weights, prices, spreads)
    return {"reversal_book": metrics(series)}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    funding = load_funding()
    prices = load_prices()
    print(f"funding {funding.shape}, prices {prices.shape}")

    score = zscores(funding)
    weights, spreads = weights_from_rank(score, prices)
    days, series = run_book(weights, prices, spreads)
    overall = metrics(series)
    overall["avg_turnover"] = float(np.mean([d.turnover for d in days])) if days else float("nan")
    overall["avg_score_spread"] = float(np.nanmean([d.score_spread for d in days])) if days else float("nan")
    overall["avg_names"] = float(np.mean([d.n_long + d.n_short for d in days])) if days else float("nan")
    overall.update(beta_to_btc(series, prices))

    yearly = {}
    for year in range(2020, 2025):
        subset = series[(series.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) & (series.index < pd.Timestamp(f"{year+1}-01-01", tz="UTC"))]
        yearly[str(year)] = metrics(subset)

    null = placebo(score, prices, overall["sharpe"])
    control = momentum_control(prices)

    payload = {
        "hypothesis": "H2 cross-sectional funding-extreme reversal",
        "params": {"z_window": Z_WINDOW, "quantile": QUANTILE, "cost_per_side_bps": COST_PER_SIDE * 10_000},
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "overall": overall,
        "yearly": yearly,
        "placebo": null,
        "momentum_control": control,
        "first_days": [asdict(d) for d in days[:3]],
    }
    (RESULTS_DIR / "h02_funding_reversal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "yearly": yearly, "placebo": null, "momentum_control": control}, indent=2))


if __name__ == "__main__":
    main()

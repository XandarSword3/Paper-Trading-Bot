"""H3 pilot: volatility-compression range breakout.

Hypothesis (pre-registered before running):
    When 24-hour realized volatility is compressed relative to its recent
    norm, a break of the prior 24-hour range marks the start of a directional
    move.  Trade the break, hold 24 hours, exit.  Compression should improve
    the breakout's follow-through versus an unconditional breakout.

Protocol:
    * discovery only: 2020-01-01 .. 2024-12-31
    * 1h bars, 36 Binance USDT perps
    * signal on the last 1h bar of a UTC day
    * enter at the next 00:00 UTC price, exit 24h later
    * 8 bps per side (16 bps round trip), equal weight across active signals
    * primary compression: rv24 < 0.75 * 30-day mean rv24
    * holdout (2025-01-01 .. 2026-10-31) is never loaded here
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
KLINES_DIR = WORKSPACE / "Trading Bot" / "liquidation-hunter" / "data" / "binance" / "klines"
RESULTS_DIR = HERE / "results"

DISCOVERY_START = pd.Timestamp("2020-01-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-01-01", tz="UTC")
COST_ROUND_TRIP = 16.0 / 10_000.0
PRIMARY = {"range_hours": 24, "vol_hours": 24, "vol_window_hours": 24 * 30, "compression": 0.75}


def load_symbol(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return frame


def symbol_daily_signals(
    frame: pd.DataFrame,
    *,
    range_hours: int,
    vol_hours: int,
    vol_window_hours: int,
    compression: float,
) -> pd.DataFrame:
    ret = frame["close"].pct_change()
    rv = ret.rolling(vol_hours, min_periods=vol_hours).std()
    baseline = rv.rolling(vol_window_hours, min_periods=vol_hours * 10).mean()
    compressed = rv < compression * baseline

    prior_high = frame["high"].rolling(range_hours, min_periods=range_hours).max().shift(1)
    prior_low = frame["low"].rolling(range_hours, min_periods=range_hours).min().shift(1)
    long_break = (frame["close"] > prior_high) & compressed
    short_break = (frame["close"] < prior_low) & compressed

    daily_long = long_break.fillna(False).resample("1D").max().astype(bool)
    daily_short = short_break.fillna(False).resample("1D").max().astype(bool)
    both = daily_long & daily_short
    daily_long = daily_long & ~both
    daily_short = daily_short & ~both

    price = frame["close"].resample("1D", label="right", closed="right").last()
    price.index.freq = None
    fwd = (price.shift(-1) / price - 1.0).shift(1)  # signal day t -> return t+1

    out = pd.DataFrame({"long": daily_long, "short": daily_short})
    out["fwd"] = fwd.reindex(out.index)
    return out


def combine(frames: dict[str, pd.DataFrame], **kwargs) -> tuple[pd.Series, pd.DataFrame]:
    rows = []
    for symbol, frame in frames.items():
        sig = symbol_daily_signals(frame, **kwargs)
        for date, row in sig.iterrows():
            if not row["long"] and not row["short"]:
                continue
            if not np.isfinite(row["fwd"]):
                continue
            direction = 1 if row["long"] else -1
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "direction": direction,
                    "fwd": float(row["fwd"]),
                    "net_return": direction * float(row["fwd"]) - COST_ROUND_TRIP,
                }
            )
    trades = pd.DataFrame(rows)
    if trades.empty:
        return pd.Series(dtype=float), trades
    daily = trades.groupby("date")["net_return"].mean()
    daily = daily[(daily.index >= DISCOVERY_START) & (daily.index < DISCOVERY_END)].sort_index()
    return daily, trades


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
    positive = float(series[series > 0].sum())
    top5 = float(series.nlargest(5).sum())
    return {
        "days": int(len(series)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown),
        "hit_rate": float((series > 0).mean()),
        "avg_daily_bps": float(series.mean() * 10_000.0),
        "top5_share_of_positive": top5 / positive if positive > 0 else float("nan"),
    }


def yearly(series: pd.Series) -> dict:
    out = {}
    for year in range(2020, 2025):
        subset = series[
            (series.index >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & (series.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ]
        out[str(year)] = metrics(subset)
    return out


def placebo(trades: pd.DataFrame, actual_sharpe: float, draws: int = 200) -> dict:
    rng = np.random.default_rng(20261126)
    sharpes: list[float] = []
    for _ in range(draws):
        sample = []
        for _, row in trades.iterrows():
            direction = int(rng.choice([-1, 1]))
            sample.append(
                {
                    "date": row["date"],
                    "net_return": direction * float(row["fwd"]) - COST_ROUND_TRIP,
                }
            )
        frame = pd.DataFrame(sample)
        daily = frame.groupby("date")["net_return"].mean().sort_index()
        sharpes.append(metrics(daily)["sharpe"])
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
    frames = {
        path.stem.replace("_1h", ""): load_symbol(path)
        for path in sorted(KLINES_DIR.glob("*_1h.parquet"))
    }
    print(f"loaded {len(frames)} symbols")

    daily, trades = combine(frames, **PRIMARY)
    overall = {**metrics(daily), "signals": int(len(trades)), "avg_signals_per_day": float(len(trades) / max(1, len(daily)))}

    sensitivity = []
    for compression in (0.60, 0.75, 0.90):
        for range_hours in (12, 24, 48):
            cfg = {**PRIMARY, "compression": compression, "range_hours": range_hours}
            d, t = combine(frames, **cfg)
            sensitivity.append({"compression": compression, "range_hours": range_hours, **metrics(d), "signals": int(len(t))})

    # Control: unconditional breakout (no compression).
    control_daily, control_trades = combine(frames, **{**PRIMARY, "compression": 1.01})
    control = {**metrics(control_daily), "signals": int(len(control_trades))}

    null = placebo(trades, overall["sharpe"])

    payload = {
        "hypothesis": "H3 volatility-compression range breakout",
        "primary": PRIMARY,
        "discovery": [str(DISCOVERY_START.date()), "2024-12-31"],
        "overall": overall,
        "yearly": yearly(daily),
        "control_unconditional_breakout": control,
        "sensitivity": sensitivity,
        "placebo": null,
        "first_trades": trades.head(3).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }
    (RESULTS_DIR / "h03_vol_breakout.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "yearly": payload["yearly"], "control": control, "placebo": null}, indent=2))


if __name__ == "__main__":
    main()

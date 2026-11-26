"""H1 pilot: fade 1-minute liquidation-style cascades.

Hypothesis (pre-registered before running):
    When a perp prints a >=3-sigma 1-minute move on >=5x its trailing 60-minute
    median volume, the move is partly forced flow and partially reverses over
    the next 30 minutes.  Fade the move (short the spike, buy the flush).

Protocol:
    * discovery only: 2025-11-01 .. 2026-07-31
    * signal at bar close, fill at next bar open, exit at open after 30 bars
    * 8 bps cost per side, no leverage, 10% of equity per trade
    * one position per symbol at a time
    * holdout is never loaded here

This is a screening pilot, not a promotion decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT.parent / "Trading Bot" / "liquidation-hunter" / "data" / "binance_1m" / "full"
RESULTS_DIR = HERE / "results"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DISCOVERY_START = pd.Timestamp("2025-11-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2026-08-01", tz="UTC")  # exclusive
COST_PER_SIDE = 8.0 / 10_000.0
TRADE_WEIGHT = 0.10
PRIMARY = {"z_thresh": 3.0, "vol_thresh": 5.0, "hold_minutes": 30}


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    direction: int
    gross_return: float
    net_return: float


def load_symbol(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=["minute", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["minute"].astype("int64"), unit="s", utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    frame = frame.loc[(frame.index >= DISCOVERY_START) & (frame.index < DISCOVERY_END)]
    return frame[["open", "high", "low", "close", "volume"]]


def simulate(
    frame: pd.DataFrame,
    symbol: str,
    *,
    z_thresh: float,
    vol_thresh: float,
    hold_minutes: int,
) -> list[Trade]:
    ret1 = frame["close"].pct_change()
    sigma = ret1.rolling(60, min_periods=60).std()
    vol_med = frame["volume"].rolling(60, min_periods=60).median()
    z = ret1 / sigma.replace(0.0, np.nan)
    vol_ratio = frame["volume"] / vol_med.replace(0.0, np.nan)

    signal = (z.abs() >= z_thresh) & (vol_ratio >= vol_thresh) & sigma.notna()
    idxs = np.flatnonzero(signal.to_numpy())
    opens = frame["open"].to_numpy()
    times = frame.index
    trades: list[Trade] = []
    last_exit = -1

    for idx in idxs:
        entry_idx = int(idx) + 1
        exit_idx = entry_idx + int(hold_minutes)
        if entry_idx <= last_exit or exit_idx >= len(frame):
            continue
        entry = float(opens[entry_idx])
        exit_price = float(opens[exit_idx])
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price):
            continue
        move = float(ret1.iloc[idx])
        direction = -1 if move > 0 else 1
        gross = direction * (exit_price / entry - 1.0)
        net = gross - 2.0 * COST_PER_SIDE
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=times[entry_idx].isoformat(),
                exit_time=times[exit_idx].isoformat(),
                direction=direction,
                gross_return=gross,
                net_return=net,
            )
        )
        last_exit = exit_idx

    return trades


def daily_returns(trades: list[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    exits = pd.to_datetime([t.exit_time for t in trades], utc=True).normalize()
    pnl = np.array([t.net_return * TRADE_WEIGHT for t in trades])
    return pd.Series(pnl, index=exits).groupby(level=0).sum().sort_index()


def metrics(daily: pd.Series) -> dict:
    if daily.empty:
        return {"trades": 0, "sharpe": float("nan"), "total_return": 0.0, "max_drawdown": 0.0,
                "hit_rate": float("nan"), "avg_trade_bps": float("nan"), "top5_share": float("nan")}
    equity = (1.0 + daily).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).min()
    std = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / std * np.sqrt(365.0)) if std > 0 else float("nan")
    positive = float(daily[daily > 0].sum())
    top5 = float(daily.nlargest(5).sum())
    return {
        "days": int(len(daily)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown),
        "top5_share_of_positive": top5 / positive if positive > 0 else float("nan"),
    }


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0, "hit_rate": float("nan"), "avg_trade_bps": float("nan")}
    nets = np.array([t.net_return for t in trades])
    return {
        "trades": int(len(trades)),
        "hit_rate": float((nets > 0).mean()),
        "avg_trade_bps": float(nets.mean() * 10_000.0),
    }


def fold_metrics(trades: list[Trade], start: pd.Timestamp, end: pd.Timestamp, label: str) -> dict:
    subset = [
        t for t in trades
        if start <= pd.Timestamp(t.exit_time) < end
    ]
    daily = daily_returns(subset)
    return {"fold": label, **metrics(daily), **trade_stats(subset)}


def placebo(frames: dict[str, pd.DataFrame], trades: list[Trade], draws: int = 200) -> dict:
    """Same number of trades per symbol, random entry times, same holding period."""
    rng = np.random.default_rng(20261124)
    counts = pd.Series([t.symbol for t in trades]).value_counts().to_dict()
    sharpes: list[float] = []
    for _ in range(draws):
        sample: list[Trade] = []
        for symbol, count in counts.items():
            frame = frames[symbol]
            opens = frame["open"].to_numpy()
            times = frame.index
            n = len(frame)
            for _ in range(int(count)):
                entry_idx = int(rng.integers(60, n - PRIMARY["hold_minutes"] - 2))
                exit_idx = entry_idx + PRIMARY["hold_minutes"]
                entry = float(opens[entry_idx])
                exit_price = float(opens[exit_idx])
                direction = 1 if rng.random() < 0.5 else -1
                gross = direction * (exit_price / entry - 1.0)
                sample.append(
                    Trade(
                        symbol=symbol,
                        entry_time=times[entry_idx].isoformat(),
                        exit_time=times[exit_idx].isoformat(),
                        direction=direction,
                        gross_return=gross,
                        net_return=gross - 2.0 * COST_PER_SIDE,
                    )
                )
        sharpes.append(metrics(daily_returns(sample))["sharpe"])
    arr = np.array([s for s in sharpes if np.isfinite(s)])
    actual = metrics(daily_returns(trades))["sharpe"]
    return {
        "draws": int(len(arr)),
        "placebo_sharpe_mean": float(arr.mean()) if len(arr) else float("nan"),
        "placebo_sharpe_p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "actual_sharpe": actual,
        "p_value": float((arr >= actual).mean()) if len(arr) and np.isfinite(actual) else float("nan"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = {symbol: load_symbol(symbol) for symbol in SYMBOLS}
    for symbol, frame in frames.items():
        print(f"{symbol}: {len(frame):,} minutes {frame.index[0].date()} .. {frame.index[-1].date()}")

    trades: list[Trade] = []
    for symbol, frame in frames.items():
        trades.extend(simulate(frame, symbol, **PRIMARY))
    trades.sort(key=lambda t: t.exit_time)

    daily = daily_returns(trades)
    overall = {**metrics(daily), **trade_stats(trades)}

    folds = [
        fold_metrics(trades, pd.Timestamp("2025-11-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC"), "f1"),
        fold_metrics(trades, pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC"), "f2"),
        fold_metrics(trades, pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC"), "f3"),
    ]

    sensitivity = []
    for z in (2.5, 3.0, 3.5):
        for vol in (3.0, 5.0, 8.0):
            for hold in (15, 30, 60):
                cfg_trades: list[Trade] = []
                for symbol, frame in frames.items():
                    cfg_trades.extend(
                        simulate(frame, symbol, z_thresh=z, vol_thresh=vol, hold_minutes=hold)
                    )
                m = {**metrics(daily_returns(cfg_trades)), **trade_stats(cfg_trades)}
                sensitivity.append({"z": z, "vol": vol, "hold": hold, **m})

    null = placebo(frames, trades)
    payload = {
        "hypothesis": "H1 cascade fade",
        "primary": PRIMARY,
        "symbols": SYMBOLS,
        "discovery": [str(DISCOVERY_START.date()), str((DISCOVERY_END - pd.Timedelta(days=1)).date())],
        "overall": overall,
        "folds": folds,
        "placebo": null,
        "sensitivity": sensitivity,
        "trades_sample": [asdict(t) for t in trades[:5]],
    }
    (RESULTS_DIR / "h01_cascade_fade.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"overall": overall, "folds": folds, "placebo": null}, indent=2))


if __name__ == "__main__":
    main()

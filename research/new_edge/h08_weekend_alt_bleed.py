"""H8: weekend alt-bleed relative value.

Hypothesis (pre-registered):
    Alt perps bleed against BTC over the weekend.  Liquidity is thinner, market
    makers widen, and alt holders de-risk into a 72-hour window with no
    traditional-market catalyst.  BTC is the safe haven inside crypto, so the
    cross-section rotates into BTC from Friday 00:00 to Monday 00:00 UTC.

Trade:
    long 0.5 BTC / short 0.5 equal-weight alt basket, opened Friday 00:00 UTC,
    closed Monday 00:00 UTC.  Dollar-neutral, 1.0 gross.

Protocol:
    * primary universe: 12 established alts with data from 2020 (low
      survivorship bias): ETH BCH XRP LTC TRX ETC LINK XLM ADA ATOM BNB DOGE
    * secondary universe: every alt available on the date (>= 5 names)
    * discovery: 2020-01-01 .. 2024-12-31
    * holdout: 2025-01-01 .. 2026-10-31
    * costs: 8 bps per side on each leg -> 16 bps per weekend on the 1.0 book
    * funding: included, from the 8h prints strictly inside the window
    * weekly returns, annualised with 52 weeks

Exploration caveat: the weekend pattern was first noticed in a full-sample
diagnostic before this file existed, so the holdout is not perfectly blind.
It is reported anyway, and the per-year breakdown is the real evidence.
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

from research.new_edge.h03_vol_breakout import KLINES_DIR, RESULTS_DIR, load_symbol

COST_PER_SIDE = 8.0 / 10_000.0
LONG_HISTORY = [
    "ETHUSDT", "BCHUSDT", "XRPUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT",
    "LINKUSDT", "XLMUSDT", "ADAUSDT", "ATOMUSDT", "BNBUSDT", "DOGEUSDT",
]
FUNDING_DIR = KLINES_DIR.parent / "funding"
DISCOVERY = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))


def load_hourly() -> pd.DataFrame:
    frames = {}
    for path in sorted(KLINES_DIR.glob("*_1h.parquet")):
        symbol = path.stem.replace("_1h", "")
        frames[symbol] = load_symbol(path)["close"]
    return pd.concat(frames, axis=1).sort_index()


def load_funding_raw() -> pd.DataFrame:
    frames = {}
    for path in sorted(FUNDING_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frames[path.stem] = frame.drop_duplicates("timestamp").set_index("timestamp")["funding_rate"]
    return pd.concat(frames, axis=1).sort_index()


def weekend_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Fridays at 00:00 UTC present in the index."""
    return [t for t in index if t.weekday() == 4 and t.hour == 0]


def window_pnl(
    closes: pd.DataFrame,
    funding: pd.DataFrame,
    entry: pd.Timestamp,
    exit_ts: pd.Timestamp,
    alts: list[str],
) -> dict | None:
    if entry not in closes.index or exit_ts not in closes.index:
        return None
    prices = closes.loc[entry, alts + ["BTCUSDT"]]
    prices_exit = closes.loc[exit_ts, alts + ["BTCUSDT"]]
    valid = [s for s in alts if np.isfinite(prices[s]) and np.isfinite(prices_exit[s]) and prices[s] > 0]
    if len(valid) < 5:
        return None
    alt_ret = float(np.mean([prices_exit[s] / prices[s] - 1.0 for s in valid]))
    btc_ret = float(prices_exit["BTCUSDT"] / prices["BTCUSDT"] - 1.0)

    f = funding.loc[(funding.index > entry) & (funding.index < exit_ts)]
    f_btc = float(f["BTCUSDT"].sum()) if "BTCUSDT" in f.columns else 0.0
    f_alt = float(f[valid].mean(axis=1).sum()) if len(f) else 0.0
    funding_pnl = 0.5 * (f_alt - f_btc)

    gross = 0.5 * btc_ret - 0.5 * alt_ret
    cost = 2.0 * 0.5 * COST_PER_SIDE * 2.0  # both legs, entry and exit
    return {
        "entry": entry,
        "alts": len(valid),
        "gross": gross,
        "funding": funding_pnl,
        "cost": cost,
        "net": gross + funding_pnl - cost,
        "btc_ret": btc_ret,
        "alt_ret": alt_ret,
    }


def build_series(closes, funding, alts: list[str], start, end, *, entry_hour=0, exit_offset_days=3, exit_hour=0) -> pd.DataFrame:
    rows = []
    for friday in weekend_starts(closes.index):
        if friday < start or friday >= end:
            continue
        entry = friday + pd.Timedelta(hours=entry_hour)
        exit_ts = friday + pd.Timedelta(days=exit_offset_days, hours=exit_hour)
        row = window_pnl(closes, funding, entry, exit_ts, alts)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def weekly_metrics(series: pd.Series) -> dict:
    if series.empty:
        return {"weeks": 0, "sharpe": float("nan")}
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    std = float(series.std(ddof=1))
    sharpe = float(series.mean() / std * np.sqrt(52.0)) if std > 0 else float("nan")
    years = len(series) / 52.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    return {
        "weeks": int(len(series)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "hit_rate": float((series > 0).mean()),
        "avg_week_bps": float(series.mean() * 10_000.0),
        "std_week_bps": float(series.std(ddof=1) * 10_000.0),
        "worst_week_bps": float(series.min() * 10_000.0),
        "t_stat": float(series.mean() / std * np.sqrt(len(series))) if std > 0 else float("nan"),
    }


def yearly(series: pd.Series) -> dict:
    return {
        str(y): weekly_metrics(series[series.index.year == y]) for y in sorted(set(series.index.year))
    }


def placebo(closes, funding, alts, start, end, actual_mean, draws=200) -> dict:
    """Random contiguous 72h windows inside each week."""
    rng = np.random.default_rng(20261126)
    fridays = [f for f in weekend_starts(closes.index) if start <= f < end]
    means = []
    for _ in range(draws):
        nets = []
        for friday in fridays:
            offset = int(rng.integers(0, 24 * 7 - 72))
            entry = friday + pd.Timedelta(hours=offset)
            exit_ts = entry + pd.Timedelta(hours=72)
            row = window_pnl(closes, funding, entry, exit_ts, alts)
            if row is not None:
                nets.append(row["net"])
        if nets:
            means.append(float(np.mean(nets)))
    arr = np.array(means)
    return {
        "draws": int(len(arr)),
        "placebo_mean_bps": float(arr.mean() * 10_000.0) if len(arr) else float("nan"),
        "placebo_p95_bps": float(np.percentile(arr, 95) * 10_000.0) if len(arr) else float("nan"),
        "actual_mean_bps": float(actual_mean * 10_000.0),
        "p_value": float((arr >= actual_mean).mean()) if len(arr) else float("nan"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    closes = load_hourly()
    funding = load_funding_raw()
    print(f"closes {closes.shape}, funding {funding.shape}")

    all_alts = [c for c in closes.columns if c != "BTCUSDT"]

    def evaluate(alts, start, end, **kwargs):
        frame = build_series(closes, funding, alts, start, end, **kwargs)
        series = pd.Series(frame["net"].to_numpy(), index=frame["entry"]).sort_index()
        out = {
            **weekly_metrics(series),
            "gross_mean_bps": float(frame["gross"].mean() * 10_000.0),
            "funding_mean_bps": float(frame["funding"].mean() * 10_000.0),
            "cost_mean_bps": float(frame["cost"].mean() * 10_000.0),
            "avg_alts": float(frame["alts"].mean()),
        }
        return out, series

    discovery_long, disc_series = evaluate(LONG_HISTORY, *DISCOVERY)
    discovery_all, _ = evaluate(all_alts, *DISCOVERY)
    holdout_long, hold_series = evaluate(LONG_HISTORY, *HOLDOUT)
    holdout_all, _ = evaluate(all_alts, *HOLDOUT)

    full_long, full_series = evaluate(LONG_HISTORY, DISCOVERY[0], HOLDOUT[1])

    sensitivity = []
    for entry_hour in (0, 8, 16):
        for exit_day, exit_hour in ((3, 0), (2, 16), (2, 0)):
            if exit_day * 24 + exit_hour <= entry_hour:
                continue
            out, _ = evaluate(LONG_HISTORY, *DISCOVERY, entry_hour=entry_hour, exit_offset_days=exit_day, exit_hour=exit_hour)
            sensitivity.append({"entry_hour": entry_hour, "exit": f"day+{exit_day} {exit_hour:02d}:00", **out})

    # Weekday counterpart: Monday 00:00 -> Friday 00:00, long BTC / short alts.
    weekday_rows = []
    for monday in [t for t in closes.index if t.weekday() == 0 and t.hour == 0]:
        if monday < DISCOVERY[0] or monday >= DISCOVERY[1]:
            continue
        row = window_pnl(closes, funding, monday, monday + pd.Timedelta(days=4), LONG_HISTORY)
        if row is not None:
            weekday_rows.append(row)
    wframe = pd.DataFrame(weekday_rows)
    weekday_series = pd.Series(wframe["net"].to_numpy(), index=wframe["entry"]).sort_index()
    weekday = weekly_metrics(weekday_series)

    null = placebo(closes, funding, LONG_HISTORY, DISCOVERY[0], DISCOVERY[1], discovery_long["avg_week_bps"] / 10_000.0)

    payload = {
        "hypothesis": "H8 weekend alt-bleed relative value",
        "primary": {
            "universe": "12 long-history alts",
            "position": "long 0.5 BTC / short 0.5 alt basket",
            "window": "Fri 00:00 -> Mon 00:00 UTC",
            "cost_per_side_bps": COST_PER_SIDE * 10_000.0,
        },
        "discovery": ["2020-01-01", "2024-12-31"],
        "holdout": ["2025-01-01", "2026-10-31"],
        "discovery_long_history": discovery_long,
        "discovery_all_alts": discovery_all,
        "discovery_yearly": yearly(disc_series),
        "discovery_weekday_counterpart": weekday,
        "discovery_sensitivity": sensitivity,
        "discovery_placebo": null,
        "holdout_long_history": holdout_long,
        "holdout_all_alts": holdout_all,
        "holdout_yearly": yearly(hold_series),
        "full_sample_long_history": full_long,
        "full_sample_yearly": yearly(full_series),
    }
    (RESULTS_DIR / "h08_weekend_alt_bleed.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("discovery_long_history", "discovery_all_alts", "discovery_yearly", "discovery_weekday_counterpart", "discovery_placebo", "holdout_long_history", "holdout_all_alts", "holdout_yearly", "full_sample_long_history", "full_sample_yearly")}, indent=2))


if __name__ == "__main__":
    main()

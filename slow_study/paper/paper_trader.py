"""12-month forward paper trader for a VIABLE frozen strategy.

Rules: identical frozen strategy code as the backtest (import from strategies.py).
Every day (run once per day after the daily candle closes):
  1. fetch latest completed daily candles from Coinbase public API,
  2. compute the frozen signal on closed bars only,
  3. compare to current paper position, log intended trade at next open,
  4. append a row to paper/log.csv with timestamps (UTC),
  5. update paper/state.json.

NO rule changes mid-run. Only data-quality checks allowed (gaps, clock,
missing candles) — those are logged, never used to alter signals.

Usage:
  python paper/paper_trader.py --strategy S1_ma_trend [--init]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import strategies as st  # noqa: E402

CB = "https://api.exchange.coinbase.com"
STATE = Path(__file__).parent / "state.json"
LOG = Path(__file__).parent / "log.csv"
START_EQUITY = st.START_EQUITY


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def fetch_daily(product):
    """Last ~400 daily candles for signal warmup."""
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=400)
    r = requests.get(f"{CB}/products/{product}/candles", params={
        "granularity": 86400,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, timeout=20, headers={"User-Agent": "slow-study-paper/1.0"})
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "vol"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    df = df.sort_values("ts").drop_duplicates("ts")
    # drop in-progress candle
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if len(df) and (now_ts - int(df["ts"].iloc[-1])) < 86400:
        df = df.iloc[:-1]
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.set_index("dt")[["open", "high", "low", "close", "vol"]]


def dq_check(df, name):
    """Data-quality checks only — logged, never alter signals."""
    issues = []
    diffs = df.index.to_series().diff().dt.days
    gaps = diffs[diffs > 1]
    if len(gaps):
        issues.append(f"{name}: {len(gaps)} date gaps, max {int(gaps.max())}d")
    if df[["open", "high", "low", "close"]].isna().any().any():
        issues.append(f"{name}: NaN OHLC present")
    if (df["high"] < df["low"]).any():
        issues.append(f"{name}: high<low rows present")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["S1_ma_trend", "S2_tsmom_voltarget", "S3_donchian55",
                             "S4_xsmom_weekly", "S5_funding_carry", "S6_buyhold"])
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()

    fn = getattr(st, args.strategy.lower().replace("s1_", "s1_").replace("-", "_"), None)
    # map strategy id -> signal function
    fmap = {
        "S1_ma_trend": st.s1_weights, "S2_tsmom_voltarget": st.s2_weights,
        "S3_donchian55": st.s3_weights, "S4_xsmom_weekly": st.s4_weights,
        "S6_buyhold": st.s6_weights,
    }
    if args.strategy == "S5_funding_carry":
        print("S5 paper trading requires perp leg — not implemented; abort.")
        return
    sig_fn = fmap[args.strategy]

    # determine needed symbols
    if args.strategy in ("S1_ma_trend", "S3_donchian55", "S6_buyhold"):
        symbols = ["BTC"]
    elif args.strategy == "S2_tsmom_voltarget":
        symbols = ["BTC", "ETH"]
    else:
        symbols = st.ALTS

    bars = {}
    for s in symbols:
        product = f"{s}-USD"
        bars[s] = fetch_daily(product)
        for issue in dq_check(bars[s], s):
            print("DQ:", issue)

    # build weight DataFrame on the union index (same as backtest)
    if args.strategy == "S4_xsmom_weekly":
        w = sig_fn(bars)
    elif args.strategy == "S2_tsmom_voltarget":
        w = sig_fn(bars)
    else:
        w = sig_fn(bars["BTC"])
    last_w = w.iloc[-1].fillna(0.0).to_dict()
    last_date = str(w.index[-1])

    state = {}
    if STATE.exists() and not args.init:
        state = json.loads(STATE.read_text())
    else:
        state = {"strategy": args.strategy, "equity": START_EQUITY,
                 "positions": {}, "started": utcnow()}

    prev_positions = state.get("positions", {})
    trades = []
    for asset, wt in last_w.items():
        prev = prev_positions.get(asset, 0.0)
        if abs(wt - prev) > 1e-9:
            trades.append({"dt": utcnow(), "signal_date": last_date, "asset": asset,
                           "target_weight": wt, "prev_weight": prev,
                           "note": "execute at next daily open (paper)"})
    state["positions"] = {k: float(v) for k, v in last_w.items() if abs(v) > 1e-9}
    state["last_signal_date"] = last_date
    state["last_run_utc"] = utcnow()
    STATE.write_text(json.dumps(state, indent=2))

    header = not LOG.exists()
    with open(LOG, "a") as f:
        if header:
            f.write("run_utc,strategy,signal_date,asset,target_weight,prev_weight,note\n")
        for t in trades:
            f.write(",".join(str(t[k]) for k in
                             ["run_utc", "strategy", "signal_date", "asset",
                              "target_weight", "prev_weight", "note"]) + "\n")
        if not trades:
            f.write(f"{utcnow()},{args.strategy},{last_date},NO_TRADE,0,0,hold\n")
    print(f"paper run complete: strategy={args.strategy} signal_date={last_date} "
          f"trades={len(trades)} state={STATE}")


if __name__ == "__main__":
    main()

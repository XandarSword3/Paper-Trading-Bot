"""
Funding-Rate Carry — forward paper-track (pre-gate diagnostic).

This is NOT a live-trading bot and never places an order on any exchange.
It does one thing: poll OKX's public funding-rate-history endpoint (the
only funding-rate source confirmed reachable from GitHub-hosted runners —
see okx_funding_rate_fetcher.py), find any settlement(s) not yet recorded,
and apply them to a virtual position so the strategy's real forward
(out-of-sample, non-fabricatable) performance starts accumulating.

Why it does NOT call research.validation.readiness_gate.check_gate():
that gate exists to stop a bot from ACTING on an unvalidated strategy.
This script never acts — it has no order-placement path at all, paper or
live. It is the thing that eventually produces the walk-forward/paper
track record Phase 5's readiness file is supposed to be built from, so
gating it on a readiness file would be circular. If this is ever extended
to place simulated orders against a broker/testnet, that extension must
call check_gate("funding_carry") first — this diagnostic tracker must not.

Capital accounting is deliberately NOT the naive "funding_rate * notional"
model in funding_rate_carry.py. A real long-spot/short-perp hedge ties up
capital for BOTH legs. Default here is the honest, no-leverage case:
real_capital = 2x notional. See CAPITAL_MULTIPLIER below to model a
leveraged variant instead — but know that leverage on the perp leg adds
liquidation risk this script still does not and cannot model from a
funding-rate print alone. That risk only ever shows up by actually running
the hedge on an exchange.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

STATE_PATH = DATA_DIR / "funding_carry_paper_state.json"
TRADES_PATH = DATA_DIR / "funding_carry_paper_trades.json"

OKX_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
INST_ID = "BTC-USDT-SWAP"

INITIAL_NOTIONAL = 100_000.0
CAPITAL_MULTIPLIER = 2.0  # honest, unlevered, full-collateral hedge (see docstring)


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    real_capital = INITIAL_NOTIONAL * CAPITAL_MULTIPLIER
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "notional": INITIAL_NOTIONAL,
        "capital_multiplier": CAPITAL_MULTIPLIER,
        "real_capital": real_capital,
        "equity": real_capital,
        "last_settlement_applied": None,
        "num_settlements": 0,
    }


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_trades() -> list:
    if TRADES_PATH.exists():
        with open(TRADES_PATH) as f:
            return json.load(f)
    return []


def save_trades(trades: list) -> None:
    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def fetch_recent_settlements(limit: int = 20) -> pd.DataFrame:
    """Pull the most recent `limit` funding settlements from OKX. Mirrors
    OKXFundingRateFetcher's parsing exactly, but single-page (no `before`
    pagination) since we only need what's new since the last run."""
    resp = requests.get(OKX_URL, params={"instId": INST_ID, "limit": limit}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "0":
        raise ValueError(f"OKX API error: {payload.get('msg')}")
    data = payload.get("data", [])
    if not data:
        raise ValueError("OKX returned no funding-rate rows")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    n_bad = df["funding_rate"].isna().sum()
    if n_bad:
        print(f"WARNING: {n_bad} settlement(s) had unparseable fundingRate — dropping them")
        df = df.dropna(subset=["funding_rate"])
    df = df.set_index("timestamp")[["funding_rate"]]
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


def run(dry_run: bool = False) -> dict:
    state = load_state()
    trades = load_trades()

    df = fetch_recent_settlements()

    last_applied = (
        pd.Timestamp(state["last_settlement_applied"], tz="UTC")
        if state["last_settlement_applied"] else None
    )
    new_rows = df[df.index > last_applied] if last_applied is not None else df

    if new_rows.empty:
        print("No new settlements since last run.")
        return state

    for ts, row in new_rows.iterrows():
        rate = float(row["funding_rate"])
        # Clip mirrors funding_rate_carry.py's guard against a bad data
        # point corrupting the whole track — real settlements never
        # approach this bound.
        rate = max(min(rate, 0.20), -0.20)
        pnl = state["equity"] * rate * (state["notional"] / state["real_capital"])
        state["equity"] += pnl
        state["num_settlements"] += 1
        state["last_settlement_applied"] = ts.isoformat()

        trades.append({
            "timestamp": ts.isoformat(),
            "funding_rate": rate,
            "pnl": pnl,
            "equity_after": state["equity"],
        })
        print(f"{ts} funding_rate={rate:.6f} pnl={pnl:+.2f} equity={state['equity']:.2f}")

    if not dry_run:
        save_state(state)
        save_trades(trades)

    total_return_pct = (state["equity"] - state["real_capital"]) / state["real_capital"] * 100
    print(f"\nApplied {len(new_rows)} new settlement(s). "
          f"Real capital=${state['real_capital']:,.0f}  "
          f"Equity=${state['equity']:,.2f}  "
          f"Total return={total_return_pct:+.3f}%")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Fetch and print but don't persist state/trades")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

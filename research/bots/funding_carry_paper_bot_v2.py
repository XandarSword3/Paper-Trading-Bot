"""
Funding-Rate Carry — forward paper-track v2 (pre-gate diagnostic).

Same non-negotiable rule as v1: this is NOT a live-trading bot and never
places an order on any exchange. It polls OKX's public endpoints (the
only ones confirmed reachable from GitHub-hosted runners — see
okx_funding_rate_fetcher.py) and applies whatever actually printed to a
virtual position. It does not call research.validation.readiness_gate
for the same reason v1 doesn't: this script has no action path, so
gating it on a readiness file it exists to help produce would be
circular.

WHAT V2 ADDS OVER V1, AND WHY V1 WASN'T ENOUGH
------------------------------------------------
v1 (funding_carry_paper_bot.py) is `state["equity"] += pnl` where pnl is
the raw funding rate applied to equity. That's a compounding calculator
bolted onto a real data feed, not a trading simulation — it can't
disagree with the backtest because it never has a chance to. Three
concrete gaps, closed in priority order:

1. FEES. v1 charges nothing to open or close. This file applies the
   same commission_pct=0.08 + slippage_pct=0.05 (= 0.13%) per-leg
   convention already used everywhere else in this repo (see
   strategies/strategy.py:apply_costs, strategies/config.py) to BOTH
   legs (spot + perp) on BOTH open and close. Four fee events total per
   position lifecycle.

2. PRICE + MARGIN + LIQUIDATION. v1 never looks at BTC's price, so it
   structurally cannot liquidate — it never models a margin account at
   all. This file fetches OKX's public mark price alongside funding,
   and simulates an actual isolated-margin perp account (separately
   from the spot leg) with a maintenance-margin check every poll. If
   the perp leg's isolated margin would have been wiped out, this
   records a paper liquidation — the spot leg is then flattened too
   (a hedge that's lost its other leg isn't a hedge anymore, and a
   responsible trader would close it rather than run a naked directional
   position by accident). A `--margin-mode cross` flag models the
   alternative: spot and perp share one equity pool, which structurally
   can't be liquidated by price alone under a matched hedge (only by a
   sustained run of negative funding) — this is the isolated-vs-cross
   trade-off from the earlier margin-mode research made visible, not
   just discussed.

3. REALISTIC SIZING. v1 hardcodes $100k notional. This file still
   defaults to $100k (so v1 and v2 stay comparable at default settings)
   but the notional is a single `--notional` flag, and — this is the
   part that actually matters — every run checks the resulting BTC
   size against OKX's real minimum order granularity for
   BTC-USDT-SWAP (0.01 BTC, confirmed against live trade prints; see
   MIN_LOT_BTC below). At $100k that friction is invisible (~$900 out
   of $100k). It only bites once notional is small enough that
   `notional / price` doesn't round cleanly to a 0.01 BTC lot — which
   is most of the point of adding the check at all. Set --notional to
   whatever you'd actually deploy to see if it bites.

WHAT THIS STILL DOESN'T MODEL (documented so nobody mistakes v2 for a
full exchange replica):
- MAINTENANCE_MARGIN_RATE below is an approximate tier-1 rate, not
  fetched live from OKX's tiered schedule (that schedule is on a page
  this fetcher doesn't parse). If you're using this to size a real
  position, pull the real tier-1 MMR for BTC-USDT-SWAP from
  https://www.okx.com/trade-market/position/margin first.
- Funding settlements are applied at the mark price fetched at RUN
  TIME, not the price that actually prevailed at each settlement
  instant. Across an 8h polling cadence with one settlement per poll
  this is a small approximation; it would matter more if this ever
  polled less often than it settles.
- OKX's actual liquidation engine has insurance-fund mechanics, partial
  liquidation tiers, and ADL — this models a single all-or-nothing
  liquidation of the isolated margin balance, which is the right order
  of magnitude but not the exact cent.
- Cross mode here approximates OKX's portfolio/multi-currency margin
  (spot BTC counted as collateral for the perp). Default OKX cross
  margin does NOT do this automatically — you'd need that mode enabled.
"""
import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

STATE_PATH = DATA_DIR / "funding_carry_paper_state_v2.json"
TRADES_PATH = DATA_DIR / "funding_carry_paper_trades_v2.json"

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OKX_MARK_PRICE_URL = "https://www.okx.com/api/v5/public/mark-price"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
INST_ID = "BTC-USDT-SWAP"

# ---- Problem 1: fees, matching the repo-wide convention exactly -----------
COMMISSION_PCT = 0.08   # strategies/config.py DEFAULT_PARAMS.commission_pct
SLIPPAGE_PCT = 0.05     # strategies/config.py DEFAULT_PARAMS.slippage_pct
FEE_PCT_PER_LEG = COMMISSION_PCT + SLIPPAGE_PCT  # 0.13%, charged on open AND close

# ---- Problem 2: margin / liquidation --------------------------------------
# Approximate tier-1 maintenance margin ratio for BTC-USDT-SWAP. This is a
# documented approximation, not a live value — see module docstring.
MAINTENANCE_MARGIN_RATE = 0.004  # 0.4%

# ---- Problem 3: realistic sizing -------------------------------------------
# Minimum BTC-USDT-SWAP order granularity. Verified against live trade prints
# on OKX (sz increments observed: 0.01, 0.02, 0.21, 2.01 ...), i.e. contracts
# trade in 0.01 BTC steps. Confirm against the live /public/instruments
# endpoint (ctVal/lotSz) if this ever needs to be exact to the cent.
MIN_LOT_BTC = 0.01

DEFAULT_NOTIONAL = 100_000.0  # PLACEHOLDER — set --notional to your real size
DEFAULT_LEVERAGE = 1.0        # 1x matches v1's "honest, unlevered" default
DEFAULT_MARGIN_MODE = "isolated"


# ----------------------------------------------------------------------------
# State I/O
# ----------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return None  # no open (or ever-opened) position yet


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


def archive_closed_position(state: dict) -> None:
    """A closed/liquidated position's state+trades get archived (not
    overwritten) so the record of what happened survives a --reopen."""
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if STATE_PATH.exists():
        shutil.move(str(STATE_PATH), str(DATA_DIR / f"funding_carry_paper_state_v2_closed_{ts_tag}.json"))
    if TRADES_PATH.exists():
        shutil.move(str(TRADES_PATH), str(DATA_DIR / f"funding_carry_paper_trades_v2_closed_{ts_tag}.json"))


# ----------------------------------------------------------------------------
# OKX data
# ----------------------------------------------------------------------------

def fetch_recent_settlements(limit: int = 20) -> pd.DataFrame:
    """Identical parsing to v1 / OKXFundingRateFetcher."""
    resp = requests.get(OKX_FUNDING_URL, params={"instId": INST_ID, "limit": limit}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "0":
        raise ValueError(f"OKX funding API error: {payload.get('msg')}")
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


def fetch_mark_price() -> float:
    """Fetch BTC-USDT-SWAP mark price — the price OKX itself uses for
    margin-ratio and liquidation checks (not last-traded price). Falls
    back to last-traded ticker price if the mark-price endpoint's shape
    ever doesn't match, rather than silently crashing the whole poll."""
    try:
        resp = requests.get(OKX_MARK_PRICE_URL, params={"instType": "SWAP", "instId": INST_ID}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") == "0" and payload.get("data"):
            return float(payload["data"][0]["markPx"])
        print(f"WARNING: mark-price endpoint returned unexpected payload: {payload}")
    except Exception as e:
        print(f"WARNING: mark-price fetch failed ({e}); falling back to ticker last price")

    resp = requests.get(OKX_TICKER_URL, params={"instId": INST_ID}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "0" or not payload.get("data"):
        raise ValueError(f"OKX ticker fallback also failed: {payload}")
    return float(payload["data"][0]["last"])


# ----------------------------------------------------------------------------
# Position lifecycle
# ----------------------------------------------------------------------------

def round_down_to_lot(btc_amount: float, lot: float = MIN_LOT_BTC) -> float:
    return (int(btc_amount / lot)) * lot


def open_position(notional: float, leverage: float, margin_mode: str, mark_price: float) -> dict:
    size_btc = round_down_to_lot(notional / mark_price)
    if size_btc < MIN_LOT_BTC:
        raise ValueError(
            f"${notional:,.0f} at ${mark_price:,.0f}/BTC is below OKX's minimum "
            f"BTC-USDT-SWAP lot ({MIN_LOT_BTC} BTC ≈ ${MIN_LOT_BTC * mark_price:,.0f}). "
            f"Cannot open a hedged position at this size."
        )

    spot_notional = size_btc * mark_price
    perp_notional = size_btc * mark_price
    unhedged_residual = notional - spot_notional  # leftover cash the lot-rounding couldn't place

    entry_fee_spot = spot_notional * FEE_PCT_PER_LEG / 100
    entry_fee_perp = perp_notional * FEE_PCT_PER_LEG / 100

    perp_isolated_margin = perp_notional / leverage  # gross, before entry fee is deducted below

    state = {
        "version": 2,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "notional_target": notional,
        "unhedged_residual_usd": unhedged_residual,
        "leverage": leverage,
        "margin_mode": margin_mode,
        "size_btc": size_btc,
        "entry_price": mark_price,
        "entry_fee_spot": entry_fee_spot,
        "entry_fee_perp": entry_fee_perp,
        "spot_cash_basis": spot_notional,          # BTC bought at cost
        "perp_isolated_margin": perp_isolated_margin - entry_fee_perp,  # net of entry fee
        "cumulative_funding": 0.0,
        "num_settlements": 0,
        "last_settlement_applied": None,
        "position_open": True,
        "liquidated": False,
        "closed": False,
        "closed_reason": None,
    }
    return state


def mark_to_market(state: dict, mark_price: float) -> dict:
    """Returns a dict of derived values without mutating state — pure
    read so it can be called for reporting even after close."""
    size_btc = state["size_btc"]
    entry_price = state["entry_price"]

    spot_value = size_btc * mark_price
    spot_unrealized_pnl = spot_value - state["spot_cash_basis"]
    perp_unrealized_pnl = size_btc * (entry_price - mark_price)  # short: gains as price falls

    perp_equity = state["perp_isolated_margin"] + perp_unrealized_pnl
    maintenance_margin = size_btc * mark_price * MAINTENANCE_MARGIN_RATE

    if state["margin_mode"] == "cross":
        # Spot value and perp margin share one pool — a matched hedge's
        # price moves cancel here by construction; only cumulative funding
        # (already folded into perp_isolated_margin) moves this number.
        combined_equity = spot_value + state["perp_isolated_margin"] + perp_unrealized_pnl
        buffer_equity = combined_equity  # whole account backs the perp leg
    else:
        combined_equity = spot_value + perp_equity
        buffer_equity = perp_equity  # only the walled-off isolated margin backs the perp leg

    return {
        "spot_value": spot_value,
        "spot_unrealized_pnl": spot_unrealized_pnl,
        "perp_unrealized_pnl": perp_unrealized_pnl,
        "perp_equity": perp_equity,
        "maintenance_margin": maintenance_margin,
        "combined_equity": combined_equity,
        "would_liquidate": buffer_equity <= maintenance_margin,
    }


def close_position(state: dict, mark_price: float, reason: str) -> dict:
    """Flatten both legs at mark_price, charging exit fees on both, and
    realize final equity. Mutates and returns state."""
    size_btc = state["size_btc"]
    mtm = mark_to_market(state, mark_price)

    exit_fee_spot = mtm["spot_value"] * FEE_PCT_PER_LEG / 100
    exit_fee_perp = (size_btc * mark_price) * FEE_PCT_PER_LEG / 100

    if reason == "liquidated":
        # Isolated margin balance is seized on liquidation — realistically
        # you get back ~0 of it, not the (already near-zero, possibly
        # slightly negative) mark-to-market perp_equity. Model the full
        # loss of what was posted, which is the honest worst case.
        final_perp_realized = 0.0
        state["liquidated"] = True
    else:
        final_perp_realized = mtm["perp_equity"] - exit_fee_perp

    final_spot_realized = mtm["spot_value"] - exit_fee_spot
    final_equity = final_spot_realized + final_perp_realized

    state["position_open"] = False
    state["closed"] = True
    state["closed_reason"] = reason
    state["closed_at"] = datetime.now(timezone.utc).isoformat()
    state["close_price"] = mark_price
    state["exit_fee_spot"] = exit_fee_spot
    state["exit_fee_perp"] = exit_fee_perp
    state["final_equity"] = final_equity
    return state


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------

def run(notional: float, leverage: float, margin_mode: str, dry_run: bool = False, reopen: bool = False) -> dict:
    state = load_state()
    trades = load_trades() if state is not None else []

    if state is not None and state.get("closed") and not reopen:
        print(
            f"Position already closed ({state['closed_reason']} at ${state.get('close_price', 0):,.2f}, "
            f"final equity ${state.get('final_equity', 0):,.2f}). Pass --reopen to start a fresh position "
            f"(the closed one is archived, not deleted)."
        )
        return state

    if state is not None and state.get("closed") and reopen:
        archive_closed_position(state)
        state = None
        trades = []

    mark_price = fetch_mark_price()

    if state is None:
        state = open_position(notional=notional, leverage=leverage, margin_mode=margin_mode, mark_price=mark_price)
        print(
            f"OPENED: {state['size_btc']} BTC each leg @ ${mark_price:,.2f} "
            f"(target ${notional:,.0f}, unhedged residual ${state['unhedged_residual_usd']:,.2f}) "
            f"leverage={leverage}x margin_mode={margin_mode} "
            f"entry_fees=${state['entry_fee_spot'] + state['entry_fee_perp']:,.2f}"
        )
        trades.append({
            "timestamp": state["started_at"],
            "event": "open",
            "mark_price": mark_price,
            "size_btc": state["size_btc"],
            "fees": state["entry_fee_spot"] + state["entry_fee_perp"],
        })
        if not dry_run:
            save_state(state)
            save_trades(trades)
        return state

    df = fetch_recent_settlements()
    last_applied = (
        pd.Timestamp(state["last_settlement_applied"], tz="UTC")
        if state["last_settlement_applied"] else None
    )
    new_rows = df[df.index > last_applied] if last_applied is not None else df

    for ts, row in new_rows.iterrows():
        rate = float(row["funding_rate"])
        rate = max(min(rate, 0.20), -0.20)  # same corrupt-data guard as v1

        funding_payment = state["size_btc"] * mark_price * rate  # short receives when rate > 0
        state["perp_isolated_margin"] += funding_payment
        state["cumulative_funding"] += funding_payment
        state["num_settlements"] += 1
        state["last_settlement_applied"] = ts.isoformat()

        trades.append({
            "timestamp": ts.isoformat(),
            "event": "funding",
            "funding_rate": rate,
            "mark_price_used": mark_price,
            "funding_payment": funding_payment,
            "cumulative_funding": state["cumulative_funding"],
        })
        print(f"{ts} funding_rate={rate:.6f} mark_price(approx)=${mark_price:,.2f} "
              f"payment={funding_payment:+.2f} cum_funding={state['cumulative_funding']:+.2f}")

    mtm = mark_to_market(state, mark_price)

    if mtm["would_liquidate"] and not state["liquidated"]:
        state = close_position(state, mark_price, reason="liquidated")
        print(
            f"*** LIQUIDATED *** perp isolated margin wiped out @ ${mark_price:,.2f} "
            f"(margin_mode={state['margin_mode']}). Spot leg flattened too "
            f"(hedge is broken, not left running naked). Final equity=${state['final_equity']:,.2f}"
        )
        trades.append({
            "timestamp": state["closed_at"],
            "event": "liquidated",
            "mark_price": mark_price,
            "final_equity": state["final_equity"],
        })
    else:
        print(
            f"MTM @ ${mark_price:,.2f}: perp_equity=${mtm['perp_equity']:,.2f} "
            f"maintenance_req=${mtm['maintenance_margin']:,.2f} "
            f"combined_equity=${mtm['combined_equity']:,.2f}"
        )
        trades.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "mark",
            "mark_price": mark_price,
            "perp_equity": mtm["perp_equity"],
            "maintenance_margin": mtm["maintenance_margin"],
            "combined_equity": mtm["combined_equity"],
        })

    if not dry_run:
        save_state(state)
        save_trades(trades)

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Fetch and print but don't persist state/trades")
    parser.add_argument("--notional", type=float, default=DEFAULT_NOTIONAL,
                         help="Target notional per leg in USD. PLACEHOLDER default of "
                              "$100k matches v1 for comparability — set this to your real "
                              "deployment size to see if lot-size friction actually bites.")
    parser.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE,
                         help="Perp leg leverage. 1x (default) matches v1's honest/unlevered "
                              "case. Raise this to explore paper liquidation scenarios.")
    parser.add_argument("--margin-mode", choices=["isolated", "cross"], default=DEFAULT_MARGIN_MODE,
                         help="isolated: perp margin is walled off from the spot leg (can be "
                              "liquidated by price alone even with a perfect hedge). cross: "
                              "spot+perp share one equity pool (approximates OKX portfolio "
                              "margin; only sustained negative funding can drain it).")
    parser.add_argument("--reopen", action="store_true",
                         help="If the current position is closed/liquidated, archive it and "
                              "open a fresh one instead of refusing to run.")
    args = parser.parse_args()
    run(notional=args.notional, leverage=args.leverage, margin_mode=args.margin_mode,
        dry_run=args.dry_run, reopen=args.reopen)

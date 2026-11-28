"""Strategy B — cash-and-carry futures basis carry.

Long spot + short dated (quarterly) futures, hold to expiry, collect basis
convergence. NO funding (dated futures don't pay funding) — basis only,
so no double-counting with Strategy A.

FROZEN RULES (fixed before any run):
- Universe: BTC and ETH Binance USD-M quarterly futures vs OKX USDT spot.
- Contract: nearest expiry with days_to_expiry > 14.
- Entry: only when annualized basis (F/S-1)*365/dte > MIN_ANN_BASIS = 3%/yr.
- Size: spot_notional = equity * 0.95 / (1 + IMR); futures notional matched.
  IMR = 0.20 (initial margin ratio, conservative; sensitivity-tested).
- Margin: isolated futures account, MMR_F = 0.50% of notional (conservative
  vs typical quarterly tier-1). Daily check on futures HIGH; top up from
  reserve if equity_f < maintenance; liquidate if reserve exhausted.
- Roll: at expiry, if next contract basis > MIN_ANN_BASIS re-enter same day.
- Costs: spot 0.15%/side (0.10 taker + 0.05 slip), futures 0.10%/side
  (0.05 taker + 0.05 slip). Paid on entry, exit, and every roll.
- Capital $1,000 per base evaluated independently.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
from engine import load_spot, START_EQUITY  # noqa: E402

QDIR = ROOT / "data" / "quarterly"

SPOT_COST = 0.0015   # per side
FUT_COST = 0.0010    # per side
IMR = 0.20           # initial margin ratio on futures notional
MMR_F = 0.005        # maintenance margin ratio (conservative)
MIN_ANN_BASIS = 0.03 # 3%/yr annualized basis threshold
DEPLOY = 0.95        # fraction of equity deployed per cycle
MIN_DTE = 14         # days to expiry minimum


def load_contracts(base):
    """Return {expiry_date: DataFrame(dt,close)} for all quarterly contracts."""
    out = {}
    for p in sorted(QDIR.glob(f"{base}USDT_*_1D.csv")):
        exp_str = p.name.split("_")[1]  # YYMMDD
        exp = pd.Timestamp(f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}", tz="UTC")
        df = pd.read_csv(p, parse_dates=["dt"]).set_index("dt").sort_index()
        out[exp] = df
    return out


def run_cash_carry(base, imr=IMR, min_basis=MIN_ANN_BASIS, cost_mult=1.0,
                   start=None, end=None):
    spot = load_spot(base)
    contracts = load_contracts(base)
    exps = sorted(contracts)
    # union index = spot dates
    idx = spot.index
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]

    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []
    reserve = START_EQUITY
    # position state
    pos = None  # dict when in market
    cost_total = 0.0
    max_util = 0.0
    liq_events = []
    basis_history = []

    def fut_price(exp, d):
        df = contracts[exp]
        if d in df.index:
            return float(df.loc[d, "close"]), float(df.loc[d, "high"])
        # use last available <= d
        sub = df[df.index <= d]
        if sub.empty:
            return None, None
        return float(sub["close"].iloc[-1]), float(sub["high"].iloc[-1])

    def spot_px(d):
        return float(spot.loc[d, "close"]) if d in spot.index else None

    def ann_basis(exp, d):
        s = spot_px(d)
        f, _ = fut_price(exp, d)
        if s is None or f is None:
            return None
        dte = (exp - d).days
        if dte <= 0:
            return None
        return (f / s - 1) * 365 / dte

    for d in idx:
        # current contract = nearest expiry with dte > MIN_DTE
        cur_exp = None
        for e in exps:
            if (e - d).days > MIN_DTE:
                cur_exp = e
                break
        s_close = spot_px(d)
        if s_close is None:
            continue

        # ---- if in position ----
        if pos is not None:
            exp = pos["exp"]
            f_close, f_high = fut_price(exp, d)
            # expiry handling: settle at expiry day's close
            if d >= exp:
                # realize: futures converges; use last fut price <= exp
                f_settle, _ = fut_price(exp, min(d, exp))
                if f_settle is None:
                    f_settle = s_close
                # close futures
                fut_pnl = pos["fut_qty"] * (pos["f_entry"] - f_settle)
                fut_fee = pos["fut_qty"] * f_settle * FUT_COST * cost_mult
                reserve += pos["margin"] + fut_pnl - fut_fee
                cost_total += fut_fee
                # close spot
                spot_proceeds = pos["spot_qty"] * s_close
                spot_fee = spot_proceeds * SPOT_COST * cost_mult
                reserve += spot_proceeds - spot_fee
                cost_total += spot_fee
                trades.append({"dt": d, "action": "EXPIRY_SETTLE", "exp": str(exp.date()),
                               "fut_pnl": fut_pnl, "basis_realized": pos.get("basis_at_entry")})
                pos = None
            else:
                # daily margin check on futures HIGH
                if f_high is not None:
                    eq_f_high = pos["margin"] + pos["fut_qty"] * (pos["f_entry"] - f_high)
                    mm_req = MMR_F * pos["fut_qty"] * f_high
                    if eq_f_high <= mm_req:
                        # try top-up from reserve
                        need = mm_req - eq_f_high + 1.0
                        take = min(need, max(reserve, 0.0))
                        reserve -= take; pos["margin"] += take
                        if pos["margin"] + pos["fut_qty"] * (pos["f_entry"] - f_high) <= mm_req:
                            # liquidated
                            loss = pos["margin"]
                            reserve += max(0.0, pos["margin"] + pos["fut_qty"] * (pos["f_entry"] - f_high) - mm_req)
                            pos["margin"] = 0.0
                            liq_events.append({"dt": d, "exp": str(exp.date()), "loss": loss})
                            trades.append({"dt": d, "action": "LIQUIDATED", "exp": str(exp.date())})
                            # keep spot? strategy is delta-neutral; on liq we hold naked spot.
                            # Conservative: also sell spot to go flat.
                            spot_proceeds = pos["spot_qty"] * s_close
                            spot_fee = spot_proceeds * SPOT_COST * cost_mult
                            reserve += spot_proceeds - spot_fee
                            cost_total += spot_fee
                            pos = None
                # mark
                if pos is not None and f_close is not None:
                    fut_eq = pos["margin"] + pos["fut_qty"] * (pos["f_entry"] - f_close)
                    spot_val = pos["spot_qty"] * s_close
                    E = reserve + spot_val + fut_eq
                    equity.loc[d] = E
                    util = (E - reserve) / E if E > 0 else 0.0
                    max_util = max(max_util, util)
                    continue

        # ---- if flat, consider entry ----
        if pos is None and cur_exp is not None:
            ab = ann_basis(cur_exp, d)
            if ab is not None and ab > min_basis:
                f_entry, _ = fut_price(cur_exp, d)
                if f_entry is None:
                    continue
                E = reserve  # all cash when flat
                X = E * DEPLOY / (1 + imr)   # spot notional
                margin = X * imr * (f_entry / s_close)  # futures notional ~ X
                # actually futures notional = fut_qty*f_entry; match to X
                fut_qty = X / f_entry
                margin = fut_qty * f_entry * imr
                if margin + X * (1 + SPOT_COST * cost_mult) > reserve:
                    continue
                spot_fee = X * SPOT_COST * cost_mult
                fut_fee = fut_qty * f_entry * FUT_COST * cost_mult
                reserve -= X + spot_fee + fut_fee + margin
                cost_total += spot_fee + fut_fee
                pos = {"exp": cur_exp, "spot_qty": X / s_close, "fut_qty": fut_qty,
                       "f_entry": f_entry, "margin": margin, "basis_at_entry": ab}
                trades.append({"dt": d, "action": "ENTRY", "exp": str(cur_exp.date()),
                               "basis_ann": ab, "spot_notional": X})
                basis_history.append({"dt": d, "exp": str(cur_exp.date()), "ann_basis": ab})

        # ---- mark if still flat ----
        if pos is None:
            equity.loc[d] = reserve

    eq = equity.dropna()
    meta = {
        "final_equity": float(eq.iloc[-1]) if len(eq) else START_EQUITY,
        "cost_total": float(cost_total),
        "n_liq": len(liq_events), "liq_events": liq_events,
        "max_util": float(max_util),
        "trades": pd.DataFrame(trades),
        "basis_history": pd.DataFrame(basis_history),
    }
    return eq, meta

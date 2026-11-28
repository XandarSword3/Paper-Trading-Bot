"""Strategy B v2 — cash-and-carry with audited margin structures.

Preserves v1 (backtest/cash_carry.py) untouched. Adds:
- Detailed liquidation-event records (all fields needed for audit).
- Two margin structures:
    isolated : USDT margin = notional*IMR posted to a segregated futures
               account; liquidation when isolated equity <= MMR*notional.
    cross    : Binance USD-M Multi-Assets Mode — the spot asset itself is held
               in the futures wallet as collateral at COLLATERAL_HAIRCUT=0.95
               (official: BTC/ETH/BNB valued at 95%, cross-margin only,
               must be transferred into the USD-M futures wallet).
               Liquidation when (USDT wallet + spot*haircut + fut_uPnL)
               <= MMR*notional.
- Liquidation Clearance Fee = 1.25% of futures notional (Binance USD-M perp
  schedule; delivery-futures rate assumed identical — UNVERIFIED, flagged).
- Mark price proxy: futures daily HIGH (real mark = index + 30s basis MA;
  on delivery day last 30min = index average — approximated, flagged).
- PnL decomposition: initial basis, convergence, spot PnL, futures PnL,
  transaction costs, liquidation losses.
- flat_days: days out of market (after liquidation until re-entry). Spot is
  sold at the liquidation-day close, so naked-spot exposure is intraday-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
from engine import load_spot, START_EQUITY  # noqa: E402

Q4H = ROOT / "data" / "quarterly_4h"


def load_contracts_4h(base):
    """{expiry Timestamp: DataFrame(dt,open,high,low,close)} of 4H klines."""
    out = {}
    for p in sorted(Q4H.glob(f"{base}USDT_*_4H.csv")):
        exp_str = p.name.split("_")[1]
        exp = pd.Timestamp(f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}", tz="UTC")
        df = pd.read_csv(p, parse_dates=["dt"]).set_index("dt").sort_index()
        out[exp] = df
    return out

SPOT_COST = 0.0015
FUT_COST = 0.0010
MMR_F = 0.005           # maintenance margin rate (assumption; perp tier-1 = 0.40%)
LIQ_FEE_RATE = 0.0125   # Binance USD-M liquidation clearance fee (perp schedule)
COLLATERAL_HAIRCUT = 0.95  # Multi-Assets Mode ETH/BTC collateral value ratio
MIN_ANN_BASIS = 0.03
DEPLOY = 0.95
MIN_DTE = 14


def run_v2(base, mode="isolated", imr=0.20, cost_mult=1.0,
           start=None, end=None, liq_fee_rate=LIQ_FEE_RATE):
    assert mode in ("isolated", "cross")
    spot = load_spot(base)
    contracts = load_contracts_4h(base)
    exps = sorted(contracts)
    idx = spot.index
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]

    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []
    liq_events = []
    usdt = START_EQUITY
    pos = None
    cost_total = 0.0
    max_util = 0.0
    flat_days = 0
    decomp = {"basis_entry_usd": 0.0, "convergence_usd": 0.0,
              "spot_pnl": 0.0, "fut_pnl": 0.0, "costs": 0.0, "liq_fees": 0.0}

    def fut_price(exp, d):
        """OKX spot candle labelled d (16:00 UTC) OPENS at d and CLOSES at
        d+24h. Match the futures 4H candle closing at d+24h (ts = d+20h);
        high = max of the six 4H candles covering (d, d+24h]."""
        df = contracts[exp]
        target = d + pd.Timedelta(hours=20)
        sub = df[df.index <= target]
        if sub.empty:
            return None, None
        close = float(sub["close"].iloc[-1])
        win = df[(df.index >= d) & (df.index <= target)]
        high = float(win["high"].max()) if len(win) else float(sub["high"].iloc[-1])
        return close, high

    def spot_px(d):
        return float(spot.loc[d, "close"]) if d in spot.index else None

    def ann_basis(exp, d):
        s = spot_px(d); f, _ = fut_price(exp, d)
        if s is None or f is None:
            return None
        dte = (exp - d).days
        if dte <= 0:
            return None
        return (f / s - 1) * 365 / dte

    def fut_pnl(p, price):
        return p["fut_qty"] * (p["f_entry"] - price)

    for d in idx:
        s_close = spot_px(d)
        if s_close is None:
            continue
        cur_exp = next((e for e in exps if (e - d).days > MIN_DTE), None)

        # ---------------- in position ----------------
        if pos is not None:
            exp = pos["exp"]
            # settle on the spot candle covering the 08:00 UTC expiry
            # settlement (label = exp - 8h, i.e. previous day 16:00)
            if d >= exp - pd.Timedelta(hours=8):
                # ---- settlement ----
                # Binance USD-M quarterly settles at 08:00 UTC on expiry day;
                # use the 4H candle closing at exp+8h (ts = exp+4h).
                df_exp = contracts[exp]
                tgt = exp + pd.Timedelta(hours=4)
                sub = df_exp[df_exp.index <= tgt]
                f_settle = float(sub["close"].iloc[-1]) if len(sub) else s_close
                fp = fut_pnl(pos, f_settle)
                ff = pos["fut_qty"] * f_settle * FUT_COST * cost_mult
                sp = pos["spot_qty"] * (s_close - pos["s_entry"])
                sf = pos["spot_qty"] * s_close * SPOT_COST * cost_mult
                basis_end = pos["fut_qty"] * (f_settle - s_close)
                conv = pos["basis_entry_usd"] - basis_end
                decomp["fut_pnl"] += fp; decomp["spot_pnl"] += sp
                decomp["convergence_usd"] += conv
                decomp["costs"] += ff + sf
                cost_total += ff + sf
                if mode == "isolated":
                    usdt += pos["margin"] + fp - ff
                else:
                    usdt += fp - ff
                usdt += pos["spot_qty"] * s_close - sf
                trades.append({"dt": d, "action": "EXPIRY_SETTLE", "exp": str(exp.date()),
                               "fut_pnl": fp, "spot_pnl": sp, "convergence": conv,
                               "basis_entry_usd": pos["basis_entry_usd"]})
                pos = None
            else:
                f_close, f_high = fut_price(exp, d)
                if f_high is not None:
                    mm_req = MMR_F * pos["fut_qty"] * f_high
                    fp_high = fut_pnl(pos, f_high)
                    if mode == "isolated":
                        collateral_high = pos["margin"] + fp_high
                    else:
                        collateral_high = (usdt + pos["spot_qty"] * s_close * COLLATERAL_HAIRCUT
                                           + fp_high)
                    if mode == "isolated" and collateral_high <= mm_req:
                        # margin defense: top up isolated margin from wallet first
                        take = min(mm_req - collateral_high + 1.0, max(usdt, 0.0))
                        usdt -= take; pos["margin"] += take; collateral_high += take
                    if collateral_high <= mm_req:
                        # ---------------- LIQUIDATION ----------------
                        # The daily HIGH only tells us liquidation happened
                        # sometime that day. Settle at the ANALYTIC liq price
                        # (equity = maintenance margin), not at the high —
                        # using the high overshoots into bankruptcy and books
                        # phantom gains/losses.
                        qty = pos["f_qty"] if "f_qty" in pos else pos["fut_qty"]
                        if mode == "isolated":
                            # margin + qty*(f_entry - P) = MMR*qty*P
                            p_liq = (pos["margin"] + qty * pos["f_entry"]) / (qty * (1 + MMR_F))
                            eq_liq = pos["margin"] + qty * (pos["f_entry"] - p_liq)
                        else:
                            coll0 = (usdt + pos["spot_qty"] * s_close * COLLATERAL_HAIRCUT
                                     + qty * pos["f_entry"])
                            p_liq = coll0 / (qty * (1 + MMR_F))
                            eq_liq = MMR_F * qty * p_liq
                        liq_price = p_liq
                        notional = qty * liq_price
                        fee = notional * liq_fee_rate * cost_mult
                        fp_liq = qty * (pos["f_entry"] - liq_price)
                        sp_val = pos["spot_qty"] * s_close
                        sp_unreal = pos["spot_qty"] * (s_close - pos["s_entry"])
                        eq_before = (usdt + sp_val + pos["margin"] + fp_high
                                     if mode == "isolated"
                                     else usdt + sp_val + fp_high)
                        if mode == "isolated":
                            remaining = max(0.0, eq_liq - fee)
                            usdt += remaining
                            fut_account_loss = eq_liq - remaining
                        else:
                            usdt += fp_liq - fee
                            fut_account_loss = fee
                        sf = sp_val * SPOT_COST * cost_mult
                        usdt += sp_val - sf
                        cost_total += sf + fee
                        decomp["liq_fees"] += fee
                        decomp["costs"] += sf + fee
                        liq_events.append({
                            "dt": d, "mode": mode, "imr": imr,
                            "contract": str(exp.date()),
                            "spot_price": s_close,
                            "fut_mark_proxy_high": f_high,
                            "liq_price_est": liq_price,
                            "fut_notional": notional,
                            "initial_margin": (pos["margin"] if mode == "isolated"
                                               else notional * imr),
                            "maint_margin": mm_req,
                            "collateral_at_high": collateral_high,
                            "equity_at_liq_est": eq_liq,
                            "fut_unreal_pnl_at_liq": fp_liq,
                            "spot_unreal_pnl": sp_unreal,
                            "eq_before_liq": eq_before,
                            "fut_account_loss": fut_account_loss,
                            "liq_fee": fee, "spot_sell_fee": sf,
                            "spot_remaining": 0.0,
                            "eq_after": usdt,
                            "reentry": None,
                        })
                        trades.append({"dt": d, "action": "LIQUIDATED",
                                       "exp": str(exp.date()), "loss": fut_account_loss,
                                       "fee": fee})
                        pos = None
                if pos is not None and f_close is not None:
                    if mode == "isolated":
                        E = usdt + pos["spot_qty"] * s_close + pos["margin"] + fut_pnl(pos, f_close)
                    else:
                        E = usdt + pos["spot_qty"] * s_close + fut_pnl(pos, f_close)
                    equity.loc[d] = E
                    util = (E - usdt) / E if E > 0 else 0.0
                    max_util = max(max_util, util)
                    continue

        # ---------------- flat: consider entry ----------------
        if pos is None and cur_exp is not None:
            ab = ann_basis(cur_exp, d)
            if ab is not None and ab > MIN_ANN_BASIS:
                f_entry, _ = fut_price(cur_exp, d)
                if f_entry is not None:
                    E = usdt
                    X = E * DEPLOY / (1 + imr)
                    fut_qty = X / f_entry
                    margin = fut_qty * f_entry * imr
                    spot_fee = X * SPOT_COST * cost_mult
                    fut_fee = fut_qty * f_entry * FUT_COST * cost_mult
                    if mode == "isolated":
                        if X + spot_fee + fut_fee + margin <= usdt:
                            usdt -= X + spot_fee + fut_fee + margin
                            cost_total += spot_fee + fut_fee
                            decomp["costs"] += spot_fee + fut_fee
                            decomp["basis_entry_usd"] += fut_qty * (f_entry - s_close)
                            pos = {"exp": cur_exp, "spot_qty": X / s_close,
                                   "fut_qty": fut_qty, "f_entry": f_entry,
                                   "s_entry": s_close, "margin": margin,
                                   "basis_entry_usd": fut_qty * (f_entry - s_close)}
                            trades.append({"dt": d, "action": "ENTRY",
                                           "exp": str(cur_exp.date()), "basis_ann": ab,
                                           "spot_notional": X, "mode": mode, "imr": imr})
                            if liq_events and liq_events[-1].get("reentry") is None:
                                liq_events[-1]["reentry"] = str(d.date())
                    else:
                        # cross: no segregated margin; wallet backs the position
                        if X + spot_fee + fut_fee <= usdt:
                            usdt -= X + spot_fee + fut_fee
                            cost_total += spot_fee + fut_fee
                            decomp["costs"] += spot_fee + fut_fee
                            decomp["basis_entry_usd"] += fut_qty * (f_entry - s_close)
                            pos = {"exp": cur_exp, "spot_qty": X / s_close,
                                   "fut_qty": fut_qty, "f_entry": f_entry,
                                   "s_entry": s_close, "margin": 0.0,
                                   "basis_entry_usd": fut_qty * (f_entry - s_close)}
                            trades.append({"dt": d, "action": "ENTRY",
                                           "exp": str(cur_exp.date()), "basis_ann": ab,
                                           "spot_notional": X, "mode": mode, "imr": imr})
                            if liq_events and liq_events[-1].get("reentry") is None:
                                liq_events[-1]["reentry"] = str(d.date())
        if pos is None:
            flat_days += 1
            equity.loc[d] = usdt

    eq = equity.dropna()
    meta = {
        "final_equity": float(eq.iloc[-1]) if len(eq) else START_EQUITY,
        "cost_total": float(cost_total),
        "n_liq": len(liq_events), "liq_events": liq_events,
        "max_util": float(max_util), "flat_days": int(flat_days),
        "decomp": decomp,
        "trades": pd.DataFrame(trades),
    }
    return eq, meta

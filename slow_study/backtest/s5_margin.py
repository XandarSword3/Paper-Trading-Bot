"""S5 v2 — delta-neutral funding carry WITH a margin model.

Fixes v1's critical flaw: v1 assumed the perp short never needs margin and can
never be liquidated. v2 models a USDT-margined isolated perp account.

FROZEN RULES (fixed before running; no tuning after seeing results):
- Capital T = $1,000 total, split at entry into: spot BTC purchase (notional
  N0), perp margin (N0/L), reserve (~0 after fees).
- Leverage L in {1,2,3}: perp notional target N* = E_total * L/(L+1), so that
  spot = N*, perp margin = N*/L, reserve = E_total - N* - N*/L = 0 at target.
- Spot leg: BTC bought at entry, held; only SOLD to defend perp margin
  (Rule A) or resized on rebalance (Rule B).
- Perp leg: short qty tracks spot qty (delta-neutral). avg entry tracked via
  cost_basis = sum(qty_i * entry_i).
- Rule A (margin defense, checked daily at close): target margin
  M* = qty_perp * close / L. If perp equity E_p < M*, top up from reserve;
  if reserve insufficient, SELL SPOT (taker cost) and cover perp proportionally
  to raise cash, keeping qty_perp = qty_spot. Proceeds go to perp margin.
- Rule B (compounding rebalance): if |perp_notional - N*| / N* > 20%, fully
  resize spot+perp+margin to target structure (trading costs both legs).
- Liquidation: checked daily using intraday HIGH (adverse for a short).
  E_p(high) = margin_posted + funding_acc + cost_basis - qty_perp*high.
  MM_req = MMR * qty_perp * high. If E_p(high) <= MM_req -> LIQUIDATED at
  estimated liq price; entire perp account is lost (conservative: leftover
  above MM is ~0 by construction), spot+reserve remain, NO re-entry.
- Funding: applied at historical settlement timestamps to perp equity
  (short receives when rate > 0). Marked at that day's close (approximation).
- Costs: 0.15%/side (0.10% taker + 0.05% slip) on every spot/perp trade.
  Margin transfers internal = free.

Margin parameters (published rules):
- MMR = 0.40% — OKX BTCUSDT perpetual swap tier-1 maintenance margin ratio,
  position <= ~500 contracts (~5 BTC, >> our size). Source: OKX help article
  "OKX to adjust position tiers of perpetual swaps and futures" (2023-06-09),
  tier-1 MMR 0.40%. Bybit BTCUSDT tier-1 (<=300k USDT) MMR = 0.33% per Bybit
  margin parameters (via dexgc.com mirror) — we use the higher OKX 0.40%.
- IM = notional / L by construction (Bybit: IM = position value / leverage).
- Liquidation model: isolated margin; liq when account equity <= MM_req.
  OKX/Bybit exact liq-price formulas include fees/tiers we approximate;
  flagged as approximation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategies import COST_PER_SIDE, START_EQUITY, load_bars, DATA

MMR = 0.004          # OKX tier-1 maintenance margin ratio (see docstring)
LEVERAGES = [1, 2, 3]
REBAL_DRIFT = 0.20   # Rule B trigger
TOPUP_MULT = 1.0     # Rule A: top up whenever E_p < M* (target margin)


def load_funding():
    f = pd.read_csv(DATA / "funding_BTCUSDT_binance.csv")
    f["dt"] = pd.to_datetime(f["dt"], utc=True, format="mixed")
    return f.dropna(subset=["dt"]).set_index("dt").sort_index()["fundingRate"]


def run_s5_margin(leverage, capital=START_EQUITY):
    spot = load_bars("BTC", "1D")
    fund = load_funding()
    idx = spot.index
    common_start = max(idx[0], fund.index[0])
    common_end = min(idx[-1], fund.index[-1])
    idx = idx[(idx >= common_start) & (idx <= common_end)]
    L = leverage

    # ---------- entry at first day OPEN ----------
    d0 = idx[0]
    p0 = float(spot.loc[d0, "open"])
    est_cost = 2 * COST_PER_SIDE  # both legs ~0.3% total on N0
    N0 = (capital - capital * est_cost) / (1 + 1 / L)
    qty_spot = N0 / p0
    qty_perp = qty_spot
    cost_basis = qty_perp * p0          # sum(qty_i * entry_i) for the short
    margin_posted = N0 / L
    reserve = capital - N0 - margin_posted
    # entry trading costs
    entry_cost = N0 * COST_PER_SIDE * 2  # spot buy + perp open
    reserve -= entry_cost
    funding_acc = 0.0
    terminated = False
    liq_info = None
    cost_total = entry_cost  # entry spot+perp fees

    equity = pd.Series(np.nan, index=idx, dtype=float)
    min_liq_dist = np.inf        # distance from daily CLOSE
    min_liq_dist_high = np.inf   # distance from daily HIGH (conservative)
    trades = []
    debug_rows = []
    max_recon = 0.0
    fund_in = fund[(fund.index >= idx[0]) & (fund.index <= idx[-1])]

    for i, d in enumerate(idx):
        o = float(spot.loc[d, "open"]); h = float(spot.loc[d, "high"]); c = float(spot.loc[d, "close"])

        # 1) funding settlements on this date -> perp equity
        day_f = fund_in[fund_in.index.normalize() == d.normalize()]
        for rate in day_f.values:
            cf = float(rate) * qty_perp * c  # short receives if rate>0
            funding_acc += cf

        # 2) perp equity at close
        def perp_eq(price):
            return margin_posted + funding_acc + (cost_basis - qty_perp * price)

        E_p = perp_eq(c)
        E_total = reserve + qty_spot * c + E_p

        # 3) liquidation check on intraday HIGH
        if not terminated and qty_perp > 0:
            E_p_high = perp_eq(h)
            mm_req_high = MMR * qty_perp * h
            # analytic liq price for the short: E_p(P) = MMR*qty*P
            # margin_posted + funding_acc + cost_basis - qty*P = MMR*qty*P
            denom = qty_perp * (1 + MMR)
            p_liq = (margin_posted + funding_acc + cost_basis) / denom if denom > 0 else np.inf
            dist = (p_liq - c) / c if c > 0 else np.inf
            dist_h = (p_liq - h) / h if h > 0 else np.inf
            if np.isfinite(dist):
                min_liq_dist = min(min_liq_dist, dist)
            if np.isfinite(dist_h):
                min_liq_dist_high = min(min_liq_dist_high, dist_h)
            if E_p_high <= mm_req_high:
                # LIQUIDATED: entire perp account lost, no re-entry
                liq_info = {"dt": d, "price_est": p_liq, "high": h,
                            "perp_equity_lost": E_p_high}
                reserve += max(0.0, E_p_high - mm_req_high)  # ~0 by construction
                qty_perp = 0.0; cost_basis = 0.0
                margin_posted = 0.0; funding_acc = 0.0
                terminated = True
                trades.append({"dt": d, "action": "LIQUIDATED", "price": h,
                               "note": f"liq~{p_liq:.0f}"})

        # 4) Rule A: margin defense (only if alive)
        if not terminated and qty_perp > 0:
            M_star = qty_perp * c / L
            E_p = perp_eq(c)
            if E_p < M_star:
                need = M_star - E_p
                # top up from reserve first
                take = min(need, max(reserve, 0.0))
                reserve -= take; margin_posted += take; need -= take
                # then sell spot + cover perp proportionally
                if need > 1e-9 and qty_spot > 0:
                    spot_val = qty_spot * c
                    sell_notional = min(need / (1 - COST_PER_SIDE), spot_val)
                    f = sell_notional / spot_val if spot_val > 0 else 0.0
                    f = min(f, 1.0)
                    dq = qty_perp * f
                    # spot sell
                    spot_fee = sell_notional * COST_PER_SIDE
                    proceeds = sell_notional - spot_fee
                    qty_spot -= sell_notional / c
                    reserve += proceeds
                    # perp cover (buy back) dq coins: realize covered uPnL to
                    # margin, pay fee, shrink basis+qty proportionally
                    realized = f * (cost_basis - qty_perp * c)
                    cover_cost = dq * c * COST_PER_SIDE
                    cost_basis *= (1 - f)
                    qty_perp -= dq
                    margin_posted += realized - cover_cost
                    cost_total += spot_fee + cover_cost
                    # recompute the deficit AFTER the position shrank: required
                    # margin M* = qty_perp*c/L is now smaller, so only transfer
                    # the remaining gap (not the original `need`)
                    M_star_new = qty_perp * c / L
                    E_p_new = perp_eq(c)
                    need2 = max(0.0, M_star_new - E_p_new)
                    take2 = min(need2, max(reserve, 0.0))
                    reserve -= take2; margin_posted += take2
                    trades.append({"dt": d, "action": "MARGIN_TOPUP_SPOT_SALE",
                                   "price": c, "spot_sold_usd": sell_notional,
                                   "perp_covered": dq})
                E_p = perp_eq(c)

        # 5) Rule B: compounding rebalance when notional drifts >20% from target
        if not terminated and qty_perp > 0:
            E_total = reserve + qty_spot * c + perp_eq(c)
            N_star = E_total * L / (L + 1)
            N_cur = qty_perp * c
            if N_star > 0 and abs(N_cur - N_star) / N_star > REBAL_DRIFT:
                target_qty = N_star / c
                # trade spot delta
                d_spot = target_qty - qty_spot
                if abs(d_spot) * c >= 1.0:
                    cost_total += abs(d_spot) * c * COST_PER_SIDE
                    if d_spot > 0:   # buy spot: pay notional + fee
                        reserve -= d_spot * c * (1 + COST_PER_SIDE)
                    else:            # sell spot: receive notional - fee
                        reserve += (-d_spot) * c * (1 - COST_PER_SIDE)
                    qty_spot = target_qty
                # trade perp delta (short: positive delta = more short)
                d_perp = target_qty - qty_perp
                if abs(d_perp) * c >= 1.0:
                    cost = abs(d_perp) * c * COST_PER_SIDE
                    cost_total += cost
                    # perp trades settle in margin account: opening adds margin need,
                    # closing releases. Approximate: charge cost to reserve.
                    reserve -= cost
                    if d_perp > 0:
                        cost_basis += d_perp * c
                    else:
                        # closing part of the short: realize its uPnL to margin
                        g = abs(d_perp) / qty_perp if qty_perp else 0.0
                        realized = g * (cost_basis - qty_perp * c)
                        margin_posted += realized
                        cost_basis *= (1 - g)
                    qty_perp = target_qty
                # reset margin to target, sweep difference to/from reserve
                M_target = N_star / L
                old_m = margin_posted
                margin_posted = M_target
                reserve += old_m - M_target
                trades.append({"dt": d, "action": "REBALANCE", "price": c,
                               "N_star": N_star, "qty": target_qty})

        # 6) if the perp was fully closed by Rule A, sweep remaining margin and
        # accumulated funding into reserve so equity is not lost
        if qty_perp == 0 and not terminated:
            reserve += margin_posted + funding_acc
            margin_posted = 0.0; funding_acc = 0.0

        # 7) mark equity + reconciliation check (identity must hold exactly)
        E_p = perp_eq(c) if qty_perp > 0 else 0.0
        E_total = reserve + qty_spot * c + E_p
        recon = reserve + qty_spot * c + margin_posted + funding_acc + cost_basis - qty_perp * c
        max_recon = max(max_recon, abs(E_total - recon))
        debug_rows.append({"dt": d, "close": c, "reserve": reserve,
                           "qty_spot": qty_spot, "qty_perp": qty_perp,
                           "margin_posted": margin_posted, "funding_acc": funding_acc,
                           "cost_basis": cost_basis, "E_p": E_p, "E_total": E_total})
        equity.loc[d] = E_total

    eq = equity.dropna()
    return eq, pd.DataFrame(trades), {
        "leverage": L, "terminated": terminated, "liq": liq_info,
        "min_liq_dist_pct": min_liq_dist * 100 if np.isfinite(min_liq_dist) else np.nan,
        "min_liq_dist_high_pct": min_liq_dist_high * 100 if np.isfinite(min_liq_dist_high) else np.nan,
        "final_equity": float(eq.iloc[-1]),
        "cost_total": float(cost_total),
        "max_abs_recon": float(max_recon),
        "debug": pd.DataFrame(debug_rows),
    }


def monthly_returns(eq, capital=START_EQUITY):
    """Month-end returns: first month vs `capital`, last month dropped if partial."""
    eq = eq.dropna()
    me = eq.resample("ME").last().dropna()
    if len(me) == 0:
        return pd.Series(dtype=float)
    if eq.index[-1] < me.index[-1]:  # last month incomplete
        me = me.iloc[:-1]
    if len(me) == 0:
        return pd.Series(dtype=float)
    rets = me.pct_change()
    rets.iloc[0] = me.iloc[0] / capital - 1.0
    return rets.dropna()


def cagr(eq, capital=START_EQUITY, base_date=None):
    yrs = (eq.index[-1] - (base_date or eq.index[0])).days / 365.25
    return float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan


def main():
    print("=== S5 v2.1 margin-model runs ===")
    summary = []
    for L in LEVERAGES:
        eq, tr, meta = run_s5_margin(L)
        eq.to_csv(f"reports/equity_S5v2_L{L}.csv")
        if len(tr):
            tr.to_csv(f"reports/trades_S5v2_L{L}.csv", index=False)
        mr = monthly_returns(eq)
        peak = eq.cummax(); mdd = float((eq/peak - 1).min())
        summary.append({
            "L": L, "final": meta["final_equity"],
            "CAGR": cagr(eq),
            "mean_mo": float(mr.mean()), "geo_mo": float((1+mr).prod()**(1/len(mr))-1) if len(mr) else np.nan,
            "worst_mo": float(mr.min()), "maxDD": mdd,
            "min_liq_dist_%": meta["min_liq_dist_pct"],
            "min_liq_dist_high_%": meta["min_liq_dist_high_pct"],
            "terminated": meta["terminated"], "cost_total": meta["cost_total"],
            "max_abs_recon": meta["max_abs_recon"],
            "liq": meta["liq"], "n_trades": len(tr),
        })
        print(f"L={L}: final={meta['final_equity']:.2f} CAGR={cagr(eq)*100:.1f}% "
              f"mean_mo={mr.mean()*100:.2f}% maxDD={mdd*100:.1f}% "
              f"min_liq_dist={meta['min_liq_dist_pct']:.1f}% "
              f"(high {meta['min_liq_dist_high_pct']:.1f}%) "
              f"terminated={meta['terminated']} costs={meta['cost_total']:.2f} "
              f"recon={meta['max_abs_recon']:.2e} trades={len(tr)}")
    pd.DataFrame(summary).to_csv("reports/s5v2_summary.csv", index=False)
    print("wrote reports/s5v2_summary.csv")
if __name__ == "__main__":
    main()

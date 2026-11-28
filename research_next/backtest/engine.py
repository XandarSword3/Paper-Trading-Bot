"""Shared backtest engine for research_next strategies.

Generalizes the audited S5 v2.1 margin model (slow_study/backtest/s5_margin.py)
to a PORTFOLIO of independent delta-neutral funding-carry positions sharing one
cash reserve pool.

Frozen conventions (inherited from S5 v2.1, documented not tuned):
- Daily bars, signals/trades at day CLOSE using only data <= that close.
- Funding settlements applied at their real timestamps but marked at the day's
  close price (same acknowledged approximation as S5 v2.1).
- Perp leg marked at the SPOT price (perp/spot basis ignored for marking;
  documented limitation — no perp OHLC history for alts).
- Isolated margin per position: IM = notional / L, MMR = 0.40% (OKX BTC tier-1;
  conservative for alts whose tier-1 MMR is similar or lower).
- Liquidation checked on the day's HIGH: liq when perp equity <= MMR*notional.
- Rule A (margin defense): top up from reserve; if still short, sell spot +
  cover perp proportionally, then transfer only the RECOMPUTED deficit.
- Rule B (rebalance): if perp notional drifts >20% from target N* = alloc*L/(L+1),
  resize spot+perp to target at close, pay costs.
- Costs: COST_PER_SIDE = 0.15% (0.10% taker + 0.05% slippage) on spot AND perp
  legs, applied to every trade including Rule A. Stress multiplier scales it.
- If a perp is fully closed by Rule A, margin+funding are swept to reserve.
- Liquidation => perp account lost, position terminated, NO re-entry that run
  (same as S5 v2.1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPOT_DIR = DATA / "spot"

MMR = 0.004
COST_PER_SIDE = 0.0015
START_EQUITY = 1000.0
REBAL_DRIFT = 0.20


def load_spot(sym):
    """OKX USDT spot 1D bars (dt, open, high, low, close, vol)."""
    df = pd.read_csv(SPOT_DIR / f"{sym}-USDT_1D.csv", parse_dates=["dt"])
    return df.set_index("dt").sort_index()


def load_funding(sym):
    """Binance USD-M perp funding history -> Series rate indexed by dt."""
    df = pd.read_csv(DATA / f"funding_{sym}USDT_binance.csv")
    df["dt"] = pd.to_datetime(df["dt"], utc=True, format="mixed")
    return df.set_index("dt")["fundingRate"].sort_index()


def load_all(symbols):
    spots, funds = {}, {}
    for s in symbols:
        try:
            spots[s] = load_spot(s)
        except FileNotFoundError:
            pass
        try:
            funds[s] = load_funding(s)
        except FileNotFoundError:
            pass
    return spots, funds


class Position:
    __slots__ = ("sym", "L", "qty_spot", "qty_perp", "cost_basis",
                 "margin_posted", "funding_acc", "terminated", "alloc",
                 "liquidated", "_min_liq_dist_high")

    def __init__(self, sym, L, alloc):
        self.sym = sym
        self.L = L
        self.alloc = alloc          # fraction of total equity targeted
        self.qty_spot = 0.0
        self.qty_perp = 0.0
        self.cost_basis = 0.0
        self.margin_posted = 0.0
        self.funding_acc = 0.0
        self.terminated = False
        self.liquidated = False
        self._min_liq_dist_high = np.inf

    def perp_eq(self, price):
        return self.margin_posted + self.funding_acc + self.cost_basis - self.qty_perp * price

    def notional(self, price):
        return self.qty_perp * price


def _enter(pos, price, reserve, cost_mult):
    """Open spot long + perp short for position's allocation. Returns cost."""
    E_alloc = pos.alloc  # absolute $ allocated (set by caller)
    N = E_alloc * pos.L / (pos.L + 1)
    M = E_alloc / (pos.L + 1)
    qty = N / price
    cost = N * COST_PER_SIDE * cost_mult * 2  # spot buy + perp open
    pos.qty_spot = qty
    pos.qty_perp = qty
    pos.cost_basis = qty * price
    pos.margin_posted = M
    return cost


def run_portfolio(spots, funds, policy, start=None, end=None,
                  cost_mult=1.0, rebal_days=7, verbose=False):
    """Run the portfolio engine.

    policy(d, ctx) -> dict {sym: alloc_fraction} evaluated on rebalance days
    (every `rebal_days` calendar days and whenever the target set changes).
    ctx provides trailing-data helpers; see strategies/*.py for examples.
    All allocations are fractions of CURRENT total equity, sum <= 1.
    """
    # union index of all spot series
    idx = None
    for s, df in spots.items():
        i = df.index
        idx = i if idx is None else idx.union(i)
    idx = idx.sort_values()
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]

    positions = {}   # sym -> Position
    reserve = START_EQUITY
    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []
    debug_rows = []
    max_recon = 0.0
    funding_paid = 0.0   # negative funding received (paid by short)
    funding_recv = 0.0   # positive funding received
    liq_events = []
    cost_total = 0.0
    max_util = 0.0       # max (spot+perp notional+margin)/equity

    # precompute per-symbol funding lookup by normalized date
    fund_by_day = {}
    for s, f in funds.items():
        fund_by_day[s] = f.groupby(f.index.normalize()).sum()

    last_rebal = None
    ctx = {"spots": spots, "funds": funds, "positions": positions,
           "equity_hist": equity, "day": None}

    for d in idx:
        ctx["day"] = d
        # mark prices for today (close of each asset; missing -> skip asset)
        px = {}
        for s, df in spots.items():
            if d in df.index:
                px[s] = float(df.loc[d, "close"])
        if not px:
            continue

        # ---- 1) funding settlements (marked at today's close) ----
        for s, pos in positions.items():
            if pos.qty_perp <= 0 or s not in px:
                continue
            day_sum = fund_by_day.get(s, pd.Series(dtype=float))
            rate = day_sum.get(d.normalize(), 0.0)
            if rate != 0.0:
                cf = float(rate) * pos.qty_perp * px[s]
                pos.funding_acc += cf
                if cf >= 0:
                    funding_recv += cf
                else:
                    funding_paid += -cf

        # ---- 2) liquidation check on intraday HIGH ----
        for s, pos in list(positions.items()):
            if pos.terminated or pos.qty_perp <= 0 or s not in spots or d not in spots[s].index:
                continue
            h = float(spots[s].loc[d, "high"])
            c = px[s]
            E_p_high = pos.perp_eq(h)
            mm_req = MMR * pos.qty_perp * h
            denom = pos.qty_perp * (1 + MMR)
            p_liq = (pos.margin_posted + pos.funding_acc + pos.cost_basis) / denom if denom > 0 else np.inf
            if np.isfinite(p_liq):
                pos._min_liq_dist_high = min(getattr(pos, "_min_liq_dist_high", np.inf), (p_liq - h) / h)
            if E_p_high <= mm_req:
                liq_events.append({"dt": d, "sym": s, "price_est": p_liq, "high": h})
                reserve += max(0.0, E_p_high - mm_req)
                pos.qty_perp = 0.0; pos.cost_basis = 0.0
                pos.margin_posted = 0.0; pos.funding_acc = 0.0
                pos.terminated = True
                pos.liquidated = True
                trades.append({"dt": d, "sym": s, "action": "LIQUIDATED", "price": h})

        # ---- 3) Rule A margin defense per position ----
        for s, pos in positions.items():
            if pos.terminated or pos.qty_perp <= 0 or s not in px:
                continue
            c = px[s]
            E_p = pos.perp_eq(c)
            M_star = pos.qty_perp * c / pos.L
            need = M_star - E_p
            if need > 1e-9:
                take = min(need, max(reserve, 0.0))
                reserve -= take; pos.margin_posted += take; need -= take
                if need > 1e-9 and pos.qty_spot > 0:
                    spot_val = pos.qty_spot * c
                    sell_notional = min(need / (1 - COST_PER_SIDE * cost_mult), spot_val)
                    f = min(sell_notional / spot_val, 1.0) if spot_val > 0 else 0.0
                    dq = pos.qty_perp * f
                    spot_fee = sell_notional * COST_PER_SIDE * cost_mult
                    proceeds = sell_notional - spot_fee
                    pos.qty_spot -= sell_notional / c
                    reserve += proceeds
                    realized = f * (pos.cost_basis - pos.qty_perp * c)
                    cover_cost = dq * c * COST_PER_SIDE * cost_mult
                    pos.cost_basis *= (1 - f)
                    pos.qty_perp -= dq
                    pos.margin_posted += realized - cover_cost
                    cost_total += spot_fee + cover_cost
                    M_star_new = pos.qty_perp * c / pos.L
                    need2 = max(0.0, M_star_new - pos.perp_eq(c))
                    take2 = min(need2, max(reserve, 0.0))
                    reserve -= take2; pos.margin_posted += take2
                    trades.append({"dt": d, "sym": s, "action": "MARGIN_TOPUP_SPOT_SALE",
                                   "price": c, "spot_sold_usd": sell_notional, "perp_covered": dq})

        # ---- 4) policy rebalance (every rebal_days or on target-set change) ----
        do_rebal = (last_rebal is None) or ((d - last_rebal).days >= rebal_days)
        target = policy(d, ctx) if do_rebal else None
        if target is not None:
            last_rebal = d
            # current target set
            tgt_syms = {s for s, a in target.items() if a > 0 and s in px}
            # close positions no longer targeted (or terminated->skip)
            for s in list(positions.keys()):
                pos = positions[s]
                if s not in tgt_syms and not pos.terminated and pos.qty_perp > 0 and s in px:
                    c = px[s]
                    # sell all spot, cover all perp
                    sell_notional = pos.qty_spot * c
                    spot_fee = sell_notional * COST_PER_SIDE * cost_mult
                    reserve += sell_notional - spot_fee
                    cover_cost = pos.qty_perp * c * COST_PER_SIDE * cost_mult
                    realized = pos.cost_basis - pos.qty_perp * c
                    reserve += realized - cover_cost
                    cost_total += spot_fee + cover_cost
                    pos.qty_spot = 0.0; pos.qty_perp = 0.0
                    pos.cost_basis = 0.0
                    reserve += pos.margin_posted + pos.funding_acc
                    pos.margin_posted = 0.0; pos.funding_acc = 0.0
                    pos.terminated = True   # closed; policy may re-open next cycle
                    trades.append({"dt": d, "sym": s, "action": "EXIT", "price": c})
            # open / resize targeted positions
            E_total_now = _total_equity(positions, px, reserve)
            for s, alloc in target.items():
                if alloc <= 0 or s not in px:
                    continue
                pos = positions.get(s)
                if pos is None:
                    pos = Position(s, 2.0, 0.0)  # default L=2; policy may override via ctx
                    positions[s] = pos
                if pos.liquidated:
                    continue  # no re-entry after liquidation (S5 v2.1 rule)
                pos.alloc = alloc
                pos.L = ctx.get("leverage", {}).get(s, 2.0)
                c = px[s]
                if pos.terminated and pos.qty_perp == 0 and pos.qty_spot == 0:
                    pos.terminated = False  # allow re-entry after clean exit
                # desired quantities
                E_alloc = E_total_now * alloc
                N_star = E_alloc * pos.L / (pos.L + 1)
                target_qty = N_star / c
                cur_notional = pos.qty_perp * c
                # resize if drift > REBAL_DRIFT or fresh entry
                if pos.qty_perp == 0 or (N_star > 0 and abs(cur_notional / N_star - 1) > REBAL_DRIFT):
                    # adjust spot
                    dq_spot = target_qty - pos.qty_spot
                    trade_notional = abs(dq_spot) * c
                    spot_cost = trade_notional * COST_PER_SIDE * cost_mult
                    if dq_spot > 0:
                        # buy spot: need cash from reserve
                        need_cash = dq_spot * c * (1 + COST_PER_SIDE * cost_mult)
                        take = min(need_cash, max(reserve, 0.0))
                        reserve -= take
                        pos.qty_spot += take / (c * (1 + COST_PER_SIDE * cost_mult))
                        cost_total += take * COST_PER_SIDE * cost_mult / (1 + COST_PER_SIDE * cost_mult)
                    else:
                        proceeds = (-dq_spot) * c * (1 - COST_PER_SIDE * cost_mult)
                        pos.qty_spot += dq_spot
                        reserve += proceeds
                        cost_total += (-dq_spot) * c * COST_PER_SIDE * cost_mult
                    # adjust perp
                    dq_perp = target_qty - pos.qty_perp
                    if abs(dq_perp) > 1e-12:
                        perp_cost = abs(dq_perp) * c * COST_PER_SIDE * cost_mult
                        cost_total += perp_cost
                        if dq_perp > 0:
                            # short more: cost basis grows
                            pos.cost_basis += dq_perp * c
                        else:
                            # cover: realize proportionally
                            f = min(-dq_perp / pos.qty_perp, 1.0) if pos.qty_perp > 0 else 1.0
                            realized = f * (pos.cost_basis - pos.qty_perp * c)
                            pos.cost_basis *= (1 - f)
                            pos.margin_posted += realized - perp_cost
                        pos.qty_perp = target_qty
                    # reset margin to target
                    M_target = N_star / pos.L
                    reserve += pos.margin_posted - M_target
                    pos.margin_posted = M_target
                    trades.append({"dt": d, "sym": s, "action": "REBALANCE" if pos.qty_perp != 0 else "ENTRY",
                                   "price": c, "qty": target_qty, "N*": N_star})
            # sweep fully-closed positions back to reserve
            for s, pos in positions.items():
                if pos.qty_perp == 0 and not pos.terminated:
                    reserve += pos.margin_posted + pos.funding_acc
                    pos.margin_posted = 0.0; pos.funding_acc = 0.0

        # ---- 5) mark equity ----
        E_total = _total_equity(positions, px, reserve)
        # capital utilization: gross notional+margin deployed / equity
        deployed = reserve * 0 + sum(
            p.qty_spot * px.get(p.sym, 0) + p.margin_posted for p in positions.values()
        )
        # utilization = (total equity - idle reserve) / equity
        util = (E_total - reserve) / E_total if E_total > 0 else 0.0
        max_util = max(max_util, util)
        # reconciliation
        recon = reserve + sum(
            p.qty_spot * px.get(p.sym, 0) + p.margin_posted + p.funding_acc + p.cost_basis - p.qty_perp * px.get(p.sym, 0)
            for p in positions.values()
        )
        max_recon = max(max_recon, abs(E_total - recon))
        equity.loc[d] = E_total
        debug_rows.append({"dt": d, "E_total": E_total, "reserve": reserve,
                           "n_pos": sum(1 for p in positions.values() if p.qty_perp > 0)})

    eq = equity.dropna()
    meta = {
        "final_equity": float(eq.iloc[-1]),
        "cost_total": float(cost_total),
        "funding_recv": float(funding_recv),
        "funding_paid": float(funding_paid),
        "liq_events": liq_events,
        "n_liq": len(liq_events),
        "max_util": float(max_util),
        "max_abs_recon": float(max_recon),
        "trades": pd.DataFrame(trades),
        "debug": pd.DataFrame(debug_rows),
        "positions": positions,
    }
    return eq, meta


def _total_equity(positions, px, reserve):
    tot = reserve
    for p in positions.values():
        c = px.get(p.sym)
        if c is None:
            continue
        tot += p.qty_spot * c + p.perp_eq(c)
    return tot


def monthly_returns(eq, capital=START_EQUITY):
    eq = eq.dropna()
    me = eq.resample("ME").last().dropna()
    if len(me) == 0:
        return pd.Series(dtype=float)
    if eq.index[-1] < me.index[-1]:
        me = me.iloc[:-1]
    if len(me) == 0:
        return pd.Series(dtype=float)
    rets = me.pct_change()
    rets.iloc[0] = me.iloc[0] / capital - 1.0
    return rets.dropna()


def cagr(eq, capital=START_EQUITY, base_date=None):
    yrs = (eq.index[-1] - (base_date or eq.index[0])).days / 365.25
    return float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan


def perf_stats(eq, mr):
    peak = eq.cummax()
    dd = eq / peak - 1
    # longest underwater: days below running peak
    uw = dd < 0
    longest = 0; cur = 0
    for v in uw:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return {
        "final": float(eq.iloc[-1]), "cagr": cagr(eq),
        "mean_mo": float(mr.mean()) if len(mr) else np.nan,
        "geo_mo": float((1 + mr).prod() ** (1 / len(mr)) - 1) if len(mr) else np.nan,
        "median_mo": float(mr.median()) if len(mr) else np.nan,
        "pct_prof": float((mr > 0).mean()) if len(mr) else np.nan,
        "pct_ge6": float((mr >= 0.06).mean()) if len(mr) else np.nan,
        "worst_mo": float(mr.min()) if len(mr) else np.nan,
        "maxDD": float(dd.min()),
        "longest_uw_days": int(longest),
    }

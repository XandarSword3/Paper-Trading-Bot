"""S5 v3 — venue-consistent Bybit backtest (BTCUSDT spot + BTCUSDT perp).

Fixes the v2.1 approximations using Bybit's own public data:
- Funding applied at the EXACT settlement timestamp (00/08/16 UTC) with the
  mark price at that instant (1H mark-price kline closing at settlement).
- Liquidation checked on 1H MARK-PRICE highs (not spot highs), MMR = 0.33%
  (Bybit BTCUSDT tier-1, riskLimitValue <= 300k USDT — verified via
  /v5/market/risk-limit).
- Perp leg marked at mark price; spot leg at spot close.
- Venue fees: spot taker 0.10% + 0.05% slip = 0.15%/side;
  perp taker 0.055% + 0.05% slip = 0.105%/side.
- Frozen strategy rules unchanged: L=1 structure, Rule A margin defense,
  Rule B 20% rebalance, no re-entry after liquidation.

Data: slow_study/data/bybit/ (fetch_bybit.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bybit"
START_EQUITY = 1000.0

SPOT_COST = 0.0015    # 0.10% taker + 0.05% slip
PERP_COST = 0.00105   # 0.055% taker + 0.05% slip
MMR = 0.0033          # Bybit BTCUSDT tier-1 maintenance margin ratio
REBAL_DRIFT = 0.20
LEVERAGE = 1


def load():
    spot = pd.read_csv(DATA / "spot_BTCUSDT_1D.csv", parse_dates=["dt"]).set_index("dt").sort_index()
    mark = pd.read_csv(DATA / "mark_BTCUSDT_1H.csv", parse_dates=["dt"]).set_index("dt").sort_index()
    fund = pd.read_csv(DATA / "funding_BTCUSDT_bybit_full.csv", parse_dates=["dt"]).set_index("dt").sort_index()["fundingRate"]
    return spot, mark, fund


def run_v3(capital=START_EQUITY, leverage=LEVERAGE, cost_mult=1.0,
           start=None, end=None, fund_sign=1.0, shock_date=None, shock_mult=1.0):
    spot, mark, fund = load()
    if shock_date is not None:
        sd = pd.Timestamp(shock_date, tz="UTC")
        for col in ("open", "high", "low", "close"):
            spot.loc[spot.index >= sd, col] *= shock_mult
            mark.loc[mark.index >= sd, col] *= shock_mult
    fund = fund * fund_sign
    idx = spot.index
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
    L = leverage
    cm = cost_mult
    sc = SPOT_COST * cm
    pc = PERP_COST * cm

    # funding events within window
    fund_in = fund[(fund.index >= idx[0]) & (fund.index <= idx[-1] + pd.Timedelta(hours=24))]
    fund_in = fund_in.sort_index()

    def mark_close_at(t):
        """Mark price at instant t = close of 1H candle ts=t-1h."""
        ts = t - pd.Timedelta(hours=1)
        sub = mark[mark.index <= ts]
        return float(sub["close"].iloc[-1]) if len(sub) else None

    def mark_highs_for_day(d):
        """1H mark candles covering spot day d (00:00 -> next 00:00)."""
        win = mark[(mark.index >= d) & (mark.index < d + pd.Timedelta(hours=24))]
        return win

    # ---- entry at first day open ----
    d0 = idx[0]
    p0 = float(spot.loc[d0, "open"])
    m0 = mark_close_at(d0) or p0
    est = 2 * sc
    N0 = (capital - capital * est) / (1 + 1 / L)
    qty_spot = N0 / p0
    qty_perp = qty_spot
    cost_basis = qty_perp * m0          # perp entry at mark price
    margin_posted = N0 / L
    reserve = capital - N0 - margin_posted - N0 * (sc + pc)
    funding_acc = 0.0
    cost_total = N0 * (sc + pc)
    terminated = False
    liq_info = None
    min_liq_dist = np.inf

    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []
    fund_events = []
    debug = []

    def perp_eq(price):
        return margin_posted + funding_acc + (cost_basis - qty_perp * price)

    for d in idx:
        o = float(spot.loc[d, "open"]); c = float(spot.loc[d, "close"])
        m_close = mark_close_at(d + pd.Timedelta(hours=24)) or c  # mark at day end

        # 1) funding settlements with timestamp-exact mark price
        day_f = fund_in[(fund_in.index >= d) & (fund_in.index < d + pd.Timedelta(hours=24))]
        for ft, rate in day_f.items():
            mp = mark_close_at(ft)
            if mp is None:
                mp = m_close
            cf = float(rate) * qty_perp * mp   # short receives if rate>0
            funding_acc += cf
            fund_events.append({"dt": ft, "rate": float(rate), "mark": mp,
                                "qty": qty_perp, "cashflow": cf})

        # 2) liquidation check on 1H mark highs (only if alive)
        if not terminated and qty_perp > 0:
            win = mark_highs_for_day(d)
            denom = qty_perp * (1 + MMR)
            p_liq = (margin_posted + funding_acc + cost_basis) / denom if denom > 0 else np.inf
            dist = (p_liq - m_close) / m_close if m_close > 0 else np.inf
            if np.isfinite(dist):
                min_liq_dist = min(min_liq_dist, dist)
            for _, bar in win.iterrows():
                hh = float(bar["high"])
                e_high = margin_posted + funding_acc + cost_basis - qty_perp * hh
                mm_req = MMR * qty_perp * hh
                if e_high <= mm_req:
                    liq_info = {"dt": d, "hour": bar["dt"], "mark_high": hh,
                                "p_liq_est": p_liq,
                                "perp_equity_lost": e_high}
                    reserve += max(0.0, e_high - mm_req)
                    qty_perp = 0.0; cost_basis = 0.0
                    margin_posted = 0.0; funding_acc = 0.0
                    terminated = True
                    trades.append({"dt": d, "action": "LIQUIDATED", "price": hh})
                    break

        # 3) Rule A: margin defense at day close (mark price)
        if not terminated and qty_perp > 0:
            M_star = qty_perp * m_close / L
            E_p = perp_eq(m_close)
            if E_p < M_star:
                need = M_star - E_p
                take = min(need, max(reserve, 0.0))
                reserve -= take; margin_posted += take; need -= take
                if need > 1e-9 and qty_spot > 0:
                    spot_val = qty_spot * c
                    sell_notional = min(need / (1 - sc), spot_val)
                    f = min(sell_notional / spot_val, 1.0) if spot_val > 0 else 0.0
                    dq = qty_perp * f
                    spot_fee = sell_notional * sc
                    proceeds = sell_notional - spot_fee
                    qty_spot -= sell_notional / c
                    reserve += proceeds
                    realized = f * (cost_basis - qty_perp * m_close)
                    cover_cost = dq * m_close * pc
                    cost_basis *= (1 - f)
                    qty_perp -= dq
                    margin_posted += realized - cover_cost
                    cost_total += spot_fee + cover_cost
                    M_star_new = qty_perp * m_close / L
                    E_p_new = perp_eq(m_close)
                    need2 = max(0.0, M_star_new - E_p_new)
                    take2 = min(need2, max(reserve, 0.0))
                    reserve -= take2; margin_posted += take2
                    trades.append({"dt": d, "action": "MARGIN_TOPUP_SPOT_SALE",
                                   "price": c, "spot_sold_usd": sell_notional,
                                   "perp_covered": dq})

        # 4) Rule B: compounding rebalance
        if not terminated and qty_perp > 0:
            E_total = reserve + qty_spot * c + perp_eq(m_close)
            N_star = E_total * L / (L + 1)
            N_cur = qty_perp * m_close
            if N_star > 0 and abs(N_cur - N_star) / N_star > REBAL_DRIFT:
                target_qty = N_star / m_close
                d_spot = target_qty - qty_spot
                if abs(d_spot) * c >= 1.0:
                    cost_total += abs(d_spot) * c * sc
                    if d_spot > 0:
                        reserve -= d_spot * c * (1 + sc)
                    else:
                        reserve += (-d_spot) * c * (1 - sc)
                    qty_spot = target_qty
                d_perp = target_qty - qty_perp
                if abs(d_perp) * m_close >= 1.0:
                    cost = abs(d_perp) * m_close * pc
                    cost_total += cost
                    reserve -= cost
                    if d_perp > 0:
                        cost_basis += d_perp * m_close
                    else:
                        g = abs(d_perp) / qty_perp if qty_perp else 0.0
                        realized = g * (cost_basis - qty_perp * m_close)
                        margin_posted += realized
                        cost_basis *= (1 - g)
                    qty_perp = target_qty
                M_target = N_star / L
                reserve += margin_posted - M_target
                margin_posted = M_target
                trades.append({"dt": d, "action": "REBALANCE", "price": m_close,
                               "N_star": N_star, "qty": target_qty})

        # 5) sweep if perp fully closed
        if qty_perp == 0 and not terminated:
            reserve += margin_posted + funding_acc
            margin_posted = 0.0; funding_acc = 0.0

        # 6) mark equity
        E_p = perp_eq(m_close) if qty_perp > 0 else 0.0
        E_total = reserve + qty_spot * c + E_p
        equity.loc[d] = E_total
        debug.append({"dt": d, "spot_close": c, "mark_close": m_close,
                      "reserve": reserve, "qty_spot": qty_spot, "qty_perp": qty_perp,
                      "margin_posted": margin_posted, "funding_acc": funding_acc,
                      "cost_basis": cost_basis, "E_total": E_total})

    eq = equity.dropna()
    meta = {
        "final_equity": float(eq.iloc[-1]),
        "terminated": terminated, "liq": liq_info,
        "min_liq_dist_pct": min_liq_dist * 100 if np.isfinite(min_liq_dist) else np.nan,
        "cost_total": float(cost_total),
        "n_funding_events": len(fund_events),
        "funding_total": float(sum(e["cashflow"] for e in fund_events)),
        "funding_paid": float(sum(e["cashflow"] for e in fund_events if e["cashflow"] < 0)),
        "funding_received": float(sum(e["cashflow"] for e in fund_events if e["cashflow"] > 0)),
        "trades": pd.DataFrame(trades),
        "fund_events": pd.DataFrame(fund_events),
        "debug": pd.DataFrame(debug),
    }
    return eq, meta


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


def perf(eq, capital=START_EQUITY):
    mr = monthly_returns(eq, capital)
    peak = eq.cummax()
    mdd = float((eq / peak - 1).min())
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan
    return {"final": float(eq.iloc[-1]), "cagr": cagr,
            "mean_mo": float(mr.mean()), "geo_mo": float((1 + mr).prod() ** (1 / len(mr)) - 1) if len(mr) else np.nan,
            "median_mo": float(mr.median()), "worst_mo": float(mr.min()),
            "maxDD": mdd, "n_months": len(mr)}


if __name__ == "__main__":
    eq, meta = run_v3()
    eq.to_csv(ROOT / "reports" / "equity_S5v3_bybit_L1.csv")
    meta["trades"].to_csv(ROOT / "reports" / "trades_S5v3_bybit_L1.csv", index=False)
    meta["fund_events"].to_csv(ROOT / "reports" / "funding_S5v3_bybit_L1.csv", index=False)
    meta["debug"].to_csv(ROOT / "reports" / "debug_S5v3_bybit_L1.csv", index=False)
    p = perf(eq)
    print("S5 v3 Bybit L1:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
    print("liq:", meta["liq"], "min_liq_dist%:", round(meta["min_liq_dist_pct"], 2) if meta["min_liq_dist_pct"] == meta["min_liq_dist_pct"] else "n/a")
    print("cost_total:", round(meta["cost_total"], 2), "funding net:", round(meta["funding_total"], 2),
          "(in", round(meta["funding_received"], 2), "out", round(meta["funding_paid"], 2), ")")
    print("n_trades:", len(meta["trades"]), "n_funding:", meta["n_funding_events"])

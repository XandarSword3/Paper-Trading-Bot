"""12-month forward PAPER trader for S5 v2 — funding carry WITH margin model.

Mirrors backtest/s5_margin.py exactly (frozen rules, see its docstring):
  - capital $1,000 split into spot BTC + perp margin (N0/L) + ~0 reserve
  - perp short qty tracks spot qty (delta-neutral)
  - Rule A: if perp equity < target margin (N/L), top up from reserve, else
    sell spot + cover perp proportionally
  - Rule B: if perp notional drifts >20% from target N*=E*L/(L+1), full rebalance
  - liquidation checked on the day's HIGH (isolated margin, MMR=0.40%)
  - funding applied at real OKX settlement timestamps (public endpoint)

Run once per day after the daily candle closes:
    python paper/paper_s5.py --init --leverage 2   # once per leverage
    python paper/paper_s5.py --leverage 2          # daily mark

State/log are per-leverage: s5_state_L{L}.json, s5_log_L{L}.csv.
NO rule changes mid-run. No orders are placed anywhere.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from strategies import COST_PER_SIDE, START_EQUITY  # noqa: E402

CB = "https://api.exchange.coinbase.com"
OKX = "https://www.okx.com"
HERE = Path(__file__).parent
MMR = 0.004          # OKX BTCUSDT-SWAP tier-1 (same as backtest)
REBAL_DRIFT = 0.20


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def fetch_btc_daily():
    """Latest completed BTC-USD daily candle -> (open, high, close, date_str)."""
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=10)
    r = requests.get(f"{CB}/products/BTC-USD/candles", params={
        "granularity": 86400,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, timeout=20, headers={"User-Agent": "slow-study-paper/1.0"})
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "vol"])
    for col in ("open", "high", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    df = df.sort_values("ts")
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if len(df) and (now_ts - int(df["ts"].iloc[-1])) < 86400:
        df = df.iloc[:-1]  # drop in-progress candle
    last = df.iloc[-1]
    dstr = str(pd.to_datetime(int(last["ts"]), unit="s", utc=True))
    return float(last["open"]), float(last["high"]), float(last["close"]), dstr


def fetch_new_funding(since_ms):
    r = requests.get(f"{OKX}/api/v5/public/funding-rate-history",
                     params={"instId": "BTC-USDT-SWAP", "limit": 100}, timeout=20)
    r.raise_for_status()
    out = []
    for row in r.json().get("data", []):
        ft = int(row["fundingTime"])
        if ft > since_ms:
            out.append({"fundingTime": ft,
                        "dt": pd.to_datetime(ft, unit="ms", utc=True).isoformat(),
                        "rate": float(row["fundingRate"])})
    return sorted(out, key=lambda x: x["fundingTime"])


def perp_eq(st, price):
    return (st["margin_posted"] + st["funding_acc"]
            + st["cost_basis"] - st["qty_perp"] * price)


def total_eq(st, close):
    return st["reserve"] + st["qty_spot"] * close + perp_eq(st, close)


def log_row(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a") as f:
        if new:
            f.write("run_utc,type,dt,price,qty_spot,qty_perp,rate,cashflow,equity,note\n")
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="open position now (first run)")
    ap.add_argument("--leverage", type=int, default=2, choices=[1, 2, 3])
    args = ap.parse_args()
    L = args.leverage
    STATE = HERE / f"s5_state_L{L}.json"
    LOG = HERE / f"s5_log_L{L}.csv"

    o, h, c, candle_date = fetch_btc_daily()

    if STATE.exists():
        st = json.loads(STATE.read_text())
    else:
        st = {"leverage": L, "qty_spot": 0.0, "qty_perp": 0.0, "cost_basis": 0.0,
              "margin_posted": 0.0, "funding_acc": 0.0, "reserve": START_EQUITY,
              "terminated": False, "last_funding_ms": 0, "last_close": None,
              "last_high": None, "last_candle": None, "started": utcnow()}

    # --- entry
    if args.init and st["qty_perp"] == 0.0 and not st["terminated"]:
        N0 = (START_EQUITY - START_EQUITY * 2 * COST_PER_SIDE) / (1 + 1 / L)
        qty = N0 / c  # enter at latest completed close (current mark)
        st["qty_spot"] = qty
        st["qty_perp"] = qty
        st["cost_basis"] = qty * c
        st["margin_posted"] = N0 / L
        st["reserve"] = START_EQUITY - N0 - st["margin_posted"] - N0 * 2 * COST_PER_SIDE
        st["last_funding_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)
        eq = total_eq(st, c)
        log_row(LOG, f"{utcnow()},ENTRY,{candle_date},{c},{st['qty_spot']:.8f},"
                     f"{st['qty_perp']:.8f},,,{eq:.2f},L{L} spot+perp opened @ {c}")
        print(f"[L{L}] entered: qty={qty:.6f} BTC @ {c}, margin={st['margin_posted']:.2f}, "
              f"reserve={st['reserve']:.2f}")

    if st["qty_perp"] > 0 or st["qty_spot"] > 0:
        # --- funding settlements (mark at current close — documented approx)
        for ev in fetch_new_funding(st["last_funding_ms"]):
            cf = ev["rate"] * st["qty_perp"] * c  # short receives if rate>0
            st["funding_acc"] += cf
            st["last_funding_ms"] = max(st["last_funding_ms"], ev["fundingTime"])
            log_row(LOG, f"{utcnow()},FUNDING,{ev['dt']},{c},{st['qty_spot']:.8f},"
                         f"{st['qty_perp']:.8f},{ev['rate']},{cf:.4f},"
                         f"{total_eq(st, c):.2f},settlement")

        # --- liquidation check on intraday HIGH
        if not st["terminated"] and st["qty_perp"] > 0:
            e_high = perp_eq(st, h)
            mm_high = MMR * st["qty_perp"] * h
            if e_high <= mm_high:
                leftover = max(0.0, e_high - mm_high)
                st["reserve"] += leftover
                st["qty_perp"] = 0.0; st["cost_basis"] = 0.0
                st["margin_posted"] = 0.0; st["funding_acc"] = 0.0
                st["terminated"] = True
                log_row(LOG, f"{utcnow()},LIQUIDATED,{candle_date},{h},"
                             f"{st['qty_spot']:.8f},0,,,{total_eq(st, c):.2f},"
                             f"liq check on high={h}")
                print(f"[L{L}] LIQUIDATED on high={h}")

        # --- Rule A: margin defense
        if not st["terminated"] and st["qty_perp"] > 0:
            m_star = st["qty_perp"] * c / L
            e_p = perp_eq(st, c)
            if e_p < m_star:
                need = m_star - e_p
                take = min(need, max(st["reserve"], 0.0))
                st["reserve"] -= take; st["margin_posted"] += take; need -= take
                if need > 1e-9 and st["qty_spot"] > 0:
                    spot_val = st["qty_spot"] * c
                    sell_notional = min(need / (1 - COST_PER_SIDE), spot_val)
                    f = min(sell_notional / spot_val, 1.0) if spot_val > 0 else 0.0
                    dq = st["qty_perp"] * f
                    proceeds = sell_notional * (1 - COST_PER_SIDE)
                    st["qty_spot"] -= sell_notional / c
                    st["reserve"] += proceeds
                    realized = f * (st["cost_basis"] - st["qty_perp"] * c)
                    cover_cost = dq * c * COST_PER_SIDE
                    st["cost_basis"] *= (1 - f)
                    st["qty_perp"] -= dq
                    st["margin_posted"] += realized - cover_cost
                    # recompute deficit AFTER position shrink (v2.1 fix)
                    m_star_new = st["qty_perp"] * c / L
                    e_p_new = perp_eq(st, c)
                    need2 = max(0.0, m_star_new - e_p_new)
                    take2 = min(need2, max(st["reserve"], 0.0))
                    st["reserve"] -= take2; st["margin_posted"] += take2
                    log_row(LOG, f"{utcnow()},MARGIN_TOPUP,{candle_date},{c},"
                                 f"{st['qty_spot']:.8f},{st['qty_perp']:.8f},,,"
                                 f"{total_eq(st, c):.2f},sold ${sell_notional:.2f} spot")
                    print(f"[L{L}] margin top-up: sold ${sell_notional:.2f} spot")

        # --- Rule B: compounding rebalance on >20% notional drift
        if not st["terminated"] and st["qty_perp"] > 0:
            e_total = total_eq(st, c)
            n_star = e_total * L / (L + 1)
            n_cur = st["qty_perp"] * c
            if n_star > 0 and abs(n_cur - n_star) / n_star > REBAL_DRIFT:
                target_qty = n_star / c
                d_spot = target_qty - st["qty_spot"]
                if abs(d_spot) * c >= 1.0:
                    if d_spot > 0:
                        st["reserve"] -= d_spot * c * (1 + COST_PER_SIDE)
                    else:
                        st["reserve"] += (-d_spot) * c * (1 - COST_PER_SIDE)
                    st["qty_spot"] = target_qty
                d_perp = target_qty - st["qty_perp"]
                if abs(d_perp) * c >= 1.0:
                    cost = abs(d_perp) * c * COST_PER_SIDE
                    st["reserve"] -= cost
                    if d_perp > 0:
                        st["cost_basis"] += d_perp * c
                    else:
                        g = abs(d_perp) / st["qty_perp"] if st["qty_perp"] else 0.0
                        st["margin_posted"] += g * (st["cost_basis"] - st["qty_perp"] * c)
                        st["cost_basis"] *= (1 - g)
                    st["qty_perp"] = target_qty
                m_target = n_star / L
                st["reserve"] += st["margin_posted"] - m_target
                st["margin_posted"] = m_target
                log_row(LOG, f"{utcnow()},REBALANCE,{candle_date},{c},"
                             f"{st['qty_spot']:.8f},{st['qty_perp']:.8f},,,"
                             f"{total_eq(st, c):.2f},N*={n_star:.2f}")
                print(f"[L{L}] rebalance: N*={n_star:.2f} qty={target_qty:.6f}")

    # --- sweep margin+funding to reserve if perp fully closed (v2.1 fix)
    if st["qty_perp"] == 0 and not st["terminated"]:
        st["reserve"] += st["margin_posted"] + st["funding_acc"]
        st["margin_posted"] = 0.0; st["funding_acc"] = 0.0

    # --- DQ checks + daily mark
    eq = total_eq(st, c)
    if st.get("last_close") and abs(c / st["last_close"] - 1) > 0.25:
        log_row(LOG, f"{utcnow()},DQ,{candle_date},{c},{st['qty_spot']:.8f},"
                     f"{st['qty_perp']:.8f},,,{eq:.2f},close moved >25%")
        print("DQ: close moved >25%")
    st["last_close"] = c
    st["last_high"] = h
    st["last_candle"] = candle_date
    st["last_run_utc"] = utcnow()
    STATE.write_text(json.dumps(st, indent=2))
    log_row(LOG, f"{utcnow()},MARK,{candle_date},{c},{st['qty_spot']:.8f},"
                 f"{st['qty_perp']:.8f},,,{eq:.2f},daily mark")
    print(f"[L{L}] mark: close={c} high={h} equity={eq:.2f} "
          f"qty_spot={st['qty_spot']:.6f} qty_perp={st['qty_perp']:.6f}")


if __name__ == "__main__":
    main()

"""S5 v3 SHADOW trader — Bybit public data only, NO orders, NO API keys.

Mirrors backtest/s5_v3_bybit.py rules forward-looking:
  - L=1 delta-neutral: long BTCUSDT spot, short BTCUSDT perp, qty equal.
  - Funding applied at exact settlement timestamps using mark price at that
    instant (1H mark-price kline closing at settlement).
  - Liquidation checked on 1H mark-price highs, MMR=0.33% (tier-1).
  - Rule A margin defense / Rule B 20% rebalance — frozen, unchanged.
  - Isolated margin model (single perp position).

Run once per day (after 00:00 UTC candle close) or more often:
    python paper/shadow_s5_bybit.py --init     # open shadow position
    python paper/shadow_s5_bybit.py            # daily mark + funding check

State: paper/shadow_s5_state.json ; Log: paper/shadow_s5_log.csv
Every funding settlement is logged with the mark price used; compare against
Bybit's funding history endpoint (same public source) — see VALIDATION notes
in reports/REPORT_S5_v3.md.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
BASE = "https://api.bybit.com"
START_EQUITY = 1000.0
SPOT_COST = 0.0015
PERP_COST = 0.00105
MMR = 0.0033
REBAL_DRIFT = 0.20
LEVERAGE = 1
STATE = HERE / "shadow_s5_state.json"
LOG = HERE / "shadow_s5_log.csv"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def get(path, params, retries=4):
    for i in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=20)
            j = r.json()
            if j.get("retCode") == 0:
                return j["result"]
            if j.get("retCode") == 10006:
                time.sleep(1.5 * (i + 1))
                continue
            raise RuntimeError(f"{path} retCode={j.get('retCode')} {j.get('retMsg')}")
        except requests.RequestException:
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{path} failed")


def fetch_spot_daily():
    """Latest completed Bybit spot BTCUSDT daily candle -> (open, high, close, dt)."""
    res = get("/v5/market/kline", {"category": "spot", "symbol": "BTCUSDT",
                                   "interval": "D", "limit": 3})
    rows = res["list"]  # newest first
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for row in rows:
        ts = int(row[0])
        if now_ms - ts >= 86400000:  # completed candle
            return float(row[1]), float(row[2]), float(row[4]), pd.to_datetime(ts, unit="ms", utc=True)
    raise RuntimeError("no completed daily candle")


def fetch_mark_1h(since_ms):
    """Mark-price 1H candles newer than since_ms (for liq check + funding marks)."""
    res = get("/v5/market/mark-price-kline", {"category": "linear", "symbol": "BTCUSDT",
                                              "interval": "60", "start": since_ms, "limit": 1000})
    rows = res.get("list", [])
    out = []
    for row in rows:
        out.append({"dt": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "open": float(row[1]), "high": float(row[2]),
                    "low": float(row[3]), "close": float(row[4])})
    return sorted(out, key=lambda x: x["dt"])


def fetch_funding_since(since_ms):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = min(since_ms, now_ms - 200 * 8 * 3600 * 1000)
    res = get("/v5/market/funding/history", {"category": "linear", "symbol": "BTCUSDT",
                                             "startTime": start, "endTime": now_ms, "limit": 200})
    out = []
    for row in res.get("list", []):
        ft = int(row["fundingRateTimestamp"])
        if ft > since_ms:
            out.append({"dt": pd.to_datetime(ft, unit="ms", utc=True),
                        "rate": float(row["fundingRate"]), "ts_ms": ft})
    return sorted(out, key=lambda x: x["ts_ms"])


def mark_close_at(mark_df, t):
    """Mark price at instant t = close of 1H candle ts=t-1h."""
    ts = t - pd.Timedelta(hours=1)
    sub = mark_df[mark_df["dt"] <= ts]
    return float(sub["close"].iloc[-1]) if len(sub) else None


def perp_eq(st, price):
    return st["margin_posted"] + st["funding_acc"] + st["cost_basis"] - st["qty_perp"] * price


def total_eq(st, spot_close, mark_close):
    return st["reserve"] + st["qty_spot"] * spot_close + perp_eq(st, mark_close)


def log_row(row):
    new = not LOG.exists() or LOG.stat().st_size == 0
    with open(LOG, "a") as f:
        if new:
            f.write("run_utc,type,dt,price,qty_spot,qty_perp,rate,cashflow,equity,note\n")
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()

    o, h, c, candle_dt = fetch_spot_daily()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    mark_df = pd.DataFrame(fetch_mark_1h(now_ms - 3 * 86400000))
    if mark_df.empty:
        raise RuntimeError("no mark price data")
    m_close = mark_close_at(mark_df, candle_dt + pd.Timedelta(hours=24)) or c

    if STATE.exists():
        st = json.loads(STATE.read_text())
    else:
        st = {"qty_spot": 0.0, "qty_perp": 0.0, "cost_basis": 0.0,
              "margin_posted": 0.0, "funding_acc": 0.0, "reserve": START_EQUITY,
              "terminated": False, "last_funding_ms": 0, "started": utcnow()}

    # ---- entry
    if args.init and st["qty_perp"] == 0.0 and not st["terminated"]:
        N0 = (START_EQUITY - START_EQUITY * (SPOT_COST + PERP_COST)) / (1 + 1 / LEVERAGE)
        qty = N0 / c
        st["qty_spot"] = qty
        st["qty_perp"] = qty
        st["cost_basis"] = qty * m_close
        st["margin_posted"] = N0 / LEVERAGE
        st["reserve"] = START_EQUITY - N0 - st["margin_posted"] - N0 * (SPOT_COST + PERP_COST)
        st["last_funding_ms"] = now_ms
        eq = total_eq(st, c, m_close)
        log_row(f"{utcnow()},ENTRY,{candle_dt.isoformat()},{c},{st['qty_spot']:.8f},"
                f"{st['qty_perp']:.8f},,,{eq:.2f},L1 spot+perp opened @ spot={c} mark={m_close}")
        print(f"[shadow] entered: qty={qty:.6f} BTC @ spot {c} / mark {m_close}, "
              f"margin={st['margin_posted']:.2f}, reserve={st['reserve']:.2f}")

    if st["qty_perp"] > 0 or st["qty_spot"] > 0:
        # ---- funding settlements (timestamp-exact mark price)
        for ev in fetch_funding_since(st["last_funding_ms"]):
            mp = mark_close_at(mark_df, ev["dt"]) or m_close
            cf = ev["rate"] * st["qty_perp"] * mp
            st["funding_acc"] += cf
            st["last_funding_ms"] = max(st["last_funding_ms"], ev["ts_ms"])
            log_row(f"{utcnow()},FUNDING,{ev['dt'].isoformat()},{mp},{st['qty_spot']:.8f},"
                    f"{st['qty_perp']:.8f},{ev['rate']},{cf:.4f},"
                    f"{total_eq(st, c, m_close):.2f},settlement mark={mp:.2f}")
            print(f"[shadow] funding {ev['dt']} rate={ev['rate']} mark={mp:.2f} cf={cf:.4f}")

        # ---- liquidation check on 1H mark highs
        if not st["terminated"] and st["qty_perp"] > 0:
            denom = st["qty_perp"] * (1 + MMR)
            p_liq = (st["margin_posted"] + st["funding_acc"] + st["cost_basis"]) / denom if denom > 0 else float("inf")
            for _, bar in mark_df.iterrows():
                hh = float(bar["high"])
                e_high = st["margin_posted"] + st["funding_acc"] + st["cost_basis"] - st["qty_perp"] * hh
                mm_req = MMR * st["qty_perp"] * hh
                if e_high <= mm_req:
                    leftover = max(0.0, e_high - mm_req)
                    st["reserve"] += leftover
                    st["qty_perp"] = 0.0; st["cost_basis"] = 0.0
                    st["margin_posted"] = 0.0; st["funding_acc"] = 0.0
                    st["terminated"] = True
                    log_row(f"{utcnow()},LIQUIDATED,{bar['dt'].isoformat()},{hh},"
                            f"{st['qty_spot']:.8f},0,,,{total_eq(st, c, m_close):.2f},"
                            f"liq check mark_high={hh} p_liq~{p_liq:.0f}")
                    print(f"[shadow] LIQUIDATED on mark high={hh}")
                    break

        # ---- Rule A: margin defense at day close (mark price)
        if not st["terminated"] and st["qty_perp"] > 0:
            m_star = st["qty_perp"] * m_close / LEVERAGE
            e_p = perp_eq(st, m_close)
            if e_p < m_star:
                need = m_star - e_p
                take = min(need, max(st["reserve"], 0.0))
                st["reserve"] -= take; st["margin_posted"] += take; need -= take
                if need > 1e-9 and st["qty_spot"] > 0:
                    spot_val = st["qty_spot"] * c
                    sell_notional = min(need / (1 - SPOT_COST), spot_val)
                    f = min(sell_notional / spot_val, 1.0) if spot_val > 0 else 0.0
                    dq = st["qty_perp"] * f
                    proceeds = sell_notional * (1 - SPOT_COST)
                    st["qty_spot"] -= sell_notional / c
                    st["reserve"] += proceeds
                    realized = f * (st["cost_basis"] - st["qty_perp"] * m_close)
                    cover_cost = dq * m_close * PERP_COST
                    st["cost_basis"] *= (1 - f)
                    st["qty_perp"] -= dq
                    st["margin_posted"] += realized - cover_cost
                    m_star_new = st["qty_perp"] * m_close / LEVERAGE
                    e_p_new = perp_eq(st, m_close)
                    need2 = max(0.0, m_star_new - e_p_new)
                    take2 = min(need2, max(st["reserve"], 0.0))
                    st["reserve"] -= take2; st["margin_posted"] += take2
                    log_row(f"{utcnow()},MARGIN_TOPUP,{candle_dt.isoformat()},{c},"
                            f"{st['qty_spot']:.8f},{st['qty_perp']:.8f},,,"
                            f"{total_eq(st, c, m_close):.2f},sold ${sell_notional:.2f} spot")
                    print(f"[shadow] margin top-up: sold ${sell_notional:.2f} spot")

        # ---- Rule B: compounding rebalance
        if not st["terminated"] and st["qty_perp"] > 0:
            e_total = total_eq(st, c, m_close)
            n_star = e_total * LEVERAGE / (LEVERAGE + 1)
            n_cur = st["qty_perp"] * m_close
            if n_star > 0 and abs(n_cur - n_star) / n_star > REBAL_DRIFT:
                target_qty = n_star / m_close
                d_spot = target_qty - st["qty_spot"]
                if abs(d_spot) * c >= 1.0:
                    if d_spot > 0:
                        st["reserve"] -= d_spot * c * (1 + SPOT_COST)
                    else:
                        st["reserve"] += (-d_spot) * c * (1 - SPOT_COST)
                    st["qty_spot"] = target_qty
                d_perp = target_qty - st["qty_perp"]
                if abs(d_perp) * m_close >= 1.0:
                    cost = abs(d_perp) * m_close * PERP_COST
                    st["reserve"] -= cost
                    if d_perp > 0:
                        st["cost_basis"] += d_perp * m_close
                    else:
                        g = abs(d_perp) / st["qty_perp"] if st["qty_perp"] else 0.0
                        st["margin_posted"] += g * (st["cost_basis"] - st["qty_perp"] * m_close)
                        st["cost_basis"] *= (1 - g)
                    st["qty_perp"] = target_qty
                m_target = n_star / LEVERAGE
                st["reserve"] += st["margin_posted"] - m_target
                st["margin_posted"] = m_target
                log_row(f"{utcnow()},REBALANCE,{candle_dt.isoformat()},{m_close},"
                        f"{st['qty_spot']:.8f},{st['qty_perp']:.8f},,,"
                        f"{total_eq(st, c, m_close):.2f},N*={n_star:.2f}")
                print(f"[shadow] rebalance: N*={n_star:.2f} qty={target_qty:.6f}")

    # ---- sweep if perp fully closed
    if st["qty_perp"] == 0 and not st["terminated"]:
        st["reserve"] += st["margin_posted"] + st["funding_acc"]
        st["margin_posted"] = 0.0; st["funding_acc"] = 0.0

    eq = total_eq(st, c, m_close)
    st["last_run_utc"] = utcnow()
    STATE.write_text(json.dumps(st, indent=2))
    log_row(f"{utcnow()},MARK,{candle_dt.isoformat()},{c},{st['qty_spot']:.8f},"
            f"{st['qty_perp']:.8f},,,{eq:.2f},spot={c} mark={m_close}")
    print(f"[shadow] mark: spot={c} mark={m_close} equity={eq:.2f} "
          f"qty_spot={st['qty_spot']:.6f} qty_perp={st['qty_perp']:.6f}")


if __name__ == "__main__":
    main()

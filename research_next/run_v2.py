"""Cash-and-carry validation V2 runner.

Produces reports_v2/:
- liq_audit_ETH.csv / liq_audit_BTC.csv  (reconstruction of v1 liquidations,
  v2 isolated imr=0.20 cost=1.0 liq_fee=0 to match v1)
- margin_sweep.csv  (isolated+cross x IMR x cost, ETH & BTC)
- equity_*.csv / trades_*.csv per config
- common_period.csv (2021-02-01+ comparison)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backtest"))
import engine as eng
import cash_carry_v2 as v2

REP = ROOT / "reports_v2"
REP.mkdir(exist_ok=True)
S5_L1 = ROOT.parent / "slow_study" / "reports" / "equity_S5v2_L1.csv"
COMMON_START = "2021-02-01"

IMRS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
COSTS = [1.0, 1.5, 2.0]


def stats(eq, meta, extra=None):
    mr = eng.monthly_returns(eq, eng.START_EQUITY)
    st = eng.perf_stats(eq, mr)
    row = {"final": st["final"], "cagr": st["cagr"], "mean_mo": st["mean_mo"],
           "geo_mo": st["geo_mo"], "median_mo": st["median_mo"],
           "pct_prof": st["pct_prof"], "pct_ge6": st["pct_ge6"],
           "worst_mo": st["worst_mo"], "maxDD": st["maxDD"],
           "longest_uw_days": st["longest_uw_days"],
           "cost_total": meta["cost_total"], "n_liq": meta["n_liq"],
           "flat_days": meta["flat_days"], "max_util": meta["max_util"]}
    if extra:
        row.update(extra)
    return row


def main():
    # ---------- 1) liquidation audit: reproduce v1 (isolated, imr=0.2, cost=1, no liq fee)
    for base in ("ETH", "BTC"):
        eq, meta = v2.run_v2(base, mode="isolated", imr=0.20, cost_mult=1.0,
                             liq_fee_rate=0.0)
        eq.to_csv(REP / f"equity_audit_{base}.csv")
        meta["trades"].to_csv(REP / f"trades_audit_{base}.csv", index=False)
        if meta["liq_events"]:
            pd.DataFrame(meta["liq_events"]).to_csv(REP / f"liq_audit_{base}.csv", index=False)
        print(f"audit {base}: n_liq={meta['n_liq']} final={meta['final_equity']:.0f}")

    # ---------- 2) margin structure sweep ----------
    rows = []
    for base in ("ETH", "BTC"):
        for mode in ("isolated", "cross"):
            for imr in IMRS:
                for cm in COSTS:
                    tag = f"{base}_{mode}_imr{int(imr*100)}_c{cm:g}"
                    eq, meta = v2.run_v2(base, mode=mode, imr=imr, cost_mult=cm)
                    eq.to_csv(REP / f"equity_{tag}.csv")
                    if len(meta["trades"]):
                        meta["trades"].to_csv(REP / f"trades_{tag}.csv", index=False)
                    d = meta["decomp"]
                    row = {"base": base, "mode": mode, "imr": imr, "cost_mult": cm,
                           **stats(eq, meta),
                           "basis_entry_usd": d["basis_entry_usd"],
                           "convergence_usd": d["convergence_usd"],
                           "spot_pnl": d["spot_pnl"], "fut_pnl": d["fut_pnl"],
                           "liq_fees": d["liq_fees"]}
                    rows.append(row)
                    print(f"{tag}: final={row['final']:.0f} liq={row['n_liq']} "
                          f"mean_mo={row['mean_mo']*100:.2f}% maxDD={row['maxDD']*100:.1f}%")
    pd.DataFrame(rows).to_csv(REP / "margin_sweep.csv", index=False)

    # ---------- 3) common-period comparison 2021-02-01+ ----------
    cp = []
    # B ETH cross imr=0.20 (candidate) and isolated imr=0.35 (candidate)
    for base, mode, imr in [("ETH", "cross", 0.20), ("ETH", "isolated", 0.35),
                            ("BTC", "cross", 0.20), ("BTC", "isolated", 0.35)]:
        eq, meta = v2.run_v2(base, mode=mode, imr=imr, cost_mult=1.0,
                             start=COMMON_START)
        eq.to_csv(REP / f"equity_common_{base}_{mode}.csv")
        mr = eng.monthly_returns(eq, eng.START_EQUITY)
        # CAGR base = equity at start (1000 by construction)
        st = eng.perf_stats(eq, mr)
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        st["cagr"] = (eq.iloc[-1] / eng.START_EQUITY) ** (1 / yrs) - 1
        cp.append({"strategy": f"B_{base}_{mode}_imr{int(imr*100)}", **st,
                   "n_liq": meta["n_liq"], "cost_total": meta["cost_total"],
                   "flat_days": meta["flat_days"]})
    # S5 1x re-based to 1000 at common start
    if S5_L1.exists():
        eq5 = pd.read_csv(S5_L1, index_col=0, parse_dates=True).iloc[:, 0]
        eq5.index = pd.to_datetime(eq5.index, utc=True)
        eq5 = eq5[eq5.index >= pd.Timestamp(COMMON_START, tz="UTC")]
        eq5 = eq5 / eq5.iloc[0] * eng.START_EQUITY
        mr = eng.monthly_returns(eq5, eng.START_EQUITY)
        st = eng.perf_stats(eq5, mr)
        yrs = (eq5.index[-1] - eq5.index[0]).days / 365.25
        st["cagr"] = (eq5.iloc[-1] / eng.START_EQUITY) ** (1 / yrs) - 1
        cp.append({"strategy": "S5_L1", **st, "n_liq": 0, "cost_total": np.nan,
                   "flat_days": 0})
        eq5.to_csv(REP / "equity_common_S5_L1.csv")
    # BTC buy & hold re-based
    btc = eng.load_spot("BTC")
    bh = btc["close"]
    bh = bh[bh.index >= pd.Timestamp(COMMON_START, tz="UTC")]
    bh = bh / bh.iloc[0] * eng.START_EQUITY
    mr = eng.monthly_returns(bh, eng.START_EQUITY)
    st = eng.perf_stats(bh, mr)
    yrs = (bh.index[-1] - bh.index[0]).days / 365.25
    st["cagr"] = (bh.iloc[-1] / eng.START_EQUITY) ** (1 / yrs) - 1
    cp.append({"strategy": "BTC_buyhold", **st, "n_liq": 0, "cost_total": np.nan,
               "flat_days": 0})
    bh.to_csv(REP / "equity_common_BTC_buyhold.csv")
    # stablecoin ~3.5%/yr analytic
    yrs = (bh.index[-1] - bh.index[0]).days / 365.25
    cp.append({"strategy": "stablecoin_3.5pct", "final": eng.START_EQUITY * (1.035 ** yrs),
               "cagr": 0.035, "mean_mo": 0.035 / 12, "geo_mo": 0.035 / 12,
               "median_mo": 0.035 / 12, "pct_prof": 1.0, "pct_ge6": 0.0,
               "worst_mo": 0.035 / 12, "maxDD": 0.0, "longest_uw_days": 0,
               "n_liq": 0, "cost_total": 0.0, "flat_days": 0})
    pd.DataFrame(cp).to_csv(REP / "common_period.csv", index=False)
    print("\nwrote", REP)


if __name__ == "__main__":
    main()

"""Run all research_next strategies + benchmarks, write CSVs and summary.

Usage: python run_all.py
Outputs: reports/equity_*.csv, reports/trades_*.csv, reports/summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "strategies"))

import engine as eng
import sA_funding_arb as sA
import sC_adaptive as sC
import cash_carry as cc

REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX"]
S5_L1 = ROOT.parent / "slow_study" / "reports" / "equity_S5v2_L1.csv"


def splits(eq):
    """Return dict name -> (eq_subset, mr_full filtered)."""
    mr_full = eng.monthly_returns(eq, eng.START_EQUITY)
    out = {}
    for name, year0 in [("FULL", None), ("2023+", 2023), ("2025+", 2025)]:
        se = eq if year0 is None else eq[eq.index.year >= year0]
        mr = mr_full if year0 is None else mr_full[mr_full.index.year >= year0]
        if len(se) == 0 or len(mr) == 0:
            continue
        # CAGR base: last equity before split
        if year0 is None:
            base_eq, base_date = eng.START_EQUITY, se.index[0]
        else:
            pre = eq[eq.index < pd.Timestamp(f"{year0}-01-01", tz=eq.index.tz)]
            base_eq, base_date = (float(pre.iloc[-1]), pre.index[-1]) if len(pre) else (eng.START_EQUITY, se.index[0])
        yrs = (se.index[-1] - base_date).days / 365.25
        cagr = (se.iloc[-1] / base_eq) ** (1 / yrs) - 1 if yrs > 0 else np.nan
        st = eng.perf_stats(se, mr)
        st["cagr"] = cagr
        out[name] = st
    return out


def run_engine_strategy(name, policy_fn, spots, funds, cost_mult, tag):
    eq, meta = eng.run_portfolio(spots, funds, policy_fn, cost_mult=cost_mult)
    eq.to_csv(REP / f"equity_{name}_{tag}.csv")
    if len(meta["trades"]):
        meta["trades"].to_csv(REP / f"trades_{name}_{tag}.csv", index=False)
    # min liq distance across positions
    min_liq = np.inf
    for p in meta["positions"].values():
        if np.isfinite(p._min_liq_dist_high):
            min_liq = min(min_liq, p._min_liq_dist_high)
    return eq, meta, (min_liq * 100 if np.isfinite(min_liq) else np.nan)


def main():
    print("loading data...")
    spots, funds = eng.load_all(SYMBOLS)
    print(f"spots: {sorted(spots)}")
    print(f"funds: {sorted(funds)}")

    results = []

    # ---------- Strategy A: multi-asset funding arb ----------
    for L in sA.LEVERAGES:
        for cm in (1.0, 1.5, 2.0):
            tag = f"L{L}_cost{cm:g}"
            eq, meta, min_liq = run_engine_strategy(
                "sA_funding_arb", sA.make_policy(funds, L), spots, funds, cm, tag)
            for sname, st in splits(eq).items():
                results.append({"strategy": f"A_funding_arb_L{L}", "cost_mult": cm,
                                "split": sname, **st,
                                "cost_total": meta["cost_total"],
                                "funding_recv": meta["funding_recv"],
                                "funding_paid": meta["funding_paid"],
                                "n_liq": meta["n_liq"], "max_util": meta["max_util"],
                                "min_liq_dist_pct": min_liq,
                                "max_recon": meta["max_abs_recon"]})
            print(f"A L={L} cost={cm}: final={meta['final_equity']:.0f} liq={meta['n_liq']} "
                  f"min_liq={min_liq:.1f}% recon={meta['max_abs_recon']:.1e}")

    # ---------- Strategy C: adaptive funding carry ----------
    for L in sC.LEVERAGES:
        for cm in (1.0, 1.5, 2.0):
            tag = f"L{L}_cost{cm:g}"
            eq, meta, min_liq = run_engine_strategy(
                "sC_adaptive", sC.make_policy(spots, funds, L), spots, funds, cm, tag)
            for sname, st in splits(eq).items():
                results.append({"strategy": f"C_adaptive_L{L}", "cost_mult": cm,
                                "split": sname, **st,
                                "cost_total": meta["cost_total"],
                                "funding_recv": meta["funding_recv"],
                                "funding_paid": meta["funding_paid"],
                                "n_liq": meta["n_liq"], "max_util": meta["max_util"],
                                "min_liq_dist_pct": min_liq,
                                "max_recon": meta["max_abs_recon"]})
            print(f"C L={L} cost={cm}: final={meta['final_equity']:.0f} liq={meta['n_liq']} "
                  f"min_liq={min_liq:.1f}% recon={meta['max_abs_recon']:.1e}")

    # ---------- Strategy B: cash-and-carry ----------
    for base in ("BTC", "ETH"):
        for cm in (1.0, 1.5, 2.0):
            tag = f"{base}_cost{cm:g}"
            eq, meta = cc.run_cash_carry(base, cost_mult=cm)
            eq.to_csv(REP / f"equity_sB_cashcarry_{tag}.csv")
            if len(meta["trades"]):
                meta["trades"].to_csv(REP / f"trades_sB_cashcarry_{tag}.csv", index=False)
            for sname, st in splits(eq).items():
                results.append({"strategy": f"B_cashcarry_{base}", "cost_mult": cm,
                                "split": sname, **st,
                                "cost_total": meta["cost_total"],
                                "funding_recv": 0.0, "funding_paid": 0.0,
                                "n_liq": meta["n_liq"], "max_util": meta["max_util"],
                                "min_liq_dist_pct": np.nan,
                                "max_recon": np.nan})
            print(f"B {base} cost={cm}: final={meta['final_equity']:.0f} liq={meta['n_liq']} "
                  f"util={meta['max_util']:.2f}")

    # ---------- Benchmarks ----------
    # S5 1x (existing audited result)
    if S5_L1.exists():
        eq = pd.read_csv(S5_L1, index_col=0, parse_dates=True).iloc[:, 0]
        eq.index = pd.to_datetime(eq.index, utc=True)
        for sname, st in splits(eq).items():
            results.append({"strategy": "BENCH_S5_L1", "cost_mult": 1.0, "split": sname,
                            **st, "cost_total": np.nan, "funding_recv": np.nan,
                            "funding_paid": np.nan, "n_liq": 0, "max_util": np.nan,
                            "min_liq_dist_pct": 47.3, "max_recon": np.nan})
        eq.to_csv(REP / "equity_BENCH_S5_L1.csv")

    # BTC buy-and-hold (OKX spot)
    btc = spots["BTC"]
    bh = (btc["close"] / btc["close"].iloc[0] * eng.START_EQUITY)
    bh = bh[bh.index >= eng.monthly_returns(bh).index.min() - pd.Timedelta(days=400)] if False else bh
    # align to engine index: just use spot index
    for sname, st in splits(bh).items():
        results.append({"strategy": "BENCH_BTC_buyhold", "cost_mult": 1.0, "split": sname,
                        **st, "cost_total": np.nan, "funding_recv": np.nan,
                        "funding_paid": np.nan, "n_liq": 0, "max_util": 1.0,
                        "min_liq_dist_pct": np.nan, "max_recon": np.nan})
    bh.to_csv(REP / "equity_BENCH_BTC_buyhold.csv")

    # Stablecoin yield ~3.5%/yr -> monthly 0.287%; cash 0%
    # represented analytically in report (no equity curve needed)

    df = pd.DataFrame(results)
    df.to_csv(REP / "summary.csv", index=False)
    print(f"\nwrote {REP/'summary.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()

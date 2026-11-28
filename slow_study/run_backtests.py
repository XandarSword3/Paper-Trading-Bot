"""Run all frozen strategies S1-S6 and write reports/results.csv + REPORT.md.

Usage: python run_backtests.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backtest"))

import strategies as st
from backtest.engine import run_backtest
from backtest.s5_carry import run_s5
from backtest import metrics as met

REPORTS = Path(__file__).parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    bars = st.load_panel(st.SYMBOLS, "1D")
    print("loaded symbols:", {k: (str(v.index[0].date()), str(v.index[-1].date()), len(v))
                              for k, v in bars.items()})

    results = {}   # name -> equity series
    trades = {}

    # S1, S3, S6 on BTC; S2 on BTC+ETH; S4 on 10 alts
    btc = bars["BTC"]
    for name, fn in [("S1_ma_trend", st.s1_weights),
                     ("S3_donchian55", st.s3_weights),
                     ("S6_buyhold", st.s6_weights)]:
        w = fn(btc)
        eq, tr = run_backtest(w, {"BTC": btc})
        results[name] = eq
        trades[name] = tr
        print(f"{name}: {len(eq)} days, final equity {eq.iloc[-1]:.2f}")

    w2 = st.s2_weights(bars)
    eq2, tr2 = run_backtest(w2, {s: bars[s] for s in w2.columns})
    results["S2_tsmom_voltarget"] = eq2
    trades["S2_tsmom_voltarget"] = tr2
    print(f"S2: {len(eq2)} days, final equity {eq2.iloc[-1]:.2f}")

    w4 = st.s4_weights(bars)
    eq4, tr4 = run_backtest(w4, {s: bars[s] for s in w4.columns if s in bars})
    results["S4_xsmom_weekly"] = eq4
    trades["S4_xsmom_weekly"] = tr4
    print(f"S4: {len(eq4)} days, final equity {eq4.iloc[-1]:.2f}")

    try:
        eq5, tr5, fund5 = run_s5()
        results["S5_funding_carry"] = eq5
        trades["S5_funding_carry"] = tr5
        print(f"S5: {len(eq5)} days, final equity {eq5.iloc[-1]:.2f}, "
              f"funding events {len(fund5)}")
    except Exception as e:
        print(f"S5 FAILED: {e}")
        results["S5_funding_carry"] = None

    # ---- metrics ----
    all_rows = []
    for name, eq in results.items():
        if eq is None or len(eq) < 30:
            continue
        rows = met.evaluate(eq, name)
        all_rows.extend(rows)
        eq.to_csv(REPORTS / f"equity_{name}.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(REPORTS / "results.csv", index=False)

    # save trades logs
    for name, tr in trades.items():
        if tr is not None and len(tr):
            tr.to_csv(REPORTS / f"trades_{name}.csv", index=False)

    # ---- verdict vs gates (evaluated on 2023+ out-of-sample split) ----
    oos = df[df["split"] == "2023_plus"].copy()
    verdicts = []
    for _, r in oos.iterrows():
        name = r["label"].split("|")[0]
        ci_excludes_zero = (r["boot_lo99"] > 0) or (r["boot_hi99"] < 0)
        dd_ok = r["max_drawdown"] >= -0.40
        viable = bool(ci_excludes_zero and dd_ok and r["mean_monthly"] > 0)
        target = bool(r["mean_monthly"] >= 0.06)
        verdicts.append({
            "strategy": name, "mean_monthly": r["mean_monthly"],
            "boot_lo99": r["boot_lo99"], "boot_hi99": r["boot_hi99"],
            "max_drawdown": r["max_drawdown"], "viable": viable, "target": target,
        })
    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(REPORTS / "verdicts.csv", index=False)
    print("\n=== VERDICTS (2023+ OOS, Bonferroni-99% CI) ===")
    print(vdf.to_string(index=False))
    any_viable = bool(vdf["viable"].any()) if len(vdf) else False
    any_target = bool(vdf["target"].any()) if len(vdf) else False
    print(f"\nANY VIABLE: {any_viable} | ANY TARGET(>=6%/mo): {any_target}")
    if not any_viable:
        print("VERDICT: no strategy found")

    write_report(df, vdf, any_viable, any_target)


def write_report(df, vdf, any_viable, any_target):
    lines = ["# Slow-Strategy Study — Results (net of costs)\n"]
    lines.append("Costs: 0.10% taker + 0.05% slippage per side (0.15%/side). "
                 "Start equity $1,000. Signals on daily close, executed next open.\n")
    lines.append("Bootstrap: circular block bootstrap over monthly returns, "
                 "block=6 months, 10,000 reps. 99% CI = Bonferroni correction for 5 strategies.\n")
    lines.append("## Period splits\n")
    lines.append("| strategy | split | n_months | mean/mo | median/mo | %mo>=6% | worst/mo | "
                 "CAGR | maxDD | underwater_d | CI99_lo | CI99_hi |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['label']} | {r['split']} | {r['n_months']} | "
            f"{r['mean_monthly']*100:.2f}% | {r['median_monthly']*100:.2f}% | "
            f"{r['pct_months_ge_6']*100:.0f}% | {r['worst_month']*100:.1f}% | "
            f"{r['cagr']*100:.1f}% | {r['max_drawdown']*100:.1f}% | "
            f"{r['longest_underwater_days']} | {r['boot_lo99']*100:.2f}% | "
            f"{r['boot_hi99']*100:.2f}% |")
    lines.append("\n## Verdicts (2023+ out-of-sample)\n")
    lines.append("| strategy | mean/mo | CI99 | maxDD | VIABLE | TARGET(>=6%/mo) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in vdf.iterrows():
        lines.append(f"| {r['strategy']} | {r['mean_monthly']*100:.2f}% | "
                     f"[{r['boot_lo99']*100:.2f}%, {r['boot_hi99']*100:.2f}%] | "
                     f"{r['max_drawdown']*100:.1f}% | {r['viable']} | {r['target']} |")
    lines.append(f"\n## FINAL VERDICT\n")
    if any_target:
        lines.append("**TARGET MET** — at least one strategy reached >=6%/month mean "
                     "net out-of-sample.")
    elif any_viable:
        lines.append("**VIABLE but below target** — a strategy has a positive OOS CI "
                     "and DD<=40%, but mean < 6%/month.")
    else:
        lines.append("**NO STRATEGY FOUND** — no strategy satisfies the viability gate "
                     "(OOS net-return CI excludes zero AND max drawdown <= 40%). "
                     "Per the frozen protocol, no further strategies will be added.")
    (REPORTS / "REPORT.md").write_text("\n".join(lines))
    print("wrote reports/REPORT.md")


if __name__ == "__main__":
    main()

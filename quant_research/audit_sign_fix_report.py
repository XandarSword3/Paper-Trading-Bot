"""
Item 2 / Item 4 audit report generator (reads committed + v2 artifacts only; writes nothing
back into them).

Outputs:
  quant_research/results/turtle_sign_fix_comparison.json  (item 2)
  quant_research/results/cycle_metrics_v2.json            (item 4 per-config cycle table)
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np

RESULTS_V1 = "quant_research/results/turtle_results.json"          # committed (direction-agnostic funding)
RESULTS_V2 = "quant_research/results/turtle_results_v2.json"       # corrected sign
DIAG_V1 = "quant_research/results/turtle_diagnostics.json"         # committed diagnostics (old sign)
DIAG_V2 = "quant_research/results/turtle_diagnostics_v2.json"      # corrected sign (+ cycle detail)
OUT_ITEM2 = "quant_research/results/turtle_sign_fix_comparison.json"
OUT_ITEM4 = "quant_research/results/cycle_metrics_v2.json"


def load(p):
    with open(p) as f:
        return json.load(f)


def implied_funding_bps(row):
    """funding bps per unit implied by the identity net = gross - fee + funding."""
    return row["net_bps"] - (row["gross_bps"] - row["fee_bps"])


def main():
    v1 = load(RESULTS_V1)
    v2 = load(RESULTS_V2)
    d1 = load(DIAG_V1)
    d2 = load(DIAG_V2)

    # ---------------- ITEM 2: per-config (36) before/after sign fix ----------------
    diag1 = {(r["asset"], r["timeframe"], r["N"], r["M"]): r for r in d1["diagnostics_individual"]}
    diag2 = {(r["asset"], r["timeframe"], r["N"], r["M"]): r for r in d2["diagnostics_individual"]}

    rows = []
    for key in sorted(diag1.keys()):
        a = diag1[key]
        b = diag2[key]
        before_net = a["net_bps"]
        after_net = b["net_bps"]
        fund_before = implied_funding_bps(a)
        fund_after = implied_funding_bps(b)
        rows.append({
            "asset": key[0], "timeframe": key[1], "N": key[2], "M": key[3],
            "net_bps_before": before_net,
            "net_bps_after": after_net,
            "delta_bps": round(after_net - before_net, 4),
            "funding_bps_per_unit_before": round(fund_before, 4),
            "funding_bps_per_unit_after": round(fund_after, 4),
            "gross_bps_after": b["gross_bps"],
            "fee_bps_after": b["fee_bps"],
            "sign_changed": (before_net > 0) != (after_net > 0) and abs(after_net) > 1e-9 and abs(before_net) > 1e-9,
        })

    # Pooled 1h N=20 M=20 (trade-weighted, from the runner) per asset + pooled
    def runner_rows(payload):
        return {(r["asset"], r["timeframe"], r["N"], r["M"]): r for r in payload["individual_runs"]}

    r1 = runner_rows(v1)
    r2 = runner_rows(v2)
    pooled = []
    for asset in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
        k = (asset, "1h", 20, 20)
        pooled.append({
            "asset": asset,
            "oos_trades": r2[k]["oos_trades"],
            "net_bps_before": r1[k]["oos_exp_bps"],
            "net_bps_after": r2[k]["oos_exp_bps"],
            "delta_bps": round(r2[k]["oos_exp_bps"] - r1[k]["oos_exp_bps"], 4),
            "daily_sharpe_after": r2[k]["oos_daily_sharpe"],
        })
    pr1 = {(p["timeframe"], p["N"], p["M"]): p for p in v1["pooled_summary"]}
    pr2 = {(p["timeframe"], p["N"], p["M"]): p for p in v2["pooled_summary"]}
    pooled_rows = []
    for key in sorted(pr1.keys()):
        tf, N, M = key
        pooled_rows.append({
            "timeframe": tf, "N": N, "M": M,
            "pooled_oos_trades": pr2[key]["pooled_oos_trades"],
            "pooled_net_bps_before": pr1[key]["pooled_oos_exp_bps"],
            "pooled_net_bps_after": pr2[key]["pooled_oos_exp_bps"],
            "pooled_delta_bps": round(pr2[key]["pooled_oos_exp_bps"] - pr1[key]["pooled_oos_exp_bps"], 4),
            "pooled_oos_stress_bps_after": pr2[key]["pooled_oos_stress_bps"],
            "verdict_before": pr1[key]["verdict"],
            "verdict_after": pr2[key]["verdict"],
        })

    payload2 = {
        "note": "net_bps_before/after are per-UNIT net expectancies from the diagnostics simulator "
                "(equal weight per unit); funding_bps_per_unit is implied by net = gross - fee + funding.",
        "configs": rows,
        "pooled_by_config": pooled_rows,
        "pooled_1h_20_20_per_asset": pooled,
        "n_sign_changes": int(sum(1 for r in rows if r["sign_changed"])),
        "n_verdict_changes": int(sum(1 for r in pooled_rows if r["verdict_before"] != r["verdict_after"])),
        "n_configs_net_moved": int(sum(1 for r in rows if abs(r["delta_bps"]) > 1e-9)),
    }
    with open(OUT_ITEM2, "w") as f:
        json.dump(payload2, f, indent=2)
    print(f"Wrote {OUT_ITEM2}")

    # ---------------- ITEM 4: per-config cycle metrics ----------------
    cyc = []
    for r in d2["diagnostics_individual"]:
        n_cycles = r["n_cycles"]
        cyc.append({
            "asset": r["asset"], "timeframe": r["timeframe"], "N": r["N"], "M": r["M"],
            "cycles": n_cycles,
            "unit_net_bps_mean (per-unit, trade-weighted within config)": r["net_bps"],
            "cycle_mean_bps (argument order)": r["cycle_net_bps_mean"],
            "cycle_se_iid": r["cycle_se_bps"],
            "cycle_se_day_clustered": r["day_clustered_se_bps"],
            "cycle_n_eff": r["n_eff_day_clustered"],
            "share_1unit_pct": r["cycle_1unit_share_pct"],
            "n_1unit_cycles": r["n_1unit_cycles"],
            "n_multiunit_cycles": r["n_multiunit_cycles"],
            "mean_1unit_cycle_bps": r["cycle_1unit_mean_bps"],
            "mean_multiunit_cycle_bps": r["cycle_multiunit_mean_bps"],
            "total_cycle_net_usd": r["cycle_dollar_pnl_total"],
            "total_cycle_entry_notional_usd": r["cycle_notional_total"],
            "dollar_weighted_cycle_bps": r["cycle_dollar_weighted_bps"],
            "dollar_pnl_per_cycle_usd": round(r["cycle_dollar_pnl_total"] / n_cycles, 6) if n_cycles else None,
            "long_cycle_share_pct": r["cycle_long_share_pct"],
            "daily_return_bps_mean": r["daily_return_bps_mean"],
            "daily_return_bps_se": r["daily_return_bps_se"],
            "daily_return_bps_t": r["daily_return_bps_t"],
            "n_days": r["n_days"],
            "equity_return_pct": r["total_return_pct"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "daily_sharpe": r["daily_sharpe"],
        })
    payload4 = {
        "definition": {
            "cycle": "one completed position lifecycle (1-4 units sharing one entry decision, "
                     "one ratcheted stop and one exit)",
            "cycle bps numerator": "sum of the cycle's unit NET dollar P&L (gross - costs + funding)",
            "cycle bps denominator (argument order)": "mean over the cycle's units of the unit's own "
                     "net/entry-notional ratio (equal weight per unit within the cycle), NOT the cycle's total dollars",
            "pooled cycle mean": "equal-weighted mean over completed cycles (each cycle one vote)",
            "dollar-weighted alternative": "sum(cycle net $) / sum(cycle entry notional $) x 10000",
        },
        "configs": cyc,
    }
    with open(OUT_ITEM4, "w") as f:
        json.dump(payload4, f, indent=2)
    print(f"Wrote {OUT_ITEM4}")

    # ---------------- console summaries ----------------
    print("\n=== ITEM 2: per-config before/after (36) ===")
    print(f"{'config':24} {'net before':>11} {'net after':>10} {'delta':>8} {'fund/unit bef':>14} {'fund/unit aft':>14} {'sign flip':>9}")
    for r in rows:
        label = f"{r['asset']} {r['timeframe']} {r['N']}/{r['M']}"
        print(f"{label:24} {r['net_bps_before']:11.3f} {r['net_bps_after']:10.3f} {r['delta_bps']:8.3f} "
              f"{r['funding_bps_per_unit_before']:14.4f} {r['funding_bps_per_unit_after']:14.4f} {str(r['sign_changed']):>9}")

    print("\n=== Pooled 1h N=20 M=20 per asset (runner) ===")
    for r in pooled:
        print(f"{r['asset']:9} {r['oos_trades']:5d} trades | net before {r['net_bps_before']:+.3f} -> "
              f"after {r['net_bps_after']:+.3f} bps (delta {r['delta_bps']:+.3f})")

    print("\n=== Pooled per config (12 rows) ===")
    for r in pooled_rows:
        print(f"{r['timeframe']:3} {r['N']:2}/{r['M']:2} | {r['pooled_oos_trades']:5d} tr | "
              f"net before {r['pooled_net_bps_before']:+7.3f} -> after {r['pooled_net_bps_after']:+7.3f} "
              f"(delta {r['pooled_delta_bps']:+7.3f}) | stress after {r['pooled_oos_stress_bps_after']:+7.3f} | "
              f"verdict {r['verdict_before']} -> {r['verdict_after']}")

    print("\n=== Item 4: cycle table (v2) ===")
    for r in cyc:
        print(f"{r['asset']:8} {r['timeframe']:3} {r['N']}/{r['M']}: cycles {r['cycles']:5d} | "
              f"1-unit {r['share_1unit_pct']:5.1f}% mean {r['mean_1unit_cycle_bps']:9.2f} | "
              f"multi mean {r['mean_multiunit_cycle_bps']:9.2f} | $/cycle {r['dollar_pnl_per_cycle_usd']:12.2f} | "
              f"cycle mean {r['cycle_mean_bps (argument order)']:9.2f} | day t {r['daily_return_bps_t']:7.2f} | "
              f"ret {r['equity_return_pct']:8.2f}% dd {r['max_drawdown_pct']:6.2f}%")


if __name__ == "__main__":
    main()

"""
Item 7 - Replay artifact v2: rebuild the Test 3 replay summary with the LEDGER
commissions (inputs/transaction_history.csv) instead of the replay's own
reconstructed commission column, and report the ledger-true net.

The committed test_3_replay_summary.json reconstructed commissions from the trade
history (`test_3_replay_trades.csv`, commission column sum -1,139.53 USDT) and gets
net -178.59 USDT. The exchange transaction ledger independently shows:
    COMMISSION      -1,131.47  (1,785 entries)
    FUNDING_FEE         +0.68
    INSURANCE_CLEAR    -18.76
    REALIZED_PNL      +960.27
    -> ledger net   -189.28 USDT, i.e. 100% of the 189.28 USDT deposited.

This script writes NEW files; the committed summary is left untouched and is
superseded by the pointers added to REPORT.md.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pandas as pd

LEDGER = "quant_research/inputs/transaction_history.csv"
REPLAY_TRADES = "quant_research/results/test_3_replay_trades.csv"
OLD_SUMMARY = "quant_research/results/test_3_replay_summary.json"
OUT = "quant_research/results/test_3_replay_summary_v2.json"

DEPOSIT_TYPES = ("TRANSFER", "STRATEGY_ACCOUNT_TRANSFER")


def main():
    tx = pd.read_csv(LEDGER)
    tx["Time"] = pd.to_datetime(tx["Time"])

    ledger_comm = tx[tx["Type"] == "COMMISSION"].groupby("Symbol")["Amount"].agg(["count", "sum"])
    ledger_funding = float(tx[tx["Type"] == "FUNDING_FEE"]["Amount"].sum())
    ledger_insurance = float(tx[tx["Type"] == "INSURANCE_CLEAR"]["Amount"].sum())
    ledger_realized = float(tx[tx["Type"] == "REALIZED_PNL"]["Amount"].sum())
    ledger_comm_total = float(tx[tx["Type"] == "COMMISSION"]["Amount"].sum())
    ledger_net = ledger_realized + ledger_comm_total + ledger_funding + ledger_insurance
    deposits = float(tx[tx["Type"].isin(DEPOSIT_TYPES)]["Amount"].sum())

    trades = pd.read_csv(REPLAY_TRADES)
    total_volume = float(trades["entry_notional"].sum() + trades["exit_notional"].sum())
    gross = float(trades["gross_pnl"].sum())
    funding = float(trades["funding_pnl"].sum())
    old_commission = float(trades["commission"].sum())

    # Per-symbol block with the ledger commission next to the replay reconstruction
    sym_rows = []
    for sym, sub in trades.groupby("symbol"):
        vol = float(sub["entry_notional"].sum() + sub["exit_notional"].sum())
        g = float(sub["gross_pnl"].sum())
        f = float(sub["funding_pnl"].sum())
        c_replay = float(sub["commission"].sum())
        lc = ledger_comm.loc[sym] if sym in ledger_comm.index else None
        c_ledger = float(lc["sum"]) if lc is not None else 0.0
        net_ledger = g + f + c_ledger
        bps_gross = (g / vol * 10000.0) if vol > 0 else 0.0
        bps_fee = (c_ledger / vol * 10000.0) if vol > 0 else 0.0
        bps_net = (net_ledger / vol * 10000.0) if vol > 0 else 0.0
        sym_rows.append({
            "symbol": sym,
            "trades": int(len(sub)),
            "volume_usd": round(vol, 2),
            "gross_pnl_usd": round(g, 2),
            "commission_replay_usd": round(c_replay, 2),
            "commission_ledger_usd": round(c_ledger, 2),
            "commission_ledger_entries": int(lc["count"]) if lc is not None else 0,
            "funding_usd": round(f, 2),
            "net_pnl_ledger_usd": round(net_ledger, 2),
            "gross_bps": round(bps_gross, 2),
            "fee_bps_ledger": round(bps_fee, 2),
            "net_bps_ledger": round(bps_net, 2),
        })

    payload = {
        "supersedes": "results/test_3_replay_summary.json (reconstructed commissions; net -178.59 USDT)",
        "commission_source": "inputs/transaction_history.csv (exchange ledger, COMMISSION entries)",
        "total_trades": int(len(trades)),
        "symbols": sorted(trades["symbol"].unique().tolist()),
        "total_volume_usd": round(total_volume, 2),
        "total_gross_pnl_usd": round(gross, 2),
        "total_commissions_replay_usd": round(old_commission, 2),
        "total_commissions_ledger_usd": round(ledger_comm_total, 2),
        "commission_ledger_entries": int(tx[tx["Type"] == "COMMISSION"].shape[0]),
        "total_funding_usd_ledger": round(ledger_funding, 2),
        "total_funding_usd_replay": round(funding, 2),
        "insurance_liquidation_usd": round(ledger_insurance, 2),
        "realized_pnl_ledger_usd": round(ledger_realized, 2),
        "total_net_pnl_ledger_usd": round(ledger_net, 2),
        "total_deposits_usd": round(deposits, 2),
        "account_wipeout_pct": round(abs(ledger_net) / deposits * 100.0, 2) if deposits > 0 else 0.0,
        "avg_gross_expectancy_bps": round(gross / total_volume * 10000.0, 2),
        "avg_fee_drag_bps_ledger": round(ledger_comm_total / total_volume * 10000.0, 2),
        "avg_net_expectancy_bps_ledger": round(ledger_net / total_volume * 10000.0, 2),
        "symbol_breakdown": sym_rows,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 96)
    print("TEST 3 REPLAY SUMMARY v2 (LEDGER COMMISSIONS)")
    print("=" * 96)
    print(f"Trades: {payload['total_trades']} | Volume: ${payload['total_volume_usd']:,.2f}")
    print(f"Gross: {payload['total_gross_pnl_usd']:+,.2f} USDT")
    print(f"Commissions: replay reconstruction {payload['total_commissions_replay_usd']:+,.2f} | "
          f"exchange ledger {payload['total_commissions_ledger_usd']:+,.2f} "
          f"({payload['commission_ledger_entries']} entries)")
    print(f"Funding (ledger): {payload['total_funding_usd_ledger']:+,.2f} | "
          f"Insurance/liquidation: {payload['insurance_liquidation_usd']:+,.2f} | "
          f"Realized PnL (ledger): {payload['realized_pnl_ledger_usd']:+,.2f}")
    print(f"LEDGER NET: {payload['total_net_pnl_ledger_usd']:+,.2f} USDT on "
          f"{payload['total_deposits_usd']:,.2f} deposited -> capital depletion "
          f"{payload['account_wipeout_pct']:.2f}%")
    print(f"bps (per traded volume): gross {payload['avg_gross_expectancy_bps']:+.2f} | "
          f"fees {payload['avg_fee_drag_bps_ledger']:+.2f} | net {payload['avg_net_expectancy_bps_ledger']:+.2f}")
    print("\nPer symbol (ledger commissions):")
    for r in sym_rows:
        print(f"  {r['symbol']:8} {r['trades']:4d} trades | gross ${r['gross_pnl_usd']:+7.2f} | "
              f"replay fees ${r['commission_replay_usd']:+8.2f} | ledger fees ${r['commission_ledger_usd']:+8.2f} "
              f"({r['commission_ledger_entries']:4d} entries) | ledger net ${r['net_pnl_ledger_usd']:+8.2f} "
              f"({r['net_bps_ledger']:+.2f} bps)")
    print(f"\nSaved {OUT}")
    return payload


if __name__ == "__main__":
    main()

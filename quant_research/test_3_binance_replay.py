"""
Test 3: Empirical Replay of Real Binance Futures Execution Archives.
Replays the user's 257 trades from:
- quant_research/inputs/trade_history.csv
- quant_research/inputs/transaction_history.csv
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any


def run_binance_replay():
    print("=" * 80)
    print("TEST 3: EMPIRICAL REPLAY OF REAL BINANCE FUTURES TRADING HISTORY")
    print("=" * 80)

    trades_path = "quant_research/inputs/trade_history.csv"
    trans_path = "quant_research/inputs/transaction_history.csv"

    if not os.path.exists(trades_path) or not os.path.exists(trans_path):
        raise FileNotFoundError("Binance history CSVs missing from quant_research/inputs/")

    df_trades = pd.read_csv(trades_path)
    df_trans = pd.read_csv(trans_path)

    df_trades["Time"] = pd.to_datetime(df_trades["Time"])
    df_trans["Time"] = pd.to_datetime(df_trans["Time"])

    df_trades = df_trades.sort_values("Time").reset_index(drop=True)
    df_trans = df_trans.sort_values("Time").reset_index(drop=True)

    print(f"Loaded {len(df_trades):,} fills across {len(df_trans):,} transaction records.")
    print(f"Time range: {df_trades['Time'].min()} to {df_trades['Time'].max()}\n")

    # Reconstruct position cycles per symbol
    symbols = df_trades["Symbol"].unique()
    all_completed_trades = []

    # Map commissions and funding from transactions
    # Note: Trade ID in trades matches Trade ID / Transaction ID in trans
    commissions_by_trade_id = {}
    for _, row in df_trans[df_trans["Type"] == "COMMISSION"].iterrows():
        t_id = row["Transaction ID"]
        commissions_by_trade_id[t_id] = row["Amount"]

    funding_records = df_trans[df_trans["Type"] == "FUNDING_FEE"].copy()

    for sym in sorted(symbols):
        sub_trades = df_trades[df_trades["Symbol"] == sym].copy()
        
        pos_qty = 0.0
        current_fills = []
        trade_index = 1

        for _, fill in sub_trades.iterrows():
            side = fill["Side"]
            qty = float(fill["Quantity"])
            price = float(fill["Price"])
            notional = float(fill["Amount"])
            t_time = fill["Time"]
            trade_id = fill["Trade ID"]
            realized_pnl = float(fill["Realized Profit"])
            
            # Commission from trans_df
            # If trade_id matched or matched by time
            comm = commissions_by_trade_id.get(trade_id, None)
            if comm is None:
                # Approximate 0.05% if missing
                comm = -notional * 0.0005
            else:
                comm = float(comm)

            delta_qty = qty if side == "BUY" else -qty
            prev_pos = pos_qty
            pos_qty = round(pos_qty + delta_qty, 8)

            current_fills.append({
                "time": t_time,
                "side": side,
                "price": price,
                "qty": qty,
                "notional": notional,
                "commission": comm,
                "realized_profit": realized_pnl,
            })

            # Check if position just closed back to 0.0
            if prev_pos != 0.0 and abs(pos_qty) < 1e-7:
                # Closed trade cycle!
                entry_fills = [f for f in current_fills if (f["side"] == "BUY" if prev_pos > 0 else f["side"] == "SELL")]
                exit_fills = [f for f in current_fills if (f["side"] == "SELL" if prev_pos > 0 else f["side"] == "BUY")]

                entry_notional = sum(f["notional"] for f in entry_fills)
                exit_notional = sum(f["notional"] for f in exit_fills)
                total_qty = sum(f["qty"] for f in entry_fills)
                direction = "LONG" if prev_pos > 0 else "SHORT"

                avg_entry = (entry_notional / total_qty) if total_qty > 0 else 0.0
                avg_exit = (exit_notional / total_qty) if total_qty > 0 else 0.0

                cycle_comm = sum(f["commission"] for f in current_fills)
                cycle_realized_pnl = sum(f["realized_profit"] for f in exit_fills)

                # Find funding fees during this cycle
                start_t = current_fills[0]["time"]
                end_t = current_fills[-1]["time"]
                sym_funding = funding_records[
                    (funding_records["Symbol"] == sym) &
                    (funding_records["Time"] >= start_t) &
                    (funding_records["Time"] <= end_t)
                ]
                cycle_funding = sym_funding["Amount"].sum()

                net_pnl = cycle_realized_pnl + cycle_comm + cycle_funding
                net_bps = (net_pnl / entry_notional * 10000.0) if entry_notional > 0 else 0.0
                gross_bps = (cycle_realized_pnl / entry_notional * 10000.0) if entry_notional > 0 else 0.0
                fee_bps = (abs(cycle_comm) / entry_notional * 10000.0) if entry_notional > 0 else 0.0
                funding_bps = (cycle_funding / entry_notional * 10000.0) if entry_notional > 0 else 0.0

                all_completed_trades.append({
                    "symbol": sym,
                    "trade_num": trade_index,
                    "direction": direction,
                    "entry_time": str(start_t),
                    "exit_time": str(end_t),
                    "duration_minutes": round((end_t - start_t).total_seconds() / 60.0, 2),
                    "quantity": total_qty,
                    "entry_price": avg_entry,
                    "exit_price": avg_exit,
                    "entry_notional": entry_notional,
                    "exit_notional": exit_notional,
                    "gross_pnl": cycle_realized_pnl,
                    "commission": cycle_comm,
                    "funding_pnl": cycle_funding,
                    "net_pnl": net_pnl,
                    "gross_bps": gross_bps,
                    "fee_bps": fee_bps,
                    "funding_bps": funding_bps,
                    "net_bps": net_bps,
                    "is_win": net_pnl > 0,
                })

                trade_index += 1
                current_fills = []
                pos_qty = 0.0

    trades_df_out = pd.DataFrame(all_completed_trades)
    print(f"Reconstructed exactly {len(trades_df_out)} closed round-trip trade cycles:")
    print(trades_df_out["symbol"].value_counts().to_string())

    # Overall Portfolio Performance
    total_trades = len(trades_df_out)
    wins = trades_df_out[trades_df_out["net_pnl"] > 0]
    losses = trades_df_out[trades_df_out["net_pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0

    total_gross = trades_df_out["gross_pnl"].sum()
    total_fees = trades_df_out["commission"].sum()
    total_funding = trades_df_out["funding_pnl"].sum()
    total_net = trades_df_out["net_pnl"].sum()

    gross_wins = wins["gross_pnl"].sum()
    gross_losses = abs(losses["gross_pnl"].sum())
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 0.0

    total_volume = trades_df_out["entry_notional"].sum() + trades_df_out["exit_notional"].sum()
    avg_net_bps = trades_df_out["net_bps"].mean()
    avg_gross_bps = trades_df_out["gross_bps"].mean()
    avg_fee_bps = trades_df_out["fee_bps"].mean()
    avg_funding_bps = trades_df_out["funding_bps"].mean()

    # Cashflow reconciliation from transaction_history.csv
    total_deposits = df_trans[df_trans["Type"].isin(["TRANSFER", "STRATEGY_ACCOUNT_TRANSFER"])]["Amount"].sum()
    insurance_clear = df_trans[df_trans["Type"] == "INSURANCE_CLEAR"]["Amount"].sum()

    # Equity curve from $1,000 baseline
    initial_baseline = 1000.0
    equity = initial_baseline
    eq_curve = [equity]
    for pnl in trades_df_out["net_pnl"]:
        equity += pnl
        eq_curve.append(equity)
    
    eq_series = pd.Series(eq_curve)
    peaks = eq_series.cummax()
    dd_series = (peaks - eq_series) / peaks * 100.0
    max_dd_pct = dd_series.max()

    summary = {
        "total_trades": total_trades,
        "symbols": trades_df_out["symbol"].value_counts().to_dict(),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_volume_usd": round(total_volume, 2),
        "total_gross_pnl_usd": round(total_gross, 2),
        "total_commissions_usd": round(total_fees, 2),
        "total_funding_usd": round(total_funding, 2),
        "insurance_liquidation_usd": round(insurance_clear, 2),
        "total_net_pnl_usd": round(total_net, 2),
        "avg_gross_expectancy_bps": round(avg_gross_bps, 2),
        "avg_fee_drag_bps": round(avg_fee_bps, 2),
        "avg_funding_bps": round(avg_funding_bps, 2),
        "avg_net_expectancy_bps": round(avg_net_bps, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_deposits_usd": round(total_deposits, 2),
        "account_wipeout_pct": round((abs(total_net) / total_deposits * 100.0) if total_deposits > 0 else 0.0, 2),
    }

    # Breakdown by symbol
    sym_breakdown = {}
    for sym in sorted(symbols):
        st = trades_df_out[trades_df_out["symbol"] == sym]
        w = st[st["net_pnl"] > 0]
        sym_breakdown[sym] = {
            "trades": len(st),
            "win_rate_pct": round(len(w) / len(st) * 100.0, 2) if len(st) > 0 else 0.0,
            "volume_usd": round(st["entry_notional"].sum() + st["exit_notional"].sum(), 2),
            "gross_pnl_usd": round(st["gross_pnl"].sum(), 2),
            "commission_usd": round(st["commission"].sum(), 2),
            "funding_usd": round(st["funding_pnl"].sum(), 2),
            "net_pnl_usd": round(st["net_pnl"].sum(), 2),
            "avg_net_bps": round(st["net_bps"].mean(), 2),
        }
    summary["symbol_breakdown"] = sym_breakdown

    # Save to results
    os.makedirs("quant_research/results", exist_ok=True)
    summary_path = "quant_research/results/test_3_replay_summary.json"
    trades_out_path = "quant_research/results/test_3_replay_trades.csv"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    trades_df_out.to_csv(trades_out_path, index=False)

    print("\n" + "=" * 80)
    print("TEST 3 EMPIRICAL REPLAY RESULTS")
    print("=" * 80)
    print(f"Total Completed Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total Volume Traded: ${total_volume:,.2f}")
    print(f"Gross Realized PnL: ${total_gross:,.2f} (+{avg_gross_bps:.2f} bps avg)")
    print(f"Commissions Paid: ${total_fees:,.2f} (-{avg_fee_bps:.2f} bps avg)")
    print(f"Funding Carry: ${total_funding:,.2f} ({avg_funding_bps:.2f} bps avg)")
    print(f"Insurance / Liquidation Clear: ${insurance_clear:,.2f}")
    print(f"Net Realized Trading PnL: ${total_net:,.2f} ({avg_net_bps:.2f} bps avg)")
    print(f"Account Deposit Basis: ${total_deposits:,.2f}")
    print(f"Capital Depletion: {summary['account_wipeout_pct']:.2f}%\n")

    print("Breakdown by Symbol:")
    for sym, m in sym_breakdown.items():
        print(f"  {sym:8} -> {m['trades']:3} trades | Win Rate: {m['win_rate_pct']:5.2f}% | Gross: ${m['gross_pnl_usd']:8.2f} | Fees: ${m['commission_usd']:8.2f} | Net: ${m['net_pnl_usd']:8.2f} ({m['avg_net_bps']:+.2f} bps)")

    print(f"\nSaved trade log to {trades_out_path}")
    print(f"Saved summary to {summary_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_binance_replay()

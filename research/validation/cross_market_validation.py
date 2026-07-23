"""
Phase 4 acceptance-check script — VALIDATION_REMEDIATION_PLAN.md.

Runs V1's frozen rule set (config.DEFAULT_PARAMS, unchanged — no
re-fitting per market) against the reference market plus each configured
cross-market leg (config.DEFAULT_CROSS_MARKET): a different asset
(ETH/USDT), a different timeframe (BTC/USDT 1h), and a different data
source (BTC/USD via Kraken — the source both live bots actually trade
against, per STRATEGY_VERSION_AUDIT.md). Every leg is restricted to the
development window (data_splits.get_development — in-sample + validation);
the true holdout is never touched here, same discipline as the rest of
this branch.

Goal, from the plan: check whether the edge is structural or a
BTC-2017-2025-shaped coincidence. This script reports the numbers side by
side — it does not compute an automated pass/fail verdict. A Sharpe ratio
alone doesn't say whether a leg had enough trades to mean anything; read
num_trades alongside it, and treat anything under ~20 trades in its
development window as too thin to conclude much either way.

Writes:
    results/cross_market_validation.csv

Usage:
    python cross_market_validation.py
    python cross_market_validation.py --initial-capital 50000
    python cross_market_validation.py --start-date 2018-01-01
"""

import argparse
import os

import pandas as pd

from config import DEFAULT_CROSS_MARKET, DEFAULT_PARAMS, RESULTS_DIR, CrossMarketLeg
from data_splits import describe_split, get_development
from strategy import TurtleDonchianStrategy

MIN_BARS_FOR_BACKTEST = 300   # below this, indicators barely finish warming up
THIN_TRADE_COUNT = 20         # below this, don't read much into the Sharpe/CAGR


def fetch_leg(leg: CrossMarketLeg, start_date: str) -> pd.DataFrame:
    """Dispatch to the right fetcher for this leg's data source."""
    if leg.source == "binance":
        from data_fetcher import download_binance_data
        return download_binance_data(symbol=leg.symbol, timeframe=leg.timeframe, start_date=start_date)
    elif leg.source == "kraken":
        from data_fetcher_kraken import download_kraken_data
        return download_kraken_data(pair=leg.symbol, timeframe=leg.timeframe, start_date=start_date)
    raise ValueError(f"Unknown data source '{leg.source}' for leg '{leg.label}'")


def run_leg(leg: CrossMarketLeg, initial_capital: float, start_date: str) -> dict:
    """Run the frozen rule set on one leg's development-window data. Returns
    a flat dict suitable for a DataFrame row; includes an 'error' key instead
    of metrics if the leg couldn't be run."""
    base = {
        "label": leg.label, "source": leg.source,
        "symbol": leg.symbol, "timeframe": leg.timeframe,
    }
    try:
        df = fetch_leg(leg, start_date)
    except Exception as e:
        return {**base, "error": f"fetch failed: {e}"}

    df_dev = get_development(df)
    if len(df_dev) < MIN_BARS_FOR_BACKTEST:
        return {**base, "error": f"only {len(df_dev)} bars in development window (need >= {MIN_BARS_FOR_BACKTEST})"}

    strategy = TurtleDonchianStrategy(DEFAULT_PARAMS)  # exact same params, no re-fitting
    strategy.run_backtest(df_dev, initial_capital=initial_capital, verbose=False)
    equity_stats = strategy.get_equity_stats(initial_capital)
    trade_stats = strategy.get_trade_stats()

    return {
        **base,
        "bars": len(df_dev),
        "window_start": str(df_dev.index[0]),
        "window_end": str(df_dev.index[-1]),
        "total_return_pct": equity_stats.get("total_return_pct", 0.0),
        "cagr_pct": equity_stats.get("cagr_pct", 0.0),
        "max_drawdown_pct": equity_stats.get("max_drawdown_pct", 0.0),
        "sharpe_ratio": equity_stats.get("sharpe_ratio", 0.0),
        "calmar_ratio": equity_stats.get("calmar_ratio", 0.0),
        "num_trades": trade_stats.get("total_trades", 0),
        "win_rate": trade_stats.get("win_rate", 0.0),
        "profit_factor": trade_stats.get("profit_factor", 0.0),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 cross-market validation")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--start-date", default="2017-01-01",
                    help="Fetch start date per leg — trimmed to each leg's own "
                         "development window regardless, so an asset with a "
                         "shorter history (e.g. ETH) just starts later.")
    return p.parse_args()


def main():
    args = parse_args()

    print(describe_split())
    print(
        "\nRunning V1's frozen rule set (config.DEFAULT_PARAMS — same entry/exit/"
        "trail/risk/pyramid values, unchanged per leg) against the reference "
        "market and each cross-market leg. All windows are the development "
        "period (in-sample + validation) only; the true holdout is never "
        "touched here.\n"
    )

    legs = [DEFAULT_CROSS_MARKET.reference] + DEFAULT_CROSS_MARKET.legs
    rows = []
    for leg in legs:
        print(f"--- {leg.label} ---")
        row = run_leg(leg, args.initial_capital, args.start_date)
        rows.append(row)
        if "error" in row:
            print(f"  SKIPPED: {row['error']}")
        else:
            thin = " (THIN — under {} trades)".format(THIN_TRADE_COUNT) if row["num_trades"] < THIN_TRADE_COUNT else ""
            print(
                f"  return={row['total_return_pct']:.1f}%  cagr={row['cagr_pct']:.1f}%  "
                f"sharpe={row['sharpe_ratio']:.2f}  max_dd={row['max_drawdown_pct']:.1f}%  "
                f"trades={row['num_trades']}{thin}  win_rate={row['win_rate']:.1f}%"
            )

    results_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "cross_market_validation.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY — read alongside num_trades, not as a standalone verdict")
    print("=" * 80)
    display_cols = [c for c in [
        "label", "bars", "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "num_trades", "win_rate", "error",
    ] if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))
    print(
        "\nIf the reference (BTC/USDT 4h, Binance) is the only leg with a real "
        "Sharpe and everything else is flat, thin-traded, or negative, that's "
        "evidence this edge is curve-fit to one asset's specific history rather "
        "than a structural breakout effect. If most legs land in the same "
        "broad range, that's evidence for a structural edge — this script "
        "doesn't decide that for you; it just puts the numbers next to "
        "each other honestly."
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

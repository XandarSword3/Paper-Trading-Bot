"""
Final holdout validation — the ONE script in this repo allowed to touch the
frozen holdout window (config.DEFAULT_SPLIT.holdout_start / holdout_end,
currently 2025-01-01 -> latest). See data_splits.py for the split definition
and its acceptance check.

DO NOT RUN THIS until Phases 2-4 of VALIDATION_REMEDIATION_PLAN.md are done:
  - Phase 2: real walk-forward OOS testing (walk_forward_test.py doesn't
    exist yet)
  - Phase 3: an honestly-labeled Monte Carlo (sequence-risk only) plus a
    deflated/probabilistic Sharpe on the walk-forward OOS returns
  - Phase 4: cross-market validation (does the edge hold on ETH/other
    sources, or is it BTC-2017-2024-shaped curve-fitting?)

Running this before those phases just produces a second in-sample number
with extra steps. The holdout is a one-shot resource — you only get to
spend it once, on parameters that are already frozen. If you re-run this
after changing anything about the strategy, you no longer have a holdout;
you have a new in-sample period that happens to be dated 2025.

Usage (only once the above are actually done):
    python final_holdout_validation.py --i-understand-this-burns-the-holdout
"""
import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_PARAMS
from data_fetcher import download_btc_data
from data_splits import get_holdout, describe_split
from strategy import TurtleDonchianStrategy


def run(initial_capital: float = 100_000.0) -> dict:
    print(describe_split())

    df = download_btc_data(timeframe="4h")
    holdout_df = get_holdout(df)

    if len(holdout_df) == 0:
        raise RuntimeError(
            "Holdout slice is empty — check config.DEFAULT_SPLIT and confirm "
            "the downloaded dataset actually reaches into the holdout window."
        )

    print(f"\nRunning FINAL holdout validation on {len(holdout_df)} candles "
          f"({holdout_df.index[0]} -> {holdout_df.index[-1]})...")

    strategy = TurtleDonchianStrategy(DEFAULT_PARAMS)
    strategy.run_backtest(holdout_df, initial_capital=initial_capital, verbose=False)

    trade_stats = strategy.get_trade_stats()
    equity_stats = strategy.get_equity_stats(initial_capital)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_start": str(holdout_df.index[0]),
        "holdout_end": str(holdout_df.index[-1]),
        "n_candles": len(holdout_df),
        "params": asdict(DEFAULT_PARAMS),
        "trade_stats": trade_stats,
        "equity_stats": equity_stats,
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (
        f"holdout_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nWritten to {out_path} — per Phase 8 (reporting discipline), this file "
          f"is the receipt for whatever number you quote. Don't re-run this looking "
          f"for a better one.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--i-understand-this-burns-the-holdout",
        action="store_true",
        help="Required. This data is spent once — confirm you actually mean it.",
    )
    args = parser.parse_args()

    if not args.i_understand_this_burns_the_holdout:
        print(
            "Refusing to run without --i-understand-this-burns-the-holdout.\n"
            "Read the module docstring first: this should be the LAST thing "
            "you run in a strategy's validation, not a routine check, and "
            "not before Phases 2-4 of the remediation plan are done."
        )
        sys.exit(1)

    run()

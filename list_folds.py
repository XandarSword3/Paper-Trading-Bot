"""
Prints the fold indices for a strategy as a JSON array, e.g. [0,1,2,...,8].

Used by the GitHub Actions workflow's setup job to build a dynamic matrix,
so the number of folds is computed from the actual data/config rather than
hardcoded in the workflow file.

Usage:
    python list_folds.py --strategy v1
"""
import argparse
import json

from config import DEFAULT_SPLIT, DEFAULT_WALK_FORWARD
from data_fetcher import download_btc_data
from data_splits import get_development
from strategy_registry import get_strategy_config
from walk_forward import generate_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    args = ap.parse_args()

    cfg = get_strategy_config(args.strategy)
    df = get_development(download_btc_data(timeframe=cfg["timeframe"]))
    folds = generate_folds(df, DEFAULT_WALK_FORWARD, DEFAULT_SPLIT)

    print(json.dumps([f.index for f in folds]))


if __name__ == "__main__":
    main()

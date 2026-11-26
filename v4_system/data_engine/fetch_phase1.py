"""Phase 1 data fetch orchestrator.

Downloads the core historical datasets and persists them as parquet with
manifest provenance + gap reports. Run:

    python -m v4_system.data_engine.fetch_phase1            # full plan
    python -m v4_system.data_engine.fetch_phase1 --probe     # availability only
    python -m v4_system.data_engine.fetch_phase1 --quick     # 1h only, 1 symbol

Depth policy (bounds Phase 1 download size while preserving ≥3y of
intraday history and full history on higher TFs):
  - 1m/5m/15m: from INTRADAY_START (default 2023-01-01)
  - 1h/4h/1d:  from each symbol's perp listing date
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from v4_system.config import REPORTS_DIR
from v4_system.data_engine import store
from v4_system.data_engine.availability_probe import run_probe
from v4_system.data_engine.binance_client import BinanceClient
from v4_system.data_engine.contract_specs import fetch_exchange_specs
from v4_system.data_engine.funding import fetch_funding_history
from v4_system.data_engine.klines import (fetch_index_price_klines,
                                           fetch_mark_price_klines,
                                           fetch_perp_klines,
                                           fetch_spot_klines)
from v4_system.data_engine.open_interest import fetch_open_interest_hist

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PERP_LISTING = {"BTCUSDT": "2019-09-01", "ETHUSDT": "2019-11-01",
                "SOLUSDT": "2020-09-01"}
SPOT_LISTING = {"BTCUSDT": "2017-08-01", "ETHUSDT": "2017-08-01",
                "SOLUSDT": "2018-01-01"}
INTRADAY_TFS = ["1m", "5m", "15m"]
HIGHER_TFS = ["1h", "4h", "1d"]
ALL_TFS = INTRADAY_TFS + HIGHER_TFS
INTRADAY_START = "2023-01-01"


def _gap_summary(df: pd.DataFrame, tf: str) -> dict:
    gaps = store.detect_gaps(df, tf)
    return {"n_gaps": len(gaps),
            "missing_bars_total": sum(g[2] for g in gaps),
            "largest": max((g[2] for g in gaps), default=0)}


def _save_klines(df: pd.DataFrame, name: str, source: str, symbol: str,
                 tf: str, gap_report: dict) -> None:
    path = store.save(df, name, source=source, symbol=symbol, timeframe=tf,
                      extra={"gaps": gap_report})
    print(f"  saved {name}: {len(df)} rows {df.index[0]} → {df.index[-1]} "
          f"-> {os.path.basename(path)}")


def run_full(intraday_start: str = INTRADAY_START,
             symbols: list[str] | None = None,
             timeframes: list[str] | None = None) -> None:
    symbols = symbols or SYMBOLS
    timeframes = timeframes or ALL_TFS
    client = BinanceClient.futures()
    spot = BinanceClient.spot()
    t0 = time.time()

    # Contract specs (needed later by risk engine; cheap single call)
    try:
        specs = fetch_exchange_specs(symbols, client)
        store.save(specs.reset_index().rename(columns={"index": "symbol"}),
                   "contract_specs", source="fapi /fapi/v1/exchangeInfo",
                   symbol="MULTI", timeframe=None)
        print(f"contract specs: {len(specs)} symbols")
    except Exception as e:
        print(f"contract specs FAILED: {e}")

    for sym in symbols:
        print(f"\n=== {sym} ===")
        for tf in timeframes:
            start = intraday_start if tf in INTRADAY_TFS else PERP_LISTING[sym]
            name = f"{sym}_perp_{tf}"
            if os.path.exists(store.dataset_path(name)):
                print(f"  {name}: exists, skipping (delete to refetch)")
                continue
            try:
                df = fetch_perp_klines(sym, tf, start=start, client=client)
                _save_klines(df, name, "fapi /fapi/v1/klines", sym, tf,
                             _gap_summary(df, tf))
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}")

        # Spot klines — higher TFs only (spot 1m depth is huge; perp is primary)
        for tf in HIGHER_TFS:
            name = f"{sym}_spot_{tf}"
            if os.path.exists(store.dataset_path(name)):
                continue
            try:
                df = fetch_spot_klines(sym, tf, start=SPOT_LISTING[sym], client=spot)
                _save_klines(df, name, "api /api/v3/klines", sym, tf,
                             _gap_summary(df, tf))
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}")

        # Funding — full history
        name = f"{sym}_funding"
        if not os.path.exists(store.dataset_path(name)):
            try:
                df = fetch_funding_history(sym, start=PERP_LISTING[sym], client=client)
                _save_klines(df, name, "fapi /fapi/v1/fundingRate", sym, "funding",
                             {})
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}")

        # Mark & index price — 1h only for Phase 1 (basis research)
        for kind, fn in (("mark", fetch_mark_price_klines),
                         ("index", fetch_index_price_klines)):
            name = f"{sym}_{kind}_1h"
            if os.path.exists(store.dataset_path(name)):
                continue
            try:
                df = fn(sym, "1h", start=PERP_LISTING[sym], client=client)
                _save_klines(df, name, f"fapi /fapi/v1/{kind}PriceKlines", sym, "1h",
                             _gap_summary(df, "1h"))
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}")

        # OI — only ~30d reachable; snapshot what exists
        name = f"{sym}_oi_1h"
        if not os.path.exists(store.dataset_path(name)):
            try:
                df = fetch_open_interest_hist(sym, period="1h", days=29, client=client)
                _save_klines(df, name, "fapi /futures/data/openInterestHist", sym,
                             "1h", {"note": "Binance caps OI history at ~30 days"})
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}")

    # Gap report
    report = {"elapsed_s": round(time.time() - t0, 1), "datasets": {}}
    for fn in sorted(os.listdir(store.DATA_DIR)) if os.path.isdir(store.DATA_DIR) else []:
        if not fn.endswith(".parquet"):
            continue
        name = fn[:-8]
        try:
            df = store.load(name)
            tf = name.rsplit("_", 1)[-1] if name.rsplit("_", 1)[-1] in store.TF_DELTA else None
            report["datasets"][name] = {
                "rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
                "gaps": _gap_summary(df, tf) if tf else {},
            }
        except Exception as e:
            report["datasets"][name] = {"error": str(e)}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "phase1_fetch_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone in {report['elapsed_s']}s. Report: reports/phase1_fetch_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run availability probe only")
    ap.add_argument("--quick", action="store_true",
                    help="BTCUSDT 1h perp only (connectivity smoke test)")
    ap.add_argument("--intraday-start", default=INTRADAY_START)
    args = ap.parse_args()

    if args.probe:
        rep = run_probe()
        for c in rep["checks"]:
            print(("OK " if c["ok"] else "FAIL"), c["name"],
                  "" if c["ok"] else str(c.get("error"))[:100])
        return
    if args.quick:
        run_full(symbols=["BTCUSDT"], timeframes=["1h"])
        return
    run_full(intraday_start=args.intraday_start)


if __name__ == "__main__":
    main()

"""Phase 1 data fetch orchestrator.

Downloads the core historical datasets and persists them as parquet with
manifest provenance + gap reports. Run:

    python -m v4_system.data_engine.fetch_phase1            # full plan
    python -m v4_system.data_engine.fetch_phase1 --probe     # availability only
    python -m v4_system.data_engine.fetch_phase1 --quick     # 1h only, 1 symbol
    python -m v4_system.data_engine.fetch_phase1 --verify    # coverage report only

Coverage (item 5): every eligible spot AND perpetual instrument in the
project listing (SYMBOLS), every project timeframe (config.TIMEFRAMES),
plus funding history, mark/index price 1h, and open-interest for every
OI period supported by the endpoint that matches a project timeframe.

Depth policy (bounds download size while preserving >=3y of intraday
history and full history on higher TFs):
  - 1m/5m/15m: from INTRADAY_START (default 2023-01-01)
  - 1h/4h/1d:  from each symbol's listing date

Resume: existing datasets are NOT re-downloaded; their tails are
refreshed incrementally (data_engine.refresh) so interrupted or stale
downloads converge without duplicating history. All writes go through
store.save (atomic temp-file + validation + os.replace) and unfinished
candles are excluded before saving (data_engine.pit). Historical
availability is unknown and left unset (NaT) — never fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from v4_system.config import REPORTS_DIR, TIMEFRAMES
from v4_system.data_engine import pit, refresh as refresh_mod, store
from v4_system.data_engine.availability_probe import run_probe
from v4_system.data_engine.binance_client import BinanceClient
from v4_system.data_engine.contract_specs import fetch_exchange_specs
from v4_system.data_engine.funding import fetch_funding_history
from v4_system.data_engine.klines import (fetch_index_price_klines,
                                           fetch_mark_price_klines,
                                           fetch_perp_klines,
                                           fetch_spot_klines)
from v4_system.data_engine.open_interest import (OI_PERIODS,
                                                  fetch_open_interest_hist)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PERP_LISTING = {"BTCUSDT": "2019-09-01", "ETHUSDT": "2019-11-01",
                "SOLUSDT": "2020-09-01"}
SPOT_LISTING = {"BTCUSDT": "2017-08-01", "ETHUSDT": "2017-08-01",
                "SOLUSDT": "2018-01-01"}
INTRADAY_TFS = ["1m", "5m", "15m"]
HIGHER_TFS = ["1h", "4h", "1d"]
ALL_TFS = INTRADAY_TFS + HIGHER_TFS
INTRADAY_START = "2023-01-01"

# Open-interest periods to fetch: project timeframes that the OI endpoint
# supports (derived, not invented).
OI_TFS = [tf for tf in TIMEFRAMES if tf in OI_PERIODS]

KLINE_FETCHERS = {
    "perp": fetch_perp_klines,
    "spot": fetch_spot_klines,
    "mark": fetch_mark_price_klines,
    "index": fetch_index_price_klines,
}


def dataset_plan(symbols: list[str] | None = None,
                 timeframes: list[str] | None = None) -> list[dict]:
    """Enumerate every dataset the project requires.

    Derived entirely from the existing listing (SYMBOLS), listing dates,
    and project configuration (TIMEFRAMES / depth policy). Each entry:
    name, kind, symbol, timeframe/period, market, start date.
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or ALL_TFS
    plan = []
    for sym in symbols:
        for tf in timeframes:
            intraday = tf in INTRADAY_TFS
            plan.append({
                "name": f"{sym}_perp_{tf}", "kind": "perp_kline",
                "symbol": sym, "timeframe": tf, "market": "perp",
                "start": INTRADAY_START if intraday else PERP_LISTING[sym],
            })
            plan.append({
                "name": f"{sym}_spot_{tf}", "kind": "spot_kline",
                "symbol": sym, "timeframe": tf, "market": "spot",
                "start": INTRADAY_START if intraday else SPOT_LISTING[sym],
            })
        plan.append({"name": f"{sym}_funding", "kind": "funding",
                     "symbol": sym, "timeframe": "funding",
                     "market": "perp", "start": PERP_LISTING[sym]})
        for kind in ("mark", "index"):
            plan.append({"name": f"{sym}_{kind}_1h", "kind": f"{kind}_kline",
                         "symbol": sym, "timeframe": "1h",
                         "market": "perp", "start": PERP_LISTING[sym]})
        for tf in OI_TFS:
            plan.append({"name": f"{sym}_oi_{tf}", "kind": "oi",
                         "symbol": sym, "timeframe": tf, "market": "perp",
                         "start": None})
    return plan


def _gap_summary(df: pd.DataFrame, tf: str) -> dict:
    gaps = store.detect_gaps(df, tf)
    return {"n_gaps": len(gaps),
            "missing_bars_total": sum(g[2] for g in gaps),
            "largest": max((g[2] for g in gaps), default=0)}


def _drop_unfinished(df: pd.DataFrame, tf: str | None) -> pd.DataFrame:
    """Exclude candles that have not closed yet (item 3 integrity)."""
    if tf in pit.CANDLE_DELTA:
        return pit.drop_unfinished_candles(df, tf, pd.Timestamp.now(tz="UTC"))
    return df


def _save(df: pd.DataFrame, name: str, source: str, symbol: str,
          tf: str | None, gap_report: dict) -> None:
    path = store.save(df, name, source=source, symbol=symbol, timeframe=tf,
                      extra={"gaps": gap_report})
    print(f"  saved {name}: {len(df)} rows {df.index[0]} -> {df.index[-1]} "
          f"-> {os.path.basename(path)}")


def _fetch_kline(entry: dict, client, spot_client) -> pd.DataFrame:
    kind = entry["kind"]
    fetcher = KLINE_FETCHERS[kind.split("_")[0]]
    cli = spot_client if kind == "spot_kline" else client
    df = fetcher(entry["symbol"], entry["timeframe"], start=entry["start"],
                 client=cli)
    return _drop_unfinished(df, entry["timeframe"])


def _refresh_existing(entry: dict, client, spot_client) -> str:
    """Tail-refresh an existing dataset; returns a status string."""
    name = entry["name"]
    kind = entry["kind"]
    try:
        if kind in ("perp_kline", "spot_kline", "mark_kline", "index_kline"):
            refresh_mod.refresh_klines_dataset(
                name, entry["symbol"], entry["timeframe"],
                kind=kind.split("_")[0] if kind != "index_kline" else "index",
                client=spot_client if kind == "spot_kline" else client)
        elif kind == "funding":
            refresh_mod.refresh_funding_dataset(name, entry["symbol"],
                                                client=client)
        elif kind == "oi":
            refresh_mod.refresh_oi_dataset(name, entry["symbol"],
                                          period=entry["timeframe"],
                                          client=client)
        else:
            return "skipped_unknown_kind"
        return "refreshed"
    except Exception as e:  # noqa: BLE001
        return f"refresh_failed: {type(e).__name__}: {e}"


def _fetch_new(entry: dict, client, spot_client) -> None:
    kind = entry["kind"]
    name, sym = entry["name"], entry["symbol"]
    if kind in ("perp_kline", "spot_kline", "mark_kline", "index_kline"):
        df = _fetch_kline(entry, client, spot_client)
        source = {"perp_kline": "fapi /fapi/v1/klines",
                  "spot_kline": "api /api/v3/klines",
                  "mark_kline": "fapi /fapi/v1/markPriceKlines",
                  "index_kline": "fapi /fapi/v1/indexPriceKlines"}[kind]
        _save(df, name, source, sym, entry["timeframe"],
              _gap_summary(df, entry["timeframe"]))
    elif kind == "funding":
        df = fetch_funding_history(sym, start=entry["start"], client=client)
        _save(df, name, "fapi /fapi/v1/fundingRate", sym, "funding", {})
    elif kind == "oi":
        df = fetch_open_interest_hist(sym, period=entry["timeframe"],
                                      days=29, client=client)
        _save(df, name, "fapi /futures/data/openInterestHist", sym,
              entry["timeframe"],
              {"note": "Binance caps OI history at ~30 days"})
    else:
        raise ValueError(f"unknown dataset kind: {kind}")


def run_full(intraday_start: str = INTRADAY_START,
             symbols: list[str] | None = None,
             timeframes: list[str] | None = None,
             client: BinanceClient | None = None,
             spot_client: BinanceClient | None = None) -> dict:
    symbols = symbols or SYMBOLS
    timeframes = timeframes or ALL_TFS
    client = client or BinanceClient.futures()
    spot_client = spot_client or BinanceClient.spot()
    t0 = time.time()
    statuses: list[dict] = []

    # Contract specs (needed later by risk engine; cheap single call)
    try:
        specs = fetch_exchange_specs(symbols, client)
        store.save(specs.reset_index().rename(columns={"index": "symbol"}),
                   "contract_specs", source="fapi /fapi/v1/exchangeInfo",
                   symbol="MULTI", timeframe=None)
        print(f"contract specs: {len(specs)} symbols")
        statuses.append({"name": "contract_specs", "status": "downloaded"})
    except Exception as e:
        print(f"contract specs FAILED: {e}")
        statuses.append({"name": "contract_specs", "status": "failed",
                         "error": f"{type(e).__name__}: {e}"})

    plan = dataset_plan(symbols, timeframes)
    for entry in plan:
        name = entry["name"]
        if os.path.exists(store.dataset_path(name)):
            status = _refresh_existing(entry, client, spot_client)
            print(f"  {name}: exists -> {status}")
            statuses.append({"name": name, "status": status})
            continue
        try:
            _fetch_new(entry, client, spot_client)
            statuses.append({"name": name, "status": "downloaded"})
        except Exception as e:  # noqa: BLE001 — record, never silently skip
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
            statuses.append({"name": name, "status": "failed",
                             "error": f"{type(e).__name__}: {e}"})

    report = verify_coverage(plan, statuses=statuses,
                            elapsed_s=round(time.time() - t0, 1))
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "phase1_fetch_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone in {report['elapsed_s']}s. "
          f"Coverage: {report['summary']['validated']}/{report['summary']['expected']} "
          f"validated, {report['summary']['failed']} failed, "
          f"{report['summary']['incomplete']} incomplete.")
    print(f"Report: reports/phase1_fetch_report.json")
    return report


def verify_dataset(name: str, timeframe: str | None = None) -> dict:
    """Validate one stored dataset. PASS requires real data checks, not
    just file existence."""
    result = {"name": name, "status": "FAIL", "checks": {}}
    try:
        df = store.load(name)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    result["checks"]["rows"] = int(len(df))
    if len(df) == 0:
        result["error"] = "empty dataset"
        return result
    result["checks"]["start"] = str(df.index[0])
    result["checks"]["end"] = str(df.index[-1])
    dupes = int(df.index.duplicated().sum())
    result["checks"]["duplicates"] = dupes
    tf = timeframe or _timeframe_from_name(name)
    # Candle rules (gaps, unfinished candles) apply only to actual kline
    # datasets — OI/funding are period/event series, not candles.
    is_candle = (tf in store.TF_DELTA and "_oi_" not in name.lower()
                 and "_funding" not in name.lower())
    if is_candle:
        gaps = store.detect_gaps(df, tf)
        result["checks"]["n_gaps"] = len(gaps)
        result["checks"]["missing_bars_total"] = sum(g[2] for g in gaps)
        if tf in pit.CANDLE_DELTA:
            unfinished = int((df.index + pit.CANDLE_DELTA[tf]
                              > pd.Timestamp.now(tz="UTC")).sum())
            result["checks"]["unfinished_candles"] = unfinished
    ok = dupes == 0 and result["checks"].get("unfinished_candles", 0) == 0
    result["status"] = "PASS" if ok else "INCOMPLETE"
    return result


def _timeframe_from_name(name: str) -> str | None:
    tail = name.rsplit("_", 1)[-1]
    return tail if tail in store.TF_DELTA else None


def verify_coverage(plan: list[dict] | None = None,
                    statuses: list[dict] | None = None,
                    elapsed_s: float | None = None) -> dict:
    """Validate every expected dataset and build the coverage summary."""
    plan = plan if plan is not None else dataset_plan()
    status_by_name = {s["name"]: s for s in (statuses or [])}
    datasets = {}
    for entry in plan:
        name = entry["name"]
        v = verify_dataset(name, entry.get("timeframe"))
        v["kind"] = entry["kind"]
        v["market"] = entry["market"]
        v["symbol"] = entry["symbol"]
        dl = status_by_name.get(name, {})
        v["download_status"] = dl.get("status", "not_attempted")
        if dl.get("error"):
            v["download_error"] = dl["error"]
        datasets[name] = v

    summary = {
        "expected": len(plan),
        "downloaded": sum(1 for d in datasets.values()
                          if d["download_status"] in ("downloaded", "refreshed")),
        "validated": sum(1 for d in datasets.values() if d["status"] == "PASS"),
        "failed": sum(1 for d in datasets.values()
                      if d["download_status"] == "failed"),
        "incomplete": sum(1 for d in datasets.values()
                          if d["status"] == "INCOMPLETE"),
        "missing": sum(1 for d in datasets.values()
                       if d["status"] == "FAIL"),
        "unsupported": 0,
    }
    return {"verified_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s, "summary": summary, "datasets": datasets}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run availability probe only")
    ap.add_argument("--quick", action="store_true",
                    help="BTCUSDT 1h perp only (connectivity smoke test)")
    ap.add_argument("--verify", action="store_true",
                    help="validate coverage of existing datasets only")
    ap.add_argument("--intraday-start", default=INTRADAY_START)
    args = ap.parse_args()

    if args.probe:
        rep = run_probe()
        for c in rep["checks"]:
            print(("OK " if c["ok"] else "FAIL"), c["name"],
                  "" if c["ok"] else str(c.get("error"))[:100])
        return
    if args.verify:
        report = verify_coverage()
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(os.path.join(REPORTS_DIR, "phase1_fetch_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        for name, d in sorted(report["datasets"].items()):
            print(f"[{d['status']:10s}] {name}: {d['checks'].get('rows', 0)} rows "
                  f"{d['checks'].get('start', '')[:10]} -> "
                  f"{d['checks'].get('end', '')[:10]} "
                  f"gaps={d['checks'].get('n_gaps', '-')} "
                  f"dupes={d['checks'].get('duplicates', '-')} "
                  f"unfinished={d['checks'].get('unfinished_candles', '-')}")
        print(f"\nSummary: {report['summary']}")
        return
    if args.quick:
        run_full(symbols=["BTCUSDT"], timeframes=["1h"])
        return
    run_full(intraday_start=args.intraday_start)


if __name__ == "__main__":
    main()

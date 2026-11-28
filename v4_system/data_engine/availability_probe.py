"""Phase 1 availability probe.

Touches every required data source with minimal requests and records:
reachability, earliest available timestamp, approximate depth, and known
limitations. Output: v4_system/reports/data_availability.json + printed
summary. This is the evidence base for DATA_AVAILABILITY.md.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from v4_system.config import REPORTS_DIR
from v4_system.data_engine.agg_trades import fetch_agg_trades
from v4_system.data_engine.binance_client import BinanceClient
from v4_system.data_engine.contract_specs import fetch_exchange_specs
from v4_system.data_engine.funding import fetch_funding_history
from v4_system.data_engine.klines import (fetch_index_price_klines,
                                           fetch_mark_price_klines)
from v4_system.data_engine.open_interest import fetch_open_interest_hist

PROBE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _probe(name: str, fn) -> dict:
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        if isinstance(result, pd.DataFrame):
            return {"name": name, "ok": True, "rows": len(result),
                    "start": str(result.index[0]), "end": str(result.index[-1]),
                    "elapsed_s": round(elapsed, 2)}
        return {"name": name, "ok": True, "result": result,
                "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        return {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 2)}


def run_probe() -> dict:
    client = BinanceClient.futures()
    spot = BinanceClient.spot()
    report = {"probed_at": datetime.now(timezone.utc).isoformat(), "checks": []}

    # 1. Connectivity / exchangeInfo
    report["checks"].append(_probe("exchangeInfo+contractSpecs",
                                  lambda: fetch_exchange_specs(PROBE_SYMBOLS, client).shape))

    # 2. Klines — earliest data via a single early-window request
    def _first_open(client_, path, sym, start_str):
        raw = client_.get(path, params={
            "symbol": sym, "interval": "1h",
            "startTime": int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000),
            "limit": 5})
        if not raw:
            raise ValueError("empty response")
        ts = pd.to_datetime(raw[0][0], unit="ms", utc=True)
        return pd.DataFrame({"first_open": [ts]}, index=[ts])

    PERP_LISTING = {"BTCUSDT": "2019-09-01", "ETHUSDT": "2019-11-01",
                    "SOLUSDT": "2020-09-01"}
    SPOT_LISTING = {"BTCUSDT": "2017-08-01", "ETHUSDT": "2017-08-01",
                    "SOLUSDT": "2018-01-01"}

    for sym in PROBE_SYMBOLS:
        report["checks"].append(_probe(
            f"perp_klines_1h_earliest[{sym}]",
            lambda s=sym: _first_open(client, "/fapi/v1/klines", s, PERP_LISTING[s])))
        report["checks"].append(_probe(
            f"spot_klines_1h_earliest[{sym}]",
            lambda s=sym: _first_open(spot, "/api/v3/klines", s, SPOT_LISTING[s])))

    # 3. Funding history depth
    for sym in PROBE_SYMBOLS:
        def _funding(s=sym):
            df = fetch_funding_history(s, start="2019-09-01")
            return df.head(1000)  # bound probe cost; full fetch later
        report["checks"].append(_probe(f"funding_history[{sym}]", _funding))

    # 4. OI history reachable window (request 35 days back — expect ~30d cap)
    for sym in PROBE_SYMBOLS:
        def _oi(s=sym):
            return fetch_open_interest_hist(s, period="1h", days=29)
        report["checks"].append(_probe(f"openInterestHist_1h[{sym}]", _oi))

    # 5. Mark & index price klines (recent window only)
    recent_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    for sym in PROBE_SYMBOLS:
        report["checks"].append(_probe(f"markPriceKlines_1h[{sym}]",
                                      lambda s=sym: fetch_mark_price_klines(s, "1h", start=recent_start)))
        report["checks"].append(_probe(f"indexPriceKlines_1h[{sym}]",
                                      lambda s=sym: fetch_index_price_klines(s, "1h", start=recent_start)))

    # 6. aggTrades bounded window
    report["checks"].append(_probe("aggTrades_1h_window[BTCUSDT]",
                                  lambda: fetch_agg_trades(
                                      "BTCUSDT",
                                      start=datetime.now(timezone.utc) - timedelta(hours=1),
                                      end=datetime.now(timezone.utc),
                                      max_rows=5_000)))

    # 7. Liquidation / order-book history availability (documented: no public REST history)
    report["checks"].append({
        "name": "liquidation_history",
        "ok": False,
        "error": "NO_PUBLIC_REST_ENDPOINT — Binance exposes liquidations only via "
                 "!forceOrder@arr websocket (live only). Historical liquidation "
                 "data requires third-party vendors or self-recording.",
        "elapsed_s": 0,
    })
    report["checks"].append({
        "name": "orderbook_history",
        "ok": False,
        "error": "NO_PUBLIC_REST_HISTORY — depth snapshots are websocket-only "
                 "(<symbol>@depth). Historical book reconstruction not possible "
                 "from Binance public API.",
        "elapsed_s": 0,
    })

    out = f"{REPORTS_DIR}/data_availability.json"
    import os
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    rep = run_probe()
    for c in rep["checks"]:
        status = "OK " if c["ok"] else "FAIL"
        detail = f"rows={c.get('rows')} {c.get('start','')}→{c.get('end','')}" if c["ok"] else c.get("error", "")[:110]
        print(f"[{status}] {c['name']}: {detail}")
    print(f"\nReport written to reports/data_availability.json")

"""Contract specifications from exchangeInfo.

Captures the fields the risk/execution engines need: tick size, step size,
min qty, min notional, and funding interval. Point-in-time caveat: Binance
adjusts these rarely but does adjust them — for strict PIT backtests we
should snapshot exchangeInfo periodically; for now we record fetch time.
"""
from __future__ import annotations

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient


def fetch_exchange_specs(symbols: list[str] | None = None,
                         client: BinanceClient | None = None) -> pd.DataFrame:
    client = client or BinanceClient.futures()
    info = client.get("/fapi/v1/exchangeInfo")

    rows = []
    for s in info["symbols"]:
        if symbols and s["symbol"] not in symbols:
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        filters = {f["filterType"]: f for f in s["filters"]}
        rows.append({
            "symbol": s["symbol"],
            "status": s["status"],
            "baseAsset": s["baseAsset"],
            "quoteAsset": s["quoteAsset"],
            "pricePrecision": s["pricePrecision"],
            "quantityPrecision": s["quantityPrecision"],
            "tick_size": filters.get("PRICE_FILTER", {}).get("tickSize"),
            "step_size": filters.get("LOT_SIZE", {}).get("stepSize"),
            "min_qty": filters.get("LOT_SIZE", {}).get("minQty"),
            "max_qty": filters.get("LOT_SIZE", {}).get("maxQty"),
            "min_notional": filters.get("MIN_NOTIONAL", {}).get("notional"),
            "maint_margin_percent": filters.get("MAINTENANCE_MARGIN", {}).get("maintMarginPercent"),
            "required_margin_percent": filters.get("MAINTENANCE_MARGIN", {}).get("requiredMarginPercent"),
            "funding_interval_hours": s.get("fundingIntervalHours"),
        })
    return pd.DataFrame(rows).set_index("symbol")

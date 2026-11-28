"""Contract specifications from exchangeInfo.

Captures the fields the risk/execution engines need: tick size, step size,
min qty, min notional, and margin percentages. Point-in-time caveat: Binance
adjusts these rarely but does adjust them — for strict PIT backtests we
should snapshot exchangeInfo periodically; for now we record fetch time.

Field sources (verified against the live API):
- Perp symbol-level margin fields (maintMarginPercent,
  requiredMarginPercent, marginAsset, liquidationFee) live at the TOP
  LEVEL of each symbol object in /fapi/v1/exchangeInfo — there is NO
  "MAINTENANCE_MARGIN" filter, which is why earlier versions stored
  nulls. Bracketed margin tiers are NOT public (see margin_tiers.py).
- Spot specs come from /api/v3/exchangeInfo (different filter set:
  NOTIONAL instead of MIN_NOTIONAL, no funding/margin fields).
- fundingIntervalHours is NOT in exchangeInfo (null there); the
  authoritative source is /fapi/v1/fundingInfo (funding_intervals.py).
"""
from __future__ import annotations

import pandas as pd

from v4_system.data_engine.binance_client import BinanceClient


def _f(filters: dict, key: str, field: str):
    return filters.get(key, {}).get(field)


def fetch_exchange_specs(symbols: list[str] | None = None,
                         client: BinanceClient | None = None) -> pd.DataFrame:
    """Perpetual contract specs from /fapi/v1/exchangeInfo."""
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
            "market": "perp",
            "status": s["status"],
            "baseAsset": s["baseAsset"],
            "quoteAsset": s["quoteAsset"],
            "pricePrecision": s["pricePrecision"],
            "quantityPrecision": s["quantityPrecision"],
            "baseAssetPrecision": s.get("baseAssetPrecision"),
            "quotePrecision": s.get("quotePrecision"),
            "tick_size": _f(filters, "PRICE_FILTER", "tickSize"),
            "step_size": _f(filters, "LOT_SIZE", "stepSize"),
            "min_qty": _f(filters, "LOT_SIZE", "minQty"),
            "max_qty": _f(filters, "LOT_SIZE", "maxQty"),
            "min_notional": _f(filters, "MIN_NOTIONAL", "notional"),
            # Symbol-level margin fields (top level of exchangeInfo symbol).
            "maint_margin_percent": s.get("maintMarginPercent"),
            "required_margin_percent": s.get("requiredMarginPercent"),
            "margin_asset": s.get("marginAsset"),
            "liquidation_fee": s.get("liquidationFee"),
            # Lifecycle / classification.
            "onboard_date": s.get("onboardDate"),
            "delivery_date": s.get("deliveryDate"),
            "underlying_type": s.get("underlyingType"),
            "underlying_sub_type": s.get("underlyingSubType"),
            # fundingIntervalHours is null in exchangeInfo; use
            # funding_intervals.py (/fapi/v1/fundingInfo) instead.
            "funding_interval_hours": None,
        })
    return pd.DataFrame(rows).set_index("symbol")


def fetch_spot_specs(symbols: list[str] | None = None,
                     client: BinanceClient | None = None) -> pd.DataFrame:
    """Spot contract specs from /api/v3/exchangeInfo.

    Spot has no margin/funding fields; min notional comes from the
    NOTIONAL filter (quoteAsset-denominated) or MIN_NOTIONAL-style
    filters depending on the symbol.
    """
    client = client or BinanceClient.spot()
    info = client.get("/api/v3/exchangeInfo")

    rows = []
    for s in info["symbols"]:
        if symbols and s["symbol"] not in symbols:
            continue
        if s.get("status") != "TRADING":
            continue
        filters = {f["filterType"]: f for f in s["filters"]}
        rows.append({
            "symbol": s["symbol"],
            "market": "spot",
            "status": s["status"],
            "baseAsset": s["baseAsset"],
            "quoteAsset": s["quoteAsset"],
            "pricePrecision": s["pricePrecision"],
            "quantityPrecision": s["quantityPrecision"],
            "baseAssetPrecision": s.get("baseAssetPrecision"),
            "quotePrecision": s.get("quotePrecision"),
            "tick_size": _f(filters, "PRICE_FILTER", "tickSize"),
            "step_size": _f(filters, "LOT_SIZE", "stepSize"),
            "min_qty": _f(filters, "LOT_SIZE", "minQty"),
            "max_qty": _f(filters, "LOT_SIZE", "maxQty"),
            "min_notional": (_f(filters, "NOTIONAL", "minNotional")
                             or _f(filters, "MIN_NOTIONAL", "notional")),
            "maint_margin_percent": None,
            "required_margin_percent": None,
            "margin_asset": None,
            "liquidation_fee": None,
            "onboard_date": None,
            "delivery_date": None,
            "underlying_type": None,
            "underlying_sub_type": None,
            "funding_interval_hours": None,
        })
    return pd.DataFrame(rows).set_index("symbol")

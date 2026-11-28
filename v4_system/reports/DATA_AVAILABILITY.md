# V4 Data Availability Report — Phase 1

Probed: 2026-11-26 (UTC). Raw machine-readable output:
`reports/data_availability.json`. Fetch results:
`reports/phase1_fetch_report.json` + `data/MANIFEST.json`.

All endpoints are Binance **public** REST — no API key, no auth, no
withdrawal permissions. Base URLs: `https://fapi.binance.com` (USDⓈ-M
perpetuals) and `https://api.binance.com` (spot).

## What IS available

| Dataset | Endpoint | Depth confirmed | Notes |
|---|---|---|---|
| Perp OHLCV + quote vol, trades, taker-buy | `GET /fapi/v1/klines` | From listing: BTC 2019-09, ETH 2019-11, SOL 2020-09 | 1500 bars/request, all TFs 1m–1d |
| Spot OHLCV | `GET /api/v3/klines` | From listing: BTC/ETH 2017-08, SOL 2018-01 | 1000 bars/request |
| Funding rate history | `GET /fapi/v1/fundingRate` | Full history from listing (probe: first 1000 rows reach back to listing month) | 8h or 4h cadence per `fundingIntervalHours` in exchangeInfo |
| Mark-price klines | `GET /fapi/v1/markPriceKlines` | Full history (liquidation reference price) | Same pagination as klines |
| Index-price klines | `GET /fapi/v1/indexPriceKlines` | Full history | **Quirk: requires `pair` param, not `symbol`** |
| Open interest history | `GET /futures/data/openInterestHist` | **~30 days only** (696 hourly rows confirmed) | Hard Binance limit — see gaps |
| Aggregate trades | `GET /fapi/v1/aggTrades` | Full history, paginated by `fromId` | ~5000 rows ≈ 7 min of BTC trading; bulk download is heavy, deferred |
| Contract specs | `GET /fapi/v1/exchangeInfo` | Current snapshot | tick/step/min-qty/min-notional/maintenance-margin/funding interval |

## What is NOT available (documented gaps)

- **Historical liquidations** — no public REST endpoint. Binance exposes
  them only via the `!forceOrder@arr` websocket (live stream). Options:
  (a) self-record going forward via websocket recorder (planned for
  paper_trading phase), (b) third-party vendors (e.g. Coinglass,
  Laevitas). Liquidation-*cascade* strategy (F) is therefore limited to
  live/paper detection unless a vendor is added.
- **Historical order-book depth** — depth is websocket-only
  (`<symbol>@depth`). No historical book reconstruction from Binance
  public API. Order-book imbalance / spread features are live-only or
  must be proxied (e.g. trade-count/volume-based estimators on klines).
- **Long-history open interest** — capped at ~30 days. OI-change features
  are usable only on recent data unless self-recorded. Mitigation:
  begin OI recording now via a scheduled job so depth accumulates.
- **Long-history aggTrades at scale** — available but expensive
  (BTCUSDT does ~500k–1M aggTrades/day). Phase 1 stores bounded windows;
  bulk backfill is a separate job if microstructure backtests need it.

## Point-in-time integrity

- Klines are indexed by **open time** (bar start) — a bar is only known
  after close; consumers must shift by one bar or use `close_time`.
- Funding/OI/mark/index rows carry exchange timestamps; no revision
  risk (exchange data is append-only).
- `exchangeInfo` is a *current* snapshot — Binance rarely changes
  tick/step sizes but does; for strict PIT backtests we should snapshot
  it periodically (noted as a TODO, low risk for 1h+ strategies).

## Phase 1 fetch policy

- Perp klines: 1m/5m/15m from **2023-01-01** (≈3y intraday depth, bounds
  download size); 1h/4h/1d from each symbol's listing (full history).
- Spot klines: 1h/4h/1d full history (spot 1m not needed for perp
  strategies).
- Funding: full history. Mark/index: 1h full history. OI: latest ~30d.
- Every dataset: parquet + MANIFEST.json entry (source, rows, range,
  sha256) + automated gap detection (>1.5× bar interval).

## Pass/fail vs PREREG

- PASS criteria met for klines (spot+perp, all TFs, ≥2y) and funding.
- OI/mark/index depth limits are quantified above → PASS-with-documented-
  gaps per PREREG.
- Remaining risk: Binance reachability. It is currently reachable from
  this machine (all probes OK), but repo history notes HTTP 451 from
  Lebanon/GitHub runners — the existing `CryptoDataDownload` fallback in
  `research/data/data_fetcher.py` remains the contingency for OHLCV only
  (no funding/OI/mark/index equivalent).

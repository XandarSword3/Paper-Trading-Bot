# Item 8 — Final Historical Data Validation

Generated: 2026-11-28T14:25:42.186326+00:00
Branch: phase2-v4-adapter

- Expected datasets: **64** (60 market + 4 metadata)
- Freshly fetched/refreshed: **0**
- PASS: **1**  INCOMPLETE: **63**  FAIL: **0**  (structural failures: **0**)

Classification: FAIL = structural integrity failure (load error, empty, duplicates, unfinished candles). INCOMPLETE = structurally valid but coverage-limited (gaps, OI ~30d cap, unknown available_at, snapshot-only metadata). PASS = structural + coverage checks pass.

## Dataset-by-dataset

| Dataset | Symbol | TF/Metric | Rows | First | Last | Gaps | Dupes | Unfin | Struct | Status | Limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT_perp_1m | BTCUSDT | 1m | 2055544 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_1m | BTCUSDT | 1m | 2055464 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 80 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_perp_5m | BTCUSDT | 5m | 411108 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_5m | BTCUSDT | 5m | 411092 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 16 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_perp_15m | BTCUSDT | 15m | 137036 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_15m | BTCUSDT | 15m | 137031 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 5 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_perp_1h | BTCUSDT | 1h | 63306 | 2019-09-08 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_1h | BTCUSDT | 1h | 81239 | 2017-08-17 | 2026-11-28 | 28 | 0 | 0 | PASS | **INCOMPLETE** | 28 gap(s) / 128 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_perp_4h | BTCUSDT | 4h | 15826 | 2019-09-08 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_4h | BTCUSDT | 4h | 20325 | 2017-08-17 | 2026-11-28 | 8 | 0 | 0 | PASS | **INCOMPLETE** | 8 gap(s) / 16 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_perp_1d | BTCUSDT | 1d | 2638 | 2019-09-08 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_spot_1d | BTCUSDT | 1d | 3390 | 2017-08-17 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_funding | BTCUSDT | funding | 7909 | 2019-09-10 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_mark_1h | BTCUSDT | 1h | 60768 | 2019-12-23 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_index_1h | BTCUSDT | 1h | 60768 | 2019-12-23 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_oi_5m | BTCUSDT | 5m | 8381 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_oi_15m | BTCUSDT | 15m | 2794 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_oi_1h | BTCUSDT | 1h | 735 | 2026-10-28 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_oi_4h | BTCUSDT | 4h | 174 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| BTCUSDT_oi_1d | BTCUSDT | 1d | 29 | 2026-10-31 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_1m | ETHUSDT | 1m | 2055544 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_1m | ETHUSDT | 1m | 2055464 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 80 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_5m | ETHUSDT | 5m | 411108 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_5m | ETHUSDT | 5m | 411092 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 16 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_15m | ETHUSDT | 15m | 137036 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_15m | ETHUSDT | 15m | 137031 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 5 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_1h | ETHUSDT | 1h | 61396 | 2019-11-27 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_1h | ETHUSDT | 1h | 81239 | 2017-08-17 | 2026-11-28 | 28 | 0 | 0 | PASS | **INCOMPLETE** | 28 gap(s) / 128 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_4h | ETHUSDT | 4h | 15349 | 2019-11-27 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_4h | ETHUSDT | 4h | 20325 | 2017-08-17 | 2026-11-28 | 8 | 0 | 0 | PASS | **INCOMPLETE** | 8 gap(s) / 16 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_perp_1d | ETHUSDT | 1d | 2558 | 2019-11-27 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_spot_1d | ETHUSDT | 1d | 3390 | 2017-08-17 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_funding | ETHUSDT | funding | 7675 | 2019-11-27 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_mark_1h | ETHUSDT | 1h | 60768 | 2019-12-23 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_index_1h | ETHUSDT | 1h | 60768 | 2019-12-23 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_oi_5m | ETHUSDT | 5m | 8372 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_oi_15m | ETHUSDT | 15m | 2791 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_oi_1h | ETHUSDT | 1h | 734 | 2026-10-28 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_oi_4h | ETHUSDT | 4h | 174 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| ETHUSDT_oi_1d | ETHUSDT | 1d | 29 | 2026-10-31 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_1m | SOLUSDT | 1m | 2055544 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_1m | SOLUSDT | 1m | 2055464 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 80 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_5m | SOLUSDT | 5m | 411108 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_5m | SOLUSDT | 5m | 411092 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 16 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_15m | SOLUSDT | 15m | 137036 | 2023-01-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_15m | SOLUSDT | 15m | 137031 | 2023-01-01 | 2026-11-28 | 1 | 0 | 0 | PASS | **INCOMPLETE** | 1 gap(s) / 5 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_1h | SOLUSDT | 1h | 54388 | 2020-09-14 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_1h | SOLUSDT | 1h | 55185 | 2020-08-11 | 2026-11-28 | 10 | 0 | 0 | PASS | **INCOMPLETE** | 10 gap(s) / 20 missing bars (documented exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_4h | SOLUSDT | 4h | 13597 | 2020-09-14 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_4h | SOLUSDT | 4h | 13801 | 2020-08-11 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_perp_1d | SOLUSDT | 1d | 2266 | 2020-09-14 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_spot_1d | SOLUSDT | 1d | 2300 | 2020-08-11 | 2026-11-27 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_funding | SOLUSDT | funding | 6876 | 2020-09-13 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_mark_1h | SOLUSDT | 1h | 54408 | 2020-09-13 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_index_1h | SOLUSDT | 1h | 54707 | 2020-09-01 | 2026-11-28 | 0 | 0 | 0 | PASS | **INCOMPLETE** | available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_oi_5m | SOLUSDT | 5m | 8365 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_oi_15m | SOLUSDT | 15m | 2788 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_oi_1h | SOLUSDT | 1h | 734 | 2026-10-28 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_oi_4h | SOLUSDT | 4h | 174 | 2026-10-30 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| SOLUSDT_oi_1d | SOLUSDT | 1d | 29 | 2026-10-31 | 2026-11-28 | — | 0 | — | PASS | **INCOMPLETE** | OI history capped ~30 days by Binance (exchange-side); available_at unknown for 100% of rows (event-time PIT only) |
| contract_specs | MULTI | metadata_snapshot | 3 | — | — | — | — | — | PASS | **INCOMPLETE** | current snapshot; no historical effective dates |
| spot_contract_specs | MULTI | metadata_snapshot | 3 | — | — | — | — | — | PASS | **INCOMPLETE** | current snapshot; no historical effective dates |
| funding_intervals | MULTI | metadata_snapshot | 3 | — | — | — | — | — | PASS | **PASS** | current interval config is a snapshot; historical effective dates only via observed settlements |
| margin_tiers | MULTI | metadata_snapshot | 68 | 2026-11-28 | 2026-11-28 | — | — | — | PASS | **INCOMPLETE** | dated snapshots only; historical effective dates UNKNOWN |

## Remaining limitations

- OI history limited to ~30 days by Binance (exchange-side cap) — tail refresh only, not full historical refetch
- Margin-tier snapshots are dated observations; historical effective dates UNKNOWN — current snapshot used as assumption
- Contract specs / spot specs / funding current config are current snapshots, not historical reconstructions
- Spot candle gaps are exchange-side and preserved (not filled)
- available_at is NaT (unknown) for historical market rows — event-time PIT only
- 2026-11-01→2026-11-28 holdout EXPOSED by Program E H09–H12 (PREREG IV.1) — not pristine
- This report validates stored datasets only; no refetch performed (Integrity Correction 3)

Machine-readable: `item8_final_validation.json`

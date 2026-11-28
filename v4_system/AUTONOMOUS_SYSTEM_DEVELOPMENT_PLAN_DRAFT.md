# Autonomous Crypto Trading System — Technical Development Plan DRAFT (rev 2)

**Status**: DRAFT for owner review. Nothing herein is approved, frozen,
registered, or authorized for implementation. No trial budget consumed.
No model trained. No experiment run.

**Date**: 2026-11-28 · **Branch**: `phase2-v4-adapter` @ `7413767`
**Governing architecture**: `v4_system/AUTONOMOUS_CRYPTO_TRADING_SYSTEM_SPEC_DRAFT.md`
(`0f45abe`, local only — not pushed).

**Rev 2 changes**: recorder/watchdog status verified (item 0); prior
carry/basis work documented and WP-05 replaced (item 1); measured
10-minute stream volumes replace estimates (item 2); live-recording
scope corrected to non-backfillable streams only (item 3); data
inventory dates corrected (item 4); trial-count definition conflict
with PREREG IV.4 resolved into a proposal (item 5); per-opportunity
table, per-work-package detail, online/offline architecture, and
per-component ML requirements added (item 6).

---

## 1. Executive summary

This plan converts the architecture draft into a dependency-aware,
prioritized development path toward the autonomous multi-asset ML
trading system. The central tension — **data accumulation takes months,
but research and infrastructure cannot wait** — is resolved by running
data collection in parallel with infrastructure and early research on
data that already exists.

**Correction vs rev 1**: funding/basis carry is NOT untested. Programs
A/B/C and hypotheses H02/H06/H10/H11 already evaluated carry, funding
reversal, basis fades, and pairs — all KILL or marginal (details §2.4).
The first new research family is therefore NOT carry; it is the
regime/return-forecasting stack plus short-horizon momentum research
(§9, §15).

**Recommended first implementation package** (details §15):
1. Extend `live_recorder` to the **top-20 USDT perps by 24h quote
   volume** for the streams that CANNOT be backfilled: liquidations,
   bookTicker, depth5. aggTrade/klines/funding/OI come from the
   Binance archive, not live recording.
2. Build the market-universe registry with eligibility intervals
   (survivorship control).
3. Build the OI archive ingestion pipeline (validated, checksum-
   verified, provenance-preserving).
4. Build the feature-store skeleton (multi-resolution, PIT-enforced,
   versioned) on existing klines.
5. Design the portfolio-level backtester (WP-07) — the largest
   infrastructure gap; can be designed on existing single-asset data.

---

## 2. Verified repository baseline

Inspected 2026-11-28, branch `phase2-v4-adapter` @ `7413767`, clean tree.
Commit `0f45abe` (architecture draft) and `7413767` (rev-1 plan) are
local-only; `origin/phase2-v4-adapter` is at `16e0673`.

### 2.1 Component status (verified by file inspection)

| Component | Path | Status | Evidence |
|---|---|---|---|
| Data engine | `v4_system/data_engine/` | **Implemented** | 17 modules: store, klines, funding, OI, contract_specs, margin_tiers, pit, manifest, fetch_phase1, final_validation, live_recorder, live_watchdog, agg_trades, binance_client, refresh, availability_probe |
| PIT controls | `data_engine/pit.py` | **Implemented, tested** | Unified `as_of`, unfinished-candle exclusion, `available_at` semantics; `tests/test_pit_integrity.py` |
| Live recorder | `data_engine/live_recorder.py` | **Implemented, young** | Binance `!forceOrder@arr` (sampled liq), `bookTicker` (3 syms), Bybit `allLiquidation` (3 syms); JSONL/min, atomic writes, gap log, reconnect |
| Live watchdog | `data_engine/live_watchdog.py` | **Implemented** | Health monitoring + Telegram outbox retry |
| V4↔QR adapter | `adapters/quant_research_data.py` | **Implemented** | PIT-labeled loading, schema mapping |
| Backtest engine | `quant_research/core/engine.py` | **Implemented, single-asset** | Next-open fills, 14/18 bps costs, funding, 1m sub-bar stops; NOT portfolio/multi-asset |
| Metrics/stats | `quant_research/core/metrics.py`, `statistical_tests.py` | **Implemented** | DSR, walk-forward, Sharpe, random baselines |
| Strategies | `quant_research/strategies/` | **Implemented, manual** | Turtle, indicator_stack, parameter_grid |
| Trial governance | `trials_registry.py`, `trials_log.csv`, `PREREG.md` | **Implemented** | Append-only, 1/40 trials used, N=1,480 |
| Family 1 runner | `run_family1.py` | **Implemented** | 5-fold WF, NO EDGE preserved |
| Tests | `tests/` | **Implemented** | 38 files, 359 tests passing |
| **ml_engine** | `v4_system/ml_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **execution_engine** | `v4_system/execution_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **risk_engine** | `v4_system/risk_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **strategy_engine** | `v4_system/strategy_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **backtesting** (v4) | `v4_system/backtesting/` | **EMPTY SHELL** | Only `__init__.py` |
| **microstructure_engine** | `v4_system/microstructure_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **macro_engine / news_engine** | `v4_system/` | **EMPTY SHELLS** | Only `__init__.py` |
| **paper_trading** | `v4_system/paper_trading/` | **EMPTY SHELL** | Only `__init__.py` |

### 2.2 Data inventory (corrected dates, from `data/MANIFEST.json`)

| Dataset | Coverage (verified) | Rows (example) |
|---|---|---|
| Perp klines 1m/5m/15m | 2023-01-01 → 2026-11-28 | BTC 1m: 2,055,544 |
| Perp klines 1h | BTC 2019-09-08, ETH 2019-11-27, SOL 2020-09-14 → 2026-11-28 | BTC 1h: 63,306 |
| Spot klines 1h | BTC/ETH 2017-08-17, SOL 2020-08-11 → 2026-11-28 | BTC spot 1h: 81,239 |
| Funding settlements | BTC 2019-09-10, ETH 2019-11-27, SOL 2020-09-13 → 2026-11-28 | BTC: 7,909 |
| OI (REST resamples 5m/15m/1h/4h/1d) | ~2026-10-28/30 → 2026-11-28 (**~30 days only**) | BTC 1h: 735 |
| Live recorder output | ~2026-11-27/28 → present (~1–2 days, growing) | ~93 MB at inspection |

64 manifest entries total. Item-8 validation: 1 PASS / 63 INCOMPLETE /
0 FAIL (INCOMPLETE = unknown `available_at` for bulk history, not
structural defects).

**Follow-up noted**: the item-8 validation report still describes OI
as "capped ~30 days" in its limitations text. That wording is accurate
for the REST-sourced data but predates the archive investigation; it
should be corrected when the OI archive ingestion pipeline lands so
the report distinguishes REST coverage from archive availability.

### 2.3 Hardware (verified)

- CPU: 13th Gen Intel i7-13650HX, **14 cores**
- RAM: **23.7 GB**
- Disk D: free: **64.4 GB** (insufficient for aggressive tick
  collection — see §8.2)
- OS: Windows, Python 3.11

Development workstation, not a production server. Adequate for
research, feature engineering, GBM training, hourly/daily backtesting.
NOT adequate for temporal deep learning at scale or production
trading — future owner decision.

### 2.4 Prior carry / basis / funding work (verified — "untested" claim removed)

Funding carry and basis strategies have been tested repeatedly. All
results below are from `v4_system/PREREG.md` §II.1 and
`research/new_edge/registry.json`:

| ID | Work | Result | Verdict |
|---|---|---|---|
| A1 | Funding-rate carry validation, BTC short-perp+long-spot, 2019-09→2024-12 | +109.9%, CAGR 15.0%, Sharpe 18.2 — **superseded** (pre fee/slippage/mark-price corrections) | Superseded by B |
| A2 | Funding carry paper bot v1 | superseded by v2 | — |
| B5 | S5 funding carry, no margin model, daily, $1k | 30.2% CAGR, 1.14%/mo (2023+), maxDD −0.7% | **VIABLE but below 6%/mo target** |
| B7 | S5 v2 margin model (isolated, OKX MMR) | L=1: 5.4% CAGR; L=2: 7.3%; **L=3: LIQUIDATED 2021-01-22** | Leverage-constrained |
| B8 | S5 v3 Bybit venue-consistent | $1,195 final, CAGR 3.36%, 0.28%/mo, maxDD −0.76%, 0 liq | ≈ stablecoin yield with funding-sign risk |
| C1 | Multi-asset funding arb, top-3 trailing-30d, weekly | $1,037–1,053 (0.9–1.3% CAGR); fails at 1.5x/2x costs | FAIL |
| C2/C3 | Cash-and-carry v1→v2 (long spot + short quarterly) | v1 invalidated (data-alignment artifact); v2: ETH 0.30%/mo, BTC 0.35%/mo | Below S5; not pursued |
| C4 | Adaptive funding carry (BTC, vol-scaled) | L=1: 3.3% CAGR; L=2: 4.5% | Below S5 |
| E-H02/H02b | Funding-extreme reversal, daily, 36 perps | net Sharpe 0.49; H02b worse at longer holds | KILL |
| E-H06 | Funding reversal + funding cash flow | net Sharpe 0.70, 2024 Sharpe −1.28 | KILL |
| E-H10/H11 | Perp/index premium fade (72h mark/index) | BTC/ETH positive was asset-specific; SOL holdout −56 bps/event, Sharpe −0.43 | KILL |
| E-H12 | BTC/ETH/SOL pairs mean-reversion | spreads trend not revert; −95 to −212 bps net/event | KILL |
| Paper bots | `research/bots/funding_carry_paper_bot{,_v2}.py` | live paper track (v2 current) | Prospective, ongoing |

**Implication**: naive carry is a known, modest, capacity-limited
result (~stablecoin yield with funding-sign risk). Any NEW carry work
must add something prior work lacked: dynamic universe breadth, ML
timing/selection overlay, or portfolio integration — not a re-run of
S5. This is why WP-05 (funding carry as "first experiment") was
replaced by WP-07 (portfolio backtester design) in FIP-1.

### 2.5 Live recorder / watchdog process status (verified 2026-11-28 ~17:30 local)

| Process | PID | Started | Command | Includes commits |
|---|---|---|---|---|
| `live_recorder` | 25824 | 2026-11-28 09:37:50 | `python -u -m v4_system.data_engine.live_recorder --flush-seconds 60` | **742edb0** (07:32:18), **0a97d96** (07:33:37), **e28fbf8** (09:36:30) — all ancestors of HEAD; recorder started 80 s after e28fbf8 |
| `live_watchdog` | 41884 | 2026-11-28 07:33:54 | `python -u -m v4_system.data_engine.live_watchdog` | 742edb0 + 0a97d96 (committed 07:33:37, 17 s before start); e28fbf8 touched only `live_recorder.py` — watchdog unaffected |

- Working folder: this repository (`v4_system/data/live/` output;
  `BTC Strategy` and `BTC Strategy-p2` resolve to the same directory).
- Streams actually recorded: Binance `!forceOrder@arr` (SAMPLED
  liquidations), `bookTicker` BTC/ETH/SOL, Bybit `allLiquidation`
  BTC/ETH/SOL. **No aggTrade, no depth streams.**
- **Lag report: NOT done** — no lag measurement artifact exists.
- **depth5 test: NOT done** before this plan — the 10-minute probe in
  §8.2 is the first depth5 measurement.

---

## 3. Complete development workstreams

### A. Market discovery and universe management
Exchange market discovery (Binance spot+futures, Bybit instruments);
instrument metadata registry; liquidity/volume/spread measurement;
eligibility filters; historical eligibility intervals (survivorship);
dynamic promotion/demotion; delisting/new-listing handling.

### B. Market-data collection and storage
Expand recorder for **non-backfillable streams only** (liquidations,
bookTicker, depth); historical backfills from Binance archive
(klines all TFs, aggTrades, OI archive, funding); storage technology;
timestamp/PIT/sequence integrity; reconnect/recovery/gaps; monitoring;
validation.

### C. Feature and market-state infrastructure
Multi-resolution feature computation (event-driven + rolling);
cross-asset features; regime/vol estimates; microstructure features;
funding/positioning features; versioning, reproducibility, missing-
data policy, timestamp alignment, PIT enforcement.

### D. ML research and model infrastructure
Modular components: regime, return/vol forecast, order-flow,
opportunity detection, ranking, strategy selection, execution,
portfolio allocation. Per-component requirements in §9.

### E. Opportunity and strategy research
Organized framework for opportunity families (§3.E table below);
per-family data/frequency/execution/cost/capital/feasibility;
family↔component mapping is many-to-many.

### F. Portfolio decision and risk infrastructure
Ranking, allocation, sizing, leverage, concurrent positions,
correlation, partial exits, reversals, hedging, capital conflicts,
turnover, drawdown, emergency controls; portfolio-level backtesting.

### G. Execution and paper trading
Exchange connectivity, order management, partial fills, cancel/
replace, maker/taker, book-aware execution, slippage/impact, funding,
reconciliation, outages, risk controls, monitoring, audit; progression
historical sim → live-data paper → (future) live.

### 3.E Opportunity-family table (required deliverable)

| Family | Required data | Frequency | Exec complexity | Cost sensitivity | Capital need | Post-cost viability (prior evidence) | ML components used | Feasibility now |
|---|---|---|---|---|---|---|---|---|
| Funding/basis carry | Funding + perp/spot klines | 8h settlements, daily rebalance | Low | Low (low turnover) | Medium | **Known: ~stablecoin yield** (B5/B8/C1–C4); needs ML overlay or breadth to improve | Regime, return forecast, opportunity detector | ✅ Data exists; re-run only with new mechanism |
| Short-term momentum | 1m–1h klines | 1m–1h | Low | High (turnover) | Medium | Unknown at 1m–15m; NO EDGE at 1h/4h (Family 1) | Return forecast, regime | ✅ Data exists (1m from 2023) |
| Mean reversion | 1m–15m klines | 1m–15m | Low | High | Medium | Unknown; A3/A4 WF failed at daily+ | Return forecast, regime | ✅ Data exists |
| Breakout/vol expansion | klines + vol | 5m–1h | Low | High | Medium | H03 KILL (latency artifact); Turtle NO EDGE | Regime, opportunity detector | ⚠️ Research needed |
| Liquidation cascades | Liquidation feed | Seconds–minutes | High | Very high | High | H01 KILL at 1m fade; needs different mechanism | Order-flow, microstructure | ❌ Data accumulating (~days) |
| Order-flow imbalance | Trades + book | Seconds | Very high | Very high | High | Unknown — no historical data | Order-flow model | ❌ Data accumulating |
| Liquidity/spread dislocations | bookTicker/depth | Seconds | Very high | Very high | Very high | Unknown — no historical data | Execution model | ❌ Data accumulating |
| Cross-asset relative value | Multi-asset klines | 1h–1d | Low | Medium | Medium | H05/H12 KILL; Family D pairs NO EDGE | Return forecast, regime | ⚠️ Data exists, prior failures |
| Statistical arbitrage | Multi-asset klines | 1h–1d | Low | Medium | Medium | H12 KILL (trend not revert) | Return forecast | ⚠️ Data exists, prior failures |
| Cross-exchange discrepancies | Multi-exchange data | Seconds | Very high | Very high | Very high | Unknown | Execution model | ❌ Needs 2nd exchange recorder |
| Market making | Full book + trades | Continuous | Very high | Extreme | Very high | Unknown | RL/advanced (deferred) | ❌ Deferred |

**Note**: one ML component can serve several families (e.g., the
return/vol forecaster feeds momentum, mean-reversion, and relative-
value); one family can use several components (e.g., carry uses
regime + forecast + execution). The mapping is many-to-many.

---

## 4. Dependency graph

```
A (Universe) ──────────────────────────────────────────────┐
B (Data collection) ──► C (Features) ──► D (ML) ──► E (Research)
     │                        │              │         │
     │                        │              ▼         ▼
     │                        │         F (Portfolio) ──► G (Execution/Paper)
     │                        │              │
     └──► B2 (Historical      └──► C2 (Feature    └──► F2 (Portfolio
          backfill)                store)              backtester)
```

**Hard dependencies**: order-flow/microstructure ML needs tick/book
data (months); portfolio backtesting needs multi-asset data +
features; paper trading needs portfolio engine + execution sim; live
trading needs paper validation + owner authorization.

**Soft dependencies (start now)**: universe registry (exchangeInfo
API only); OI archive ingestion (HTTP + validation); feature-store
skeleton (existing klines); portfolio backtester design (existing
single-asset data sufficient for architecture).

---

## 5. Immediate vs parallel vs deferred tasks

### Immediate (start now, current code+data)
| Task | Why immediate |
|---|---|
| Extend live_recorder: top-20 perps, liquidations+bookTicker+depth5 (non-backfillable only) | Every day without recording is permanently lost data |
| Universe registry + eligibility intervals | Pure metadata; survivorship control; no trading risk |
| OI archive ingestion pipeline | Unlocks ~5yr OI history; validation design exists |
| Feature-store skeleton on existing klines | Foundation for all ML |
| Portfolio backtester design (WP-07) | Largest infrastructure gap; designable now |

### Parallel (run alongside immediate)
| Task | Why parallel |
|---|---|
| Historical kline backfill for expanded universe | Independent; Binance archive |
| aggTrades selective archive download (BTC/ETH recent) | Independent; enables microstructure research later |
| Regime estimator (unsupervised, existing klines) | No new data needed |
| Data-quality/validation framework extension | Generalizes item-8 validator |

### Data-dependent (cannot validate until data accumulates)
| Task | Data needed | Wait |
|---|---|---|
| Order-flow/microstructure models | Trade/book history | Months of recording |
| Liquidation-cascade strategies | Liquidation history | Months |
| Spread/dislocation strategies | bookTicker/depth history | Months |
| Cross-exchange arbitrage | Multi-exchange synchronized data | After recorder expansion |
| Sub-minute strategies | 1s bars from trades | After trade data + aggregation |

### Infrastructure-dependent
| Task | Depends on |
|---|---|
| Paper trading | Portfolio engine + execution sim + live data |
| Live trading | Paper validation + owner authorization |
| Learned strategy selection | Multiple validated opportunity families |
| Learned portfolio allocation | Validated ranking + risk framework |

### Deferred
| Task | Why deferred |
|---|---|
| Options strategies | No data source; infrastructure absent |
| Market making | Requires depth + capital + low latency |
| Transformer/temporal deep models | Insufficient data; GBM/linear unexhausted |
| Reinforcement learning | Premature before supervised baseline |
| News/macro features | No data source; lower priority |

---

## 6. Prioritized work packages (detailed)

Priority = strategic value × capabilities unlocked ÷ (difficulty + cost
+ data wait). Each WP lists: objective, reuse, new work, dependencies,
tests, acceptance, complexity, resources, unlocks.

### WP-01 — Extend live_recorder (P0)
- **Objective**: record non-backfillable streams for top-20 perps.
- **Reuse**: `live_recorder.py` (reconnect, gap log, atomic JSONL,
  watchdog), `live_watchdog.py`.
- **New work**: symbol list from universe registry or static top-20;
  add `depth5@100ms` (or `depth20@100ms` — owner decision §14);
  keep bookTicker+liquidations; per-symbol stream management; storage
  rotation/retention.
- **Dependencies**: none (starts collecting immediately).
- **Tests**: unit tests for stream parsing, gap logging, reconnect;
  24h soak test; storage-growth measurement.
- **Acceptance**: ≥7 days continuous, ≥15/20 symbols healthy, gap rate
  <1%, measured storage within budget.
- **Complexity**: Medium. **Resources**: ~6 GB/day download,
  ~0.3–0.6 GB/day stored (§8.2). **Unlocks**: all microstructure
  research.

### WP-02 — Universe registry + eligibility intervals (P0)
- **Objective**: dynamic universe with survivorship control.
- **Reuse**: `binance_client.py`, `contract_specs.py`, `manifest.py`
  patterns.
- **New work**: `universe_registry` module — exchangeInfo polling,
  eligibility filters (volume, spread, history, data quality),
  eligibility-interval history table, promotion/demotion logic.
- **Dependencies**: none.
- **Tests**: filter unit tests; survivorship reconstruction test
  (delisted instrument recoverable); interval-history integrity.
- **Acceptance**: ≥50 instruments tracked with eligibility intervals;
  point-in-time universe reconstruction works.
- **Complexity**: Low-Med. **Unlocks**: survivorship-controlled
  backtests; dynamic universe.

### WP-03 — OI archive ingestion pipeline (P0)
- **Objective**: ingest Binance OI archive (2020-09/2021-12→present,
  5m) with validation.
- **Reuse**: `store.py` (atomic parquet+manifest), `fetch_phase1.py`
  patterns, `final_validation.py` classification.
- **New work**: archive downloader (data.binance.vision), checksum
  verification, timestamp-convention classifier (2026-06-25 change),
  resample to existing OI TFs, manifest integration.
- **Dependencies**: none (HTTP only).
- **Tests**: checksum validation; convention classification per file;
  resample correctness; gap detection.
- **Acceptance**: OI history back to archive start for BTC/ETH/SOL;
  conventions classified; validation report updated (fixes the
  "capped ~30 days" wording noted in §2.2).
- **Complexity**: Medium. **Unlocks**: 5yr OI features; positioning
  research.

### WP-04 — Feature-store skeleton (P0)
- **Objective**: multi-resolution, PIT-enforced, versioned features.
- **Reuse**: `pit.py` (cutoff semantics), `store.py`, adapter patterns.
- **New work**: `feature_store` module — feature registry, versioning,
  missing-data policy, cross-asset alignment, event-time vs receipt-
  time handling, feature hashing for reproducibility.
- **Dependencies**: existing klines (available now).
- **Tests**: PIT leakage tests; reproducibility (same data → same
  features); missing-data handling; cross-asset timestamp alignment.
- **Acceptance**: features reproducible; PIT tests pass; versioning
  works; usable by ML components.
- **Complexity**: Medium. **Unlocks**: all ML components.

### WP-05 — (REMOVED: was "funding carry research")
Funding carry is already tested (§2.4). Any new carry work requires a
NEW mechanism (ML overlay, universe breadth, portfolio integration)
and is deferred to after WP-04/WP-07 produce the infrastructure to
evaluate it properly. Replaced in FIP-1 by WP-07.

### WP-06 — Historical kline backfill, expanded universe (P1)
- **Objective**: klines for top ~50 perps, all TFs, from Binance archive.
- **Reuse**: `klines.py`, `fetch_phase1.py`, `store.py`.
- **New work**: batch archive downloader; universe-driven symbol list;
  incremental refresh.
- **Dependencies**: WP-02 (universe list) recommended but not blocking.
- **Tests**: gap detection; manifest integrity; spot-check vs API.
- **Acceptance**: ≥50 symbols × 6 TFs backfilled; validation passes.
- **Complexity**: Low. **Unlocks**: broader research universe.

### WP-07 — Portfolio-level backtester (P1, promoted into FIP-1)
- **Objective**: multi-asset, multi-position, concurrent-position
  simulation with capital allocation, correlation, partial exits,
  hedging, turnover limits.
- **Reuse**: `quant_research/core/engine.py` cost/fill/funding
  primitives; `core/metrics.py`, `walk_forward.py`.
- **New work**: new engine in `v4_system/backtesting/` — portfolio
  state, position ledger, capital allocator interface, correlation
  matrix, risk-gate hooks, marked-to-market equity curve.
- **Dependencies**: none for design (single-asset data sufficient);
  multi-asset data for full validation (WP-06).
- **Tests**: **parity test** — single-asset single-position case must
  match existing engine exactly; constraint enforcement tests;
  cost/funding correctness.
- **Acceptance**: parity test passes; portfolio constraints enforced;
  realistic costs; documented limitations.
- **Complexity**: High. **Unlocks**: portfolio research; realistic
  multi-strategy evaluation; paper-trading target architecture.

### WP-08 — Regime estimator (P1)
- **Objective**: unsupervised regime classification (trend/range/vol).
- **Reuse**: existing klines; `core/indicators.py`.
- **New work**: clustering/HMM on returns/vol/correlation features;
  regime labels as features; stability analysis.
- **Dependencies**: WP-04 (features).
- **Tests**: regime persistence; out-of-sample stability; label
  interpretability.
- **Acceptance**: stable regimes across walk-forward; usable as
  gating feature.
- **Complexity**: Medium. **Unlocks**: strategy gating; context
  features for all models.

### WP-09 — aggTrades archive download + 1s bar builder (P1)
- **Objective**: historical aggTrades for priority symbols; 1s bars.
- **Reuse**: `agg_trades.py` downloader; `build_15s_bars.py` pattern.
- **New work**: archive aggTrade downloader (data.binance.vision —
  large, selective); 1s/1m bar aggregation; storage partitioning.
- **Dependencies**: none (archive).
- **Tests**: bar reproducibility; boundary correctness (existing
  `test_agg_trades_boundary.py` pattern).
- **Acceptance**: BTC/ETH aggTrades for selected period; 1s bars
  validated against 1m klines.
- **Complexity**: Medium. **Resources**: multi-GB download; ~100s GB
  stored. **Unlocks**: microstructure research foundation.

### WP-10 — Return/vol forecaster, GBM baseline (P1)
- **Objective**: first predictive ML component.
- **Reuse**: WP-04 features; LightGBM (new dependency — engineering
  decision); `core/metrics.py` for eval.
- **New work**: `ml_engine/` module — dataset builder, LightGBM
  quantile/regression, walk-forward eval, model registry.
- **Dependencies**: WP-04.
- **Tests**: leakage tests; walk-forward integrity; calibration.
- **Acceptance**: trained model with OOS metrics; registered as
  research artifact (NOT a trading trial until converted to strategy
  per §11.2).
- **Complexity**: Medium. **Unlocks**: core predictive capability.

### WP-11 — Opportunity ranker, rule-based first (P2)
- **Objective**: rank candidate opportunities across instruments.
- **Reuse**: WP-07 portfolio engine; WP-10 forecasts.
- **New work**: scoring function (expected net edge, confidence,
  regime fit, liquidity, correlation); constraint-aware selection.
- **Dependencies**: research results (WP-10+), WP-07.
- **Tests**: ranking correctness; constraint respect; backtest
  integration.
- **Acceptance**: ranker produces ordered candidates; allocator
  respects limits in backtest.
- **Complexity**: Medium.

### WP-12 — Execution simulator, book-aware (P2)
- **Objective**: realistic execution modeling (spread crossing,
  market impact, partial fills).
- **Reuse**: WP-01 bookTicker/depth data; engine cost primitives.
- **New work**: book-aware fill model; square-root impact model;
  partial-fill simulation; maker/taker logic.
- **Dependencies**: WP-01 (data accumulating), WP-07.
- **Tests**: fill realism vs recorded data; impact model sanity.
- **Acceptance**: simulated fills statistically consistent with
  observed book data.
- **Complexity**: High.

### WP-13 — Paper-trading engine (P2)
- **Objective**: live-data paper trading with order management.
- **Reuse**: WP-07 portfolio engine; WP-12 execution sim; exchange
  WS connectivity patterns from recorder.
- **New work**: `paper_trading/` module — order management, position
  reconciliation, funding application, risk controls, audit logs.
- **Dependencies**: WP-07, WP-12, live data (WP-01).
- **Tests**: reconciliation tests; outage/rejection handling; risk-
  control enforcement.
- **Acceptance**: ≥30 days paper trading without critical bugs;
  reconciliation matches exchange.
- **Complexity**: High.

### WP-14 — Order-flow/microstructure models (P2)
- **Objective**: short-term impact/imbalance prediction.
- **Reuse**: WP-04 features; WP-09 trade data.
- **New work**: microstructure features (order-flow imbalance, trade
  signing, book pressure); GBM/temporal models.
- **Dependencies**: WP-01 (months of data), WP-09.
- **Tests**: as WP-10 plus microstructure-specific leakage checks.
- **Acceptance**: OOS predictive metrics; only then consider trading
  trial registration.
- **Complexity**: High. **Data wait**: months.

### WP-15 — Learned strategy selection / allocation (P3)
- **Objective**: replace rule-based selection with learned models.
- **Dependencies**: WP-11 validated; multiple opportunity families
  with evidence.
- **Complexity**: High. **Deferred until justified.**

### WP-16 — Cross-exchange expansion (P3)
- **Objective**: Bybit/other exchange data + arbitrage research.
- **Dependencies**: WP-01 patterns proven; WP-02 universe extended.
- **Complexity**: High.

### WP-17 — Options integration (P3)
- **Objective**: options data + strategies.
- **Dependencies**: options data source (none exists); new
  infrastructure.
- **Complexity**: Very high. **Deferred.**

---

## 7. Concrete milestones and acceptance criteria

### M1 — Expanded market discovery and eligibility registry
- **Deliverable**: universe_registry; exchangeInfo polling;
  eligibility filters; eligibility-interval history.
- **Acceptance**: ≥50 instruments with eligibility intervals;
  survivorship reconstruction test passes.
- **Unlocks**: dynamic universe; survivorship-controlled backtests.

### M2 — Reliable multi-symbol data collection
- **Deliverable**: extended live_recorder for **top-20 perps**
  (liquidations + bookTicker + depth5); gap logging; health
  monitoring. *(Symbol count aligned with FIP-1: 20.)*
- **Acceptance**: ≥7 days continuous for ≥15/20 symbols; gap rate
  <1%; storage within budget.
- **Unlocks**: microstructure research; realistic execution sim.

### M3 — Trade and order-book data pipelines
- **Deliverable**: aggTrade archive ingestion; 1s/1m bar builder;
  bookTicker spread features; depth ingestion with sequence
  integrity.
- **Acceptance**: 1s bars reproducible; book state reconstructable;
  sequence-gap detection works.
- **Unlocks**: order-flow features; microstructure models.

### M4 — Historical and prospective data validation
- **Deliverable**: extended validation framework; OI archive
  ingested+validated; aggTrades spot-checked.
- **Acceptance**: all datasets PASS/INCOMPLETE/FAIL with provenance;
  OI conventions classified; report wording corrected (§2.2 follow-up).
- **Unlocks**: trustworthy features; OI research.

### M5 — Reproducible feature infrastructure
- **Deliverable**: feature store (multi-res, PIT, versioned,
  missing-data policy); feature registry.
- **Acceptance**: reproducible features; PIT tests pass; versioning;
  cross-asset alignment.
- **Unlocks**: all ML components.

### M6 — Multi-asset portfolio simulation
- **Deliverable**: portfolio backtester (WP-07).
- **Acceptance**: parity vs single-asset engine; constraints enforced;
  realistic costs.
- **Unlocks**: portfolio research; realistic evaluation.

### M7 — Initial ML component infrastructure
- **Deliverable**: ml_engine with regime estimator + return/vol
  forecaster (GBM baseline); training/eval pipeline; model registry.
- **Acceptance**: models train on features; walk-forward eval;
  DSR integration; versioning.
- **Unlocks**: predictive components; first ML research results.

### M8 — Opportunity ranking and allocation
- **Deliverable**: rule-based ranker; allocator; sizer; risk gate.
- **Acceptance**: ranker scores; allocator respects limits; risk gate
  vetoes; backtested on portfolio engine.
- **Unlocks**: multi-opportunity trading.

### M9 — Execution simulation and paper trading
- **Deliverable**: book-aware execution sim; paper-trading engine.
- **Acceptance**: realistic fills; reconciliation matches; ≥30 days
  paper without critical bugs.
- **Unlocks**: prospective validation.

### M10 — Integrated autonomous decision loop
- **Deliverable**: full pipeline on paper, live data.
- **Acceptance**: ≥6 months paper; vs benchmarks; risk controls
  functional; audit trail complete.
- **Unlocks**: evidence for live-trading decision.

---

## 8. Data-collection and storage strategy

### 8.1 What to collect live vs backfill (corrected — item 3)

**Live-record ONLY what cannot be backfilled** (no historical source):

| Stream | Symbols | Historical source? | Priority | Rationale |
|---|---|---|---|---|
| Liquidations (Binance `!forceOrder@arr` sampled + Bybit `allLiquidation` full) | Top-20 perps | ❌ None | **P0** | Irreplaceable; cascade research |
| bookTicker (top-of-book) | Top-20 perps | ❌ None | **P0** | Spread/dislocation; execution sim |
| Depth (depth5@100ms or depth20@100ms) | Top-20 perps (depth5 measured; depth20 = owner decision on bandwidth) | ❌ None | **P0/P1** | Order-book imbalance; impact modeling |

**Backfill from Binance archive (data.binance.vision) — do NOT
live-record:**

| Data | Historical source | Priority | Notes |
|---|---|---|---|
| aggTrades | ✅ Archive (daily/monthly, large) | P1 | Selective: BTC/ETH/SOL recent years first; multi-GB per symbol-year |
| Klines all TFs | ✅ Archive + REST | P1 | Universe expansion (WP-06) |
| Funding settlements | ✅ Archive/REST | P1 | Already have BTC/ETH/SOL; extend to universe |
| OI history | ✅ Archive (5m, 2020-09/2021-12→) | P0 | WP-03; timestamp conventions classified per file |

**aggTrade WS stream caveat (measured)**: the `@aggTrade` stream on
`fstream.binance.com` produced **zero messages** in both the 10-min
probe and a 30-s single-symbol test, while `@trade`, `@bookTicker`,
and `@depth5@100ms` on the same connection flowed normally. Either the
stream name is unsupported on this endpoint or it is silently dropped.
Since aggTrade history comes from the archive anyway, this does not
block anything; if tick-level live data is needed later, `@trade`
(verified flowing: ~36 msg/s BTC) or a corrected aggTrade subscription
must be used. Recorded as a finding, not worked around.

### 8.2 Measured stream volumes (10-min probe, 2026-11-28 17:19–17:29 local)

**Method**: `v4_system/data/volume_probe/probe.py` — separate WS
connection, 60 streams (20 symbols × bookTicker+aggTrade+depth5),
auto-reconnect (30 reconnects; connections lasted ~10–60 s each —
Binance appears to cycle connections; counters accumulated across
reconnects with small gaps, so rates are slight underestimates).
Symbols = top-20 USDT perps by 24h quote volume at probe time
(included tokenized equities e.g. SOXLUSDT/XAUUSDT/NVDAUSDT — the
universe filter in WP-02 should exclude or separately classify
these). Output: `v4_system/data/volume_probe/result.json`.

**Measured raw download rates** (JSON text as received):

| Group | Stream | msgs/s | KB/s | **Download GB/day** |
|---|---|---|---|---|
| Top-3 (BTC/ETH/SOL) | bookTicker | 80.3 | 16.3 | **1.41** |
| Top-3 | depth5@100ms | 6.7 | 2.8 | **0.24** |
| Top-3 | aggTrade | 0 | 0 | 0 (see caveat §8.1) |
| Top-20 | bookTicker | 279.5 | 56.9 | **4.92** |
| Top-20 | depth5@100ms | 31.4 | 13.1 | **1.13** |
| Top-20 | aggTrade | 0 | 0 | 0 |

**Stored-size estimates** (separate from download — item 2 requirement):
- JSONL raw ≈ download GB/day above.
- Parquet columnar compression of numeric WS payloads typically
  achieves **~10–20× reduction** (assumption, not measured — to be
  verified on first week of data). Estimated **stored GB/day**:
  - bookTicker top-20: ~0.25–0.5 GB/day → **~90–180 GB/year**
  - depth5 top-20: ~0.06–0.11 GB/day → **~20–40 GB/year**
  - liquidations: negligible (<1 GB/year)
  - **Total live-recording stored: ~0.3–0.6 GB/day → ~110–220 GB/year**
- If depth20@100ms instead of depth5: payload ~4× levels → download
  ~4.5 GB/day, stored ~0.25–0.45 GB/day → **~90–165 GB/year additional**.
  (Rev-1's "8.6 GB/day" depth figure was wrong; measured depth5 is
  1.13 GB/day download for top-20; depth20 estimate above replaces it.)

**Universe size decision**: live-recording universe = **top-20 USDT
perps by 24h quote volume** (measured feasible: ~6 GB/day download,
~0.3–0.6 GB/day stored). The *tradable/research* universe (WP-02
eligibility filters) may be larger for klines-only data — klines are
cheap (~MBs/symbol/year) — but live high-frequency streams stay at
top-20 until storage/bandwidth justify expansion. M2 and FIP-1 both
use 20 — aligned.

**Storage budget reality**: 64.4 GB free now. At ~0.3–0.6 GB/day
stored, ~3–7 months of headroom without expansion or retention
policy. A retention/compaction policy (e.g., parquet-ize JSONL older
than 7 days, then drop raw) is required — engineering recommendation.

### 8.3 Timestamp and PIT for high-frequency data
- Preserve exchange event timestamp (ms) AND receipt timestamp (µs).
- `available_at` = receipt time for live; NaT for bulk downloads.
- Order-book: sequence-number gap detection; missing diffs invalidate
  book state (resync from snapshot).
- `pit.py` generalizes: event-time cutoff + availability cutoff +
  unfinished-interval exclusion.

---

## 9. ML development strategy

### 9.1 Online vs offline architecture (required deliverable)

```
OFFLINE (research/training)                    ONLINE (live inference)
─────────────────────────                      ──────────────────────
Binance archive / stored data                  Live WS feeds (WP-01)
      │                                              │
      ▼                                              ▼
Feature store (batch, WP-04)              Feature calculator (streaming,
      │                                   same feature definitions,
      ▼                                   PIT-enforced at each tick)
Training pipelines (WP-10+)                      │
      │                                   Model inference (versioned,
      ▼                                   loaded from model registry)
Walk-forward eval / DSR                    Opportunity scores
      │                                   (same models, same features)
      ▼                                   │
Backtest (WP-07 portfolio engine) ◄──────► Decision layer (WP-11 ranker,
      │                                   WP-13 risk gate)
      ▼                                   │
Research verdicts (registered trials)     Orders → paper/live exchange
```

**Key invariant**: online and offline feature calculators MUST produce
identical values for identical inputs — enforced by shared feature
definitions (WP-04) and parity tests. PIT enforcement applies to both
paths. Online path prioritizes low latency (streaming, minimal
buffering); offline path prioritizes correctness and reproducibility.

### 9.2 Component-by-component plan (required detail per component)

| Component | Problem | Input data | Output | Model family | Training data | Eval metric | Compute | Data today? | Timing |
|---|---|---|---|---|---|---|---|---|---|
| Regime estimator | Classify market state | Multi-TF returns/vol/corr (WP-04) | Regime label + confidence | GMM/k-means, HMM, GBM classifier | Existing klines 2019/2023→ | Cluster stability, persistence, OOS consistency | Low (CPU) | ✅ Yes | **Now** (WP-08) |
| Return/vol forecaster | Predict returns/vol/quantiles | Multi-TF features (WP-04) | Return dist, vol, quantiles per horizon | LightGBM quantile, GARCH, linear | Existing klines | OOS RMSE/quantile loss, calibration, DSR if traded | Low-Med (CPU) | ✅ Yes | **Now** (WP-10) |
| Order-flow model | Predict short-term impact/imbalance | Trades+book (WP-01/09) | Impact prob, imbalance signal | GBM, shallow temporal (TCN/GRU) | Prospective tick/book (months) | OOS AUC/IC, latency-adjusted | Med-High | ❌ Months | After WP-01 data |
| Opportunity detector | Score candidate opportunities | Features + opportunity labels | Opportunity score | GBM, LambdaRank | Features + labeled outcomes | OOS ranking quality (NDCG/AUC) | Low-Med | ⚠️ Partial | After WP-10 |
| Opportunity ranker | Rank across instruments/families | Opportunity scores + constraints | Ranked candidate list | Rule-based → LambdaRank | Backtest outcomes | Portfolio-level backtest P&L | Low | ⚠️ Partial | WP-11 (P2) |
| Strategy selector | Choose best family for regime | Regime + scores | Strategy choice | Rules → contextual bandit | Regime + outcomes | OOS selection quality | Low | ❌ | Later (P3) |
| Execution model | Order type/timing/aggressiveness | Book+trades (WP-01/12) | Execution plan | Supervised fill models → RL later | Prospective book data | Fill quality vs benchmark, impact | Med-High | ❌ | After WP-12 |
| Portfolio allocator | Capital allocation/sizing/hedging | Positions+correlations+risk | Allocation vector | Constrained opt → risk-parity → RL later | Portfolio backtests | Risk-adjusted return, constraint adherence | Med | ⚠️ Partial | WP-07/11 (P1/P2) |

### 9.3 Model family guidance
- **Start**: LightGBM (tabular), linear models (baseline), unsupervised
  clustering/HMM (regime).
- **Add when justified**: temporal models (TCN/GRU) for sequence data;
  RL for execution/portfolio — only after supervised baselines fail.
- **Never without evidence**: transformers, deep RL.

### 9.4 Model lifecycle
- Version every model (data hash, features, hyperparams, metrics).
- Walk-forward OOS only; DSR for multiple testing.
- Paper trading before any live consideration; monitor vs backtest.
- Rollback on degradation or drift; registry extends
  `trials_registry.py` pattern.

---

## 10. Portfolio and execution infrastructure strategy

### 10.1 Portfolio backtester: new engine (WP-07)
- Existing `quant_research/core/engine.py` is single-asset,
  bar-driven — insufficient for multi-position/multi-asset/concurrent
  requirements.
- **Build new** in `v4_system/backtesting/` reusing cost/fill/funding
  primitives; portfolio state, position ledger, allocator interface,
  correlation matrix, risk-gate hooks, marked-to-market equity.
- **Parity acceptance**: single-asset single-position case must match
  existing engine exactly.

### 10.2 Execution simulation progression
1. Historical sim (current): next-open fills, fixed slippage.
2. Book-aware sim (WP-12): spread crossing, square-root impact,
   partial fills, maker/taker.
3. Paper trading (WP-13): live data, order management, latency sim.
4. Live (future, owner-authorized).

### 10.3 Paper-trading engine requirements (WP-13)
Exchange connectivity (testnet first); order management (place,
cancel, replace, partial fills); maker/taker + book-aware timing;
position reconciliation; funding/margin tracking; outage/rejection
handling; risk controls (max position/exposure/kill-switch);
monitoring (P&L, exposure, latency, fill quality, audit logs).

---

## 11. Research governance requirements

### 11.1 What requires trial registration
Any **strategy experiment** testing a trading hypothesis on historical
data and consuming research budget (PREREG IV.4). Each registered
config = architecture × horizon × feature set × threshold combination.
Budget: 1/40 used, 39 remaining. N_trials = 1,480.

### 11.2 Trial-count definition — conflict resolution (item 5, PROPOSAL for owner)

**The conflict**: rev-1 §11.2 said model training/eval without P&L
doesn't count as a trial. PREREG IV.4 says "each registered (model
architecture × horizon × feature set × threshold set) combination
consumes the trial budget" and "feature-selection sweeps, horizon
searches, and threshold grids each count their evaluated combinations"
— i.e., **selection on historical data counts regardless of metric**
(RMSE/IC or P&L). IV.4 is also unclear whether one registered trial
with a config matrix consumes 1 or N of the 40 budget slots (Family 1
used 1 slot but added 24 to DSR N).

**Proposed single definition** (for owner decision):

> **A trial-budget slot is consumed by one registered experiment
> family** — a frozen (hypothesis × instrument-set × timeframe-set ×
> model/feature/horizon/threshold matrix) specification registered
> BEFORE evaluation. Within a family, every distinct evaluated
> configuration adds to the DSR trial count N (per IV.4's "genuine
> selection opportunities"), but consumes only the family's single
> budget slot. Infrastructure work, unsupervised exploration with no
> selection, and paper trading consume no slots. Any sweep over
> model/feature/horizon/threshold alternatives on historical data —
> whether scored by P&L, RMSE, IC, or any other metric — is a
> selection and must be inside a registered family; unregistered
> sweeps are prohibited.

This works with 39 remaining slots: each new research family (e.g.,
"ML return forecaster → trading strategy", "1m momentum family")
registers once with a frozen matrix; the matrix size affects DSR N,
not the slot count. **Owner must approve or amend.**

### 11.3 Governance constraints
- PREREG binding; amendments for new families.
- Append-only registry; no deletions.
- DSR with correct N; no p-hacking; walk-forward only.
- Nov-2026 holdout exposed — all history ≤2026-11-28 is development
  data; future-only holdout per PREREG IV.3.
- Costs: 14/18 bps RT base/stress; funding applied.
- No silent budget increases.

---

## 12. Resource and feasibility analysis

### 12.1 Against current hardware (i7-13650HX, 14 cores, 23.7GB RAM, 64GB free)

| Task | Feasible now? | Notes |
|---|---|---|
| Live recording top-20 (liq+bookTicker+depth5) | ✅ Yes | ~6 GB/day download, network-bound; ~0.3–0.6 GB/day stored |
| Kline/funding/OI backfill | ✅ Yes | Existing downloader |
| OI archive ingestion | ✅ Yes | ~70MB compressed; hours |
| Feature engineering on klines | ✅ Yes | Pandas/polars |
| GBM training (LightGBM) | ✅ Yes | Fast on tabular |
| Portfolio backtesting (hourly/daily) | ✅ Yes | Manageable |
| Tick-level feature engineering | ⚠️ Marginal | Needs polars/DuckDB; large data |
| Temporal deep model training | ❌ No | Needs GPU/cloud; defer |
| High-frequency online inference | ❌ No | Needs low-latency infra |
| Full-depth book collection | ⚠️ Marginal | depth20 ~4.5 GB/day download; limit scope |

### 12.2 When additional resources justified
- Storage >200GB tick data → cloud storage or larger disk.
- Temporal model training → GPU instance.
- Production trading → VPS/cloud low-latency.
- Multi-exchange → more bandwidth.

### 12.3 Feasibility verdict
Development/research platform feasible on current hardware for: data
collection, feature engineering, GBM ML, portfolio backtesting at
hourly/daily, paper trading. NOT feasible for: tick-level deep
learning, high-frequency online trading, production deployment.

---

## 13. Risks and bottlenecks

| Risk | Impact | Mitigation |
|---|---|---|
| Data accumulation too slow | Months before microstructure research | Start recording NOW; P0 streams |
| Storage exhaustion (64GB free) | Collection stops | Retention/compaction policy; storage budget decision |
| NO EDGE prior (1,480 trials + Program E all-KILL) | ML may also find nothing | Realistic expectations; diverse families; regime conditioning |
| Overfitting complex models | False positives | DSR, walk-forward, registry, simple-first |
| Execution reality gap | Backtest ≠ live | Book-aware sim, paper trading, conservative costs |
| Regime shift | Model decay | Regime estimator, monitoring, retraining policy |
| Survivorship bias | Inflated universe performance | Eligibility-interval registry (WP-02) |
| Single workstation failure | Data loss | Backup; cloud redundancy for critical data |
| Scope creep | Nothing finishes | Milestone gates; P0–P3 discipline; owner checkpoints |
| aggTrade WS stream silent | No live tick data via that stream | Use archive for history; `@trade` stream verified flowing if live ticks needed |

**Largest bottleneck**: data accumulation time for microstructure
research — cannot be accelerated, only started earlier.

---

## 14. Owner decisions vs engineering decisions

### Requires owner approval
| Decision | Why owner-only | Recommended default |
|---|---|---|
| **Storage budget** | ~110–220 GB/year live recording + backfills vs 64GB free | Approve ~500GB–1TB budget; parquet+retention policy |
| **Compute upgrade / cloud** | Temporal models, production trading | Defer until GBM baseline exhausted |
| **Exchange/account access** | Trading requires API keys | Paper/testnet first; no live without authorization |
| **Live trading authorization** | Real money risk | Only after ≥6mo paper + sign-off |
| **Risk appetite / limits** | Leverage, drawdown, position sizes | Conservative start (2x cap, 20% MDD) |
| **Research governance** | Trial budget, PREREG amendments, §11.2 definition | Approve §11.2 proposal; keep 39 slots |
| **Strategic scope** | Which families first | Regime+forecast infra first; then 1m–15m momentum/mean-reversion; carry only with new mechanism |
| **Cross-exchange timing** | Bybit/other exchanges | After Binance system validated |
| **Options inclusion** | New asset class | Deferred until core proven |
| **depth5 vs depth20** | Bandwidth/storage tradeoff | Start depth5 (measured 1.13 GB/day); upgrade if research justifies |

### Engineering recommendations (no owner approval needed)
- Code organization, data schemas, recorder reliability design,
  testing strategy, storage benchmarks, internal interfaces,
  dependency ordering within WPs.

---

## 15. Recommended first implementation package (rev 2)

**Package FIP-1: Foundation + Data Accumulation** (4–6 weeks)

| WP | Task | Effort | Parallel? |
|---|---|---|---|
| WP-01 | Extend live_recorder: top-20 perps, liquidations+bookTicker+depth5 (non-backfillable only) | 1–2 wk | Yes (background recording) |
| WP-02 | Universe registry + eligibility intervals | 1 wk | Yes |
| WP-03 | OI archive ingestion pipeline | 1 wk | Yes |
| WP-04 | Feature-store skeleton (multi-res, PIT, versioned) | 1–2 wk | Yes |
| WP-07 | Portfolio backtester design (parity-tested architecture) | 2 wk | Yes (design on existing data) |

**WP-05 (funding carry) removed** — already tested (§2.4); any new
carry work needs a new mechanism and waits for WP-04/WP-07.

**Acceptance for FIP-1**:
- Recorder running top-20 symbols (liq+bookTicker+depth5), ≥7 days,
  gap rate <1%, storage within budget.
- Universe registry tracking ≥50 instruments with eligibility history.
- OI archive ingested for BTC/ETH/SOL, validated, conventions
  classified, report wording corrected.
- Feature store producing versioned, PIT-enforced features.
- Portfolio backtester designed with parity-test plan; core engine
  skeleton implemented.

**What FIP-1 unlocks**: data accumulation (irreversible), survivorship
control, OI history, feature infrastructure, portfolio-engine
foundation, validated end-to-end pipeline.

**What FIP-1 does NOT do**: no microstructure models (needs months of
data), no paper trading, no live trading, no registered experiments.

---

*End of draft rev 2. Awaiting owner review. Nothing herein is
approved, frozen, registered, or authorized for implementation.*

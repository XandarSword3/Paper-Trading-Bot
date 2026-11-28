# Autonomous Crypto Trading System — Technical Development Plan DRAFT

**Status**: DRAFT for owner review. Nothing herein is approved, frozen,
registered, or authorized for implementation. No trial budget consumed.
No model trained. No experiment run. No data downloaded.

**Date**: 2026-11-28 · **Branch**: `phase2-v4-adapter` @ `0f45abe`
**Governing architecture**: `v4_system/AUTONOMOUS_CRYPTO_TRADING_SYSTEM_SPEC_DRAFT.md`
(commit `0f45abe`, local only — not pushed).

---

## 1. Executive summary

This plan converts the architecture draft into a dependency-aware,
prioritized development path toward the autonomous multi-asset ML
trading system. The central tension — **data accumulation takes months,
but research and infrastructure cannot wait** — is resolved by running
data collection in parallel with infrastructure and early research on
data that already exists.

**Recommended first implementation package** (details in §15):
1. Extend `live_recorder` to top ~20 perp symbols: add `aggTrade`,
   `bookTicker`, and depth streams; keep existing liquidation streams.
2. Build the market-universe registry with eligibility intervals
   (survivorship control) — pure metadata work, no trading risk.
3. Build the OI archive ingestion pipeline (validated, checksum-verified,
   provenance-preserving) — unlocks 5 years of OI history.
4. Build the feature-store skeleton (multi-resolution, PIT-enforced,
   versioned) on existing klines.
5. First research family: **funding/basis carry** — data exists today,
   low execution complexity, days-long holding period, untested as carry.

Everything else sequences behind these, with explicit acceptance gates.

---

## 2. Verified repository baseline

Inspected 2026-11-28, `phase2-v4-adapter` @ `0f45abe`, clean tree.
Commit `0f45abe` confirmed present (`git cat-file -t` → `commit`).
Note: `0f45abe` and `2becfe0` are **local-only**; `origin/phase2-v4-adapter`
is at `16e0673`. The architecture doc states inspection commit `2becfe0`
— accurate (it was HEAD when written); it was then committed as `0f45abe`.
No discrepancy beyond the expected commit-after-inspection offset.

### 2.1 Component status (verified by file inspection, not docs)

| Component | Path | Status | Evidence |
|---|---|---|---|
| Data engine | `v4_system/data_engine/` | **Implemented** | 17 modules: store, klines, funding, OI, contract_specs, margin_tiers, pit, manifest, fetch_phase1, final_validation, live_recorder, live_watchdog, agg_trades, binance_client, refresh, availability_probe |
| PIT controls | `data_engine/pit.py` | **Implemented, well-tested** | Unified `as_of`, unfinished-candle exclusion, `available_at` semantics; `tests/test_pit_integrity.py` (21KB) |
| Live recorder | `data_engine/live_recorder.py` | **Implemented, young** | Binance `!forceOrder@arr` (sampled), `bookTicker` (3 syms), Bybit `allLiquidation`; JSONL/minute, atomic writes, gap log, reconnect; ~93 MB live data after ~1–2 days |
| Live watchdog | `data_engine/live_watchdog.py` | **Implemented** | Health monitoring |
| V4↔QR adapter | `adapters/quant_research_data.py` | **Implemented** | PIT-labeled dataset loading, schema mapping |
| Backtest engine | `quant_research/core/engine.py` | **Implemented, single-asset** | Next-open fills, 14/18 bps costs, funding, 1m sub-bar stops; NOT portfolio/multi-asset |
| Metrics/stats | `quant_research/core/metrics.py`, `statistical_tests.py` | **Implemented** | DSR, walk-forward, Sharpe, random baselines |
| Strategies | `quant_research/strategies/` | **Implemented, manual** | Turtle, indicator_stack, parameter_grid — all rule-based |
| Trial governance | `trials_registry.py`, `trials_log.csv`, `PREREG.md` | **Implemented** | Append-only, budget 1/40 used, N=1,480 |
| Family 1 runner | `run_family1.py` | **Implemented** | 5-fold WF, NO EDGE preserved |
| Tests | `tests/` | **Implemented** | 38 test files, 359 tests passing |
| **ml_engine** | `v4_system/ml_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **execution_engine** | `v4_system/execution_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **risk_engine** | `v4_system/risk_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **strategy_engine** | `v4_system/strategy_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **backtesting** (v4) | `v4_system/backtesting/` | **EMPTY SHELL** | Only `__init__.py` |
| **microstructure_engine** | `v4_system/microstructure_engine/` | **EMPTY SHELL** | Only `__init__.py` |
| **macro_engine / news_engine** | `v4_system/` | **EMPTY SHELLS** | Only `__init__.py` |
| **paper_trading** | `v4_system/paper_trading/` | **EMPTY SHELL** | Only `__init__.py` |

### 2.2 Data inventory (verified)

- **Historical klines**: 60 parquet datasets, BTC/ETH/SOL perp+spot,
  1m/5m/15m/1h/4h/1d, 2023-01→2026-11-28, manifest+sha256, validated
  (1 PASS / 63 INCOMPLETE / 0 FAIL — INCOMPLETE due to unknown
  `available_at`, not structural defects).
- **Funding**: history + interval metadata.
- **OI**: REST ~30 days only; archive (2020-09/2021-12→present, 5m)
  investigated but NOT ingested; timestamp-convention change 2026-06-25
  unresolved.
- **Live**: ~93 MB, ~1–2 days, 3 symbols bookTicker + liquidations.
- **Missing**: tick/trade history, order-book depth, multi-asset
  universe, options, news/macro, cross-exchange market data.

### 2.3 Hardware (verified)

- CPU: 13th Gen Intel i7-13650HX, **14 cores**
- RAM: **23.7 GB**
- Disk D: free: **64.4 GB**
- OS: Windows, Python 3.11

This is a **development workstation, not a production server**. It is
adequate for research, feature engineering, and moderate backtesting.
It is NOT adequate for: high-frequency tick collection at scale,
training large temporal models, or running a production trading system.
Those require either a VPS/cloud instance or significant hardware
upgrade — an owner decision (§14).

---

## 3. Complete development workstreams

### A. Market discovery and universe management
- Exchange market discovery (Binance spot+futures, Bybit instruments).
- Instrument metadata registry (tick size, step size, min notional,
  leverage, status).
- Liquidity/volume/spread measurement and eligibility filters.
- Historical eligibility intervals (survivorship-bias control).
- Dynamic promotion/demotion as characteristics change.
- Delisting/new-listing handling.

### B. Market-data collection and storage
- Expand recorder: aggTrade, bookTicker (more symbols), depth,
  liquidations (more symbols/exchanges), klines (more symbols).
- Historical backfills: klines (Binance archive), OI archive,
  aggTrades (large, selective).
- Storage: partitioned parquet vs time-series DB (see §8).
- Timestamp precision, event vs receipt time, PIT, sequence integrity.
- Reconnection, recovery, gap handling, monitoring, validation.

### C. Feature and market-state infrastructure
- Multi-resolution feature computation (event-driven + rolling).
- Cross-asset features, regime/vol estimates, microstructure features.
- Funding/positioning features.
- Feature versioning, reproducibility, missing-data handling,
  timestamp alignment, PIT enforcement.

### D. ML research and model infrastructure
- Modular components: regime, return/vol forecast, order-flow,
  opportunity detection, ranking, strategy selection, execution,
  portfolio allocation.
- Per-component: problem, data, output, model family, eval, data
  readiness, development timing.
- Model versioning, deployment, monitoring, rollback.

### E. Opportunity and strategy research
- Organized framework for 11+ opportunity families.
- Per-family: data, frequency, execution complexity, costs, capital,
  feasibility.
- Distinguish opportunity family from ML component (many-to-many).
- Prioritization by research value, not ease of coding.

### F. Portfolio decision and risk infrastructure
- Opportunity ranking, capital allocation, position sizing, leverage.
- Concurrent positions, correlation, partial entries/exits, reversals,
  hedging, capital conflicts, turnover, drawdown, emergency controls.
- Portfolio-level backtesting (existing engine insufficient).

### G. Execution and paper trading
- Exchange connectivity, order management, partial fills,
  cancel/replace, maker/taker, book-aware execution, slippage/impact,
  funding, reconciliation, outages, risk controls, monitoring, audit.
- Progression: historical sim → live-data paper → (future) live.

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

**Critical path**: B (data) → C (features) → D (ML) → E (research) →
F (portfolio) → G (execution). But A, B2, C2, F2 can proceed in
parallel with early D/E work on existing data.

**Hard dependencies**:
- Order-flow/microstructure ML (D) requires tick/book data (B) — months.
- Portfolio backtesting (F2) requires multi-asset data (B) + features (C).
- Paper trading (G) requires portfolio engine (F) + execution sim.
- Live trading requires paper-trading validation + owner authorization.

**Soft dependencies** (can start early):
- Universe registry (A) needs only exchangeInfo API — start now.
- OI archive ingestion (B2) needs only HTTP + validation — start now.
- Feature store skeleton (C2) needs only existing klines — start now.
- Funding carry research (E) needs only existing funding+klines — start now.

---

## 5. Immediate vs parallel vs deferred tasks

### Immediate (start now, current code+data)
| Task | Why immediate |
|---|---|
| Extend live_recorder (more symbols, aggTrade, depth) | Every day without recording is permanently lost data |
| Universe registry + eligibility intervals | Pure metadata; enables survivorship control; no trading risk |
| OI archive ingestion pipeline | Unlocks 5yr OI history; validation gates already designed |
| Feature-store skeleton on existing klines | Foundation for all ML; reusable across families |
| Funding carry research (first experiment) | Data exists; low complexity; untested as carry |

### Parallel (run alongside immediate)
| Task | Why parallel |
|---|---|
| Historical kline backfill for expanded universe | Independent of recorder; uses Binance archive |
| aggTrades selective download (BTC/ETH recent) | Large but independent; enables microstructure research later |
| Portfolio backtester design (extend or new engine) | Can be designed/built on existing single-asset klines |
| Regime estimator (unsupervised, on existing klines) | No new data needed; foundational for strategy gating |
| Data-quality/validation framework extension | Generalizes item-8 validator to new streams |

### Data-dependent (cannot validate until data accumulates)
| Task | Data needed | Wait time |
|---|---|---|
| Order-flow/microstructure models | Tick/trade/book history | Months of recording |
| Liquidation-cascade strategies | Liquidation history | Months of recording |
| Spread/dislocation strategies | bookTicker/depth history | Months of recording |
| Cross-exchange arbitrage | Multi-exchange synchronized data | After recorder expansion |
| High-frequency (sub-minute) strategies | 1s bars from trades | After trade data + aggregation |

### Infrastructure-dependent (requires another component first)
| Task | Depends on |
|---|---|
| Paper trading | Portfolio engine + execution sim + live data |
| Live trading | Paper-trading validation + owner authorization |
| Learned strategy selection | Multiple validated opportunity families |
| Learned portfolio allocation | Validated opportunity ranking + risk framework |

### Deferred (valuable eventually, not justified yet)
| Task | Why deferred |
|---|---|
| Options strategies | No options data/feed; infrastructure absent |
| Market making | Requires book depth + capital + low latency; far future |
| Transformer/temporal deep models | Insufficient data; GBM/linear unexhausted |
| Reinforcement learning | High risk/complexity; premature before supervised baseline |
| News/macro features | No data source; low priority vs market data |

---

## 6. Prioritized work packages

Priority = strategic value × capabilities unlocked ÷ (difficulty + cost
+ data wait). P0 = start now, P1 = next, P2 = after P0/P1, P3 = deferred.

| ID | Work package | Priority | Complexity | Data dep | Unlocks |
|---|---|---|---|---|---|
| WP-01 | Extend live_recorder: top ~20 perps, aggTrade+bookTicker+depth | **P0** | Medium | None (starts collecting) | All microstructure research; months of data |
| WP-02 | Universe registry + eligibility intervals | **P0** | Low-Med | exchangeInfo API | Survivorship control; dynamic universe |
| WP-03 | OI archive ingestion pipeline | **P0** | Medium | HTTP (archive) | 5yr OI features; positioning research |
| WP-04 | Feature-store skeleton (multi-res, PIT, versioned) | **P0** | Medium | Existing klines | All ML components |
| WP-05 | Funding carry research (first experiment) | **P0** | Low-Med | Existing funding+klines | First ML/trading result; validates pipeline |
| WP-06 | Historical kline backfill (expanded universe) | **P1** | Low | Binance archive | More instruments for research |
| WP-07 | Portfolio backtester (multi-asset, multi-position) | **P1** | High | Existing klines (single-asset OK for design) | Portfolio research; realistic evaluation |
| WP-08 | Regime estimator (unsupervised) | **P1** | Medium | Existing klines | Strategy gating; context features |
| WP-09 | aggTrades selective download + 1s bar builder | **P1** | Medium | Binance archive | Microstructure research foundation |
| WP-10 | Return/vol forecaster (GBM baseline, 1m–1h) | **P1** | Medium | WP-04 + klines | Core predictive component |
| WP-11 | Opportunity ranker (rule-based first) | **P2** | Medium | WP-05, WP-10 results | Multi-opportunity selection |
| WP-12 | Execution simulator (book-aware, impact) | **P2** | High | WP-01 data + WP-07 | Realistic execution evaluation |
| WP-13 | Paper-trading engine | **P2** | High | WP-07, WP-12, live data | Prospective validation |
| WP-14 | Order-flow/microstructure models | **P2** | High | WP-01 (months of data) | Fast strategies |
| WP-15 | Learned strategy selection / allocation | **P3** | High | WP-11 validated | Autonomous adaptation |
| WP-16 | Cross-exchange expansion | **P3** | High | WP-01 + Bybit recorder | Arbitrage; redundancy |
| WP-17 | Options integration | **P3** | Very high | No data source | New asset class |

**Why P0 items deserve top priority**:
- WP-01: data accumulation is irreversible — every day delayed is
  permanently lost. Highest strategic value per effort.
- WP-02: cheap, unblocks survivorship control, needed before any
  multi-asset backtest.
- WP-03: unlocks 5 years of OI history; validation design already done.
- WP-04: foundation for all ML; can be built on existing data now.
- WP-05: proves the research pipeline end-to-end on real data; lowest
  risk first experiment.

---

## 7. Concrete milestones and acceptance criteria

### M1 — Expanded market discovery and eligibility registry
- **Deliverable**: `universe_registry` module; exchangeInfo polling;
  eligibility filters; eligibility-interval history; promotion/demotion.
- **Acceptance**: registry tracks ≥50 instruments with eligibility
  intervals; survivorship-bias test passes (delisted instrument
  reconstructable); unit tests for filters.
- **Unlocks**: dynamic universe; survivorship-controlled backtests.

### M2 — Reliable multi-symbol data collection
- **Deliverable**: extended live_recorder (≥20 perps: aggTrade,
  bookTicker, depth, liquidations); gap logging; health monitoring.
- **Acceptance**: ≥7 days continuous recording for ≥10 symbols; gap
  rate <1%; data validation passes; storage growth within budget.
- **Unlocks**: microstructure research; realistic execution sim.

### M3 — Trade and order-book data pipelines
- **Deliverable**: aggTrade→1s/1m bar builder; bookTicker spread
  features; depth snapshot/diff ingestion with sequence integrity.
- **Acceptance**: 1s bars reproducible from trades; book state
  reconstructable; sequence-gap detection works.
- **Unlocks**: order-flow features; microstructure models.

### M4 — Historical and prospective data validation
- **Deliverable**: extended validation framework (item-8 generalized);
  OI archive ingested + validated; aggTrades spot-checked.
- **Acceptance**: all datasets classified PASS/INCOMPLETE/FAIL with
  provenance; OI archive timestamp conventions classified per file.
- **Unlocks**: trustworthy features; OI research.

### M5 — Reproducible feature infrastructure
- **Deliverable**: feature store (multi-resolution, PIT-enforced,
  versioned, missing-data policy); feature registry.
- **Acceptance**: features reproducible from raw data; PIT tests pass;
  feature versioning works; cross-asset alignment correct.
- **Unlocks**: all ML components; consistent research.

### M6 — Multi-asset portfolio simulation
- **Deliverable**: portfolio-level backtester (multi-position,
  multi-asset, correlation, capital allocation, partial exits).
- **Acceptance**: parity test vs single-asset engine on single-asset
  case; portfolio constraints enforced; realistic costs.
- **Unlocks**: portfolio research; realistic strategy evaluation.

### M7 — Initial ML component infrastructure
- **Deliverable**: ml_engine with regime estimator + return/vol
  forecaster (GBM baseline); training/eval pipeline; model registry.
- **Acceptance**: models train on features; walk-forward eval works;
  DSR integration; model versioning.
- **Unlocks**: predictive components; first ML experiment results.

### M8 — Opportunity ranking and allocation
- **Deliverable**: rule-based opportunity ranker; capital allocator;
  position sizer; risk gate (exposure/correlation/concentration).
- **Acceptance**: ranker scores candidates; allocator respects limits;
  risk gate vetoes violations; backtested on portfolio engine.
- **Unlocks**: multi-opportunity trading; autonomous selection.

### M9 — Execution simulation and paper trading
- **Deliverable**: execution simulator (order types, timing, impact);
  paper-trading engine (live data, order management, reconciliation).
- **Acceptance**: paper trades fill realistically; reconciliation
  matches exchange; ≥30 days paper trading without critical bugs.
- **Unlocks**: prospective validation; live-readiness assessment.

### M10 — Integrated autonomous decision loop
- **Deliverable**: full pipeline: universe → data → features → models →
  ranking → allocation → execution → risk, running live on paper.
- **Acceptance**: ≥6 months paper trading; performance vs benchmarks;
  all risk controls functional; audit trail complete.
- **Unlocks**: evidence for live-trading decision (owner authorization).

---

## 8. Data-collection and storage strategy

### 8.1 What to collect immediately (once authorized)
Every day of delay permanently loses data for streams with no
historical source. Priority order:

| Stream | Symbols | Historical source? | Priority | Rationale |
|---|---|---|---|---|
| Liquidations (Binance+Bybit) | Top ~20 perps | ❌ None | **P0** | Irreplaceable; cascade research |
| bookTicker | Top ~20 perps | ❌ None | **P0** | Spread/dislocation research; execution sim |
| aggTrade | Top ~20 perps | ⚠️ Archive (large) | **P0** | Trade flow; 1s bars; microstructure |
| Depth (snapshot+diff or @100ms) | Top ~10 perps | ❌ None | **P1** | Order-book imbalance; high bandwidth |
| Klines (all TFs) | Top ~50 perps | ✅ Archive | P1 | Universe expansion; backfill possible |
| Funding/OI | Top ~50 perps | ✅ Archive/REST | P1 | Positioning features |

### 8.2 Storage growth estimates (assumptions shown)

**Assumptions**:
- Top 20 perps, aggTrade: ~10–50 trades/sec/symbol avg (BTC higher,
  alts lower). Say ~20/sec avg × 20 syms × 86,400 s/day = ~34.5M
  trades/day. Each trade ~100 bytes JSON → ~3.5 GB/day raw, ~0.5–1 GB
  compressed parquet. **~200–350 GB/year** for 20 symbols.
- bookTicker: ~1–10 updates/sec/symbol. Say ~5/sec × 20 × 86,400 =
  ~8.6M/day × ~80 bytes = ~0.7 GB/day raw, ~100 MB compressed.
  **~35 GB/year**.
- Depth @100ms: 10 levels × 2 sides × ~20 bytes × 10/sec × 20 syms =
  ~8.6 GB/day raw — **very large**; recommend depth for top 5–10 only,
  or snapshot+diff (diffs smaller). **~1–3 TB/year** if full depth.
- Liquidations: sparse (events, not continuous) — negligible (<1 GB/yr).
- Klines: existing 60 datasets ~small; expanding to 50 symbols × 6 TFs
  ~few GB/year.

**Total realistic year-1 storage**: ~300–500 GB without full depth;
~1–3 TB with full depth for many symbols. **Current disk: 64.4 GB
free — insufficient for aggressive collection.** Storage expansion or
cloud storage is an owner decision (§14).

### 8.3 Storage technology recommendation
- **Klines, funding, OI, features**: partitioned parquet (existing
  store.py pattern) — works well, columnar, compresses well.
- **Tick/trade/book data**: parquet partitioned by date/symbol works
  for batch research but is poor for high-frequency online queries.
  **ClickHouse or QuestDB** justified when: (a) >100 GB tick data,
  (b) need sub-second queries, (c) concurrent online+offline access.
  Until then, parquet + DuckDB for analytics is sufficient and simpler.
- **Recommendation**: start with partitioned parquet + DuckDB; migrate
  to ClickHouse/QuestDB when tick data exceeds ~100 GB or online
  low-latency queries become necessary. Engineering decision, not
  owner decision — but storage budget is owner decision.

### 8.4 Timestamp and PIT for high-frequency data
- Preserve exchange event timestamp (ms) AND receipt timestamp (µs).
- `available_at` = receipt time for live data; NaT for bulk downloads.
- Order-book: sequence-number gap detection; missing diffs invalidate
  book state (must resync from snapshot).
- Existing `pit.py` generalizes: event-time cutoff + availability
  cutoff + unfinished-interval exclusion.

---

## 9. ML development strategy

### 9.1 Component-by-component plan

| Component | Problem | Data | Output | Model family | Data today? | Timing |
|---|---|---|---|---|---|---|
| Regime estimator | Classify market state | Multi-TF returns/vol/corr | Regime label + confidence | GMM/k-means, HMM, GBM classifier | ✅ Yes | **Now** (WP-08) |
| Return/vol forecaster | Predict returns/vol/quantiles | Multi-TF features | Return dist, vol, quantiles | LightGBM quantile, GARCH, linear | ✅ Yes | **Now** (WP-10) |
| Order-flow model | Predict short-term impact/imbalance | Trades+book | Impact prob, imbalance signal | GBM, shallow temporal | ❌ Months | After WP-01 data |
| Opportunity detector | Score candidate opportunities | Features+labels | Opportunity score | GBM, ranking | ⚠️ Partial | After WP-05/10 |
| Opportunity ranker | Rank across instruments/families | Opportunity scores | Ranked list | LambdaRank, rule-based first | ⚠️ Partial | After WP-11 |
| Strategy selector | Choose best family for regime | Regime+scores | Strategy choice | Contextual bandit, rules first | ❌ | Later |
| Execution model | Order type/timing/aggressiveness | Book+trades | Execution plan | Supervised fill models, RL later | ❌ | After WP-12 |
| Portfolio allocator | Capital allocation/sizing/hedging | Positions+correlations | Allocation vector | Constrained opt, risk-parity, RL later | ⚠️ Partial | After WP-07 |

### 9.2 Model family guidance
- **Start**: LightGBM (tabular), linear models (baseline/calibration),
  unsupervised clustering/HMM (regime).
- **Add when justified**: temporal models (TCN/GRU) for sequence data;
  RL for execution/portfolio — only after supervised baselines fail.
- **Never without evidence**: transformers, deep RL — no demonstrated
  need; high complexity/risk.

### 9.3 Model lifecycle
- Version every model (training data hash, features, hyperparams,
  metrics).
- Evaluate on walk-forward OOS only; DSR for multiple testing.
- Deploy to paper trading first; monitor live performance vs backtest.
- Rollback on performance degradation or data drift.
- Registry: extend `trials_registry.py` pattern to model registry.

---

## 10. Portfolio and execution infrastructure strategy

### 10.1 Portfolio backtester: extend or new?
- **Existing `quant_research/core/engine.py`**: single-asset,
  bar-driven, next-open fills, funding, 1m sub-bar stops. Well-tested
  for single-instrument strategies.
- **Portfolio requirements**: multi-position, multi-asset, concurrent,
  correlation-aware, capital allocation, partial exits, hedging,
  turnover limits, portfolio drawdown.
- **Recommendation**: **build a new portfolio-level engine** that
  reuses the existing engine's cost/fill/funding primitives but adds
  portfolio state management. Extending the single-asset engine would
  create an unmaintainable hybrid. The new engine lives in
  `v4_system/backtesting/` (currently empty shell — intended location).
- **Acceptance**: parity test — portfolio engine on single-asset
  single-position case must match existing engine results exactly.

### 10.2 Execution simulation progression
1. **Historical sim** (current): next-open fills, fixed slippage.
2. **Book-aware sim**: use bookTicker/depth to model spread crossing,
   market impact (square-root law), partial fills.
3. **Paper trading**: live data, realistic order management, exchange
   latency simulation.
4. **Live** (future, owner-authorized): real orders, real fills.

### 10.3 Paper-trading engine requirements
- Exchange connectivity (testnet first where available).
- Order management: place, cancel, replace, partial fills.
- Maker/taker execution, order-book-aware timing.
- Position reconciliation (internal vs exchange).
- Funding application, margin tracking.
- Outage/rejection handling, retry logic.
- Risk controls: max position, max exposure, kill-switch.
- Monitoring: P&L, exposure, latency, fill quality, audit logs.

---

## 11. Research governance requirements

### 11.1 What requires trial registration
- Any **strategy experiment** that tests a trading hypothesis and
  consumes research budget (per PREREG IV.4).
- Each registered config = architecture × horizon × feature set ×
  threshold combination.
- Current budget: 1/40 used, 39 remaining. N_trials = 1,480.

### 11.2 What does NOT require registration
- **Infrastructure development**: data collection, feature stores,
  portfolio engine, execution sim, paper trading — these are
  engineering, not experiments.
- **Unsupervised exploration**: regime clustering, data-quality checks,
  descriptive statistics — no trading hypothesis tested.
- **Model development without trading evaluation**: training a return
  forecaster and measuring RMSE is not a trading experiment; only when
  it's converted to a trading strategy and evaluated on P&L does it
  become a registered trial.
- **Paper trading**: prospective validation, not historical experiment.

### 11.3 Governance constraints
- PREREG is binding; amendments required for new experiment families.
- Append-only trial registry; no deletions.
- DSR with correct N_trials; no p-hacking.
- Walk-forward only; no random splits.
- November 2026 holdout exposed — all history ≤2026-11-28 is
  development data; future-only holdout per PREREG IV.3.
- Cost model: 14/18 bps RT base/stress; funding applied.
- No silent budget increases.

---

## 12. Resource and feasibility analysis

### 12.1 Against current hardware (i7-13650HX, 14 cores, 23.7GB RAM, 64GB free)

| Task | Feasible now? | Notes |
|---|---|---|
| Live recording (20 syms, aggTrade+bookTicker) | ✅ Yes | Network-bound, not CPU; ~3.5GB/day storage |
| Kline/funding/OI backfill | ✅ Yes | Existing downloader works |
| OI archive ingestion (~5,900 files) | ✅ Yes | ~70MB compressed; hours of download |
| Feature engineering on klines | ✅ Yes | Pandas/polars sufficient |
| GBM training (LightGBM) | ✅ Yes | Fast on tabular data |
| Portfolio backtesting (daily/hourly) | ✅ Yes | Manageable compute |
| Tick-level feature engineering | ⚠️ Marginal | 34M trades/day × 20 syms = large; needs efficient polars/DuckDB |
| Temporal deep model training | ❌ No | Needs GPU or cloud; defer |
| High-frequency online inference | ❌ No | Needs low-latency infra; defer |
| Full-depth order-book collection | ⚠️ Marginal | Bandwidth/storage heavy; limit to top symbols |

### 12.2 When additional resources justified
- **Storage >200GB tick data**: cloud storage or larger disk.
- **Temporal model training**: GPU instance (cloud) or local GPU.
- **Production trading**: VPS/cloud with low-latency exchange
  connectivity; current workstation unsuitable.
- **Multi-exchange collection**: more bandwidth; possibly dedicated
  connection.

### 12.3 Feasibility verdict
The **development and research platform** is feasible on current
hardware for: data collection (network-bound), feature engineering,
GBM-based ML, portfolio backtesting at hourly/daily resolution, and
paper trading. **Not feasible** for: tick-level deep learning,
high-frequency online trading, production deployment. Those are
future infrastructure decisions, not current blockers.

---

## 13. Risks and bottlenecks

| Risk | Impact | Mitigation |
|---|---|---|
| Data accumulation too slow | Months before microstructure research | Start recording NOW; prioritize P0 streams |
| Storage exhaustion (64GB free) | Collection stops | Storage expansion decision; parquet compression; retention policy |
| NO EDGE prior (1,480 trials) | ML may also find nothing | Realistic expectations; funding carry first (untested); diverse families |
| Overfitting complex models | False positives | DSR, walk-forward, trial registry, simple-first principle |
| Execution reality gap | Backtest ≠ live | Book-aware sim, paper trading, conservative cost model |
| Regime shift | Model decay | Regime estimator, continuous monitoring, retraining policy |
| Survivorship bias | Inflated universe performance | Eligibility-interval registry (WP-02) |
| Single workstation failure | Data loss | Backup strategy; cloud redundancy for critical data |
| Scope creep | Nothing finishes | Milestone gates; P0/P1/P2/P3 discipline; owner checkpoints |

**Largest bottleneck**: data accumulation time for microstructure
research. Cannot be accelerated — only started earlier.

---

## 14. Owner decisions vs engineering decisions

### Requires owner approval
| Decision | Why owner-only | Recommended default |
|---|---|---|
| **Storage budget** | 300GB–3TB/year for tick data; 64GB free now | Approve ~500GB–1TB budget; start parquet, migrate if needed |
| **Compute upgrade / cloud** | Temporal models, production trading need more | Defer until GBM baseline exhausted; cloud GPU when needed |
| **Exchange/account access** | Trading requires API keys, permissions | Paper/testnet first; no live without explicit authorization |
| **Live trading authorization** | Real money risk | Only after ≥6mo paper + owner sign-off |
| **Risk appetite / limits** | Leverage, max drawdown, position sizes | Start conservative (2x leverage cap, 20% MDD, as Family 1) |
| **Research governance changes** | Trial budget, PREREG amendments | Keep 39 remaining; amend PREREG for new families |
| **Strategic scope** | Which opportunity families first | Funding carry + 1m momentum first (data exists) |
| **Cross-exchange timing** | Bybit/other exchanges | After Binance system validated |
| **Options inclusion** | New asset class, new infrastructure | Deferred until core proven |

### Engineering recommendations (no owner approval needed)
- Code organization (module structure in empty shells).
- Data schemas (parquet partitioning, feature store format).
- Recorder reliability design (reconnect, gap handling, buffering).
- Testing strategy (unit, integration, parity tests).
- Storage benchmarks (parquet vs DuckDB vs ClickHouse).
- Internal interfaces (feature store API, model registry API).
- Dependency ordering within work packages.

---

## 15. Recommended first implementation package

**Package FIP-1: Foundation + Data Accumulation** (4–6 weeks)

| WP | Task | Effort | Parallel? |
|---|---|---|---|
| WP-01 | Extend live_recorder: top 20 perps, aggTrade+bookTicker+depth(top5) | 1–2 wk | Yes (background recording) |
| WP-02 | Universe registry + eligibility intervals | 1 wk | Yes |
| WP-03 | OI archive ingestion pipeline | 1 wk | Yes |
| WP-04 | Feature-store skeleton (multi-res, PIT, versioned) | 1–2 wk | Yes |
| WP-05 | Funding carry research (first experiment) | 1 wk | After WP-04 |

**Acceptance for FIP-1**:
- Recorder running ≥20 symbols, ≥7 days, gap rate <1%.
- Universe registry tracking ≥50 instruments with eligibility history.
- OI archive ingested for BTC/ETH/SOL, validated, timestamp conventions classified.
- Feature store producing versioned, PIT-enforced features from klines.
- Funding carry experiment registered, run, reported (any verdict).

**What FIP-1 unlocks**: data accumulation (irreversible), survivorship
control, OI history, feature infrastructure, first research result,
validated end-to-end pipeline.

**What FIP-1 does NOT do**: no microstructure models (needs months of
data), no portfolio engine, no paper trading, no live trading.

---

*End of draft. Awaiting owner review. Nothing herein is approved,
frozen, registered, or authorized for implementation.*

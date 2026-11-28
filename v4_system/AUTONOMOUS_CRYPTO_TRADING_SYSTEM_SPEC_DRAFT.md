# Autonomous Crypto Trading System — Architecture Specification DRAFT

**Status**: DRAFT for owner review. Nothing in this document is approved,
frozen, registered, or authorized for implementation. No trial budget is
consumed. No model has been trained. No experiment has been run.

**Date**: 2026-11-28 · **Branch**: `phase2-v4-adapter` @ `2becfe0`
**Supersedes (partially)**: `v4_system/ML_RESEARCH_SPEC_DRAFT.md` — see §12.

---

## 1. Executive summary and actual project objective

### 1.1 The actual objective

Build an **autonomous, ML-driven cryptocurrency trading system** that
competes with — and is evaluated against — human discretionary trading
on measurable net performance. The system must exploit what automation
gives that discretion cannot: continuous multi-market monitoring,
multi-timeframe analysis, consistent execution, simultaneous position
management, and adaptation to changing market conditions.

This is explicitly **not** "another technical-indicator strategy with an
ML label attached." The prior research program (Program D + Phase 2
Families) tested manually parameterized directional hypotheses and
found NO EDGE across ~1,480 registered trials. That result is evidence
about *those hypotheses*, not about ML — but it is also a warning: any
new architecture must be evaluated with the same statistical rigor
(Dsr, random baselines, cost stress, walk-forward) and must not assume
that sophistication implies edge.

### 1.2 What this document is

An honest architecture and planning document. It distinguishes what
exists and is verified, what is proposed, what requires research, what
requires data that does not yet exist, and what requires explicit owner
approval. It does not authorize implementation.

---

## 2. Verified current repository state and capabilities

Inspected 2026-11-28 on `phase2-v4-adapter` @ `2becfe0`, clean tree.

### 2.1 What exists and works (verified by code inspection + test suite)

| Component | Location | Verified capability |
|---|---|---|
| Historical kline store | `v4_system/data_engine/store.py`, `klines.py`, `fetch_phase1.py` | 60 parquet datasets: perp+spot candles 1m/5m/15m/1h/4h/1d for BTC/ETH/SOL, manifest with sha256, gap detection, atomic writes |
| Funding data | `data_engine/funding.py`, `funding_intervals.py` | Historical funding settlements + interval metadata |
| OI (REST) | `data_engine/open_interest.py` | ~30-day history only (Binance cap); 5m/15m/1h/4h/1d resamples |
| Contract/margin metadata | `contract_specs.py`, `margin_tiers.py` | Current snapshots; historical effective dates UNKNOWN |
| PIT controls | `data_engine/pit.py` | Unified `as_of` cutoff, unfinished-candle exclusion, `available_at` semantics (NaT for bulk history), strict/non-strict modes — **well-designed, reusable** |
| Live recorder | `data_engine/live_recorder.py` | Binance `!forceOrder@arr` (SAMPLED liquidations), `bookTicker` top-of-book (3 symbols), Bybit `allLiquidation` (fuller); JSONL per-minute files, atomic writes, gap logging, auto-reconnect |
| Backtest engine | `quant_research/core/engine.py` | Next-open fills, taker fee+slippage cost model (14/18 bps RT), funding settlement, 1m sub-bar stop/target resolution, pyramid invariant — **audited, single-asset, bar-driven** |
| Metrics/statistics | `quant_research/core/metrics.py`, `statistical_tests.py`, `walk_forward.py` | DSR, walk-forward, per-trade Sharpe, random-entry baselines |
| Strategy implementations | `quant_research/strategies/` | Turtle/Donchian, indicator stack, parameter-grid — all manually parameterized |
| V4↔QR adapter | `v4_system/adapters/quant_research_data.py` | Dataset loading with explicit `PIT_LABEL`, schema mapping |
| Trial governance | `v4_system/trials_registry.py`, `trials_log.csv`, `PREREG.md` | Append-only registry, budget enforcement (1/40 used), DSR N=1,480 |
| Family 1 runner | `v4_system/run_family1.py` | 5-fold WF, 24 config-assets, NO EDGE result preserved |
| Validation reports | `v4_system/reports/`, `data_engine/final_validation.py` | Corrected item-8 validator (1 PASS / 63 INCOMPLETE / 0 FAIL) |

### 2.2 What exists but is incomplete or uncertain

- **Live data is ~1–2 days old** (recording began ~2026-11-27/28). BookTicker
  covers only BTC/ETH/SOL top-of-book; no depth, no trades feed, no
  multi-symbol coverage. Prior bookTicker instability was noted in
  earlier sessions — the recorder has gap logging but long-run stability
  is unproven.
- **OI history**: REST gives ~30 days; the Binance public archive
  (investigated 2026-11-28) extends to 2020-09/2021-12 at 5m resolution
  but has an unresolved timestamp-convention change (2026-06-25) and
  possible retrospective relabeling. Not yet ingested.
- **No tick/trade data**: aggTrades exist as a downloader
  (`data_engine/agg_trades.py`) but no historical tick store is
  populated; Binance public archive has aggTrades but they are large
  (multi-GB per symbol-year).
- **No order-book depth history**: only top-of-book bookTicker, and only
  prospectively. Depth snapshots/diffs are not recorded.
- **No options data**, no news/macro feeds, no cross-exchange market
  data beyond BTC/ETH/SOL liquidations.
- **Empty engine shells**: `v4_system/ml_engine`, `execution_engine`,
  `risk_engine`, `strategy_engine`, `backtesting`, `microstructure_engine`,
  `macro_engine`, `news_engine`, `paper_trading` contain only `__init__.py`
  — the architecture was planned but never implemented.

### 2.3 What is missing entirely

- Multi-asset universe discovery/selection.
- Multi-timeframe (sub-minute) data infrastructure.
- Portfolio-level decision/execution engine.
- Any trained ML model.
- Cross-exchange arbitrage infrastructure.
- Options/derivatives beyond perpetuals.

**Honest assessment**: the project has a solid *data governance and
single-asset backtesting* foundation, a young live recorder, and a
rigorous research methodology. It does **not** yet have the data
breadth, model infrastructure, or decision architecture required for
the autonomous-system objective. The gap is large but the foundation
(PIT, costs, statistics, governance) is genuinely reusable.

---

## 3. Proposed end-to-end system architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MARKET UNIVERSE LAYER                     │
│  Discovery → Eligibility filters → Dynamic watchlist (spot+perp)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ selected instruments
┌──────────────────────────▼──────────────────────────────────────┐
│                      DATA LAYER (multi-resolution)               │
│  Trades/ticks · bookTicker · depth (prospective) · klines 1s→1d   │
│  Funding · OI (archive+REST) · liquidations · cross-exchange     │
│  PIT enforcement at every read (existing pit.py generalized)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ point-in-time feature streams
┌──────────────────────────▼──────────────────────────────────────┐
│                    FEATURE & STATE LAYER                         │
│  Per-instrument features @ multiple resolutions                  │
│  Cross-asset features · regime/state estimates · microstructure  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    MODEL LAYER (modular, §6)                     │
│  Regime · Return/vol forecasting · Order-flow · Opportunity      │
│  detection · Strategy selection · Execution · Portfolio          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ candidate opportunities + forecasts
┌──────────────────────────▼──────────────────────────────────────┐
│              OPPORTUNITY RANKING & PORTFOLIO LAYER (§8)          │
│  Rank candidates → allocate capital → size → risk-check → execute │
└──────────────────────────┬──────────────────────────────────────┘
                           │ orders
┌──────────────────────────▼──────────────────────────────────────┐
│                    EXECUTION LAYER                               │
│  Order-type selection · timing · impact-aware slicing · fills    │
│  Realistic costs (fees, slippage, funding, liquidity)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ fills / positions
┌──────────────────────────▼──────────────────────────────────────┐
│                    RISK & GOVERNANCE LAYER                       │
│  Exposure limits · correlation limits · drawdown kill-switches   │
│  Trial registry · preregistration · audit trail                  │
└─────────────────────────────────────────────────────────────────┘
```

**Status tags**: Data Layer (partially existing), PIT (existing),
Backtest engine (existing, single-asset), everything else (proposed).

---

## 4. Dynamic multi-asset universe design

### 4.1 Market discovery
- **Source**: Binance exchangeInfo (spot + USDⓈ-M futures), Bybit
  instruments API. Poll periodically; maintain a symbol registry.
- **Asset classes**: spot pairs, USDT-margined perpetuals (primary),
  coin-margined perps (secondary), options (future — Deribit/ Binance
  options; deferred until core system proven).
- **New listings**: auto-detected on exchangeInfo diff; enter a
  probationary watchlist, not the tradable universe.

### 4.2 Eligibility filters (measurable, not arbitrary)
An instrument enters the tradable universe only when ALL hold:
- **Liquidity**: 24h quote volume ≥ threshold (e.g., top ~100 perps);
  bid-ask spread ≤ X bps at bookTicker level.
- **History**: ≥ N days of 1m klines available (enough for feature
  warmup + backtest).
- **Data quality**: gap rate below threshold; no unresolved schema
  anomalies.
- **Execution feasibility**: tick size/step size/min notional compatible
  with intended position sizes; leverage available.
- **Stability**: not in delisting notice period; funding/OI data
  available.

Instruments are **re-evaluated continuously** and can be demoted
(liquidity decay, spread widening, data gaps) or promoted. The
universe is a dynamic set, not a fixed list — no arbitrary cap on
count; the count emerges from the filters.

### 4.3 Survivorship bias control
- The universe registry records **every** instrument's eligibility
  intervals with timestamps (entry/exit, reason).
- Historical evaluation must reconstruct the universe **as it was at
  each historical date**, not today's survivors. Instruments that were
  eligible then but delisted since must remain in historical tests.
- New listings enter only after their eligibility criteria are met —
  no retroactive inclusion.
- This requires storing eligibility history, which the current manifest
  does not do — **new component required**.

### 4.4 Cross-exchange
- Phase 1: Binance primary (deepest data). Bybit liquidations already
  recorded; Bybit klines/bookTicker can be added.
- Cross-exchange price discrepancies are an opportunity family (§7) but
  require synchronized multi-exchange data — deferred until single-
  exchange system is validated.

**Status**: Proposed. Requires implementation. No approval yet.

---

## 5. Multi-timeframe architecture

### 5.1 Resolution hierarchy
The system operates at **multiple simultaneous resolutions**, each
serving a different decision frequency:

| Tier | Resolution | Decision frequency | Purpose |
|---|---|---|---|
| Fast | Trades/ticks, seconds | Event-driven (sub-second to seconds) | Order-flow, microstructure, execution timing |
| Intermediate | 1s–1m bars | Seconds to minutes | Short-term momentum, mean reversion, breakout |
| Standard | 5m–1h bars | Minutes to hours | Swing signals, regime context |
| Slow | 4h–1d bars | Hours to days | Trend regime, portfolio context, funding cycles |

**These are candidate resolutions to investigate, not approved fixed
timeframes.** The architecture supports per-opportunity-family decision
frequencies: an order-flow opportunity may live seconds; a funding
carry opportunity lives days.

### 5.2 Data requirements per tier
- **Fast**: trade-by-trade or aggTrades (Binance public archive has
  aggTrades; live WS `aggTrade`/`trade` streams available). BookTicker
  for spread. Depth diffs for order-book imbalance (prospective only).
- **Intermediate**: 1s bars are NOT natively available from Binance
  klines (min 1m); must be built from trades or use 1m klines. 1m/5m
  klines exist historically.
- **Standard/Slow**: existing 1h/4h/1d datasets.

### 5.3 Hierarchical interaction without leakage
- Fast tier produces short-horizon signals and execution timing.
- Intermediate tier confirms/filters fast signals (e.g., order-flow
  spike + 1m momentum alignment).
- Slow tier sets regime/context (trend, volatility, funding regime)
  that gates which opportunity families are active.
- **Information flow is strictly downward in time**: slow features are
  computed on closed slow bars; fast features on the latest closed fast
  bar; no tier sees another tier's future. The existing `pit.py`
  cutoff mechanism generalizes to enforce this per-resolution.
- Latency: fast tier must run on live WS data with minimal buffering;
  slow tier can run on batch. The architecture separates online
  (fast/intermediate) from offline (slow, research) computation.

**Status**: Proposed. Requires new data infrastructure (tick/trade
store, 1s aggregation). Requires research on which resolutions carry
signal.

---

## 6. ML component architecture

A modular architecture where each component has a defined responsibility,
model family, data needs, and evaluation. **Prediction models are
separated from decision/execution models.**

### 6.1 Component responsibilities and candidate model families

| Component | Responsibility | Candidate models | Data needs | Status |
|---|---|---|---|---|
| **Regime/state estimator** | Classify market regime (trend/range/high-vol/low-vol), detect anomalies, estimate correlation structure | HMM, clustering (GMM/k-means), gradient-boosted classifier, change-point detection | Long history of returns, vol, correlations | Requires research |
| **Return/vol forecaster** | Predict return distribution, volatility, quantiles at multiple horizons | LightGBM quantile regression, GARCH-family, distributional RL, conformal prediction | Multi-timeframe features | Requires research |
| **Order-flow/microstructure model** | Predict short-term price impact, order-book imbalance effects, liquidation cascade probability | Gradient boosting on microstructure features, shallow temporal models (TCN/GRU), point-process models | Tick/trade/book data — **mostly unavailable historically** | Requires data + research |
| **Opportunity detector** | Score candidate opportunities across instruments/families | Learned ranking (LambdaRank), classification, anomaly detection | Features + opportunity labels | Requires research |
| **Strategy selector** | Choose which opportunity type to act on given regime | Contextual bandit, RL (careful), rule-based gating initially | Regime + opportunity scores | Requires research; start rule-based |
| **Execution model** | Order type, timing, aggressiveness, slicing to minimize impact | RL for execution (advanced), supervised fill-probability models, simple heuristics first | Order-book + trade data | Requires data + research |
| **Portfolio allocator** | Capital allocation, sizing, hedging, exposure limits | Constrained optimization, risk-parity, RL (advanced), rule-based first | Positions, correlations, risk estimates | Requires research; start rule-based |

### 6.2 Model family guidance (honest assessment)

- **Gradient boosting (LightGBM/XGBoost)**: strong baseline for tabular
  features; fast to train; interpretable-ish. Use for return/vol
  forecasting, opportunity scoring. Low risk.
- **Linear models (logistic/ridge)**: interpretable baselines; use for
  calibration and sanity checks. Low risk.
- **Temporal models (TCN, GRU, Transformer)**: potentially valuable for
  sequence data (order flow, multi-timeframe). High data requirements,
  high compute, high overfitting risk. Defer until tabular baselines
  are exhausted.
- **Deep learning (general)**: only justified if tabular methods fail
  AND sufficient data exists. Currently insufficient data for most
  components.
- **Reinforcement learning**: high risk, high complexity, hard to
  validate, easy to overfit to backtest. Defer; consider only for
  execution/portfolio after supervised components are proven.
- **Ensembles**: combine diverse models; reduces variance. Natural fit
  once multiple components exist.

**Principle**: start with the simplest model that can capture the
signal; add complexity only when simpler models demonstrably fail on
held-out data. Every model addition is a registered trial.

### 6.3 Prediction vs decision separation
- **Prediction models** output forecasts (returns, vol, probabilities,
  regime labels).
- **Decision models** (opportunity ranking, strategy selection,
  portfolio allocation) consume predictions + constraints to produce
  actions. These are initially rule-based/optimization, upgraded to
  learned models only when justified.

**Status**: Proposed. Requires research. No approval yet.

---

## 7. Opportunity and strategy research universe

Classification of opportunity families by feasibility with current/prospective data:

| Family | Required data | Timeframe | Holding | Exec complexity | Capital/liquidity | Post-cost viability | Model needs | Feasibility |
|---|---|---|---|---|---|---|---|---|
| Short-term momentum | 1m–1h klines | 1m–1h | Minutes–hours | Low | Medium | Uncertain (Family 1 NO EDGE at 1h/4h) | GBM/linear | **Research first** — test at shorter TFs |
| Mean reversion | 1m–15m klines | 1m–15m | Minutes | Low | Medium | Uncertain | GBM/linear | Research |
| Breakout/vol expansion | klines + vol | 5m–1h | Hours | Low | Medium | Uncertain | GBM/regime | Research |
| Liquidation cascades | Liquidation feed (live only, ~days old) | Seconds–minutes | Seconds–minutes | High (fast execution) | High (liquidity) | Unknown — needs prospective data | Microstructure model | **Blocked on data** — accumulate live |
| Order-flow imbalance | Trades + bookTicker/depth | Seconds | Seconds–minutes | High | High | Unknown — needs tick/book data | Microstructure model | **Blocked on data** |
| Liquidity/spread dislocations | bookTicker (3 symbols, live) | Seconds | Seconds | Very high | Very high | Unknown | Execution model | **Blocked on data + capital** |
| Cross-asset relative value | Multi-asset klines (have BTC/ETH/SOL) | 1h–1d | Hours–days | Low | Medium | Family D tested pairs — NO EDGE | GBM/linear | Research (pairs failed before) |
| Statistical arbitrage | Multi-asset klines | 1h–1d | Hours–days | Low | Medium | Uncertain | Cointegration/ML | Research |
| Funding/basis carry | Funding + perp/spot | 8h settlements | Days | Low | Medium | Family tested funding-sign — NO EDGE; carry unexplored | Rule-based + ML overlay | Research (lowest-hanging) |
| Cross-exchange discrepancies | Multi-exchange data | Seconds | Seconds | Very high | Very high | Unknown | Execution model | **Blocked on data** |
| Market making | Full book + trades | Seconds | Continuous | Very high | Very high | Unknown | RL/advanced | **Blocked on data + capital** — defer |

### 7.1 Recommended research order
1. **Funding/basis carry** — data exists (funding history + perp/spot
   klines), low execution complexity, holding period days (less cost
   sensitivity), not yet properly tested as a *carry* strategy (only
   funding-sign was tested).
2. **Short-term momentum/mean-reversion at 1m–15m** — data exists
   (1m klines back to 2023), tests whether edge exists at faster
   timeframes where Family 1 didn't look.
3. **Regime-conditional strategies** — use regime estimator to gate
   existing signals; tests whether conditioning improves the NO EDGE
   families.
4. **Order-flow/microstructure** — only after live data accumulates
   (months, not days).
5. **Cross-exchange, market-making** — last; require infrastructure
   and capital not yet justified.

**Status**: Investigation complete. Specific families require owner
approval before any trial registration.

---

## 8. Autonomous decision and portfolio-management architecture

### 8.1 Hierarchical decision process
```
Opportunity candidates (from model layer)
        │
        ▼
┌─────────────────────┐
│ Opportunity Ranker  │  Score each candidate: expected net edge,
│                     │  confidence, regime fit, liquidity, correlation
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Portfolio Allocator │  Select subset under capital/exposure/correlation
│                     │  constraints; determine size, leverage, hedge
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Execution Planner   │  Order type, timing, slicing, aggressiveness
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Risk Gate           │  Final veto: exposure limits, drawdown,
│                     │  concentration, turnover, capital conflicts
└─────────┬───────────┘
          ▼
      Orders → Exchange
```

### 8.2 Decision capabilities required
- **Which asset/market**: ranker scores across universe.
- **Which opportunity/strategy**: strategy selector picks best-fit
  family for current regime.
- **Direction/exposure**: from forecast sign + confidence.
- **Position size**: from expected edge, volatility, risk budget
  (e.g., fixed fractional or Kelly-fraction with cap).
- **Leverage**: within registered limits (Family 1 used 2x cap).
- **Entry/exit conditions**: model-driven, not fixed indicator rules.
- **Partial entries/exits**: scale in/out based on confidence updates.
- **Add/reduce/reverse**: position management as continuous decision,
  not one-shot entry/exit.
- **Hedging**: offset correlated exposures (e.g., long BTC + short ETH
  if correlation spike).
- **Capital allocation/reallocation**: dynamic across opportunities.
- **Reject insufficient edge**: minimum net-edge threshold (as in
  Family 1: ≥5 bps net per trade).

### 8.3 Overtrading/correlation/concentration controls
- **Turnover limit**: max trades per day/week per instrument.
- **Correlation limit**: max portfolio exposure to correlated assets
  (e.g., BTC+ETH+SOL often correlate >0.8 — treat as one risk bucket).
- **Concentration limit**: max % capital per instrument/strategy.
- **Capital conflict resolution**: ranker assigns priority; lower-
  ranked opportunities rejected if capital exhausted.
- **Cooldown**: after a loss, temporary pause on that instrument/
  strategy (prevents revenge-trading patterns).
- **Drawdown kill-switch**: portfolio-level and per-strategy; halt on
  breach (existing risk concepts, not yet implemented as code).

### 8.4 Risk as integral architecture
Risk is not an afterthought filter — it is a **first-class layer** that:
- Constrains the allocator's feasible set (no allocation may violate
  limits).
- Monitors live exposure continuously.
- Can override model decisions (veto power).
- Is backtested alongside strategies (stress scenarios, tail events).

**Status**: Proposed. Requires implementation. Rule-based first;
learned components only after validation. No approval yet.

---

## 9. Data and infrastructure requirements

### 9.1 Honest data inventory

| Data type | Historical availability | Prospective collection | Gap |
|---|---|---|---|
| 1m–1d klines (perp+spot, BTC/ETH/SOL) | ✅ 2023-01→2026-11, validated | ✅ recorder | Only 3 symbols; no other alts |
| Funding settlements | ✅ history + intervals | ✅ | Good |
| OI (REST) | ⚠️ ~30 days only | ✅ | **Major gap** — archive exists but not ingested, timestamp semantics unresolved |
| OI (archive) | ⚠️ 2020-09/2021-12→present, 5m, schema stable, labeling change 2026-06-25 | n/a | Not ingested; needs validation pipeline |
| Liquidations | ❌ none historical | ✅ Binance (sampled) + Bybit (full), ~1-2 days | **Major gap** — months needed |
| bookTicker (top-of-book) | ❌ none historical | ✅ 3 symbols, ~1-2 days, stability unproven | **Major gap** |
| Order-book depth | ❌ none | ❌ not recorded | **Major gap** — must start recording |
| Trades/aggTrades | ⚠️ downloader exists, no store populated | ✅ WS available | **Major gap** — large data volume |
| 1s bars | ❌ | Can build from trades | Requires trade data first |
| Cross-exchange (Bybit klines etc.) | ❌ | Partial (liquidations only) | **Gap** |
| Options | ❌ | ❌ | Not started |
| News/macro | ❌ | ❌ | Not started |

### 9.2 Data accumulation strategy
- **Immediate**: extend live_recorder to more symbols (top ~20-50
  perps by volume), add aggTrade stream, add depth (incremental
  depth@100ms or snapshot+diff), add Bybit klines/bookTicker.
- **Historical**: ingest OI archive (after validation pipeline);
  download aggTrades for priority symbols (large — start with BTC/ETH
  recent years); download additional altcoin klines for universe
  expansion.
- **Storage**: current parquet store works for klines; tick/trade data
  needs columnar format (parquet partitioned by date/symbol) or a time-
  series DB (e.g., QuestDB/ClickHouse) — decision required.
- **Timestamp precision**: exchange timestamps are ms; recorder adds
  `_recv_ts` (µs). For tick data, preserve both event time and receipt
  time (latency measurement).
- **Freshness**: live recorder flushes every 60s; for fast strategies,
  need sub-second or streaming feature computation — architecture must
  separate online (low-latency) from offline (research) paths.
- **Recovery**: gap logging exists; need automated gap-detection +
  backfill-from-archive for klines (Binance archive has historical
  klines; liquidations/bookTicker have NO historical source — gaps are
  permanent).
- **Monitoring**: recorder health, data freshness, gap alerts.

### 9.3 PIT for high-frequency data
Existing `pit.py` handles candles and event series. For tick/trade data:
- Event time = exchange timestamp; receipt time = `_recv_ts`.
- `available_at` = receipt time for live data (real), NaT for
  historical bulk downloads.
- Order-book snapshots/diffs need sequence-number gap detection
  (missing diffs corrupt the book state).

**Status**: Partially existing. Major expansion required. No approval
yet.

---

## 10. Evaluation framework and human-performance comparison

### 10.1 Measurable evaluation criteria
- **Net profitability**: after all costs (fees, slippage, funding,
  borrow if applicable).
- **Risk-adjusted returns**: Sharpe, Sortino, Calmar.
- **Max drawdown**: portfolio-level, marked-to-market.
- **Tail risk**: VaR/CVaR, worst-day/week losses, stress scenarios.
- **Capital efficiency**: return on deployed capital, not just
  notional.
- **Consistency**: performance across regimes, folds, instruments.
- **Execution quality**: slippage vs model, fill rates, latency.
- **Coverage**: number of instruments traded, opportunity detection
  rate.
- **Simultaneity**: performance with multiple concurrent positions
  (not just sequential single-asset tests).
- **Regime robustness**: performance in trending vs ranging vs high-vol.
- **Capacity**: performance vs position size (market impact).

### 10.2 Human benchmarks
Fair comparison requires:
- **Discretionary trader baselines**: e.g., published crypto fund
  returns (CoinShares indices), top-trader leaderboards (with caveats
  about survivorship and self-reporting), or a panel of human traders
  trading the same capital/rules prospectively.
- **Simpler automated baselines**: buy-and-hold, equal-weight basket,
  the existing Turtle strategy (already NO EDGE), random entry.
- **Data required**: audited track records, same cost assumptions,
  same universe, same period. Public fund indices are the most
  objective benchmark; leaderboards are biased.

### 10.3 Honest framing
- The objective is to **outperform human discretionary trading on
  net risk-adjusted basis**, not to promise beating every trader.
- Evidence required: prospective paper-trading results over sufficient
  time (≥6 months), across regimes, with realistic costs, compared to
  benchmarks.
- No optimizing for headline returns while ignoring risk, leverage,
  or survivorship bias.

**Status**: Proposed framework. Requires implementation of portfolio-
level backtesting and paper-trading infrastructure. No approval yet.

---

## 11. Development roadmap

### Phase 0 — Foundation hardening (existing, reusable)
- ✅ PIT controls, cost model, statistics, trial governance, data store.
- 🔲 Extend live_recorder: more symbols, aggTrade, depth, Bybit klines.
- 🔲 OI archive ingestion pipeline (after validation).
- 🔲 Universe registry with eligibility history (survivorship control).

### Phase 1 — Data accumulation (prospective, months)
- 🔲 Record tick/trade/book data for top ~20-50 perps.
- 🔲 Build 1s/1m bars from trades.
- 🔲 Ingest historical klines for expanded universe.
- 🔲 Ingest OI archive.
- 🔲 Download aggTrades for priority symbols (BTC/ETH/SOL recent).

### Phase 2 — Single-opportunity ML research (first experiments)
- 🔲 Regime estimator (unsupervised first).
- 🔲 Return/vol forecaster at 1m–1h (GBM baseline).
- 🔲 Funding carry strategy (rule-based + ML overlay).
- 🔲 Short-term momentum/mean-reversion at 1m–15m.
- Each is a registered trial family with walk-forward + DSR.

### Phase 3 — Portfolio & decision architecture
- 🔲 Opportunity ranker.
- 🔲 Portfolio allocator (rule-based optimization first).
- 🔲 Execution planner (simple heuristics first).
- 🔲 Risk gate (exposure/correlation/concentration limits).
- 🔲 Multi-asset, multi-position backtesting engine (extend existing
  single-asset engine or build new).

### Phase 4 — Paper trading (prospective validation)
- 🔲 Live paper-trading with real-time data.
- 🔲 Compare to benchmarks.
- 🔲 Accumulate ≥6 months of prospective results.

### Phase 5 — Controlled live deployment (only after evidence + approval)
- 🔲 Small capital, strict limits.
- 🔲 Gradual scaling.
- 🔲 Continuous monitoring and kill-switches.

**Dependency order**: data before models, models before portfolio,
portfolio before paper trading, paper trading before live. Do NOT
build models before the data infrastructure supports them. Do NOT
restrict to 3 assets/1 timeframe/2 models for convenience — but do NOT
build everything at once either.

**Status**: Proposed roadmap. Requires owner approval for each phase.

---

## 12. Existing ML draft assessment

`v4_system/ML_RESEARCH_SPEC_DRAFT.md` (commit `2becfe0`) was written
before this broader objective was articulated. Assessment:

### 12.1 What remains useful
- **PIT/leakage controls** (§6 of draft): closed bars, purge/embargo,
  event-time PIT — these generalize to any ML work.
- **Cost model integration**: 14/18 bps RT, next-open fills — reusable.
- **Trial-budget discipline**: registered config matrix, DSR — reusable.
- **Walk-forward methodology**: 5-fold chronological — reusable.
- **OI data contract**: raw-label preservation, convention
  classification — reusable for archive ingestion.
- **Baselines**: random-entry, majority-class — reusable.

### 12.2 What was an unapproved assumption (not a project requirement)
- **1-hour decision frequency**: arbitrary choice; the broader objective
  explicitly wants seconds-to-minutes investigation.
- **4-hour prediction horizon**: arbitrary; different opportunity
  families need different horizons.
- **Three-asset universe (BTC/ETH/SOL)**: arbitrary restriction; the
  objective requires dynamic multi-asset.
- **Two-model restriction (LR + LightGBM)**: arbitrary; the objective
  requires modular multi-component architecture.
- **Single prediction target (4h return)**: too narrow; the system
  needs regime, vol, order-flow, opportunity, execution, portfolio
  predictions.

### 12.3 What is too narrow
- The draft treats ML as a single predictive model producing long/short/
  no-trade signals. The broader objective requires a **decision and
  portfolio system** that ranks opportunities, allocates capital,
  manages multiple positions, and executes intelligently.
- The draft has no concept of market universe, regime adaptation,
  multi-timeframe hierarchy, or execution intelligence.

### 12.4 Disposition
- **Retain as a possible small research experiment**: the draft's
  Family ML-1 (4h return prediction on BTC/ETH/SOL with LR+LightGBM)
  could be run as ONE experiment within the broader architecture —
  specifically as a "standard timeframe return forecaster" component.
- **Do NOT treat its constraints as binding**: the 1h/4h/3-asset/2-model
  choices were drafting conveniences, not approved requirements.
- **Supersede with this document**: this architecture spec is the
  broader framework; the ML draft becomes a component-level experiment
  spec if/when approved.

**Status**: Assessment complete. Draft retained but not binding.
No modification to the draft made.

---

## 13. Unresolved design decisions requiring owner approval

1. **Universe scope**: how many instruments to target initially
   (top 20? 50? all eligible)? What liquidity threshold?
2. **Timeframe priority**: which resolution tier to research first
   (1m momentum? funding carry at 8h? order-flow once data exists)?
3. **Data infrastructure**: parquet vs time-series DB for tick data?
   Storage budget?
4. **Live recorder expansion**: which symbols/streams to add first?
   Depth recording (bandwidth/storage cost)?
5. **OI archive ingestion**: approve full ingestion (5,900 files,
   ~70MB compressed) or defer?
6. **First ML experiment**: which opportunity family to research
   first (funding carry? 1m momentum? regime-conditional)?
7. **Portfolio architecture**: rule-based allocator first, or invest
   in learned allocation earlier?
8. **Paper trading infrastructure**: build new or extend existing?
9. **Human benchmark**: which benchmark to use (fund indices?
   leaderboards? prospective panel)?
10. **Capital/risk limits**: what are the registered risk limits for
    the eventual live system?
11. **Trial budget allocation**: how to allocate the remaining 39
    trials across the broader architecture's experiment families?
12. **Cross-exchange**: when to add Bybit/other exchanges?
13. **Options**: whether/when to include options strategies.
14. **This document's status**: approve as the governing architecture
    draft, or request revisions?

---

## 14. Risks, feasibility limitations, and implementation blockers

### 14.1 Fundamental risks
- **Data sparsity for fast strategies**: order-flow/microstructure
  strategies need tick/book data that doesn't exist historically and
  is only now being collected prospectively. Months of accumulation
  needed before any research.
- **NO EDGE prior**: 1,480 trials found no edge in manual strategies.
  ML may find edge, but the base rate of success is low; expect many
  failed experiment families.
- **Overfitting risk**: complex models + limited data = overfitting.
  The trial registry and DSR help but don't eliminate it.
- **Execution reality**: backtests assume fills at next-open with
  fixed slippage; real execution at scale has market impact, partial
  fills, latency. Fast strategies are especially vulnerable.
- **Regime dependence**: crypto markets regime-shift; a model trained
  on 2023-2025 may fail in 2026+ conditions.
- **Survivorship bias**: dynamic universe must be handled carefully;
  delisted coins' data may be incomplete.
- **Capital constraints**: market-making and some arbitrage require
  capital/liquidity beyond current scope.
- **Computational cost**: tick-level ML on many instruments is
  expensive; infrastructure must be sized realistically.

### 14.2 Implementation blockers (hard)
- **No historical tick/trade/book data** — cannot research fast
  strategies until prospective collection accumulates.
- **OI archive not ingested** — OI features limited to 30 days.
- **No multi-asset universe infrastructure** — universe registry,
  eligibility filters, survivorship control not built.
- **No portfolio-level backtester** — existing engine is single-asset,
  bar-driven; multi-position, multi-asset, multi-timeframe simulation
  not built.
- **No live execution infrastructure** — paper_trading engine is an
  empty shell; no order management, no exchange connectivity for
  trading (only data).
- **No options data/feed** — options strategies blocked.

### 14.3 Feasibility assessment
- **Achievable with current data**: funding carry, 1m–1h momentum/
  mean-reversion, regime-conditional strategies on BTC/ETH/SOL (and
  expanded klines for more alts).
- **Achievable with near-term data accumulation**: order-flow signals
  (once months of tick/book data exist), liquidation-cascade signals
  (once liquidation history accumulates).
- **Achievable with significant new infrastructure**: cross-exchange
  arbitrage, market making, options strategies.
- **Not achievable today**: any strategy requiring historical tick/
  book data, any live trading (no execution infrastructure), any
  multi-asset portfolio system (no portfolio engine).

### 14.4 Recommended immediate next steps (require approval)
1. Approve this document as the governing architecture draft (or
   request revisions).
2. Approve Phase 0 data infrastructure expansion (recorder + OI
   archive + universe registry).
3. Approve the first research family (recommend: funding carry or
   1m momentum — data exists today).
4. Decide on data storage technology for tick data.
5. Allocate trial budget across experiment families.

---

*End of draft. Awaiting owner review. Nothing herein is approved,
frozen, registered, or authorized for implementation.*

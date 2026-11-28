# V4 System — Pre-Registration (PREREG)

This document is written **before** any strategy testing. It defines the
hypotheses, trial budget, and pass/fail thresholds for each phase. Every
trial run is logged in `v4_system/trials_log.csv`. If a phase fails its
thresholds, the result is reported as-is — no tuning against the holdout.

## Global rules (all phases)

- **Capital**: $1,000 starting equity, simulated.
- **Ambition target**: ≥6%/month net. This is a target, not a pass
  criterion. If unattainable under credible risk limits, we report the
  highest defensible return.
- **Costs**: Binance USDⓈ-M taker fee 0.05% (VIP0), maker 0.02%. Sensitivity
  at 1x, 1.5x, 2x base costs is mandatory.
- **Validation**: chronological walk-forward with purged/embargoed windows;
  final holdout is genuinely untouched until the end.
- **No look-ahead**: point-in-time data only. News features use
  first-observed timestamps, never revised article text.
- **Trial budget**: each phase has a fixed number of allowed
  strategy/parameter trials. Exceeding it requires explicit user approval
  and is logged.
- **"No tradable edge found" is a valid, reportable result.**

## Phase 1 — Data Engine & Availability (this phase)

**Goal**: a reproducible, point-in-time data pipeline for BTC, ETH, SOL
(and extensible to other liquid perps) covering OHLCV (1m–1D, spot +
perp), funding, open interest, mark/index price, and trade-level data —
plus a written account of what historical data does **not** exist.

**Hypothesis (H1)**: Binance USDⓈ-M public REST endpoints can supply
sufficient historical depth for walk-forward research on 1h+ timeframes,
but several required datasets (liquidations, order-book history, long
funding/OI history) have hard availability limits that must be documented
and worked around.

**Trial budget**: 0 strategy trials. This phase produces data + an
availability report only. Endpoint probes are logged but are not
"trials" in the statistical sense.

**Pass/fail thresholds**:

- PASS: klines (spot + perp) retrievable for BTC/ETH/SOL at 1m, 5m, 15m,
  1h, 4h, 1D with ≥2 years of history; funding-rate history retrievable;
  every dataset has a manifest entry (source, rows, range, sha256).
- PASS (with documented gaps): OI/mark/index history limited to recent
  windows is acceptable **if** the limitation is quantified and recorded.
- FAIL: any core OHLCV dataset cannot be retrieved at all, or data is
  silently incomplete (gaps not detected and reported).

**Deliverables**: `v4_system/data_engine/`, `v4_system/data/`,
`v4_system/reports/DATA_AVAILABILITY.md`.

## Phase 2+ (pre-registered skeleton, thresholds to be finalized per phase)

- **Phase 2 — Technical baseline**: single-instrument trend/momentum
  baseline on 1h/4h. Thresholds: 95% CI lower bound on net expectancy > 0
  at 1.5x costs, max DD ≤ 20%, ≥300 OOS trades.
- **Phase 3 — Microstructure features**: incremental value of OI, funding,
  CVD, book imbalance over technical baseline. Same thresholds.
- **Phase 4 — ML models**: LightGBM/XGBoost baselines per horizon with
  purged walk-forward. Thresholds: calibrated probabilities, OOS AUC/edge
  significantly > baseline, same trading thresholds.
- **Phase 5 — News engine**: event study + news-augmented strategies.
  Additional threshold: news features must beat technical-only baseline
  OOS, else news is dropped from the integrated system.
- **Phase 6 — Strategy integration + leverage/risk**: combined decision
  engine, leverage ladder 1x–50x, liquidation modeling. Thresholds: same
  expectancy/DD/trade-count gates at each leverage; report highest
  defensible return.
- **Phase 7 — Paper trading automation**: live paper deployment, no real
  orders, no withdrawal-enabled keys.

---

# PART II — Research History, Cost Model, Trial Budget, Freeze & Holdout

*Written 2026-11-28, after remediation items 1–6 of the V4 data-engine
program were completed, and BEFORE any Phase 2 strategy experiment. Every
number below is reconstructed from committed artifacts; sources are cited
per row. Nothing here is invented: where a metric or date is unavailable
it is marked **UNKNOWN**.*

## II.1 Complete history of previous experiments

Six research programs have been run in this repository. They are listed
chronologically. "Superseded" means a later, corrected artifact replaced
the earlier one; both are listed.

### Program A — Legacy funding-carry validation & paper track (2026-07)

| # | Experiment | Config | Data | Result | Source |
|---|---|---|---|---|---|
| A1 | Funding-rate carry validation | short perp + long spot, BTC, $100k | funding 2019-09-10→2024-12-31, 5,819 settlements | +109.9% total, CAGR 15.0%, maxDD −1.5%, Sharpe 18.2, 87.2% periods profitable | `funding_carry_results.json` (commit `6383612`) |
| A2 | Funding carry paper bot v1 | same, paper track | live 2026-07 | superseded by v2 (fee/slippage drag, mark-price feed, liquidation sim) | commits `82300c0`, `9c52762` |
| A3 | Walk-forward mean-reversion mr_1h | 2y train / 6m test / 6m step, 9 folds | OOS 2020-05-15→2024-11-15 | 375 trades, Sharpe −2.91, return −63.5%, maxDD −64.7% — **FAIL** | `walk_forward_results_mr_1h.json` |
| A4 | Walk-forward mean-reversion mr_4h | same folds | Bitstamp BTCUSD proxy (Binance unreachable) | acceptance check passed; results in same file family — **FAIL** (negative OOS) | `walk_forward_results_mr_4h.json` |
| A5 | Trend Momentum Pyramid (TMP) | live strategy | live paper | paper-trading system, not a backtest trial | commit `7b3a4e3` |
| A6 | RGMT / OmniPulse paper systems | daily rebalance, offline replay 70/30, 5 pre-registered variants, kill gate | live/paper | paper-trading systems with readiness gates; not backtest trials | commits `1877d15`, `d373a80`, `c3e6a9b` |

**Note on A1:** the +15% CAGR / Sharpe 18.2 figure was produced before the
fee/slippage drag and mark-price corrections of A2 and before the
slow_study margin model (Program B). It is **superseded** by B3/B4/B5 and
must not be quoted as a validated result.

### Program B — slow_study daily-frequency study (2026-11)

Daily candles, $1,000 start, costs 0.10% taker + 0.05% slippage per side
(0.15%/side), signals on daily close executed next open. Data: OKX spot
1D 12 majors 2019-01→2026-11-25; Binance funding full history.

| # | Strategy | FULL CAGR | 2023+ mean/mo | 2023+ CI99 | maxDD | Verdict | Source |
|---|---|---|---|---|---|---|---|
| B1 | S1 MA trend | 28.0% | 1.80% | [−0.91%, 5.55%] | −30.9% | not viable | `slow_study/reports/REPORT.md` |
| B2 | S2 TSMOM vol-target | 30.1% | 2.50% | [−1.28%, 7.35%] | −46.0% | not viable | same |
| B3 | S3 Donchian 55 | 39.4% | 2.32% | [−1.48%, 6.81%] | −39.6% | not viable | same |
| B4 | S4 cross-sectional momentum | −13.1% | −1.86% | [−6.03%, 0.94%] | −70.0% | not viable | same |
| B5 | S5 funding carry (no margin model) | 30.2% | 1.14% | [0.56%, 1.86%] | −0.7% | **VIABLE but below 6%/mo target** | same |
| B6 | S6 buy & hold (benchmark) | 41.8% | 3.82% | [−1.25%, 8.97%] | −53.1% | benchmark | same |

| # | Experiment | Result | Source |
|---|---|---|---|
| B7 | S5 v2 margin model (isolated, OKX MMR 0.40%) | L=1: $1,434 (5.4% CAGR, maxDD −0.9%, min liq dist 47.3%); L=2: $1,626 (7.3%); **L=3: LIQUIDATED 2021-01-22** | `slow_study/reports/REPORT_S5_margin.md` |
| B8 | S5 v3 Bybit venue-consistent (mark price, exact funding timestamps, MMR 0.33%) | $1,195 final, CAGR 3.36%, mean 0.28%/mo, maxDD −0.76%, 0 liquidations; negative-funding stress −3.71% CAGR; ≈ stablecoin yield with funding-sign risk | `slow_study/reports/REPORT_S5_v3.md` |

### Program C — research_next income study (2026-11)

$1,000 total capital, frozen rules, cost stress 1.0/1.5/2.0x on 0.15%/side.
Data: OKX spot 1D 12 majors 2019-01→2026-11-25; Binance funding; Binance
quarterly futures.

| # | Strategy | Result | Source |
|---|---|---|---|
| C1 | A multi-asset funding arb (top-3 trailing-30d funding, weekly rebalance) | $1,037–$1,053 final (0.9–1.3% CAGR); loses to S5 1x; loses money at 1.5x/2.0x costs — **FAIL** | `research_next/reports/REPORT.md` |
| C2 | B cash-and-carry v1 (long spot + short quarterly, isolated IMR 20%) | ETH $1,652 (13.6% CAGR), BTC $1,212 (5.0%), 8–9 liquidations each — **INVALIDATED** (see C3) | same |
| C3 | B cash-and-carry v2 (corrected data alignment, 4H futures marks) | **v1 numbers were a data-alignment artifact** (OKX 16:00 UTC candle-open labels vs Binance 00:00 UTC daily candles → look-ahead in every basis calc). Corrected: ETH cross-margin IMR=20% ~$1,145 (0.30%/mo, maxDD −3.9%, 0 liq); BTC cross max-deploy ~$1,177 (0.35%/mo, maxDD −4.8%); isolated margin unprofitable or liquidation-prone at every IMR. **Below S5 1x on return AND drawdown — not pursued** | `research_next/reports_v2/REPORT_V2.md` |
| C4 | C adaptive funding carry (BTC carry, weekly exposure from trailing funding + 30d vol) | L=1: $1,134 (3.3% CAGR, maxDD −0.6%, 0 liq); L=2: $1,187 (4.5%, min liq dist 14%) — **below S5 1x** | `research_next/reports/REPORT.md` |

### Program D — quant_research pre-registered program (2026-10-10 → 2026-10-14)

Pre-registration commit `888867e` (PREREG before data ingestion, verified
24 min before first data file). Trial log `quant_research/trials.csv`
(2,650 rows → 1,354 unique configs after dedup of an aborted duplicate
run). Master report `quant_research/REPORT.md`.

| # | Experiment | Config | Data | Result | Source |
|---|---|---|---|---|---|
| D1 | Test 1 mechanical indicator stack | 4 variants (1A–1D) × 3 TFs (1m/3m/5m), SOL perp | IS 2024-01→2025-03 | all 12 fail: net expectancy −11 to −23 bps | REPORT §6 |
| D2 | Test 2 parameter grid | 1,296 configs (BB/RSI/ADX/stop/TP/time-stop), walk-forward 5 folds | IS/OOS 2024-01→2026-03 | zero OOS edge; top IS candidate CFG_1142: stitched −76.14%, net −18.25 bps, PF 0.65 — **FAIL** | REPORT §6, `test_2_results.json` |
| D3 | Turtle/Donchian breakout | 36 configs (N∈{20,55}, M∈{10,20} × SOL/BTC/ETH × 5m/15m/1h), 4-unit pyramiding | IS/OOS 2024-01→2026-03 | all 36 FAIL (net −5.38 to −33.19 bps pooled); funding-sign defect found & fixed (36/36 moved, 0 verdict changes) | REPORT §10, §13 |
| D4 | Test 3 user execution replay | 257 real Binance fills, 2025-04-04→2025-05-12 | live ledger | account 100% depleted; net −$189.28 on $189.28 deposits; pooled net −3.08 bps/trade (equal-weighted) — **NO EDGE**; superseded summary artifact documented | REPORT §9 |
| D5 | Entry autopsy (1m + 15s) | 13 indicators × 1,000 matched random controls | 2025-04-04→2025-05-12 | intraday indicators indistinguishable from random; only daily context survives Holm; taker print costs +5.14 bps vs fill-minute open | REPORT §15, §18 |
| D6 | Amendment C candidate (daily filter, 1 unit) | 72 configs (36 C1 + 36 C0) | IS/OOS 2024-01→2026-03 | **NO EDGE**; runner 1-unit defect found & corrected (v1 ran 4 units); 3 configs clear 5/6 hurdles, all fail DSR | REPORT §19 |
| D7 | Amendment D maker entry | 24 configs (R1 maker vs R0 taker, 1h) | IS/OOS 2024-01→2026-03 | **NO EDGE**; maker worth +5.09/+5.00 bps on N=20, −3.68/−3.95 bps on N=55 (adverse selection); all fail DSR at N=1,450 | REPORT §20 |
| D8 | Amendment E funding-sign | 6 configs (U1 taker / U2 maker, pro-side) | IS/OOS 2024-01→2026-03 | **NO EDGE**; pooled OOS −17.99 (U1) / −13.77 (U2) bps; runner defects fixed pre-run | REPORT §21 |

**Hold-out status after Program D:** ETH 2026-04-01→2026-10-09
**FRESH** (never touched — no config cleared the hurdles, so the veto
hold-out was never opened); BTC 2025+ and SOL 2026-04+ **BURNED**.

### Program E — research/new_edge hypothesis lab (2026-11)

Pre-registered hypotheses H01–H12, discovery/holdout split, hard gates
(net Sharpe ≥1.0, DD ≤20%, ≥100 trades, placebo p<0.05, DSR≥0.95).

| # | Hypothesis | Result | Source |
|---|---|---|---|
| E1 | H01 liquidation cascade fade | KILL | `research/new_edge/FINDINGS.md` |
| E2 | H02 funding reversal (+H02b hold variant) | KILL | same |
| E3 | H03 latency / H04 beta overshoot | KILL | same |
| E4 | H05 cross-sectional momentum (+H05b) | KILL | same |
| E5 | H06 funding cash / H07 TSMOM | KILL | same |
| E6 | H08 weekend alt bleed | KILL (effect real 2020–2024, t=4.0, but 31–39 bps/week gross vs 16 bps cost, gone 2025–2026) | same |
| E7 | H09 OI pilot | KILL (data insufficient) | same |
| E8 | H10 premium fade (+H11 held-out SOL check) | KILL (BTC/ETH positive was asset-specific; SOL holdout −56 bps net/event, Sharpe −0.43, DD −99%) | same |
| E9 | H12 pairs mean-reversion | KILL (crypto pairs trend, not revert; −95 to −212 bps net/event) | same |

Data: Binance perp 1m 2025-11-01→2026-10-31 (20 symbols), 1h
2020-01-01→2026-10-31 (36 symbols), funding 2020-01→2026-10-31,
order-flow bars 2026-10-23→2026-11-22 (1 month, internally inconsistent —
not usable). **Survivorship bias disclosed**: all symbols still listed in
2026.

### Program F — V4 data engine (2026-11-26 → 2026-11-28)

Phase 1 data engineering only. **0 strategy trials.** Deliverables:
point-in-time data engine, atomic storage, manifest provenance,
historical coverage expansion (item 5), contract metadata / funding
intervals / margin-tier snapshots (item 6). See
`v4_system/reports/DATA_AVAILABILITY.md` and
`v4_system/reports/phase1_fetch_report.json`.

### Experiment count summary

- Named strategy experiments/backtests reconstructed: **~45** (A1–A6,
  B1–B8, C1–C4, D1–D8, E1–E9).
- Registered parameter configurations (Program D alone): **1,354 unique**
  (+72 Amendment C re-logged, +24 D, +6 E → N_trials = 1,456 at end of
  Program D).
- Programs A/B/C/E ran **without** a formal pre-registration (rules frozen
  before runs in B/C/E, but no DSR accounting); Program D is the only
  fully pre-registered program with trial accounting.

## II.2 Historical data usage reconstruction

### Chronological table of date ranges actually consumed

| Period | Program(s) | Instruments | Timeframe(s) | Role |
|---|---|---|---|---|
| 2018-05→2026-07 | A4 | BTC | 4h (Bitstamp proxy) | walk-forward OOS |
| 2019-01→2026-11-25 | B1–B8 | BTC (+ETH in C3) | 1D | full-period backtests |
| 2019-09→2024-12 | A1 | BTC | funding 8h | carry validation |
| 2020-01→2026-10-31 | E1–E9 | 36 perps | 1h | discovery + holdout |
| 2020-01→2026-11-25 | C1–C4 | 12 majors + BTC/ETH quarterly | 1D + funding + futures | income study |
| 2020-05→2024-11 | A3 | BTC | 1h | walk-forward OOS |
| 2021-02→2026-11-25 | C3 | ETH (+BTC) | 1D spot + quarterly 4H | cash-and-carry v2 |
| 2021-07-05→2026-11-26 | B8 | BTC | Bybit daily/1H/mark/funding | venue-consistent S5 |
| 2023-01→2026-11-28 | F | BTC/ETH/SOL | 1m/5m/15m spot+perp | v4 data engine (fetched, not yet consumed by strategies) |
| 2024-01→2025-03 | D1–D3 | SOL (+BTC/ETH in D3) | 1m/3m/5m/15m/1h | IS (selection) |
| 2025-04→2026-03 | D1–D3 | SOL (+BTC/ETH) | same | OOS (walk-forward) |
| 2025-04-04→2025-05-12 | D4–D5 | SOL/BTC/ETH/BNB | 1m + 15s aggTrades | user replay + autopsy |
| 2025-11-01→2026-10-31 | E1–E9 | 20 perps | 1m | intraday pilots |
| 2026-04-01→2026-10-09 | D (reserved) | ETH | all | **FRESH hold-out, never touched** |
| 2026-04-01→2026-10-31 | E (partial) | 36 perps | 1h | new_edge holdout slice |
| 2026-10-23→2026-11-22 | E7 | 20 perps | order-flow | microstructure (unusable) |
| 2026-11-01→2026-11-28 | **nobody** | BTC/ETH/SOL | all v4 datasets | **unused by any experiment** |

### Overlaps and contamination

- **2024-01→2026-03** was used as IS+OOS by Program D, as part of the
  1h universe by Program E, and as part of daily history by Programs
  B/C. Three independent programs consumed overlapping windows.
- **BTC 2025+ and SOL 2026-04+** are burned (Program D). **ETH
  2026-04-01→2026-10-09** is fresh but reserved by Program D's veto
  protocol — V4 Phase 2 will NOT reuse it.
- **2026-10-10→2026-10-31** was consumed by Program E's 1h holdout slice.
- **2026-11-01→2026-11-28** (~27 days) is the only window not consumed by
  any intraday experiment. Caveat: Programs B/C observed the same
  calendar window at **daily** frequency (different timeframe, different
  strategies) — disclosed, not hidden.
- Datasets fetched by the V4 engine (item 5) but **not yet consumed** by
  any strategy: spot intraday 1m/5m/15m (2023-01→2026-11-28), OI (~30d),
  margin-tier snapshots, funding-interval table.

### ETH-specific usage

ETH appears in: A1 (BTC only — no), B (BTC only — no), C3 cash-and-carry
(2021-02→2026-11-25, daily + quarterly), D3/D6/D7/D8 (IS 2024-01→2025-03,
OOS 2025-04→2026-03, holdout 2026-04→2026-10-09 fresh), E (1h universe
2020→2026-10-31), F (v4 engine, unconsumed). **ETH has the richest
history of any symbol and the only unburned holdout window in the
repository.**

## II.3 Full trading-cost specification for Phase 2

| Cost component | Value | Status | Evidence |
|---|---|---|---|
| Taker fee | 0.05% (5 bps) per side | **VERIFIED** | Binance USDⓈ-M VIP0 schedule; confirmed against user ledger (1,785 COMMISSION entries) in Program D Test 3 |
| Maker fee | 0.02% (2 bps) per side | **VERIFIED** | same schedule; maker fills not present in user ledger (99% taker) — fee taken from published schedule |
| Bid-ask spread | not directly measured | **UNKNOWN** | no historical book data (websocket-only); proxy: taker print crosses +5.14 bps vs fill-minute open (Program D autopsy, 2025-04→05) |
| Slippage (base) | 2 bps per side | **ASSUMPTION** (conservative vs measured) | Program D cost model; user fills show +5.14 bps cross on breakout entries |
| Slippage (stress) | 4 bps per side | **STRESS ASSUMPTION** | Program D stress schedule |
| Market impact | not modelled | **UNKNOWN** | no book data; mitigated by stress slippage + position-size caps |
| Partial fills | maker: strict-through rule (fill only if 1m tape trades strictly through limit) | **MODELLED** (Program D Amendment D) | adverse selection measured: maker worth +5 bps on N=20, −4 bps on N=55 |
| Execution delay | signals on closed bar t, fills at open[t+1] | **CONVENTION** | zero-lookahead rule used by all programs |
| Funding payments | actual settlement timestamps, rate × notional, direction-aware (long pays positive rate, short receives) | **VERIFIED convention** | Program D funding-sign defect found & fixed; funding history full from listing |
| Funding interval | current 8h for BTC/ETH/SOL (fundingInfo, 2026-11-28); SOL historical excursion to 2h/4h in 2022-11 (observed) | **VERIFIED current / OBSERVED history** | `funding_intervals.parquet` |
| Borrowing/financing | N/A for USDⓈ-M perp (no borrow); spot margin not used | **N/A** | — |
| Liquidation fee | BTC/ETH 1.25%, SOL 1.5% (`liquidationFee` in exchangeInfo, 2026-11-28) | **VERIFIED current**; historical schedule UNKNOWN | `contract_specs.parquet`; the 1.25% figure in Program C was a delivery-futures assumption — the perp value is now sourced directly |
| Maintenance margin | maintMarginPercent 2.5% / requiredMarginPercent 5.0% for BTC/ETH/SOL (exchangeInfo, 2026-11-28); bracketed tiers snapshot 2026-11-28 10:49 UTC (12/12/10 brackets) | **VERIFIED current snapshot** | `contract_specs.parquet`, `margin_tiers.parquet`; **historical effective dates UNKNOWN** |
| Margin-tier history | not available from exchange | **UNKNOWN** | snapshots are dated observations, not reconstructed history; Phase 2 must use current snapshot as assumption and flag it |
| Position-size constraints | min notional (BTC 50 / ETH 20 / SOL 5 USDT), lot/tick sizes | **VERIFIED current** | `contract_specs.parquet` |
| Data/signal latency | recorder + watchdog live; historical data PIT-clean | **VERIFIED** | v4 items 1–3 |
| Exchange outages / WS gaps | recorder logs gaps; historical klines have exchange-side gaps (documented, not filled) | **OBSERVED** | item 5 verification |
| Rejected orders / reconnection | paper-trading phase concern | **N/A for backtest** | — |

**Rules:** no zero-cost assumptions; missing values are stress-tested,
not ignored; historical fee/margin schedules are NOT assumed constant —
where only current snapshots exist, the snapshot is used with an explicit
"current-snapshot assumption" flag in every result.

## II.4 Phase 2 trial budget

| Limit | Value | Rationale |
|---|---|---|
| Max total strategy trials (Phase 2) | **40** | Program D alone ran 1,354 configs and found nothing; DSR penalty at N=1,456 already makes p<0.05 nearly unreachable for marginal edges. 40 is enough for a technical baseline family with walk-forward, small enough to keep DSR meaningful. |
| Max per strategy family | **12** | one family = one mechanism (e.g. trend, mean-reversion, breakout). 12 configs per family allows a small parameter grid without data mining. |
| Max hyperparameter configurations per trial | **200** | grid search within a trial counts as one trial but every config is logged; 200 keeps compute bounded. |
| Max repeated trials per configuration | **3** | a 4th run of the same config requires explicit user approval and is logged as a new trial. |
| Model/feature selection | **frozen before holdout** | features selected on IS+OOS only; holdout never used for selection. |
| Validation | chronological walk-forward, 5 folds, purged/embargoed windows | matches Program D convention. |
| Stop condition for a family | if a family fails all 12 configs on OOS net expectancy ≤ 0 at 1.5x costs, the family is closed and reported as NO EDGE | prevents endless tweaking. |
| Failed-experiment documentation | every trial appended to `v4_system/trials_log.csv` with params, result, net expectancy, OOS trades, max DD, pass/fail | matches Program D governance. |
| Post-holdout modification | any change after holdout exposure requires a NEW registration and a NEW holdout | prevents holdout reuse. |

**Budget is NOT spent by this document.** Phase 2 has not started.

## II.5 Freeze commit and holdout definition

1. **Freeze commit**: the commit that introduces this Part II document
   (hash recorded in `v4_system/trials_log.csv` and the item-7 final
   report). From that commit onward, the research specification is
   frozen: hypotheses, cost model, trial budget, evaluation criteria.
2. **Holdout period**: **2026-11-01 00:00:00 UTC → 2026-11-28** (end of
   available v4 data). ~27 days.
3. **Holdout validity evidence**:
   - No intraday experiment (Programs D, E) consumed data after
     2026-10-31 (Program E 1h holdout slice ends 2026-10-31).
   - Programs B/C consumed the same calendar window at **daily**
     frequency only (different timeframe, different strategies) —
     disclosed contamination risk: LOW but non-zero.
   - The V4 engine fetched this window on 2026-11-28; no strategy has
     read it.
4. **Previous exposure**: BTC/SOL 2026-04→2026-10 burned; ETH
   2026-04→2026-10-09 reserved by Program D (not reused); everything
   ≤2026-10-31 used by at least one program.
5. **Holdout usage rules**: single-touch evaluation of the frozen
   candidate only; no feature selection, hyperparameter tuning, or
   strategy selection on holdout data; results reported as-is.
6. **Honest limitation**: 27 days is SHORT. A 1h strategy produces
   ~650 bars; a 4h strategy ~160 bars. The holdout can confirm or veto
   a candidate but cannot establish statistical significance alone.
   This is reported honestly: the holdout is a **veto gate**, not a
   proof of edge. If Phase 2 candidates need more holdout data, the
   only defensible option is to WAIT for new data to accumulate (the
   recorder is running) and extend the holdout forward — never to
   reach backward into burned windows.
7. **Post-freeze changes**: any modification to strategy, features,
   costs, or evaluation criteria after the freeze commit requires a new
   PREREG amendment (dated, appended, committed BEFORE the change runs)
   and, where the change affects the candidate, a new holdout period.

## II.6 Research integrity rules (explicit prohibitions)

The following are prohibited in Phase 2 and all future phases:

- **Lookahead bias** — no future information in features, signals, or
  fills; signals on closed bars, fills at next open.
- **Survivorship bias** — universe fixed at listing; delisted symbols
  are not silently dropped (Program E disclosed this bias; V4 universe
  is BTC/ETH/SOL, all still listed — risk is low but noted).
- **Future funding or margin information** — funding rates and margin
  tiers are used only at their observation timestamps; `available_at`
  PIT filtering is mandatory.
- **Test-set optimization** — holdout is never used for selection.
- **Repeated unregistered experiments** — every trial logged before
  results are seen.
- **Cherry-picking profitable date ranges** — all windows reported,
  including negative ones.
- **Ignoring transaction costs** — no zero-cost assumptions; stress
  costs mandatory.
- **Omitting failed trials** — `trials_log.csv` is append-only.
- **Changing evaluation criteria after observing results** — criteria
  frozen in Part I and Part II; changes require amendment.

**Required practices**: chronological walk-forward validation; strict
separation of training / validation / holdout; every experiment
configuration, code commit, and dataset version traceable (git hash +
manifest entry); "no tradable edge found" is a valid, reportable result.

## II.7 Scope statement

This Part II modifies documentation only. No strategy, ML model,
indicator, or backtesting calculation was changed. No Phase 2 experiment
was run. No new historical data was fetched. The live recorder and
watchdog were not altered. No live orders were placed. Previously
recorded experiment results were not changed.

---

# PART III � Phase 2 Family 1 Registration (Turtle/Donchian + Daily Trend Filter)

*Registered 2026-11-28 on branch `phase2-v4-adapter`, BEFORE any Phase 2
historical strategy evaluation. This amendment does not modify the
freeze commit or holdout definition in Part II.5.*

## III.1 Relationship to prior research

This family reuses the EXACT Program D Amendment C machinery:
`quant_research/strategies/turtle_strategy.py` (Donchian breakout,
Wilders ATR sizing, pyramiding, 2N stop, 1m sub-bar resolution) and
`quant_research/candidate_runner.py` (`daily_trend_map` /
`apply_daily_mask`, the audited daily EMA(20) trend gate). The daily
filter's motivation comes from Amendment C results (C1 improved 11 of 12
pooled pairings vs C0) � a direction worth keeping alive, NOT a proven
edge. Program D tested 5m/15m/1h on the quant_research archive with
IS 2024-01?2025-03 / OOS 2025-04?2026-03 and found NO EDGE (0/24 pooled
configurations cleared all hurdles). Family 1 is a NEW registered trial
under the V4 budget on V4 data � it is not a repeat of Amendment C
(different data source, different date coverage, 1h/4h only, max_units=1,
daily filter mandatory).

## III.2 Strategy definition (frozen)

- **Universe**: BTCUSDT, ETHUSDT, SOLUSDT USD?-M perpetuals.
- **Decision timeframes**: 1h and 4h (4h is a new timeframe application
  of identical rules � same channels, sizing, stops; NOT a parameter
  optimization).
- **Entry (long)**: close[t] > highest high of prior N bars (excluding
  bar t); **entry (short)**: close[t] < lowest low of prior N bars.
  Fill at open[t+1]. Daily trend gate: long entries only when the
  previous COMPLETED UTC day's close > its EMA(20) on daily closes
  resampled from 1m perp data; short entries only when trend == -1.
  Missing/warmup trend blocks the direction (no default).
- **Exit**: 2N ATR protective stop (N = channel lookback, ATR = Wilder
  ATR(N)); M-bar Donchian channel exit; intrabar stop-vs-level
  ambiguity resolved by 1m sub-bars ([bar_open, next_open) window,
  chronological, no lookahead); missing sub-bars -> documented
  pessimistic stop-first fallback, counted in `ambiguous_bars`.
- **Sizing**: 1% of equity risk per unit, stop distance 2*ATR;
  pyramiding DISABLED (max_units=1); portfolio gross leverage cap 2.0x;
  minimum order $10 notional.
- **Funding**: actual settlement timestamps, direction-aware
  (long pays positive rate, short receives; reversed when negative).
- **Costs**: base = taker 5 bps + slippage 2 bps per side (14 bps RT);
  stress = taker 5 bps + slippage 4 bps per side (18 bps RT). No maker
  fills (execution methodology is taker-only). Spread/market impact
  UNKNOWN -> covered by stress slippage, never zero.

## III.3 Configuration matrix (8 configurations, within 12-trial family cap)

Channel pairs (N, M) � the four pairs documented in Program D:
(20,10), (20,20), (55,10), (55,20).
Timeframes: 1h, 4h.
Matrix = 4 pairs x 2 TFs = **8 configurations**, single arm (daily filter ON, max_units=1). Each configuration runs on ALL THREE universe assets - asset is a universe dimension, not a configuration parameter (8 configs x 3 assets = 24 runs).
(daily filter ON, max_units=1). No other parameters vary.
`bars_per_year`: 1h = 8760, 4h = 2190.

## III.4 Date ranges

- **Development window**: 2024-01-01 ? 2026-10-31 UTC for BTCUSDT and
  SOLUSDT. For ETHUSDT: 2024-01-01 ? 2026-03-31 UTC � the ETH
  2026-04-01?2026-10-09 window remains RESERVED per Part II.2 and is
  excluded from Family 1.
- **Holdout**: 2026-11-01 ? 2026-11-28 UTC � ENTIRELY EXCLUDED. No
  Family 1 result may use it; it remains the single-touch veto gate.
- Prior exposure is acknowledged: BTC 2025+ and SOL 2026-04+ were used
  by earlier programs (Part II.2). Family 1 results on those windows
  are therefore exploratory, not pristine; the DSR trial count
  (1,354 prior + Family 1) accounts for selection pressure.

## III.5 Validation protocol

- **Five chronological walk-forward folds** over the development
  window per symbol/TF: fold boundaries computed chronologically
  (expanding train, forward test), no shuffling, no overlap leakage.
- **Metrics**: net expectancy (bps/trade), OOS trade count, max
  drawdown, daily Sharpe (sqrt(365)), deflated Sharpe ratio at
  N_trials = 1,354 + registered Phase 2 trials, block-bootstrap CI.
- **Pass hurdles** (frozen): net expectancy > 0 after base costs;
  positive under stress costs; DSR >= 0.95; max drawdown <= 20%;
  >= 30 OOS trades per configuration; beats random-entry baseline.
- **Stopping rules**: family stops after 12 registered trials OR when
  all 8 configurations fail hurdles; failed trials are recorded, never
  omitted.

## III.6 Budget allocation

- Family 1: **1 trial** containing 8 configurations (of 12 allowed).
- Global Phase 2 budget: 40 trials total; this registration consumes
  1 upon execution (registration row appended to
  `v4_system/trials_log.csv` BEFORE any results).
- Repeats of identical (family, config) limited to 3.

## III.7 Data & PIT labelling

All Family 1 inputs load via `v4_system/adapters/quant_research_data.py`
with `strict=False` and carry the label
`EVENT-TIME PIT � HISTORICAL AVAILABILITY UNKNOWN`. Results must be
reported with this label; strict-PIT claims are prohibited. OI features
excluded (~30-day coverage). Margin tiers NOT used for historical
liquidation claims (effective dates unknown).

## III.8 Scope statement

This amendment registers a strategy family and budget only. No strategy
code was modified (`turtle_strategy.py`, `candidate_runner.py` logic
unchanged � only the series-based `daily_trend_from_daily_close`
extraction, behaviour-identical). No historical backtest was run. No
holdout data was accessed. No live orders. `master` is untouched.

## III.9 Amendment � Drawdown methodology (2026-11-28)

*Clarifies the "max drawdown <= 20%" hurdle in III.5. No threshold,
configuration, date range or budget is changed.*

- **Configuration-level maximum drawdown** =
  `max(fold_1_MDD, ..., fold_5_MDD)` � the WORST OOS fold's marked
  maximum drawdown. Every OOS fold must individually satisfy marked
  MDD <= 20% (equivalent, since the max is taken over folds).
- **Independent fold accounting**: each walk-forward fold is an
  independent simulation with its own initial equity. Fold equity
  curves are NEVER concatenated or stitched; no artificial equity
  continuity is created between folds.
- **Marked equity** = cash equity (realized PnL + fees + slippage +
  funding, each applied exactly once) PLUS unrealized PnL of open
  units marked at the bar close. Position sizing and leverage caps
  continue to use cash equity only � strategy rules unchanged.
- **Cash-only drawdown** (realized PnL + funding, no unrealized
  marking) is reported as a supplementary diagnostic per fold and per
  configuration; it is NOT the acceptance metric.
- **Sampling frequency**: the marked equity curve is sampled once per
  DECISION-BAR CLOSE (1h or 4h, matching the configuration timeframe).
  It is therefore a CLOSE-SAMPLED mark-to-market drawdown. It does
  NOT capture the worst intrabar equity excursion between decision
  bars; 1m sub-bars are used only for exit-order resolution, not for
  equity marking. Any change to intrabar equity sampling requires a
  separate amendment.
- **Open positions at fold end**: excluded from trade statistics
  (never completed) but included in marked equity at the final bar.

## III.10 Amendment � Random-entry baseline for Family 1 (2026-11-28)

*Implements the "beats random-entry baseline" hurdle from III.5,
adapted from the audited `quant_research/turtle_random_entry_v3.py`
construction to V4 data and Family 1 folds. Registered BEFORE any
Family 1 result exists.*

- **Null construction**: for each config x asset x fold, R = 200
  random-entry simulations through the SAME `run_turtle_simulation`
  code path (Donchian M-bar exits, 2N ATR stop, max_units=1, 1% risk,
  2x leverage cap, next-open fills, 1m sub-bar exit resolution,
  direction-aware funding, base 14 bps RT costs).
- **Random signals**: entry channels replaced by sparse triggers at
  bars drawn uniformly without replacement from the fold's TEST window
  only (warm-up bars never receive signals). Per-fold signal count =
  the strategy's OOS trade count in that fold; direction mix = the
  strategy's realised long share in that fold.
- **Daily trend filter is NOT applied to the null** � it is part of
  the strategy under test, not of the random-entry null.
- **Statistic**: pooled OOS net expectancy (bps/trade) across the five
  folds per config x asset, base costs.
- **p-value**: `p_beats` = fraction of the 200 null runs whose pooled
  expectancy >= the strategy's. Hurdle: `p_beats <= 0.05` (one-sided).
- **Seeds**: deterministic `zlib.crc32("family1|asset|tf|N|M|run")` �
  bit-reproducible across machines.
- **Scope**: baseline runs under BASE costs only; the stress-cost
  hurdle is evaluated separately on the strategy itself.

## III.11 Amendment — DSR semantics correction and trial-count reconstruction (2026-11-28)

*Integrity correction issued AFTER the Family 1 execution completed.
No strategy result, trade record, cost assumption or fold boundary is
modified; only the statistical interpretation of the already-computed
DSR values and the trial-count constant are corrected.*

### III.11.1 The defect

`quant_research/core/metrics.compute_dsr()` returns a **p-value**
(`1 - Phi(z)`): small values indicate the observed Sharpe exceeds the
expected maximum under the null of N competing trials. Part III.5 was
written as "DSR >= 0.95", which reads the p-value as a confidence
score — inverted semantics. Program D's own runners
(`turtle_runner.py`, `maker_runner.py`, `funding_runner.py`) used the
correct direction (`dsr_p < 0.05`); only the Family 1 runner
(`v4_system/run_family1.py`) implemented the inverted check
(`dsr >= 0.95`) and stored the value under the ambiguous field name
`"dsr"`.

### III.11.2 The correction

- Acceptance now requires `dsr_p_value <= 0.05` (p-value semantics,
  matching Program D's `dsr_p < 0.05`).
- The pooled-stats field is renamed `dsr_p_value`; no field named
  `dsr` remains in new outputs.
- `compute_dsr()` itself is UNCHANGED — its mathematics were verified
  against an independent SciPy implementation
  (`tests/test_sharpe_and_dsr.py`); no defect was found in the
  function, only in the runner's comparison direction and naming.

### III.11.3 Trial-count reconstruction

The pre-correction constant was `N_TRIALS_DSR = 1354 + 1 = 1355`.
Reconstruction:

- Program D ended at **1,456** registered configurations
  (1,354 base + 72 Amendment C re-logged + 24 Amendment D maker +
  6 Amendment E funding-sign), per PREREG II.1.
- Family 1 evaluated **8 channel configs x 3 assets = 24 distinct
  config-asset hypotheses**. Walk-forward folds and base/stress cost
  scenarios are robustness checks of the SAME hypothesis, not
  independent trials, and are NOT counted.
- Corrected count: **N = 1,456 + 24 = 1,480**.

Sensitivity: the DSR p-value depends on N only through
`E[max SR | null] ~ sqrt(2 ln N) / sqrt(T)`; between N=1,455 and
N=1,480 the p-value moves by <0.001 for every Family 1 config-asset,
so the verdict is insensitive to the exact count within this range.

### III.11.4 Effect on Family 1

All 24 config-assets already FAILED the random-entry baseline hurdle
(`p_beats` 0.19-0.60 >> 0.05), so the corrected DSR direction cannot
change any acceptance classification: the verdict remains
**NO EDGE (0/24)**. The corrected DSR p-values are reported in
`v4_system/results/family1_turtle_dt-001-f4f572efaa88b092_dsr_corrected.json`;
the original results artifact is preserved byte-for-byte.

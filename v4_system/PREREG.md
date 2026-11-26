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

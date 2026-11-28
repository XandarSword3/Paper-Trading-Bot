# V4 ML Research Specification — DRAFT (not frozen, not registered)

**Status**: DRAFT for review. This document is NOT appended to the binding
preregistration (`v4_system/PREREG.md`) and does NOT declare any freeze.
No trial budget is spent by this document. No model has been trained.

**Date**: 2026-11-28 · **Branch**: `phase2-v4-adapter` @ `16e0673`

---

## 1. Research hypothesis

**H-ML1**: On 1-hour decision bars, a supervised model using only
information available at bar close can predict the sign and magnitude of
the BTCUSDT/ETHUSDT/SOLUSDT perpetual return over the next **4 hours**
well enough that, after 14 bps round-trip costs (18 bps stress), a
thresholded long/short/no-trade policy produces positive net expectancy
out-of-sample — an edge that is not explained by random entries,
majority-class prediction, or the already-rejected Turtle family.

This is a **supervised-learning research program**, not a manually
parameterized technical strategy. All prior directional families
(Turtle/Donchian, mean reversion, funding-sign, cross-sectional
momentum, pairs, premium fade) are documented NO EDGE / KILL in
`v4_system/PREREG.md` Parts I–IV and `research/new_edge/FINDINGS.md`;
the ML family must beat those same hurdles, not reuse their signals.

**Instruments**: BTCUSDT, ETHUSDT, SOLUSDT perpetuals (USDT-margined).
**Decision bar**: 1h closed candles. **Prediction horizon**: 4h forward.

## 2. Model architectures (fixed — no additions in Family ML-1)

### 2.1 Logistic Regression (interpretable baseline)
- Input: standardized feature vector at decision bar close.
- Output: probability `p_up` of positive 4h forward return (binary target,
  §5) — or class probabilities for the 3-class variant if selected.
- Role: statistical baseline; coefficients are inspectable for economic
  plausibility; guards against LightGBM overfitting narrative.

### 2.2 LightGBM (nonlinear gradient boosting)
- Input: same feature vector.
- Output: same target as logistic regression.
- Role: captures nonlinear interactions (e.g., OI×price divergence,
  funding×volatility regimes) that linear models cannot.

**Excluded from Family ML-1** (explicit): transformers, neural networks,
reinforcement learning, any third architecture. Adding one requires a new
amendment and consumes trial budget.

### 2.3 From prediction to action
A predicted probability/score is **not** directly a trade. The decision
rule is:

```
trade_long  if  expected_net_edge_long  >  threshold_long
trade_short if  expected_net_edge_short >  threshold_short
else         no-trade
```

where `expected_net_edge` maps the model output to an expected 4h return
minus costs:

- For binary target: `E[r_4h | signal]` is estimated from the model's
  predicted class probability calibrated on the validation fold, minus
  the **14 bps RT cost** (18 bps stress). A trade is taken only when the
  calibrated expected net edge exceeds the threshold.
- For regression target (if selected, §5): `E[r_4h]` directly minus costs.
- Thresholds are **fixed before training** (§7.4) — they are part of the
  registered configuration, not tuned on results.

**Key principle**: probability of an up-move alone is insufficient
evidence; the decision requires expected *net* edge after costs, with
the cost model identical to Family 1 (`CostModel`: taker 0.05% +
slippage 0.02% per side = 14 bps RT; stress 18 bps RT).

## 3. Feature specification (fixed initial set)

All features computed on **closed 1h bars only**, using data with
event-time ≤ decision timestamp. No feature may use information from
the current unfinished bar or any future bar.

### 3.1 Price & volume (per instrument)
- OHLCV of the decision bar and trailing bars.
- Log returns over windows: 1h, 4h, 24h, 72h, 168h (7d).
- EMA relationships: close vs EMA(20), EMA(50), EMA(200); EMA20−EMA50,
  EMA50−EMA200 spreads (normalized by ATR).
- RSI(14), MACD(12,26,9) line/signal/histogram.
- ATR(14) and realized volatility over 24h/168h.
- Donchian position: (close − lowest_low_20) / (highest_high_20 − lowest_low_20);
  Bollinger %B (20, 2σ).
- Relative volume: volume / SMA(volume, 168).

### 3.2 Funding (per instrument)
- Latest settled funding rate (8h settlements; use last settlement
  strictly before decision time).
- Funding change: Δ over 1, 3, 7 settlements.
- Funding direction: sign and magnitude of trailing 7-day mean funding.
- Hours since last settlement (settlement timing feature).

### 3.3 Cross-asset (BTC/ETH/SOL panel)
- Each instrument's returns vs BTC's returns over 4h/24h/72h (beta proxy).
- Cross-asset momentum: BTC 24h/72h return as a feature for ETH/SOL
  (and symmetric where justified — BTC model may use ETH/SOL returns
  as risk-sentiment features).
- Rolling 72h correlation of each asset's 1h returns with BTC's.
- Relative strength: asset return minus BTC return over 24h/72h.

### 3.4 Open interest (conditional — see §4)
**Proposed limited derived set** (not all 8 archive fields):
- `oi_change_1h`, `oi_change_4h`, `oi_change_24h`: % change in
  `sum_open_interest`.
- `oi_value_change_4h`: % change in `sum_open_interest_value` (USD terms).
- `oi_price_divergence_4h`: sign(oi_change_4h) × sign(price_return_4h)
  disagreement flag — OI up + price down (short buildup) vs OI up +
  price up (long buildup).
- `toptrader_ls_change_24h`: change in `sum_toptrader_long_short_ratio`.
- `taker_ls_change_4h`: change in `sum_taker_long_short_vol_ratio`.

**Excluded**: raw levels of all ratio columns (non-stationary, scale
drift), `count_*` variants (redundant with `sum_*` for direction),
any field not listed above. Each included field must pass timestamp
and schema validation (§4) before use; if validation fails, OI features
are dropped entirely and the model runs without them (documented, not
silently imputed).

## 4. OI archive data contract

Source: `data.binance.vision` `data/futures/um/daily/metrics/{SYM}/`
daily zips (official Binance archive; investigation report 2026-11-28).

**Ingestion must preserve, per file**:
- Original `create_time` strings exactly as published (no shifting,
  no re-interpretation at ingest).
- Original 8-column schema verbatim.
- Symbol, source file name, archive date (from filename), retrieval
  timestamp (UTC), S3 ETag, `.CHECKSUM` value, S3 `LastModified`.
- Duplicate-row and missing-row diagnostics per file.
- Timestamp-convention classification per file (§4.1).
- Provenance and data-quality flags.

**PIT label**: `EVENT-TIME PIT — HISTORICAL AVAILABILITY UNKNOWN`.
`available_at` remains NaT for all historical rows. Archive retrieval
time and S3 `LastModified` are **not** publication times and must not
be treated as such.

### 4.1 June 25, 2026 convention change — handling
- Documented (binance-public-data issue #491): `create_time` labeling
  flipped end-of-period → start-of-period at 2026-06-25 00:00 UTC.
- **Not assumed definitive for every file**: issue #509 leaves
  retrospective-relabeling questions open. Current sampled files
  (2021-12 → 2026-11) all show uniform `00:00…23:55` labels, but this
  may reflect relabeling.
- **Policy**: preserve raw labels; classify each file's label convention
  at ingest (first/last label vs file date); store the classification;
  do **not** auto-shift timestamps. For feature construction, treat the
  OI observation at label T as belonging to the interval ending at T
  (conservative: usable for decisions at T + 5min under either
  convention — a 5-minute ambiguity, immaterial at 1h decision bars but
  explicit in code).
- Archive OI stays in **separate datasets** from existing REST OI
  (`*_oi_*`) until consistency is validated; the REST 30-day tail and
  archive history are never silently merged.

## 5. ML target definition

### 5.1 Options compared
| Approach | Pros | Cons |
|---|---|---|
| Continuous forward return | Directly maps to expectancy; no threshold ambiguity | Heavy-tailed; MSE dominated by outliers; needs robust scaling |
| Binary up/down | Simple, calibrated probabilities | Ignores magnitude; ~50/50 class balance uninformative |
| 3-class long/neutral/short | Matches trading decision; neutral = no-trade | Neutral band is arbitrary; class imbalance |

**Recommended primary target: continuous forward log-return**
`y = log(close_{t+4h} / close_{t+1h_open})` — i.e., return from the
**next bar's open** (executable entry) to the close 4 hours later,
computed on closed bars only. Rationale: directly estimates expected
net edge; avoids arbitrary class boundaries; the trading threshold
(§2.3) handles the long/short/no-trade decision downstream.

Binary classification is retained as a **secondary diagnostic**
(sign of y) for interpretability, not as the primary target.

### 5.2 Label construction rules
- Entry reference: **open of bar t+1** (first tradable price after the
  decision bar t closes) — no same-bar fills.
- Exit reference: **close of bar t+4** (4 hourly bars after entry bar).
- Forward return uses only future prices for the *label*; features never
  see them.
- **Overlapping labels**: consecutive 1h decisions produce overlapping
  4h windows — handled by purge/embargo in CV (§7.2), not by thinning
  (thinning would discard data).
- **Transaction costs**: 14 bps RT base / 18 bps stress, identical to
  Family 1 `CostModel`; funding applied where positions span settlements
  (same engine conventions as `run_turtle_simulation`).
- **Neutral/no-trade**: |expected net edge| ≤ threshold → no position.
  Thresholds fixed pre-training (§7.4).
- **Minimum expected advantage**: trade only if calibrated
  `E[net_return] > 0` after costs; the threshold grid (§7.4) defines the
  minimum economically meaningful edge (e.g., ≥ 5 bps net per trade,
  matching Family 1's expectancy hurdle).

## 6. Data availability & leakage controls

- **Event-time PIT only** for all historical research: label
  `EVENT-TIME PIT — HISTORICAL AVAILABILITY UNKNOWN`. All history
  ≤ 2026-11-28 is development data (PREREG IV.2); the November 2026
  window is exposed and prohibited for selection.
- **One common decision timestamp** across all feature sources:
  features at decision time T use only bars/events with timestamp ≤ T
  (closed bars; funding settlements with settlement time ≤ T; OI labels
  ≤ T with the §4.1 conservative lag).
- **No unfinished candles**: decision bars must be closed (close_time ≤ T).
- **No future event timestamps**: funding/OI events dated after T excluded.
- **Conservative OI lag**: OI label T usable for decisions at T+5min or
  later (1h bars make this immaterial, but enforced in code).
- **Purge + embargo**: between train and validation folds, drop all
  samples whose 4h label window overlaps the boundary; embargo = 4h
  (one full horizon) on each side.
- **No random splitting** — chronological walk-forward only.
- **No hidden backfill**: archive files are immutable snapshots;
  relabeling risk (§4.1) is flagged per file, not corrected silently.
- **Future live data**: the recorder continues collecting; once the
  spec+code freeze occurs (PREREG IV.3), post-freeze observations carry
  real `available_at` and enable a strict availability-time evaluation
  on the forward holdout.

## 7. Evaluation design

### 7.1 Walk-forward folds (5 chronological)
Development window: listing → 2026-10-31 (all history is development;
November 2026 excluded per PREREG IV.2). Proposed folds (to be finalized
at freeze with exact dates):

| Fold | Train | Validation (OOS) |
|---|---|---|
| F1 | earliest → 2023-12-31 | 2024-01-01 → 2024-06-30 |
| F2 | → 2024-06-30 | 2024-07-01 → 2024-12-31 |
| F3 | → 2024-12-31 | 2025-01-01 → 2025-06-30 |
| F4 | → 2025-06-30 | 2025-07-01 → 2025-12-31 |
| F5 | → 2025-12-31 | 2026-01-01 → 2026-10-31 |

(Exact boundaries adjusted at freeze so each validation window has
≥ ~6 months and ≥ 30 expected OOS trades per config-asset.)

### 7.2 Purge & embargo
- Purge: remove training samples whose label window [entry, exit]
  overlaps the validation window.
- Embargo: 4h gap between train end and validation start (and vice
  versa for any refit).

### 7.3 Model fitting & calibration
- Standardization: fit on train fold only (mean/std), apply to val.
- Logistic regression: L2 regularization, class-weight balanced if
  needed; calibration via isotonic or Platt on a train-internal split.
- LightGBM: fixed hyperparameters registered pre-training (no tuning
  on OOS); early stopping on a train-internal validation slice only.
- Both models refit per fold; OOS predictions pooled across folds for
  evaluation (same pooling convention as Family 1).

### 7.4 Fixed decision thresholds (registered, not tuned)
Threshold grid is part of the configuration matrix (§9). Candidate
net-edge thresholds: {0, 5, 10} bps per 4h trade — three values, fixed
before any training. The selected threshold is part of the registered
config; no post-hoc threshold search on OOS.

### 7.5 Cost model
Identical to Family 1: base 14 bps RT (taker 0.05% + slippage 0.02%
per side), stress 18 bps RT; funding applied on held positions;
next-open fills; 1m sub-bar exit resolution where applicable.

### 7.6 Statistical evaluation
- Pooled OOS net expectancy (bps/trade), per-trade Sharpe, profit
  factor, max drawdown (marked equity, worst-fold convention from
  PREREG III.9).
- **DSR p-value ≤ 0.05** with N_trials per PREREG IV.4 (current 1,480
  + ML configs added at registration).
- **Random-entry baseline**: same construction as Family 1
  (`random_baseline_family1.py` adapted to ML decision times),
  p_beats ≤ 0.05.
- **Majority-class baseline**: always-long / always-short /
  majority-class predictor — must be beaten.
- **Turtle reference**: Family 1 Turtle results are a historical
  reference point, NOT a model-selection target.
- Minimum ≥ 30 OOS trades per config-asset; stress-cost positivity;
  worst-fold marked MDD ≤ 20%.

## 8. OI archive ingestion decision

**Does Family ML-1 require the archive?** The OI feature set (§3.4) is
conditional: without archive history, OI features exist only for the
last ~30 days — insufficient for training. The price/volume/funding/
cross-asset features alone are available historically. **Decision:
OI features are INCLUDED in the spec but gated on successful archive
ingestion validation**; if ingestion is not approved, Family ML-1 runs
without OI features (documented limitation) — the spec does not block
on it.

**If approved**, proposed ingestion:
- Scope: BTCUSDT, ETHUSDT, SOLUSDT daily metrics, 2020-09-01/2021-12-01
  → present.
- Expected files: ~2,250 days (BTC) + ~1,820 days × 2 (ETH, SOL) ≈
  **~5,900 daily zips** (~12 KB each ≈ **~70 MB compressed**, ~500 MB
  parsed CSV).
- Download: sequential with resume (skip files whose checksum+ETag
  already recorded); `.CHECKSUM` verification per file; S3 `LastModified`
  recorded as provenance (not publication time).
- Deduplication: exact-duplicate timestamps within file (early BTC
  files) and across file boundaries (the 2026-06-25 duplicate-label
  issue) — keep first, log count.
- Timestamp-validation gates: per-file label-range check vs filename
  date; convention classification; anomaly flags; sort by create_time.
- Regression tests: schema stability, dedup correctness, convention
  classifier, checksum verification, no-timestamp-shift invariant.
- **No download performed yet** — awaiting approval.

## 9. Trial-budget allocation (PREREG IV.4)

Smallest useful fixed matrix — every cell is a registered selection
opportunity:

| Dimension | Values | Count |
|---|---|---|
| Architecture | Logistic Regression, LightGBM | 2 |
| Feature set | Full (§3.1–3.4), No-OI (§3.1–3.3 only) | 2 |
| Threshold | 0, 5, 10 bps net edge | 3 |
| Instrument | BTC, ETH, SOL (evaluated per-asset, pooled verdict) | 3 |

**Registered configurations: 2 × 2 × 3 = 12 per asset = 36 config-assets**,
analogous to Family 1's 24. Walk-forward folds and cost scenarios are
robustness checks, not separate trials. DSR N grows by 36 at
registration (1,480 → 1,516).

**Explicitly excluded**: horizon variations (4h fixed), additional
architectures, feature-set permutations beyond the two listed,
threshold searches beyond the three values. Any expansion requires a
new amendment and budget.

## 10. Holdout policy

Per PREREG IV.3: the holdout begins only after the final ML spec and
code are frozen. The freeze commit hash and UTC timestamp will be
recorded in `trials_log.csv` and a dated PREREG amendment **before**
any post-freeze observation is read. Only observations after that
timestamp qualify; sufficiency target ≥ 6 months (daily-equivalent)
adjusted for 1h/4h frequency; single-touch evaluation procedure
defined in advance. No existing historical date is designated
untouched.

## 11. Known limitations

- **Event-time PIT only**: historical `available_at` unknown for all
  sources; results are exploratory, not strict-availability proofs.
- **OI labeling ambiguity**: 2026-06-25 convention change + possible
  retrospective relabeling (issues #491/#509) — mitigated by raw-label
  preservation and conservative lag, not eliminated.
- **Overlapping labels**: 4h horizon on 1h decisions → autocorrelated
  samples; purge/embargo mitigates but does not remove dependence;
  DSR and block-bootstrap CIs account for it partially.
- **Regime coverage**: training data includes multiple regimes but no
  guarantee the forward holdout resembles them.
- **Cost model**: no historical spread/impact data; 14/18 bps RT is a
  convention, not a measurement — stress scenario partially covers it.
- **Multiple testing**: 1,480+ prior trials; DSR threshold is strict;
  a marginal ML result is more likely a false positive than a true edge
  given the program's consistent NO EDGE history.
- **November 2026 exposure**: all history ≤ 2026-11-28 is development
  data; no pristine historical holdout exists.

## 12. Exact conditions required before training

1. This draft reviewed and approved; appended to `PREREG.md` as a
   dated amendment (Family ML-1 registration).
2. OI ingestion decision made (approve archive ingestion or run
   No-OI variant); if approved, ingestion code + tests merged first.
3. Feature pipeline implemented with the §6 leakage controls and
   §4.1 timestamp policy; unit tests for purge/embargo, closed-bar
   enforcement, OI lag.
4. Trial registration in `trials_log.csv` (36 config-assets, budget
   check: 1/40 used → 37 remaining after registration).
5. Freeze commit + UTC timestamp recorded; forward holdout clock starts.
6. Only then: model fitting on train folds, OOS evaluation on
   validation folds, baselines, DSR, report.

---

*End of draft. Awaiting review before any registration or training.*

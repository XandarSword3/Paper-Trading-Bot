# Quantitative Strategy Pre-Registration Plan (PREREG)

**Date of Pre-Registration:** 2026-10-10  
**Repository Branch:** `quant-research-edge-test`  
**Purpose:** Pre-register hypotheses, indicator stacks, parameter grids, execution bias rules, cost models, data splits, and pass/fail criteria BEFORE touching data or executing tests. This document commits the research plan to prevent post-hoc hypothesis generation, p-hacking, and parameter data mining.

---

## 1. Governance and Trial Logging
* Every strategy configuration and parameter set evaluated in this research pipeline MUST be appended to [`quant_research/trials.csv`](file:///d:/Work/Attempts%20to%20Code/BTC%20Strategy/BTC%20Strategy/quant_research/trials.csv).
* The trial log records: `trial_id, timestamp_utc, test_name, asset, timeframe, strategy_name, params_json, is_return_pct, is_expectancy_bps, oos_return_pct, oos_expectancy_bps, status`.
* The total trial count will be reported in all summaries and factored into the Deflated Sharpe Ratio calculation to penalize multi-hypothesis testing.

---

## 2. Data Splits and Hold-Out Integrity

### 2.1 Fixed Chronological Splits
* **In-Sample (Selection / Training):** `2024-01-01 00:00:00 UTC` to `2025-03-31 23:59:59 UTC` (15 months).
* **Walk-Forward Out-Of-Sample (OOS):** `2025-04-01 00:00:00 UTC` to `2026-03-31 23:59:59 UTC` (12 months).
* **Final Hold-Out:** `2026-04-01 00:00:00 UTC` to `2026-10-09 23:59:59 UTC` (6+ months).
* **User Live Window (Benchmark):** `2025-04-04 00:00:00 UTC` to `2025-05-12 23:59:59 UTC`.

### 2.2 Hold-Out Status
* **BTC Hold-Out:** **BURNED.** Audit confirms `data/trades_v4.json`, `data/BTCUSDT_4h.csv`, `tmp_results.json`, and bot states (`bot_state_v4.json`) previously ingested and scored BTC data through August 2026. Therefore, the BTC hold-out cannot be treated as clean out-of-sample data. All BTC results will be explicitly labeled as OOS/Historical.
* **SOL & ETH Hold-Outs:** **FRESH AND FROZEN.** The period `2026-04-01` to `2026-10-09` will remain untouched. It will only be evaluated ONCE at the very end with frozen parameters. No tuning or iteration is permitted after hold-out exposure.

### 2.3 Rolling Walk-Forward Scheme
To fit data starting `2024-01-01` with OOS starting `2025-04-01`:
* **Window Dimensions:** 12-Month Train / 3-Month Test / 3-Month Step.
* **Folds:**
  * **Fold 0:** Train `2024-01-01` → `2024-12-31` | Test (OOS) `2025-01-01` → `2025-03-31`
  * **Fold 1:** Train `2024-04-01` → `2025-03-31` | Test (OOS) `2025-04-01` → `2025-06-30`
  * **Fold 2:** Train `2024-07-01` → `2025-06-30` | Test (OOS) `2025-07-01` → `2025-09-30`
  * **Fold 3:** Train `2024-10-01` → `2025-09-30` | Test (OOS) `2025-10-01` → `2025-12-31`
  * **Fold 4:** Train `2025-01-01` → `2025-12-31` | Test (OOS) `2026-01-01` → `2026-03-31`
* **Selection Filter:** Combos with $< 100$ completed trades during the training window are strictly disqualified.
* **Selection Metric:** **Net Expectancy in basis points (bps) per trade**, NOT Sharpe Ratio.

---

## 3. Realistic Cost and Execution Friction Model

All calculations apply the following friction:
* **Taker Fee:** $0.05\%$ per side ($5\text{ bps}$). (Confirmed from real user fill logs; 99% taker).
* **Base Slippage:** $0.02\%$ per side ($2\text{ bps}$).
  * **Total Base Round-Trip Execution Cost:** $2 \times (5 + 2) = 14\text{ bps}$.
* **Stress Slippage:** $0.04\%$ per side ($4\text{ bps}$).
  * **Total Stress Round-Trip Execution Cost:** $2 \times (5 + 4) = 18\text{ bps}$.
* **Funding Rate Cash Flows:** Real funding settlements applied at exact timestamps from historical perpetual funding archives (never assuming a rigid 8h schedule). Long positions pay $\text{rate} \times \text{notional}$; short positions receive $\text{rate} \times \text{notional}$ (reversed if funding rate is negative).
* **Fills:** Signals evaluate strictly on closed bars $t$. Fills occur at `open[t+1]`. Zero lookahead.
* **Intrabar Resolution:** When stop loss and take profit are both touched within a higher timeframe candle, 1-minute klines determine the exact sequence. If ambiguous on 1m, the stop loss is assumed to have hit first.
* **Pyramiding Invariant:** At most one entry or pyramid add per bar (`assert_no_same_bar_pyramiding`).

---

## 4. Test 1 — Mechanical Reconstruction of Indicator Stack

**Asset:** SOLUSDT Perpetual  
**Timeframes:** 1m, 3m, 5m  
**Indicator Definitions:**
1. **Bollinger Bands:** Length 20, Standard Deviation Multiplier 2.0.
2. **RSI:** Length 6 (ultra-short momentum).
3. **Stochastic RSI:** RSI Length 14, Stochastic Length 14, Smooth %K 3, Smooth %D 3.
4. **MACD:** Fast 12, Slow 26, Signal 9.
5. **Moving Averages (EMAs):** 20, 50, 80, 200.
6. **ADX:** Length 14 (trend strength filter).
7. **Volume MA:** 20-period simple moving average of volume.

### Test 1 Mechanical Strategy Variants
* **Variant 1A — Multi-EMA Trend Alignment & ADX Filter:**
  * Long: EMA 20 > EMA 50 > EMA 80 > EMA 200, ADX > 25, close crosses above EMA 20.
  * Short: EMA 20 < EMA 50 < EMA 80 < EMA 200, ADX > 25, close crosses below EMA 20.
  * Exit: Close crosses opposite EMA 50 or trailing ATR stop ($3.0\times\text{ATR}$).
* **Variant 1B — Bollinger & RSI Momentum Mean Reversion:**
  * Long: Close <= Lower BB, RSI(6) < 20, StochRSI %K < 20 & %K crosses above %D.
  * Short: Close >= Upper BB, RSI(6) > 80, StochRSI %K > 80 & %K crosses below %D.
  * Exit: Close touches Middle BB (Basis) or hard stop at $2.0\times\text{ATR}$ or max hold 15 bars.
* **Variant 1C — MACD Trend Breakout with Volume Confirmation:**
  * Long: MACD Histogram > 0, MACD Line crosses above Signal, Volume > 1.5x Volume MA, Close > EMA 50.
  * Short: MACD Histogram < 0, MACD Line crosses below Signal, Volume > 1.5x Volume MA, Close < EMA 50.
  * Exit: Opposite MACD signal or $2.5\times\text{ATR}$ trailing stop.
* **Variant 1D — Confluence Stack (Full Indicator Agreement):**
  * Long: EMA 20 > EMA 50, Close > Middle BB, RSI(6) > 50, MACD Line > Signal, ADX > 20, Volume > Volume MA.
  * Short: EMA 20 < EMA 50, Close < Middle BB, RSI(6) < 50, MACD Line < Signal, ADX > 20, Volume > Volume MA.
  * Exit: Close crosses Middle BB or trailing stop.

---

## 5. Test 2 — Parameter Grid Exploration

Pre-registered search space for Test 2:
* **BB Multiplier:** [1.5, 2.0, 2.5]
* **RSI(6) Extreme Thresholds:** [15/85, 20/80, 25/75]
* **ADX Filter Threshold:** [20, 25, 30]
* **Stop Loss Multiple ($N\times\text{ATR}$):** [1.5, 2.0, 3.0]
* **Take Profit / Trailing Multiplier:** [2.0, 3.5, 5.0, basis_reversion]
* **Time Stop (Max Hold Bars):** [10, 20, 40, unlimited]

Total combinations per timeframe: $3 \times 3 \times 3 \times 3 \times 4 \times 4 = 1,296$ configs.  
All runs logged in `quant_research/trials.csv`.

---

## 6. Pass / Fail Decision Criteria

A strategy configuration is deemed to have a **statistically meaningful edge** ONLY if ALL of the following criteria are met:
1. **Net Expectancy Hurdle:** Net expectancy $\ge +5\text{ bps}$ per trade after $14\text{ bps}$ base round-trip costs and funding payments.
2. **Stress Cost Viability:** Net expectancy $> 0\text{ bps}$ after $18\text{ bps}$ stress round-trip costs.
3. **Statistical Significance:** Deflated Sharpe Ratio (DSR) $p\text{-value} < 0.05$ after accounting for total trials logged in `trials.csv`.
4. **Sample Size:** Completed trades $\ge 100$ in-sample and $\ge 50$ out-of-sample.
5. **Drawdown Tolerance:** Max Drawdown $< 30\%$ on stitched walk-forward equity.
6. **Profit Factor:** Gross Profit / Gross Loss $\ge 1.25$ after all fees.

Any configuration failing to clear these hurdles will be objectively labeled **NO EDGE**. A clean "no edge" result is accepted as a successful scientific conclusion.

---

## 7. Pre-Registration Addendum: Classical Turtle / Donchian Breakout & Test 3 User Replay

**Date of Addendum:** 2026-10-10  
**Commit Status:** Committed strictly BEFORE running any Turtle simulation or Test 3 CSV replay.

### 7.1 Scope and Parameter Search Space
Evaluation of the classic trend-following Turtle / Donchian Breakout system across three major perpetual contracts:
* **Assets:** SOLUSDT, BTCUSDT, ETHUSDT
* **Timeframes:** 5m, 15m, 1h (resampled from continuous 1m tick/kline archives)
* **Channel Lookbacks:**
  * Entry Breakout Period $N \in \{20, 55\}$ (System 1: 20-bar breakout; System 2: 55-bar breakout)
  * Exit Breakout Period $M \in \{10, 20\}$ (System 1: 10-bar breakout; System 2: 20-bar breakout)
  * Total Distinct Configurations: $2 \times 2 = 4$ parameter sets per Asset-Timeframe combination $\times 3 \text{ assets} \times 3 \text{ timeframes} = \mathbf{36}\text{ distinct configurations}$.
  * No other variants, filters, or ad-hoc indicators are permitted.

### 7.2 Position Sizing, Pyramiding, and Risk Rules
* **Risk Per Unit:** $1.0\%$ of account equity per trade/unit.
* **Stop Loss:** $2.0 \times \text{ATR}(N)$ from the initial unit entry price.
* **Pyramiding:** Up to 4 units maximum, spaced at $0.5 \times \text{ATR}(N)$ intervals as price moves in the trade's favor.
* **Pyramid Stop Ratchet:** When a new pyramid unit is added, the stop loss for ALL accumulated units ratchets to $2.0 \times \text{ATR}(N)$ below/above the newest unit entry price.
* **Invariant:** Maximum 1 unit added per candle bar (`assert_no_same_bar_pyramiding`).
* **Leverage Cap:** Portfolio gross notional strictly capped at $2.0\times$ portfolio equity.

### 7.3 Execution and Intrabar Resolution
* **Order Execution:** Closed bar signals, fills strictly at next bar open (`open[t+1]`). Zero lookahead.
* **Friction Model:** $14\text{ bps}$ round-trip base ($5\text{ bps}$ fee $+ 2\text{ bps}$ slip per side), $18\text{ bps}$ round-trip stress ($5\text{ bps}$ fee $+ 4\text{ bps}$ slip per side).
* **Funding Settlement:** Exact historical perpetual funding settlements applied at empirical settlement timestamps.
* **1m Sub-Bar Resolution:** All 5m, 15m, and 1h bars pass 1m sub-bars into `QuantEngine.resolve_intrabar_exit` to resolve candle stop-loss vs. take-profit / channel-exit ordering. If ambiguous at 1m, the stop loss is assumed to have triggered first.

### 7.4 Data Windows and Hold-Out Protocol
* **In-Sample (IS):** `2024-01-01 00:00:00 UTC` to `2025-03-31 23:59:59 UTC`.
* **Out-Of-Sample (OOS):** `2025-04-01 00:00:00 UTC` to `2026-03-31 23:59:59 UTC`.
* **Burned Windows (OOS Only):**
  * **BTC 2025+:** **BURNED.** Ingested during legacy backtests, bot states, and Test 3. Must be reported as OOS only, never as clean hold-out.
  * **SOL 2026-04-01+:** **BURNED.** Evaluated during Test 2 (`CFG_1142`). Must be reported as OOS only.
* **Frozen Veto Hold-Out:**
  * **ETH 2026-04-01+:** **FRESH.** Reserved strictly as a single-shot veto hold-out.
  * **Hold-Out Access Rule:** The ETH hold-out window will be touched ONCE, and ONLY for configurations that pass all in-sample, out-of-sample, and stress hurdles. Its trade count and metrics will be recorded without parameter alteration.

### 7.5 Pass / Fail Decision Criteria
A configuration is deemed to possess a viable quantitative edge if and only if:
1. **Net Expectancy Hurdle:** Net expectancy $\ge +5.0\text{ bps}$ per trade after $14\text{ bps}$ RT base costs and funding.
2. **Stress Hurdle:** Net expectancy $> 0.0\text{ bps}$ after $18\text{ bps}$ RT stress costs.
3. **Pooled Sample Size:** Completed trade count $\ge 150$ trades **pooled across the three assets** (SOL, BTC, ETH) for the parameter configuration.
4. **Statistical Significance:** Deflated Sharpe Ratio (DSR) $p\text{-value} < 0.05$ factoring in the 36 new parameter sets plus the 1,318 historical configurations ($N_{\text{trials}} = 1,354$).
5. **Drawdown Tolerance:** Max Drawdown $< 30\%$ on walk-forward stitched equity.
6. **Profit Factor:** $\ge 1.25$ after all friction.

### 7.6 Test 3: Empirical Replay of Binance Execution CSVs
* Replay the two user execution CSVs to be placed in `quant_research/inputs/`.
* Match exact execution timestamps, prices, and quantities against `QuantEngine`.
* Deconstruct every trade into gross return, exchange commissions, execution slippage, funding carry, and net bps.
* Determine whether the live portfolio achieved positive net alpha over passive buy-and-hold and benchmark funding costs.

### 7.7 Trial Logging Governance
* All 36 configurations, calibration runs, and walk-forward evaluations will be appended to `quant_research/trials.csv`.
* No trial row may be deleted or modified.

---

## 8. Amendment A - dated, appended (2026-10-13)

Appended by the round-2 audit. Everything below is **additional** to the text above; no
clause above is modified, renumbered or deleted. Amendment A records two things the
original pre-registration left unspecified and registers one further single-shot hold-out
test that is **written but deliberately not executed**.

### A.1 Exit-trigger convention for the Turtle / Donchian system (retrospective; documentation only)

Section 7.3 does not state *how* the Donchian channel exit and the 2N stop are each
detected, so the committed runners chose a convention implicitly. This audit makes it
explicit and **adopts the committed convention as canonical** (a documentation decision,
not a rule change):

* **Donchian channel exit = CLOSE-based.** The prior-`M`-bar channel (`price < min(low[-M:])` for a long, `price > max(high[-M:])` for a short) is tested on the **closed** bar, and the exit fills at the **next bar's open**. It is *not* an intrabar touch.
* **2.0 x ATR(N) stop = INTRABAR-touch.** The stop is tested against the bar's high/low and fills at the stop level (plus slippage), independent of the close.
* **Ambiguity counter.** A bar is *ambiguous* when the intrabar stop condition and the close-based channel-exit condition both hold on the same bar (the two same-side levels of `QuantEngine.resolve_intrabar_exit_levels`). With `df_1m` the 1m sub-bars order them; without it the fallback is **stop-first (pessimistic)**, as section 7.3 states.
* **Documented, not silently repaired.** The regenerated round-2 numbers keep the stop-first fallback so they stay comparable with the committed artifacts; the 1m-wired variant is run as a separate sensitivity file (`results/turtle_1m_sensitivity.json`) rather than overwriting the committed results. The committed runners never passed `df_1m`, so every committed Turtle number uses the fallback; 35 of 36 configs move when `df_1m` is wired (see REPORT section 13).

### A.2 Entry autopsy: real fills vs matched random timestamps (new analysis; method fixed before inspection)

This analysis is registered here so that its definitions cannot be tuned to its result. It
uses only committed inputs and writes a new artifact; no committed result is overwritten.

* **Universe.** `quant_research/inputs/trade_history.csv` - 1,786 fills forming 257 flat-to-flat position cycles across 4 symbols, 2025-04-04 to 2025-05-12.
* **Entry definition (read off the archive, not invented).** An *entry* is a fill that **opens** a position or **adds** to it (same side, larger absolute size). Reductions and closes are not entries. The primary test unit is the 257 opening fills, one per cycle; the 746 add-fills are reported as a separately labelled secondary block, because an add is conditional on a position that is already working.
* **Clock.** The CSV `Time` column carries no timezone. It is read as **UTC+3**, pinned by the ledger: Binance settles perpetual funding at 00:00/08:00/16:00 UTC and all 50 `FUNDING_FEE` rows are stamped 03:00/11:00/19:00. The UTC+0 reading is re-run end to end and reported as a labelled sensitivity.
* **As-of rule.** Every indicator value is read from the last 1m bar that **closed strictly before** the entry timestamp. The entry bar and every later bar are excluded.
* **Indicator family (13 tests).** Bollinger(20,2) %B, RSI(6), StochRSI %K, StochRSI %D, MACD(12,26,9) histogram in bps, distance from EMA 20 / 50 / 80 / 200 in bps, ADX(14), the 1D trend direction (`+1` if the last completed daily close is above its daily EMA20, else `-1`), the daily distance from EMA20 in bps, and a close-to-close 1D direction variant. Indicator implementations are the repository's own `quant_research/core/indicators.py`.
* **Control.** 1,000 independent random timestamp sets; one control per real entry, matched on (symbol, side, hour-of-day UTC), drawn with replacement from the 1m bars inside the symbol's own trading window +/- 1 day.
* **Statistic.** Difference of means (real minus the mean of the 1,000 control-set means), with a 95% CI that combines the bootstrap SE of the real mean with the SD of the set means; two-sided empirical p-value `(1 + #{|null set mean - null mean| >= |real - null mean|}) / (1 + 1000)`; **Holm** correction within the 13-indicator family and within the 4-horizon family of each forward-move block. Sample sizes, effect sizes and CIs are reported even when a difference is not significant.
* **Forward moves.** 1, 5, 15 and 60 minutes, direction-adjusted gross bps before fees, measured from four explicitly labelled reference prices: the **actual fill price** (the only tradable one, and the only one that cannot be contaminated by intra-minute drift), the open of the fill bar, the close of the fill bar, and the next bar's open.
* **Limitation fixed in advance.** 15-second charts cannot be rebuilt from 1m data; the intra-minute path is unknown and is only bracketed. BNBUSDT has no 1m candle file and is excluded from every indicator and forward-return statistic.
* **Decision rule.** If no indicator family member survives Holm, and no forward horizon differs from the matched random control under a post-fill reference price, the report must say plainly that the entries are indistinguishable from random on the tested evidence.

### A.3 ETHUSDT 1h N=20 / M=20 on the frozen hold-out (WRITTEN, NOT RUN)

* **Configuration.** ETHUSDT 1h, N=20, M=20, the Turtle / Donchian system exactly as specified in section 7 - no parameter alteration of any kind.
* **Window.** `2026-04-01 00:00:00 UTC` to the end of available data. This is the FRESH window reserved in section 7.4.
* **Single hypothesis, fixed before any run.** H1: the **unit-pooled net expectancy** (net bps per position unit, the same statistic published in `results/cycle_metrics_v2.json`) is greater than zero over the window against 14 bps round-trip base costs and empirical funding.
* **Veto rule.** If unit-pooled net bps <= 0 the configuration is **vetoed permanently**: no parameter change, no window change, no re-run. A positive result does not by itself promote the configuration - it must still clear every hurdle in section 7.5 on the windows those hurdles are defined on.
* **Required reporting, fixed before any run.** Unit-pooled net bps; the **number of completed position cycles and of units (entries)**; pooled gross bps; cost and funding bps; mean unit holding time. The reported trade count must be the **actual** count returned by the run, not an extrapolation.
* **Sample-size reference.** On the 12-month OOS window the same configuration produced 223 cycles / 507 units (`results/random_entry_baseline_v2.json`, `strategy_cycles` / `strategy_trades`).
* **Why it is not executed.** Section 7.4 permits exactly one touch of the ETH hold-out, and only for configurations that clear all in-sample, out-of-sample and stress hurdles. ETHUSDT 1h N=20/M=20 does not: its Deflated Sharpe p-value is `1.0` against the `< 0.05` hurdle (`results/turtle_diagnostics_v2.json`, `diagnostics_individual`, ETHUSDT/1h/20/20: `n_trades` 507, `net_bps` 16.98, `cycle_dsr_p_value` 1.0). Under this pre-registration the hold-out may not be opened, so the amendment is recorded and left **unexecuted**.

## 9. Amendment B - dated, appended (2026-10-13)

Appended by the round-3 audit (indicator validation against Binance's own numbers, and the
fifteenth-second bar rebuild). Everything below is **additional** to the text above; no
clause above is modified, renumbered or deleted. Amendment B registers one deviation that
has already been fixed and one further analysis whose method is fixed here.

### B.1 Bollinger band convention: sample versus population standard deviation (defect found, fixed, registered retrospectively)

* **What A.2 registered.** `quant_research/core/indicators.py::compute_indicators` as the source of every indicator condition in the entry autopsy.
* **Defect.** `close.rolling(20).std()` defaults to `ddof=1` (sample standard deviation); Binance's BOLL(20,2) uses `ddof=0` (population). The same defect was present a second time in `test_2_runner.py`, which constructs its own bands rather than calling the module.
* **Evidence.** REPORT section 17 validates the module against the values published on a SOLUSDT-perp 3m chart: `ddof=0` reproduces UP/MB/DN with a total absolute error of 0.0054 against 0.0198 for `ddof=1`, and the artifact `results/indicator_validation.json` plus `tests/test_indicator_binance_ground_truth.py` pin both the published numbers and the fact that the population convention is the closer of the two.
* **Fix.** `std(ddof=0)` in both places. This is a correction against external ground truth, not a parameter search: no threshold or hypothesis in this pre-registration is defined on the band width, the single exception being Test 2's `bb_mult` grid, which is the one searched quantity affected.
* **Consequence for committed artifacts.** `test_1_results.json`, `calibration_results.json`, `test_2_results.json` and `test_2_results_v2.json` were produced before the fix and are stale with respect to the code; `test_1`, `test_2` and the calibration were **not** re-run by this audit. `results/entry_autopsy.json` is deliberately left untouched as the pre-fix record and `results/entry_autopsy_v2.json` is the post-fix re-run; REPORT 17.4 and `tests/test_entry_autopsy_artifact.py` assert that exactly the `bb_pctb` row moves between them.
* **Registration status.** Retrospective: the defect was found and fixed before this amendment was written. It is recorded so the deviation is auditable rather than silent.

### B.2 Fifteen-second entry autopsy (new analysis; method fixed here)

Section A.2 fixed the limitation "15s charts cannot be rebuilt from 1m data". This amendment registers the analysis that removes it.

* **Data.** Binance USDT-M daily aggTrades archives from `data.binance.vision` for the fill window 2025-04-04 to 2025-05-12 plus padding days, rebuilt into 15s OHLCV candles by `quant_research/build_15s_bars.py`. The committed artifact `results/aggtrades_15s_provenance.json` records every source file, its size and sha256, the trades and buckets it contributed, every hole in the resulting tape, and a cross-check that re-aggregating the 15s bars to 1 minute reproduces Binance's own 1m kline closes exactly.
* **Bar conventions.** A bucket stamped T covers [T, T+15s) and closes at T+15s. The as-of bucket is `floor(entry_utc,'15s') - 15s`, i.e. the last bucket that had closed strictly before the entry. `b` is the bucket containing the print and `m0` the first bucket of the print's minute.
* **Forward references, fixed for comparability.** Minute-anchored: `actual_fill_price` and `open_of_fill_minute` end at `close[m0 + 4h - 1]`, `close_of_fill_minute` and `next_minute_open` start at `close[m0 + 3]` / `open[m0 + 4]` and end at `close[m0 + 4h + 3]`. For h = 1 those are the same instants section A.2 uses, so the two blocks compare the same trades at two bar sizes rather than two differently anchored conventions.
* **Everything else is A.2 verbatim, imported rather than reimplemented**: entry definition and the open/add split, the UTC+3 primary and UTC+0 sensitivity clocks, the 13-condition indicator family, 1,000 matched (symbol, side, hour-of-day) control sets drawn from the same symbol's window +/- 1 day, 300 bars of warmup, the difference-of-means statistic with its CI and two-sided empirical p-value, and Holm within the 13-indicator family and the 4-horizon family.
* **Two measures only 15s can make, declared here with no hypothesis claimed.** (a) the print's position inside its own 15s bucket, `side * (fill / open[b] - 1) * 1e4`, against the whole-minute version section A.2 reports; (b) the direction-adjusted path at the close of each of the four buckets of the print's minute, `side * (close[b + k] / fill - 1) * 1e4`.
* **Gap policy.** The downloaded tape has holes (not every day was archived for every symbol). An indicator may never be read across one: an entry is dropped unless `run_len`, the number of contiguous bars ending at its as-of bucket, is at least the warmup plus one. A forward window may never cross one either: `run_len_fwd` must cover the whole 60-minute endpoint, for real entries and control draws alike. Dropped entries are counted in the artifact.
* **Decision rule.** Identical to A.2.
* **Honest statement of timing.** A pilot run of `quant_research/entry_autopsy_15s.py` on SOLUSDT only was executed and its output inspected before this amendment was written. What changed afterwards: three defect fixes (index arithmetic when computing minute anchors for the 2-D control draws, an out-of-range index in the control mask, and switching the as-of lookup to `searchsorted` so that a gappy file cannot be silently mis-indexed), the addition of the two gap guards, and the switch of the forward references from bucket-anchored to minute-anchored so the blocks are comparable. No indicator, control construction, statistic, threshold or hypothesis was altered in response to the pilot's numbers. The three-symbol run reported in REPORT section 18 is the run this amendment registers.

## 10. Amendment C - dated, appended (2026-10-13); committed BEFORE running any simulation

**Status.** Appended and committed strictly BEFORE any simulation of the candidate set named
below. Every number quoted as a design basis is taken from an artifact that was already
committed at the time of writing (`entry_autopsy.json`, `entry_autopsy_15s.json`,
`test_3_proper_results.json`, `turtle_results_v2.json`); the candidate design is therefore
derived from measured facts, not from the candidate's own output.

### C.1 Why this candidate, in terms of what the audit measured

1. **Intraday indicators are provably uninformative here.** All 13 intraday conditions are
   indistinguishable from matched random at 1m (REPORT 15) AND at 15s (REPORT 18). The only
   conditions that survive Holm are daily-context ones: `adx14` (real 28.19 vs random 25.28),
   `daily_trend_ema20` (real 0.388 vs random 0.620) and `daily_dist_ema20_bps`
   (real 402.5 vs random 570.4), each at Holm p = 0.0130, in the primary open block of
   `entry_autopsy.json`. The only measured signal lives on the DAILY timeframe, so the
   candidate's only discretionary filter is a daily one - no intraday oscillator is admitted.
2. **The edge is at long holding horizons, if it exists at all.** Direction-adjusted forward
   moves from the actual fill price are -1.14 bps at 1m, -0.40 at 5m, +2.56 at 15m and +5.87
   at 60m (primary open block, `entry_autopsy.json`). A 14 bps round-trip base cost and the
   +5 bps section 6/7.5 net-expectancy hurdle therefore demand multi-hour to multi-day holds:
   gross edge per trade must approach or exceed ~20 bps, which intraday scalping horizons
   cannot reach at these costs. The candidate is therefore a higher-timeframe trend system
   with few trades, not a scalper.
3. **Execution is a measured cost, and adds are measured to be worse.** The taker print
   crosses +5.14 bps against the fill-minute open (95% CI 3.75 to 6.61) and +1.86 bps inside
   its own 15s bucket; the ADD fills measure +2.54 bps inside their own bucket against
   +1.86 for opens (`entry_autopsy_15s.json`). Pyramid adds are exposure taken while a
   position is already working, with no measured edge of their own - so the candidate runs
   at **strictly 1 unit** (no pyramiding).
4. **The real account was destroyed by overtrading, not by one bad bet.** The ledger
   reconciliation puts the true max drawdown at 100.0% (`test_3_proper_results.json`) across
   1,003 entries in 39 days. A survivable risk layer is not optional: 1% risk per trade,
   2.0 x ATR(N) stop and the 2.0x leverage cap of section 7.2 are retained unchanged.

### C.2 Exactly what is registered

* **Base system (unchanged from section 7).** Classical Donchian breakout: entry on a close
  beyond the prior N-bar high/low, exit on the prior M-bar channel, 2.0 x ATR(N) intrabar
  stop, N in {20, 55}, M in {10, 20}, assets SOL/BTC/ETH, timeframes 5m/15m/1h = 36
  configurations per arm. Signals on closed bar t, fills at open[t+1], 1m sub-bar intrabar
  resolution with the stop-first pessimistic fallback, 14 bps RT base / 18 bps RT stress,
  empirical funding settlements at their exact timestamps.
* **C1 (primary, "DT-Filter 1-unit"):** the base system PLUS a daily-trend directional
  filter, at 1 unit (pyramiding disabled). The filter is the audited condition verbatim:
  the previous COMPLETED daily close versus its own EMA(20), computed by resampling the 1m
  close series to 1D and seeding the EMA at the archive start (the identical construction
  used by `entry_autopsy.py`). Long entries are allowed only when that daily trend is +1,
  short entries only when it is -1. Bars with no completed prior daily day (archive start
  warmup) admit no signal.
* **C0 ("1-unit" yardstick):** the base system at 1 unit WITHOUT the daily filter. Two
  attributions come out of this design: (a) C1 versus C0 isolates the daily filter at equal
  risk layering; (b) C0 versus the committed 4-unit `turtle_results_v2.json` isolates
  pyramiding at equal signals.
* **72 new configurations total are registered: 36 x C1 + 36 x C0. Nothing is searched
  beyond this grid; no parameter may be re-tuned; no configuration may be dropped for its
  result; all 72 are reported regardless of outcome.**
* **Trials and DSR:** all 72 are appended to `trials.csv` (section 7.7). The Deflated Sharpe
  Ratio hurdle uses N_trials = 1,354 historical + 72 new = **1,426**, computed on the pooled
  OOS per-trade Sharpe at T = pooled OOS trade count (the same canonical computation the
  committed runner documented).

### C.3 Decision rule (fixed before running)

A C1 or C0 configuration carries a viable edge if and only if it clears section 7.5's six
hurdles: (1) pooled OOS net expectancy >= +5.0 bps after 14 bps RT base and funding;
(2) pooled OOS expectancy > 0.0 bps after 18 bps RT stress; (3) pooled OOS trades >= 150;
(4) DSR p-value < 0.05 at N_trials = 1,426; (5) max drawdown < 30% on stitched equity -
defined here as the continuous 2024-01-01 to 2026-03-31 equity per asset, verdict taken as
the MAXIMUM of the three assets' max drawdowns; (6) pooled OOS profit factor >= 1.25 after
all friction - defined as pooled gross profit over pooled gross loss across the three assets.

Consequences, fixed now: if no configuration clears all six hurdles the verdict is
**NO EDGE** and no hold-out is touched; if one or more clear all six, the ETH 2026-04-01 +
frozen veto hold-out is accessed exactly once, at frozen parameters, per section 7.4, and
its trade count and metrics are recorded without alteration. A NO EDGE result is accepted as
a successful scientific conclusion, consistent with section 6.

### C.4 Anti-garden-of-forking-paths clause

The section 15/17/18 autopsy already established that intraday indicator conditions carry no
measured edge at 1m or 15s resolution; this amendment therefore admits no intraday filter,
no maker-entry variant, no cost-model change and no exit-rule change. Any future variant of
that kind must be separately pre-registered BEFORE it runs, and previously touched windows
remain burned per section 7.4.
## 11. Amendment D - dated, appended (2026-10-13); committed BEFORE running any simulation

**Status.** Appended and committed strictly BEFORE any maker-fill simulation runs. The
candidate design below is derived only from artifacts already committed at the time of
writing (`entry_autopsy.json`, `entry_autopsy_15s.json`, `amendment_c_results.json`),
per section C.4's clause that a maker-entry variant "must be separately pre-registered
BEFORE it runs".

### D.1 Why the maker entry is the one measured lever left

1. **Execution is the dominant measured per-trade leak.** The taker print crosses
   +5.14 bps against its own fill-minute open (95% CI 3.75 to 6.61) and +1.86 bps inside
   its own 15s bucket (`entry_autopsy.json`, `entry_autopsy_15s.json`). A taker entry
   therefore pays roughly the 5 bps break-out premium ON TOP of the 14 bps published
   round-trip cost. The only measured condition that survives Holm (daily trend) and the
   only band with positive OOS expectancy (1h, C1 20/10 +10.26 and 20/20 +28.98 bps, both
   DSR-failing at N=1,426) are unchanged inputs here; nothing else about the skeleton is
   reconsidered.
2. **The 15s tape quantifies the chase.** The post-print 15s path stays adverse
   (-0.79 / -1.12 / -1.44 / -1.10 bps at 15/30/45/60 s), i.e. the taker consistently
   fills after the cross has extended. A resting limit order at the pre-cross price
   converts that chase into a better entry or into no fill - both outcomes are priced
   by this design rather than assumed away.
3. **No maker fill exists in the archive.** 99% of the user's 1,786 archived fills are
   taker prints, so the maker FEE is not measurable from this account's data. It is
   taken from Binance USDT-M's published VIP0 schedule (maker 0.02% = 2 bps; the same
   schedule whose 5 bps taker leg the ledger confirms). This is declared as an input,
   not a searched quantity.

### D.2 Exactly what is registered

* **Base system (unchanged from section 7 / C.2).** Donchian N in {20, 55}, M in
  {10, 20}; signals on closed 1h bar t, 1h only (the band whose forward horizons the
  autopsy measured as the only reachable ones, and the only band positive OOS under
  Amendment C); daily-trend directional filter ON in BOTH arms (it is the only signal
  with measured value, and holding it fixed makes the R1 versus R0 subtraction a pure
  execution attribution); 1 unit; 1%-risk sizing, 2.0 x ATR(N) intrabar stop with the
  stop-first pessimistic fallback, close-based M-bar channel exit filling at next open,
  2.0x leverage cap, empirical funding at exact timestamps - all verbatim section 7.2/7.3
  machinery.
* **R1 (primary, "maker 1h"):** on each qualifying signal bar, place a LIMIT order at
  open[t+1] moved 5 bps passively (limit = open[t+1] x (1 - 5e-4) for a long,
  x (1 + 5e-4) for a short). The offset is FIXED at 5 bps, anchored to the measured
  +5.14 bps cross; it is not searched. The order is live for K = 3 closed 1h bars
  (3 hours) and is then cancelled; an unfilled signal produces no trade and no re-arm
  until the next qualifying signal bar. A fill is registered only when the 1m tape
  trades STRICTLY THROUGH the limit (low < limit for a long, high > limit for a short -
  strict inequality, so a mere touch does not fill, modeling queue priority), at the
  limit price, with a 2 bps maker fee and ZERO entry slippage. Exits are unchanged and
  remain taker.
* **R0 (taker yardstick):** the identical system with the taker entry of section 7.3
  (fill at open[t+1], 5 bps + 2 bps), run by the same runner so the comparison is
  code-identical. R1 minus R0 isolates the maker conversion (fee saving + entry-price
  improvement) net of adverse selection and missed fills.
* **Cost schedule.** Base: entry maker 2 bps fee + 0 slippage, exit taker 5 bps + 2 bps
  slippage = 9 bps RT. Stress: the entry additionally requires the 1m tape to trade
  through the limit by a further 1 bps (strictly worse fill probability, harsher adverse
  selection) and exits at 5 bps + 4 bps slippage = 11 bps RT.
* **24 new configurations total: 12 x R1 + 12 x R0 (3 assets x 4 channel pairs, 1h).
  Nothing is searched beyond this grid; no parameter may be re-tuned; no configuration
  may be dropped for its result; all 24 are reported regardless of outcome.**
* **Trials and DSR:** all 24 are appended to `trials.csv` (section 7.7). The deflated
  Sharpe hurdle uses N_trials = 1,426 (section C.2) + 24 = **1,450**, computed on the
  pooled OOS per-trade Sharpe at T = pooled OOS trade count, the same canonical
  computation as C.2.

### D.3 Decision rule (fixed before running)

A configuration carries a viable edge if and only if it clears the same six hurdles as
C.3, verbatim: (1) pooled OOS net expectancy >= +5.0 bps after the base schedule and
funding; (2) pooled OOS expectancy > 0.0 bps under the stress schedule; (3) pooled OOS
trades >= 150; (4) DSR p-value < 0.05 at N_trials = 1,450; (5) max drawdown < 30% on
stitched equity, defined as the continuous 2024-01-01 to 2026-03-31 equity per asset,
verdict taken as the MAXIMUM of the three assets' max drawdowns; (6) pooled OOS profit
factor >= 1.25, defined as pooled gross profit over pooled gross loss.

Consequences, fixed now: if R1 and R0 both fail every configuration, the verdict is
**NO EDGE (maker variant)** and no hold-out is touched; if one or more configurations
clear all six hurdles, the ETH 2026-04-01+ frozen veto hold-out is accessed exactly
once, at frozen parameters, per section 7.4, and its trade count and metrics recorded
without alteration. Diagnostics that do not enter the verdict (fill rate, missed-signal
count, R1-minus-R0 expectancy delta per channel pair) are reported alongside every
configuration to make the adverse-selection trade-off auditable.

### D.4 Anti-garden-of-forking-paths clause

The entry offset (5 bps), the order lifetime (3 bars), the strict-through fill rule and
the stress trade-through (1 bps) are fixed HERE and may not be re-chosen after seeing
results. A failed round does not license a second round with a different offset or
lifetime: any such variant is a new amendment, committed before it runs, with its own
DSR penalty. Previously touched windows remain burned per section 7.4.

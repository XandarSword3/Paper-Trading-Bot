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


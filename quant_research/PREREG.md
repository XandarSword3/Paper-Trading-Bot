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

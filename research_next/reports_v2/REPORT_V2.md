# Cash-and-Carry Validation V2 — Audit & Corrected Results

**Date:** 2026-11-26. **Scope:** Strategy B (long spot + short quarterly
futures) from `research_next`. All previous B results are superseded by this
report. Original v1 files preserved untouched.

---

## 0. Headline answer

**No.** With correctly aligned data, ETH (and BTC) cash-and-carry does **not**
produce a sustainable automated return that beats the existing S5 1x benchmark,
and the previously reported $1,652 / 13.6% CAGR ETH result was a **data
alignment artifact**, not a real edge. Details below.

---

## 1. Audit of the v1 implementation (`backtest/cash_carry.py`)

### How v1 worked
- Universe: BTC/ETH Binance USD-M quarterly futures vs OKX USDT spot, $1,000
  per base, DEPLOY=0.95, spot notional `X = E*0.95/(1+IMR)`, IMR=20%.
- Entry when annualized basis `(F/S−1)*365/dte > 3%/yr`, nearest contract with
  dte>14; hold to expiry; roll if next basis >3%.
- Isolated futures margin `margin = X*IMR`; daily liquidation check on the
  futures daily HIGH vs `MMR_F=0.5%` of notional; top-up from wallet, then
  liquidate (lose margin account, sell spot, go flat).
- Costs: spot 0.15%/side, futures 0.10%/side. No liquidation fee.

### Liquidation reconstruction (ETH, v1 parameters)
`reports_v2/liq_audit_ETH.csv` — 9 events, all on contract ETHUSDT quarterly,
spot ~$1.5k–4.7k, futures notional ~$1.07k–1.55k, isolated margin ~$206–279,
maintenance ~$5.3–7.8. Events cluster around ETH rallies (2023-01, 2023-12,
2024-02/03, 2024-11, 2025-05/07/08, 2026-08). Several are **repeated
liquidations of the same strategy** (e.g., 2024-02-19 and 2024-03-05 on the
same 2024-03-29 contract; 2025-07-14 and 2025-08-12 on 2025-09-26) — the
strategy re-entered after each liquidation and was liquidated again. They are
NOT nine independent positions; they are 9 forced exits of one rolling
strategy. After each liquidation the spot leg was sold at the same day's
close, so naked-spot exposure was intraday-only; the strategy was flat until
re-entry (0–4 days).

### Accounting verification
- Equity is marked at daily CLOSE, so the deeply negative isolated equity
  seen at the daily HIGH (bankruptcy overshoot) never enters the equity
  curve — no phantom gain/loss from that path. Verified by reconciliation.
- v1 had **no liquidation clearance fee** and used daily futures HIGH as the
  mark proxy. v2 adds the fee and settles at the analytic liquidation price.
- **Critical bug found (data, not accounting):** OKX spot 1D candles are
  stamped at candle OPEN time (16:00 UTC, covering 16:00→16:00 next day),
  while Binance futures daily candles are 00:00→24:00 UTC. v1 matched a
  futures "close" that was up to 8h in the future relative to the spot close,
  injecting look-ahead into every basis calculation and daily mark. This
  materially inflated ETH results (and distorted BTC). **All v1/v2-first-pass
  B numbers are invalid.**

---

## 2. Corrected data alignment

- Fetched Binance USD-M quarterly **4H** klines
  (`data/fetch_quarterly_4h.py` → `data/quarterly_4h/`).
- OKX spot candle labelled `d` (16:00 UTC) closes at `d+24h`; the matching
  futures mark is the 4H candle closing at `d+24h` (ts = `d+20h`). The
  liquidation-check high is the max of the six 4H candles covering the spot
  candle window. Verified against Binance ETHUSDT perp 4H (matches OKX spot
  within ~0.2%).
- Settlement: Binance USD-M quarterly settles 08:00 UTC on expiry Friday;
  futures settle price = 4H candle closing at exp+8h; spot sold at the close
  of the spot candle covering settlement (~8h later — documented
  approximation).
- Engine: `backtest/cash_carry_v2.py` (v1 preserved). Liquidation settles at
  the analytic liq price (equity = maintenance margin), not the daily high.

---

## 3. Official specifications verified

- **USD-M quarterly futures** (BTCUSDT/ETHUSDT_YYMMDD): max leverage 50x,
  24/7, USDT settlement, tick 0.1/0.01. **Multi-Assets Mode is supported**
  for quarterly contracts (Binance announcement 2023-08-16).
- **Multi-Assets Mode** (cross margin): BTC/ETH/BNB accepted as collateral at
  **95% value (5% haircut)**, cross-margin ONLY, assets must be transferred
  into the USD-M futures wallet. Source: Binance FAQ "What Is Multi-Assets
  Mode" + Multi-Assets Info page. → Structure B is real and officially
  supported, but requires spot and futures on the SAME venue (Binance), not
  OKX spot + Binance futures. Our spot prices are OKX; Binance spot ≈ OKX
  (venue basis ignored — flagged).
- **Liquidation Clearance Fee**: 1.25% of position notional (USD-M perp
  schedule; delivery-futures rate assumed identical — UNVERIFIED).
- **Mark price** (quarterly): Price Index + 30s moving average of basis;
  last 30 min of delivery day = index average. We use 4H high — approximation.
- **MMR**: perp tier-1 = 0.40% at our size; quarterly delivery tier table not
  retrieved → MMR_F=0.5% assumed (conservative). IMR is our chosen collateral
  (20–50%), not the exchange minimum.

---

## 4. Corrected results — common period 2021-02-01 → 2026-11-25, $1,000

| strategy | final | CAGR | mean/mo | geo/mo | worst/mo | maxDD | %mo≥6% | liq | flat days |
|---|---|---|---|---|---|---|---|---|---|
| B_ETH_cross_imr20 | $1,120 | 2.9% | 0.246% | 0.242% | −2.1% | −3.3% | 0% | 0 | 154 |
| B_BTC_cross_imr20 | $1,146 | 3.5% | 0.294% | 0.289% | −2.9% | −4.0% | 0% | 0 | 95 |
| B_ETH_isolated_imr35 | $984 | −0.4% | −0.027% | −0.033% | −2.6% | −8.9% | 0% | 6 | 154 |
| B_BTC_isolated_imr35 | $977 | −0.6% | −0.042% | −0.051% | −6.4% | −6.9% | 0% | 5 | 104 |
| **S5 1x (benchmark)** | **$1,295** | **4.5%** | **0.374%** | **0.373%** | **−0.13%** | **−0.3%** | 0% | 0 | 0 |
| BTC buy & hold | $3,615 | 38.6% | 4.29% | 3.33% | −23.2% | −53.3% | 43% | 0 | 0 |
| Stablecoin ~3.5%/yr | $1,145 | 3.5% | 0.292% | 0.292% | +0.29% | 0% | 0% | 0 | 0 |

Notes:
- ETH spot data (OKX ETH-USDT) starts **2022-12-17**; ETH B effectively starts
  2023-01 (first entry 2023-01-13). BTC quarterly futures start 2021-03.
- `flat_days` = days out of market (basis < 3%/yr threshold or waiting for a
  contract with dte>14). ETH is flat ~21% of days.
- Full-history (2023-01→2026-11 for ETH, 2021-04→ for BTC) results are in
  `reports_v2/margin_sweep.csv`; they tell the same story.

## 5. Margin-structure experiment (ETH, cost 1.0x, full history)

| mode | IMR | final | mean/mo | maxDD | liq |
|---|---|---|---|---|---|
| isolated | 20% | $946 | −0.11% | −13.4% | 8 |
| isolated | 25% | $951 | −0.09% | −9.6% | 7 |
| isolated | 30% | $990 | −0.01% | −7.6% | 6 |
| isolated | 35% | $984 | −0.03% | −8.9% | 6 |
| isolated | 40% | $995 | −0.00% | −7.8% | 4 |
| isolated | 50% | $1,086 | +0.18% | −2.8% | 2 |
| cross | 20% | $1,120 | +0.25% | −3.3% | 0 |
| cross | 35% | $1,106 | +0.22% | −2.9% | 0 |
| cross | 50% | $1,095 | +0.20% | −2.7% | 0 |

BTC isolated IMR=20%: $1,121, 9 liqs; BTC cross IMR=20%: $1,146, 0 liqs.
Max-deploy cross (IMR→0, spot=95% of equity): BTC $1,177 (0.35%/mo, maxDD
−4.8%), ETH $1,145 (0.30%/mo, maxDD −3.9%) — still below S5 1x.

**Reading:** isolated margin at any tested IMR either loses money or earns
less than cross margin, because liquidations force early exits that forfeit
remaining basis. Cross margin (spot as collateral at 95%) eliminates
liquidations entirely in-sample — the collateral moves with the hedge — but
capital efficiency drops and returns fall to ~0.25–0.35%/month.

## 6. Cost stress (corrected, common period where applicable)

ETH cross IMR=20%: 1.0x → $1,120; 1.5x → $1,085; 2.0x → $1,052.
BTC cross IMR=20%: 1.0x → $1,146; 1.5x → $1,110; 2.0x → $1,075.
Both remain profitable at 2x costs but below S5 1x ($1,295) and below
stablecoin yield on a risk-adjusted basis (stable $1,145 with zero DD).

## 7. Return decomposition (ETH cross IMR=20%, full history)

From `margin_sweep.csv` decomp columns: initial basis captured ≈ $360–400,
realized convergence ≈ $380–400 (no double count — convergence = entry basis
− exit basis per cycle), spot PnL ≈ +$1,500, futures PnL ≈ −$1,070 (the two
largely cancel; residual = convergence + delta mismatch from
`spot_qty=X/S` vs `fut_qty=X/F`), transaction costs $68–75, liquidation fees
$0. Roll costs are inside transaction costs (each roll pays spot+futures
fees again).

## 8. Viability assessment

Criteria: zero liquidations, credible collateral buffer, profitable at 2x
costs, works on common period, no unavailable functionality.

- **Best B candidate: BTC cross-margin, max-deploy (IMR≈0)** — $1,177,
  0.35%/mo, maxDD −4.8%, 0 liquidations, $1,090 at 2x costs. Requires Binance
  Multi-Assets Mode (officially supported for USD-M quarterly) and Binance
  spot (not OKX). 
- **It still loses to S5 1x** ($1,295, 0.37%/mo, maxDD −0.3%) on return AND
  drawdown, and adds venue-basis risk (OKX vs Binance prices), collateral
  haircut risk (5%), and operational complexity (wallet transfers, auto-
  exchange rules).
- No isolated-margin configuration is deployable: every IMR ≤40% either loses
  money or has 4–9 liquidations; IMR=50% still has 2–3 liquidations and only
  ~0.2%/mo.

## 9. Unresolved assumptions / limitations

- OKX vs Binance spot basis ignored (prices match within ~0.2% on checked
  days, but not verified across all history).
- Quarterly delivery-futures MMR/IMR tier tables not retrieved; MMR_F=0.5%
  assumed; liquidation clearance fee 1.25% taken from perp schedule.
- Mark price approximated by 4H high; real mark = index + 30s basis MA.
- Settlement: spot sold ~8h after futures settlement (no 08:00 UTC spot
  price used).
- Survivorship-biased universe; ETH spot history starts 2022-12-17.
- Daily-granularity simulation; intraday margin calls between 4H checks not
  modelled (conservative direction unknown).
- Cross-margin auto-exchange / negative-balance mechanics not modelled
  (would only trigger in bankruptcy scenarios that never occurred).

## 10. Final answer

**No — ETH cash-and-carry cannot produce a genuinely sustainable automated
return on $1,000 that beats what we already have.** The earlier 13.6% CAGR /
$1,652 result was produced by a spot/futures timestamp misalignment (OKX 16:00
UTC candle-open labels vs Binance 00:00 UTC daily candles) that injected
look-ahead into every basis calculation. After correction:

- ETH cross-margin cash-and-carry: ~0.25–0.30%/month, maxDD −3 to −4%, zero
  liquidations, profitable at 2x costs — but **below S5 1x** (0.37%/mo,
  maxDD −0.3%) and below BTC buy-and-hold by an order of magnitude.
- Isolated margin: unprofitable or liquidation-prone at every tested IMR.
- The honest conclusion: at $1,000 capital, quarterly-futures basis carry is
  a real but thin premium (~3–8%/yr on deployed notional) that does not
  clear the risk/return bar set by S5 1x, and it introduces venue, haircut,
  and operational risks that S5 does not have.

**Recommendation: keep S5 1x as the deployable strategy; do not pursue B
further unless the basis threshold or universe changes materially.**

# S5 v3 — Live-Execution Validation & Deployment Readiness

**Date:** 2026-11-26. **Strategy:** S5 1x delta-neutral BTC funding carry
(long BTCUSDT spot + short BTCUSDT perp, $1,000 total, isolated margin).
**Venue selected:** Bybit (see §2). All previous research versions preserved.

---

## 1. Phase 1 — Engine audit & reproduction

Audited `backtest/s5_carry.py` (v1), `backtest/s5_margin.py` (v2/v2.1),
`paper/paper_s5.py`, funding datasets, equity/trade logs.

**Reproduced:** S5 L1 full history (2020-01→2026-11, Coinbase spot + Binance
funding) final = **$1,434.00**, mean monthly +0.44%, worst month −0.53%,
maxDD −0.30%, 0 liquidations, reconciliation error < 1e-12. Matches the
published v2.1 results exactly.

**Audit findings (v2.1 approximations, all fixed in v3):**
- Funding was applied at the **daily close price** for all settlements that
  day (up to ~16h stale) — fixed: timestamp-exact mark price at settlement.
- Liquidation checked on **spot daily HIGH** as mark proxy — fixed: 1H
  mark-price highs from Bybit.
- Venue mismatch: Coinbase spot + Binance funding — fixed: single venue
  (Bybit) for spot, perp, funding, and mark price.
- Perp marked at spot close — fixed: perp marked at mark price.
- Fees were uniform 0.15%/side — fixed: venue schedule (spot 0.15%,
  perp 0.105% incl. 0.05% slippage assumption).
- Rule A fires **very frequently** (121 trades over 5.4y, mostly tiny
  top-ups of $1–90). Each is a real taker trade on a live venue; at $1,000
  scale the per-trade minimums are not binding for BTC, but the operational
  noise is real. Documented, not changed (frozen rules).

## 2. Phase 2 — Venue investigation

| criterion | OKX | Bybit |
|---|---|---|
| Lebanon on excluded list? | **No** (Risk & Compliance Disclosure, updated 2026-07-08) | **No** (Platform Terms cl. 11.3, effective 2026-10-16) |
| Derivatives for Lebanon? | Not separately excluded; KYC nationality-based screening | Same; Standard KYC mandatory, restrictions by nationality |
| Spot + perp same venue? | Yes | Yes |
| Deep public funding history? | **No** — public endpoint serves ~3 months only | **Yes** — `/v5/market/funding/history` back to 2020-03 |
| Historical mark price? | Limited (recent only) | **Yes** — `/v5/market/mark-price-kline` full history |
| Risk-limit tiers via API? | Yes | Yes (`/v5/market/risk-limit`) |
| Fees (non-VIP, taker) | spot 0.08%, perp 0.05% | spot 0.10%, perp 0.055% |
| Funding interval | 8h (00/08/16 UTC) | 8h (00/08/16 UTC) |
| Mark price methodology | index + basis MA | index + basis MA (public docs) |

**Fiat on-ramp caveat (both venues):** Banque du Liban Article 10 prohibits
Lebanese banks from facilitating virtual assets. A Lebanese user must
already hold crypto (e.g., USDT) and transfer it on-chain. Account approval
is discretionary. No circumvention assumed.

**Decision: Bybit.** It is the only candidate with deep, public,
venue-consistent funding + mark-price history, enabling a true
venue-specific backtest. OKX remains a valid live alternative (cheaper
perp fees) but cannot be backtested venue-consistently with public data.

## 3. Phase 3 — Funding & price timing (fixed in v3)

- Funding events: exact settlement timestamps (00/08/16 UTC), 5,913 events
  in window. Notional = `qty_perp × mark_price(settlement instant)` where
  mark = close of the 1H mark-price kline ending at settlement.
- Sign convention: `fundingRate > 0` → longs pay shorts → short receives.
  Verified against Bybit docs.
- Missing/duplicate detection: funding series checked for gaps/dupes
  (none found in window); shadow trader logs every settlement for
  reconciliation.
- Liquidation: 1H mark-price highs vs analytic liq price, MMR = 0.33%
  (Bybit BTCUSDT tier-1, verified via risk-limit API: tier-1 cap 300k USDT,
  MMR 0.33%, IMR 0.66%, max lev 150x).

## 4. Phase 4 — Capital & margin ($1,000 total, L=1)

Structure at entry: spot notional N0 ≈ $498.5, perp margin N0 ≈ $498.5,
reserve ≈ $1.3 after entry costs. All trades (entry, Rule A top-ups, Rule B
rebalance, covers) charged venue fees + 0.05% slippage. No cross-margin
transfers assumed (isolated perp account; spot held in spot wallet — a real
venue requires the spot leg to be sold to fund margin defense, exactly as
modelled).

## 5. Phase 5 — Shadow trader (public data, no orders)

`paper/shadow_s5_bybit.py` — fetches Bybit spot daily candle, 1H mark-price
klines, and funding history; mirrors all frozen rules; logs every decision
to `paper/shadow_s5_log.csv`; state in `paper/shadow_s5_state.json`.

**Status:** initialized 2026-11-26 — entered 0.008249 BTC @ spot 60,460.4 /
mark 60,430.7, margin $498.73, reserve $1.28, equity $998.73. Two daily
marks logged successfully. Funding settlements will be picked up at the next
00/08/16 UTC boundary and logged with the exact mark price used.

**14-day protocol:** run `python paper/shadow_s5_bybit.py` once daily (after
00:00 UTC). After 14 days, reconcile: (a) every FUNDING log row vs Bybit
funding history (same source — verifies timestamp + mark-price capture, not
independent settlement data); (b) equity path vs backtest engine run on the
same window; (c) any Rule A/B triggers vs manual calculation. **Limitation:**
without API keys we cannot compare against the exchange's own account-level
settlement records — that requires a funded test account (out of scope).

## 6. Phase 6 — Economic viability (Bybit venue data, 2021-07-05 → 2026-11-26)

| scenario | final | CAGR | mean/mo | worst/mo | maxDD | liq | costs | funding net |
|---|---|---|---|---|---|---|---|---|
| **Base (1.0x)** | **$1,195** | **3.36%** | **0.28%** | −0.49% | −0.76% | 0 | $13.7 | +$209 |
| 1.5x costs | $1,188 | 3.24% | 0.27% | −0.51% | −0.78% | 0 | $20.4 | +$208 |
| 2.0x costs | $1,180 | 3.12% | 0.26% | −0.53% | −0.80% | 0 | $27.2 | +$208 |
| Negative funding (pay every settlement) | $816 | −3.71% | −0.31% | −1.52% | −18.6% | 0 | $13.1 | −$171 |
| BTC +40% shock (2024-01-01) | $1,193 | 3.33% | 0.28% | −0.49% | −0.76% | 0 | $14.7 | +$209 |
| BTC +80% shock | $1,192 | 3.31% | 0.27% | −0.49% | −0.76% | 0 | $16.3 | +$209 |
| BTC −50% crash | $1,196 | 3.38% | 0.28% | −0.49% | −0.76% | 0 | $14.3 | +$210 |

- Min liquidation distance (close): **77.6%** — at L=1 the short would need
  BTC to roughly double before liquidation; Rule A defends far earlier.
- Funding received $248 / paid $39 over 5.4y on ~$500–1,000 notional.
- Return on total account equity ≈ CAGR 3.4% (capital fully deployed).
- vs stablecoin ~3.5%/yr: **S5 v3 ≈ stablecoin yield with −0.8% maxDD and
  funding-sign risk**; vs v2.1 Coinbase/Binance ($1,434, 4.5% CAGR) the
  venue-consistent Bybit window (starts 2021-07) and venue fees explain most
  of the gap.

**Downtime scenario:** if Bybit is unreachable, positions remain open on the
exchange (isolated margin, no auto-liquidation from our side); funding and
liquidation continue exchange-side. Mitigation: alerts + manual check
(§7). **Depeg scenario:** USDT depeg affects both legs' margin currency;
not modelled — flagged as unresolved.

## 7. Phase 7 — Automation architecture (design only, no orders)

```
scheduler (cron/systemd, every 15 min + daily after 00:05 UTC)
  └─ shadow/production runner
       ├─ connectivity check (Bybit /v5/market/time, 3 retries, backoff)
       ├─ data freshness check (spot candle, mark 1H, funding history)
       ├─ funding settlement monitor (poll funding/history every 5 min
       │    around 00/08/16 UTC; apply at exact timestamp)
       ├─ position reconciliation (exchange positions vs internal state;
       │    mismatch > tolerance → halt + alert)
       ├─ margin monitor (perp equity vs M*; alert at 80% of defense trigger)
       ├─ circuit breakers:
       │    • max-loss: halt if equity < $900 (−10%)
       │    • missing-data: halt if any feed stale > 2h
       │    • duplicate-order: idempotency keys + order-state journal
       ├─ decision engine (frozen rules only; no parameter changes)
       ├─ execution adapter (Bybit v5 signed endpoints; spot + linear perp)
       ├─ persistent log (SQLite/CSV, append-only, daily rotation)
       ├─ daily portfolio summary (equity, funding received, margin ratio,
       │    liq distance, costs) → Telegram bot alert
       └─ API key handling: env vars / secrets manager; trade-only key,
            NO withdrawal permission, IP whitelist, read+trade scopes only
```

Fail-safe defaults: any data/connectivity failure → no new orders, alert,
keep existing hedge (it is self-protecting: delta-neutral, isolated margin,
exchange-side liquidation as last resort).

## 8. Final decision

1. **Can S5 1x operate on a Lebanese user's eligible exchange account?**
   **Yes, conditionally.** Neither OKX nor Bybit excludes Lebanon, but KYC is
   nationality-based and discretionary, and Banque du Liban Article 10 blocks
   bank-mediated funding — the user must already hold crypto and transfer it
   on-chain. Bybit is the chosen venue (deep public data, verified specs).
2. **Expected net return on $1,000?** ≈ **3.1–3.4%/yr** (CAGR), ~0.26–0.28%/month,
   maxDD < 1%, zero liquidations in 5.4y of venue-consistent backtest.
3. **How much survives venue fees and funding?** Funding net +$209 over 5.4y;
   total costs $13.7 (1.0x) to $27.2 (2.0x). At 2x costs CAGR drops only
   3.36% → 3.12%. Fees are NOT the binding constraint; the funding rate level
   is.
4. **Remaining liquidation and operational risks?** Liquidation risk is
   negligible at L=1 (77% distance, Rule A defense, exchange-side liq as
   backstop). Operational risks: exchange downtime (hedge persists, funding
   continues), API/key compromise (mitigated: no withdrawal scope), USDT
   depeg (unmodelled), Rule A's frequent tiny trades (operational noise,
   ~$14 total cost), and the negative-funding regime (−3.7%/yr if funding
   flips sign persistently).
5. **Is it worth deploying with real capital?**
   **Marginally yes — as a small, fully-automated, monitored allocation, not
   as a return engine.** It roughly matches stablecoin yield (~3.5%) with
   slightly better drawdown characteristics but adds exchange, key-management,
   and funding-regime risk. It does NOT beat buy-and-hold (38% CAGR, −53% DD)
   on return, and its edge over stablecoins is within noise. Deploy only if:
   (a) the user already holds USDT on Bybit/OKX, (b) the shadow trader passes
   the 14-day validation, (c) automation + alerts from §7 are implemented,
   and (d) the user accepts ~3%/yr as the realistic expectation. Otherwise,
   holding USDT/stablecoin yield is the simpler equivalent.

## 9. Unresolved assumptions

- Bybit spot taker 0.10% / perp taker 0.055% non-VIP rates — verify against
  the user's actual account tier at deployment.
- Slippage 0.05%/side is an assumption; BTCUSDT is deep but $500 clips are
  small — likely conservative.
- USDT depeg / margin-currency risk not modelled.
- Exchange downtime scenario analysed qualitatively only.
- Shadow trader validates data/timing, not account-level settlement records
  (requires funded test account).
- Bybit mark-price kline close at settlement instant used as settlement
  price proxy; Bybit settles funding at mark price — exact tick-level mark
  at 00/08/16 UTC not retrievable historically.
- Rule A/B evaluated at daily close in backtest; live trader runs daily —
  intraday margin calls between checks are handled exchange-side (isolated
  margin liquidation) but not pre-empted.

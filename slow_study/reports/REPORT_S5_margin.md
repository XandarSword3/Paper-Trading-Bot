# S5 v2 — Funding Carry WITH Margin Model

Frozen rules in `backtest/s5_margin.py` docstring. Capital $1,000 total.

Margin: MMR=0.40% (OKX BTCUSDT-SWAP tier-1, OKX help 2023-06-09; Bybit tier-1 MMR=0.33% per Bybit margin parameters — higher OKX value used). IM = notional/leverage by construction. Isolated-margin liquidation approximation; exact exchange liq-price formulas not replicated.

## 1. S5 v1 reconciliation (the unmodelled-margin version)

- Window 2020-01-01 -> 2026-11-25, qty=0.13955 BTC constant.

- Cumulative funding received (short perp): **$5,177.51**

- Total trading costs: $28.28 (entry $3.00, exit $25.28)

- Equity gain: $5,149.23 = funding $5,177.51 − costs $28.28 + residual $0.00 (spot/perp basis noise)

- Perp/spot leg notional grew $1,001 -> $8,427; leg-notional/equity ratio: 1.00x -> max **3.95x** (implicit leverage, unmodelled margin)

- v1 final equity $6,149.23 on $1,000 — NOT achievable: perp leg alone reached −$16,405 unrealised at BTC $124,720 (2025-10-06) with zero margin modelled.


## 2. S5 v2 results by leverage (total capital basis)

| L | final $ | CAGR | mean/mo (arith) | geo/mo | worst/mo | maxDD | min liq dist | terminated | costs $ |
|---|---|---|---|---|---|---|---|---|---|
| 1x | 1434 | 5.4% | 0.44% | 0.44% | -0.5% | -0.9% | 47.3% | no | 26 |
| 2x | 1626 | 7.3% | 0.60% | 0.59% | -0.7% | -1.2% | 0.5% | no | 35 |
| 3x | 2046 | 10.9% | 1.84% | 1.14% | -29.3% | -65.0% | -14.7% | YES | 10 |

*min liq dist = closest approach of liquidation price to the intraday HIGH (conservative); close-based distance in `s5v2_summary.csv`. Reconciliation check (equity vs components): max abs diff 6.82e-13 $ — identity holds.


### Split detail (mean monthly, arithmetic)

| L | FULL | 2023+ | 2025+ |
|---|---|---|---|
| 1x | 0.44% | 0.28% | 0.15% |
| 2x | 0.60% | 0.37% | 0.21% |
| 3x | 1.84% | 3.13% | -0.24% |

### Liquidation events

- L=1x: survived. Min distance-to-liquidation 47.3% from intraday high (51.3% from close).

- L=2x: survived. Min distance-to-liquidation 0.5% from intraday high (3.2% from close).

- **L=3x LIQUIDATED** 2021-01-22 est liq ~$28894 (day high $33867), perp equity lost $-137. After liquidation: holds remaining spot+cash, no re-entry.


## 3. Cost drag

- L=1x: total trading costs **$26.32** over 6.9y ≈ $3.82/yr (0.38%/yr of initial capital). Includes entry, margin-defense spot sales + perp covers, and rebalances.

- L=2x: total trading costs **$35.38** over 6.9y ≈ $5.13/yr (0.51%/yr of initial capital). Includes entry, margin-defense spot sales + perp covers, and rebalances.

- L=3x: total trading costs **$10.31** over 6.9y ≈ $1.49/yr (0.15%/yr of initial capital). Includes entry, margin-defense spot sales + perp covers, and rebalances.


## 4. Negative-funding stress (30 days)

- Worst rolling 30-day cumulative funding in sample: **-1.31%** (window ending 2020-04-10).

- L=1x: at terminal perp notional ~$687, a repeat of that window would cost ~$-9 to the short (paid, not received). In v2 this drains perp equity -> triggers Rule A spot sales -> position shrinks.

- L=2x: at terminal perp notional ~$1,039, a repeat of that window would cost ~$-14 to the short (paid, not received). In v2 this drains perp equity -> triggers Rule A spot sales -> position shrinks.

- L=3x: at terminal perp notional ~$0, a repeat of that window would cost ~$-0 to the short (paid, not received). In v2 this drains perp equity -> triggers Rule A spot sales -> position shrinks.

- Funding flipped negative ~14% of all 8h periods historically; 2026 negative share 23.8%. S5 carry is regime-dependent: it earns only when funding is positive.


## 5. vs stablecoin yield

- DefiLlama yields.llama.fi/pools (fetched 2026-11-26): mainstream stable lending ≈ Aave v3 USDC 2.99%, Spark USDS ~3.4-3.8%, Maple USDC ~5.3% (credit risk), Jupiter USDC 4.6%. Centralised equivalents (e.g. exchange earn) ~3-5%/yr — **could not verify** a specific current CEX rate.

- S5v2 L=1x geo ~0.4-0.5%/mo ≈ 5-6%/yr vs stable ~3-5%/yr: carry premium is thin and comes with liquidation/ops risk stable yield doesn't have.


## 6. What could NOT be verified / approximations

- Exact OKX/Bybit liquidation price formulas (fees, tier tables, ADL) — approximated as isolated margin, liq when equity <= MMR*notional, checked on daily highs.

- Funding marked at daily close, not settlement-time price (3 settlements/day share one price — acknowledged approximation, needs intraday data to fix).

- USD (Coinbase) vs USDT (perp margin) basis ignored.

- Margin transfers assumed free/instant; no withdrawal fees modelled.

- After liquidation the sim holds remaining spot+cash (still BTC-exposed) — a real trader might exit fully; flagged as a modelling choice.

- Rule A/B thresholds (M*=N/L, 20% drift) are frozen choices, not optimised.

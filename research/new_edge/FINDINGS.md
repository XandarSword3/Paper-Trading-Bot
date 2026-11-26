# New Edge Lab — Findings

**Bottom line: 9 pre-registered trials, 0 deployable edges.** The full log is in
`registry.json`; raw payloads are in `results/`. Nothing here is wired into
production, and nothing should be traded with real capital.

## What was tested

| ID | Idea | Discovery (net) | Holdout / kill test | Verdict |
| :-- | :--- | :--- | :--- | :--- |
| H1 | 1m liquidation-cascade fade | Sharpe -18.0, 6,613 trades | gross edge ~0 | KILL |
| H2 | Cross-sectional funding reversal, daily | Sharpe 0.49 (gross 1.54) | cost 8.4 bps/day | KILL |
| H2b | Same, 3/5/7-day holds | 0.28 / -0.08 / -0.28 | signal decays in 1 day | KILL |
| H3 | Vol-compression breakout, 1h | Sharpe 4.68 | 1-min delay -> -1.08 | KILL (fill artifact) |
| H4 | BTC-shock beta overshoot, 5m | Sharpe -19.0 | gross +1.5 bps/event | KILL |
| H5 | Cross-sectional momentum, weekly | Sharpe 0.81, p=0.005 | holdout -0.43 | KILL |
| H6 | Funding reversal + funding cash | Sharpe 0.70 | 2024 Sharpe -1.28 | KILL |
| H7 | 12-month TSMOM, monthly | Sharpe 0.36, DD -72% | placebo p=0.17 | KILL |
| H8 | Weekend alt/BTC relative value | -1.64 (mirror +0.52) | holdout gross ~0 | KILL |

## The five things this lab actually established

1. **Same-bar fills manufacture edges.** H3 looked like the best strategy in the
   repository (Sharpe 4.68 discovery, 5.35 holdout, 943% CAGR) and was entirely
   gone with a one-minute entry delay. Any backtest that fills at the close of
   the signal bar is not evidence.

2. **The real gross signals are regime bets.** Funding crowding (gross Sharpe
   1.54) and cross-sectional momentum (0.91) both existed, but both were
   carried by 2021/2023 and both died in the 2025-2026 holdout. A 0.81 Sharpe
   discovery with a 0.5% placebo p-value still failed out of sample. That is
   exactly what the holdout is for.

3. **Intraday signals are an order of magnitude too small.** H1 and H4 both
   found gross edges of roughly 1-2 bps per trade. At 8 bps/side (16 bps round
   trip) there is no way to monetise them, and even at maker fees they do not
   clear the bar.

4. **Calendar effects are real but too small.** Alts reliably outperform BTC
   over the weekend from 2020 to 2024 (t = 4.0 for the spread), but the gross
   edge is 31-39 bps/week against 16 bps/week of round-trip cost, and it
   disappeared in 2025-2026. The pre-registered direction was also backwards,
   which is itself a useful reminder to check the sign of every spread.

5. **The dataset has survivorship bias.** Every symbol in
   `data/binance/klines` is still listed in 2026. Any book that buys losers or
   shorts alts is flattered by the coins that are missing because they died.
   H5 and H8 should be read with that discount.

## What the data cannot answer

- Order-flow and liquidity edges: `data/flow_bars` is one month long and
  internally inconsistent (`trade_count` is always zero, `buy_vol`/`sell_vol`
  do not reconcile with `signed_flow`). It is not usable for a microstructure
  edge.
- Crowding / squeeze mechanics: no open interest, no liquidation prints.
- Basis and cross-exchange effects: no spot prices, one venue.
- Anything involving delisted symbols: the history is not there.

## What would change the answer

In priority order:

1. **Open interest + liquidation prints per symbol** (hourly is enough). This
   would let H2/H4 be tested as crowding mechanics rather than price proxies.
2. **Spot vs perp prices on the same venue.** Basis, funding dispersion and
   spot-led price discovery are separate, structural edges.
3. **A second exchange.** Cross-venue funding dispersion and lead-lag are the
   most likely place for a retail-accessible edge that is not already
   arbitraged on Binance.
4. **Delisted-symbol history.** Without it, every cross-sectional result is
   optimistic and should not be promoted.

## How to reproduce

```powershell
python research/new_edge/h01_cascade_fade.py
python research/new_edge/h02_funding_reversal.py
python research/new_edge/h02b_funding_reversal_hold.py
python research/new_edge/h03_vol_breakout.py
python research/new_edge/h03_diagnostics.py
python research/new_edge/h03_latency.py
python research/new_edge/h04_beta_overshoot.py
python research/new_edge/h05b_csm_momentum.py
python research/new_edge/h06_funding_cash.py
python research/new_edge/h07_tsmom.py
python research/new_edge/h08_weekend_alt_bleed.py
```

`h05_cross_sectional_momentum.py` is the first (sign-bugged) H5 implementation;
its payload is kept for the audit trail. `h05b` is the corrected test and the
one recorded in the registry.

## Honesty notes

- Costs are 8 bps/side in every primary test (Binance taker + slippage). The
  low-turnover survivors were also checked at 4 bps; none became viable.
- H8's holdout is not perfectly blind: the weekend pattern was first seen in a
  full-sample diagnostic before `h08` existed. The per-year breakdown is
  reported so the reader can discount it.
- No parameter was tuned on the holdout. Every variant is logged as its own
  trial, including the ones that only exist to check a sign or an accounting
  omission.

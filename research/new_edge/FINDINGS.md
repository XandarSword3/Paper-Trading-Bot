# New Edge Lab — Findings

**Bottom line: 12 pre-registered trials (plus one live pilot), 0 deployable edges.** The full log is in
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
| H9 | OI crowding, live 21-day pilot | gross +12.4 bps/event | net -3.6 bps after cost | PILOT KILL |
| H10 | Perp/index premium fade, 8h | gross +7.4 bps/event | holdout +1.6 bps; net -8.6 | KILL |
| H11 | Same 72h basis fade on held-out SOL | net -56 bps/event, Sharpe -0.43 | placebo p=0.61 | KILL |
| H12 | BTC/ETH/SOL pairs mean-reversion | net -95 to -212 bps/event | spread trends, holdout -0.80 | KILL |

## The eight things this lab actually established

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

6. **Basis reversion is real but sub-scale.** H10 faded a 2-sigma perp/index
   premium and found a genuine +7.4 bps gross edge per 8h event (t=1.38),
   rising to +11.8 bps (t=2.23) on the smoother mark/index premium. Both are
   below the 16 bps round trip, and the holdout gross fell to +1.6 / -3.9 bps.
   The only positive-net version is a 72h mark/index hold (+9.3 bps discovery,
   +5.5 bps holdout) with a -62% drawdown and t<1. That is a diagnostic, not a
   strategy.

7. **A single-asset positive is not an edge.** The 72h basis fade was the only
   variant in the lab with positive net expectancy in both discovery and
   holdout. Frozen and re-run on held-out SOL (H11), it lost -56 bps net per
   event, Sharpe -0.43, drawdown -99%, placebo p=0.61. The BTC/ETH result was
   asset-specific noise. Every candidate here has now been checked on data
   that did not choose it.

8. **Crypto pairs trend, they do not revert.** H12 faded 2-sigma BTC/ETH/SOL
   log spreads and lost 95-212 bps net per event; the spread keeps going.
   The momentum side was positive in-sample but mixed out of sample. This is
   the opposite of the equity pairs prior and it kills the simplest
   market-neutral idea available on this data.

## What the data cannot answer

- Order-flow and liquidity edges: `data/flow_bars` is one month long and
  internally inconsistent (`trade_count` is always zero, `buy_vol`/`sell_vol`
  do not reconcile with `signed_flow`). It is not usable for a microstructure
  edge.
- Crowding / squeeze mechanics: the cached history has no open interest and no
  liquidation prints. The public Binance API *is* reachable without keys, but
  `/futures/data/openInterestHist` only serves ~21 days (500 hourly bars), so
  OI cannot be backtested over 2020-2026. The 21-day pilot (H9) found a
  +12.4 bps gross fade per 8h event (t=1.70), which is below the 16 bps round
  trip and is not evidence of anything on its own.
- Basis and cross-exchange effects: no spot prices, one venue.
- Anything involving delisted symbols: the history is not there.

## What would change the answer

In priority order:

1. **Start recording open interest, funding and 1m bars forward, now.** The
   public API works without keys, but only serves ~21 days of OI history, so
   the only way to get a real OI sample is to record it. Re-test H9 in 3-6
   months.
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
python research/new_edge/h10_premium_fade.py
python research/new_edge/h11_sol_basis_check.py
python research/new_edge/h12_pairs.py
```

H10 uses the deeper `v4_system/data/` mark, index and perp parquet files
(spot/perp/index 1h from 2019-12). SOL is excluded there because its index
series is corrupted around the 2022-11-09 FTX crash. H11 is the held-out-asset
check of H10's 72h variant and uses SOL from 2021-04-01, excluding the FTX week.

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

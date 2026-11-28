# Slow-Strategy Study — can a slow automated crypto strategy net ~6%/month on $1,000?

Research / paper only. No real orders, no API keys, no wallet credentials.

## Frozen protocol (fixed BEFORE any backtest; no tuning after seeing results)

| ID | Strategy | Rule |
|----|----------|------|
| S1 | BTC 50/200 MA trend | long BTC when close > SMA50 AND SMA50 > SMA200, else flat |
| S2 | TSMOM 90d, BTC+ETH | long/flat per asset when 90d return > 0; weight = min(1, 50% ann. vol target / 30d realized vol); gross capped at 100% |
| S3 | Donchian 55 breakout, BTC | flat -> long when close > highest high of prior 55 bars; long -> flat when close < lowest low of prior 55 bars |
| S4 | Cross-sectional momentum | 10 alt majors (excl. BTC/ETH), 90d momentum, top 3 equal-weight, rebalance every 7 days, long-only |
| S5 | Funding carry (benchmark) | long BTC spot + short BTC-USDT-SWAP perp, same coin qty, hold; collect/pay historical funding |
| S6 | Buy-and-hold BTC | baseline |

All signals on daily CLOSE, executed next day OPEN. 4h data is downloaded per
spec but all strategy rules are daily-scale.

## Costs (non-negotiable)

- Taker fee 0.10% + slippage 0.05% = **0.15% per side** on every trade.
- Funding cashflows applied at historical funding timestamps (S5).
- Min order notional floor **$1** (conservative; exchange minimums are cents).
- $1,000 starting equity. Cash earns 0%.

## Data

- OHLCV: **Coinbase Exchange public REST** (`/products/{id}/candles`), no key.
  - Why not OKX for prices: verified OKX `/market/candles` serves at most the
    most recent **1440 bars** per bar size (daily -> ~Dec 2022 only). Coinbase
    serves full history back to 2019-01-01 (or listing date, whichever later).
  - Files: `data/{BTC,ETH,...}-USD_{1D,4H}.csv`. In-progress candle dropped.
- Funding: **Binance fapi** `fundingRate` for BTCUSDT (deep history 2020-01 ->
  now). Why: OKX `/public/funding-rate-history` only serves ~3 months publicly
  (verified — pagination returns empty beyond that). Binance is reachable from
  this machine; it is only blocked in the user's CI. Substitution documented.
- Universe = today's top-liquidity majors -> **survivorship bias: NOT
  point-in-time**. Delisted/worse performers are absent; results are biased up.

## Run commands

```powershell
cd slow_study
python fetch_coinbase.py          # OHLCV daily+4h, all symbols (resumable, --force to redo)
python fetch_funding_binance.py   # deep BTC funding history for S5
python fetch_data.py              # OPTIONAL: shallow OKX candles+funding (kept for reference)
python run_backtests.py           # runs S1-S6, writes reports/
```

Outputs: `reports/results.csv` (all metrics x period splits),
`reports/verdicts.csv` (gate evaluation), `reports/REPORT.md` (tables+verdict),
`reports/equity_*.csv`, `reports/trades_*.csv`.

## Metrics & gates

- Monthly returns (month-end equity), mean/median, % months >= 6%, CAGR,
  max drawdown, worst month, longest time underwater (days below HWM).
- Circular block bootstrap over monthly returns, block=6 months, 10,000 reps.
  95% CI and Bonferroni-corrected 99% CI (5 strategies).
- Splits: FULL / 2019-2022 / **2023+ (out-of-sample, gates evaluated here)**.
- **VIABLE**: OOS net-return CI excludes zero AND maxDD <= 40%.
- **TARGET**: OOS mean net return >= 6%/month.
- Neither -> "no strategy found", STOP (no rescue strategies).
- If VIABLE: 12-month forward paper trading with frozen rules — see
  `paper/` (only data-quality checks allowed mid-run).

## Results (2023+ out-of-sample, net of costs, Bonferroni-99% CI)

| strategy | mean/mo | CI99 | maxDD | VIABLE | TARGET |
|---|---|---|---|---|---|
| S1 MA trend | 1.80% | [-0.91%, 5.55%] | -30.9% | no | no |
| S2 TSMOM vol-target | 2.50% | [-1.28%, 7.35%] | -46.0% | no | no |
| S3 Donchian 55 | 2.32% | [-1.48%, 6.81%] | -39.6% | no | no |
| S4 XS momentum | -1.86% | [-6.03%, 0.94%] | -70.0% | no | no |
| S5 funding carry (v1, no margin) | 1.14% | [0.56%, 1.86%] | -0.7% | **yes** | no |
| S6 buy-and-hold | 3.82% | [-1.25%, 8.97%] | -53.1% | no | no |

**Verdict: no strategy reaches the ~6%/month target.** S5 (delta-neutral
funding carry) is VIABLE — its OOS net-return CI excludes zero — but earns
only ~1.1-1.4%/month.

> **S5 correction (2026-11-26, v2.1 after external audit):** v1 ignored perp
> margin/liquidation entirely (perp leg hit −$16.4k unrealised). Rebuilt as
> `backtest/s5_margin.py` with isolated-margin model (MMR 0.40%, OKX tier-1),
> 1x/2x/3x leverage, margin-defense and compounding-rebalance rules,
> intraday-high liquidation checks. Result: **L=1x → $1,434, L=2x → $1,626,
> L=3x LIQUIDATED 2021-01-22** (see `reports/REPORT_S5_margin.md`). The v1
> "VIABLE" label does not survive the margin model — realistic carry is
> ~0.4-0.6%/month. Audit fixes: Rule A margin transfer recomputed after
> position shrink, all trade costs recorded, equity swept on full close,
> liquidation distance measured from intraday high (**L=2x came within 0.5%
> of liquidation**), CAGR/monthly returns based on the full $1,000.
> Also fixed: funding CSV timestamp parsing had silently dropped ~45% of
> settlement events (v1 understated funding; corrected numbers above).

Per the frozen protocol, no leveraged variants of S1-S4/S6 were run and no
rescue strategies were added.

## S5 v3 — live-execution validation (2026-11-26)

`reports/REPORT_S5_v3.md` — venue-consistent Bybit backtest (spot + perp +
funding + mark price all from Bybit public data), timestamp-exact funding,
1H mark-price liquidation checks, MMR 0.33% (tier-1 verified via risk-limit
API). Result: **$1,195 final / 3.36% CAGR / maxDD −0.76% / 0 liquidations**
over 2021-07→2026-11; survives 2x costs and ±50–80% BTC shocks; loses money
only if funding flips sign persistently (−3.7%/yr). Shadow trader:
`paper/shadow_s5_bybit.py` (public data, no orders, initialized 2026-11-26).
Verdict: marginally deployable (~stablecoin yield), conditional on 14-day
shadow validation + automation from the report.

Reproduce:
```powershell
python data/fetch_bybit.py        # venue-consistent dataset
python backtest/s5_v3_bybit.py    # backtest + reports
python paper/shadow_s5_bybit.py --init   # shadow trader (run daily)
```

## Forward paper trading (S5 v2, 12 months)

`paper/paper_s5.py` mirrors the margin model (per-leverage state files):

```powershell
cd slow_study
python paper/paper_s5.py --init --leverage 2   # once per leverage (1/2/3)
python paper/paper_s5.py --leverage 2          # once per day after candle close
```

- State: `paper/s5_state_L{L}.json`; every signal/fill/mark logged with UTC
  timestamps to `paper/s5_log_L{L}.csv`.
- Funding applied at real OKX settlement timestamps (public endpoint).
- Liquidation checked on each day's high; Rule A/B identical to backtest.
- Only data-quality checks run mid-run — logged as `DQ` rows, never used to
  change rules.

## Look-ahead & bias checklist

- [x] Signals use only data up to and including bar close t; execution at open t+1.
- [x] In-progress daily/4h candle excluded from history.
- [x] Rolling windows use only trailing data (shift(1) where "prior N bars" required).
- [x] S4 momentum computed only on rebalance dates; weights held between.
- [x] S2 vol/momentum use trailing 90d/30d windows only.
- [x] Costs charged on every trade incl. entries, exits, rebalances.
- [x] Funding applied at historical timestamps, not assumed schedule.
- [x] Min-order floor enforced; sub-$1 orders skipped.
- [ ] Survivorship: universe is current top-liquidity, NOT point-in-time —
      stated bias, results biased upward. Could not verify delisting-free universe.
- [ ] S5 funding source is Binance BTCUSDT perp, not OKX — rates are highly
      correlated across venues but not identical; documented approximation.
- [ ] OKX 4h history limited to ~240 days; 4h delivered from Coinbase instead.
- [ ] No borrow costs / no spot-perp basis modeling in S5 (delta-neutral approx).
- [ ] Taxes, exchange outages, and ADL/liquidation risk not modeled (spot+small size).

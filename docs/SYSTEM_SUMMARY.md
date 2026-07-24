# BTC Turtle-Donchian Strategy — System Summary

## Executive Summary

A backtesting, walk-forward validation, and paper-trading framework for a
Turtle-inspired Donchian breakout strategy, in two variants: V1 (4H
timeframe) and V4 (1H, higher frequency). This doc is a short orientation —
for the full validation methodology and current pass/fail status, see
**[TESTING_AND_OPTIMIZATION_GUIDE.md](TESTING_AND_OPTIMIZATION_GUIDE.md)**,
which is the authoritative source and is kept current with the actual gate
files in `data/`.

## Two numbers exist for each strategy — know which one you're looking at

**In-sample backtest** (from `main.py`'s flat grid search over 2017–2023,
same data used to pick parameters as to score them):

| | V1 (4H) | V4 (1H) |
|---|---|---|
| Total Return | 855% | 1572% |
| Sharpe | 1.98 | 1.20 |
| Max Drawdown | -39.3% | -65% |

**Walk-forward out-of-sample** (`data/walk_forward_results_v1.json` /
`_v4.json` — each fold scored only on data its optimizer never saw):

| | V1 (4H) | V4 (1H) |
|---|---|---|
| Total Return | +120.5% | +489.4% |
| Sharpe | **0.69** | **-0.39** |
| Max Drawdown | -61.7% | **-121.3%** |

The OOS numbers are the ones that matter for deciding whether the edge is
real. V4's negative OOS Sharpe and a max drawdown past -100% (meaning the
stitched equity curve went negative in at least one fold) are why V4 is not
currently trusted — consistent with its confirmed losing real paper-trading
record (see below).

## Current live-trading readiness

| Strategy | Paper trades logged | Gate status | Reason |
|---|---|---|---|
| V1 | 0 | ❌ NOT READY | No paper track record yet — nothing to evaluate |
| V4 | 104 | ❌ NOT READY | Confirmed loss: -28.7% return, Sharpe -1.36, 39.8% max drawdown, 20.2% win rate |

Both `data/readiness_v1.json` and `data/readiness_v4.json` currently set
`ready_for_live: false`. `research/bots/bot_runner.py` checks this gate
before every run and will not trade either strategy live until it flips.

## Optimized parameters (V1, canonical: `research/strategies/config.py`)

| Parameter | Original (TradingView) | Optimized V1 |
|-----------|------------------------|--------------|
| Entry Length | 20 | 40 |
| Exit Length | 10 | 16 |
| Trail Multiplier | 2.5 | 4.0 |
| Risk % | 1.5% | 1.0% |
| Direction | Both | Long Only |

The original TradingView parameters returned **-84.4%** in-sample — the
optimization step was necessary, but "beats -84%" is a low bar and doesn't
by itself establish the optimized version is robust; that's what walk-forward
testing checks.

## Actual current project structure

```
Paper-Trading-Bot/
├── main.py                       # Legacy 7-phase in-sample pipeline (NOT the validation pipeline)
├── research/
│   ├── strategies/                config.py (DEFAULT_PARAMS, DataSplitConfig, GateThresholds), strategy.py
│   ├── data/                      data_fetcher.py, data_fetcher_kraken.py, data_splits.py, metrics_utils.py
│   ├── validation/                walk_forward.py, monte_carlo.py, deflated_sharpe.py, cross_market_validation.py,
│   │                               readiness_gate.py, build_readiness_gates.py, final_holdout_validation.py
│   ├── analysis/                  regime_analysis.py, regime_simulation.py, survivability.py, sp500_reinvest.py
│   └── bots/                      bot_runner.py (unified V1/V4 runner, checks readiness gate), telegram_bot.py
├── backend/                       FastAPI app (main.py), SQLAlchemy models (models.py), JSON→DB migration (backfill.py)
├── frontend/                      index.html dashboard
├── data/                          readiness_v1.json, readiness_v4.json, walk_forward_results_*.json, trades*.json
├── docs/                          This file, TESTING_AND_OPTIMIZATION_GUIDE.md, CRITICAL_WARNINGS.md, etc.
└── docs/archive/backups/          Superseded strategy variants (V2/V3, old V4 modules) — see backups/README.md
```

Note: `paper_bot.py`, `github_bot.py`, `github_bot_v4.py`, and `dashboard.py`
referenced in older documentation and commit history no longer exist — they
were consolidated into `research/bots/bot_runner.py` and `frontend/index.html`.

## Yearly performance breakdown (V1, in-sample)

| Year | Return | Status |
|------|--------|--------|
| 2018 | -2.9% | Bear market |
| 2019 | +101.4% | Best year |
| 2020 | +81.7% | Strong bull |
| 2021 | -7.7% | Choppy top |
| 2022 | -22.6% | Bear market |
| 2023 | +74.7% | Recovery |
| 2024 | +32.2% | Solid growth |
| 2025 | +6.9% | Partial year |

This breakdown is computed the same in-sample way as the 855% headline
(one backtest over the whole period) — useful for seeing which regimes the
rules struggled in, not as a forward-looking return estimate.

## Not modeled by any of the above

- Black swan events (exchange failures, delistings)
- Regulatory/market-structure change
- Psychological trading errors
- Extended bear markets beyond what's in the 2017–2025 window
- Full execution slippage/costs beyond the fixed assumptions baked into the backtest

See [CRITICAL_WARNINGS.md](CRITICAL_WARNINGS.md) for the full risk
disclosure and [TESTING_AND_OPTIMIZATION_GUIDE.md](TESTING_AND_OPTIMIZATION_GUIDE.md)
for how to run every phase yourself and verify the numbers above.

---

*Last updated: July 24, 2026, cross-checked against `data/readiness_v1.json`,
`data/readiness_v4.json`, `data/walk_forward_results_v1.json`, and
`data/walk_forward_results_v4.json`.*

# 🐢 BTC Turtle-Donchian Trading Strategy System

A comprehensive algorithmic trading system implementing a Turtle-inspired Donchian Channel breakout strategy for Bitcoin. This project includes backtesting, optimization, Monte Carlo simulation, paper trading bots, and live trading capabilities.

## 📊 Overview

This system transforms the classic Turtle Trading strategy into a modern cryptocurrency trading bot with extensive backtesting and risk analysis. Starting from TradingView parameters that **lost 84%**, we optimized to achieve **855%+ returns** — but that number is an **in-sample backtest headline**, not a validated result. See the callout below before trusting any return figure in this README.

> ⚠️ **Read this before trusting any return number in this document.** The 855% (V1) / 1572% (V4) figures throughout this README come from a single in-sample backtest — the same data used to pick parameters is used to score them. Walk-forward testing on data the optimizer never saw tells a different story: V1's out-of-sample Sharpe is 0.69 (vs. 1.98 in-sample) with a -61.7% drawdown, and V4's out-of-sample Sharpe is **negative** with a drawdown past -100%. V4 additionally has a **confirmed losing real paper-trading track record**. Neither strategy is currently authorized to trade live (`ready_for_live: false` for both). Full details: **[TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md)**.

### Key Achievements
- ✅ **8+ years of historical backtesting** (Aug 2017 - Dec 2025) — in-sample
- ✅ **Multiple strategy versions** (V1: 4H timeframe, V4: 1H high-frequency)
- ✅ **Parameter optimization** across 2,800+ combinations — in-sample only, see walk-forward results for the validated numbers
- ✅ **Walk-forward out-of-sample validation** across 9 rolling folds per strategy
- ✅ **Monte Carlo & deflated-Sharpe testing** on out-of-sample returns, correcting for how many parameter combinations were tried
- ✅ **Fail-closed live-trading gate** — bots default to log-only unless a strategy's own paper-trading record clears explicit thresholds
- ✅ **Automated paper trading** via a scheduled GitHub Actions runner (Kraken data)
- ✅ **Live trading integration** with Telegram bot notifications
- ✅ **Interactive web dashboard** for real-time analysis

## 🏗️ Architecture & Restructure

```
Paper-Trading-Bot/
├── backend/            FastAPI Application & SQLAlchemy DB Models (SQLite / PostgreSQL)
│   ├── main.py                   FastAPI app + endpoints (dashboard data, trades, readiness)
│   ├── models.py                 ORM schemas (Strategy, Trade, EquitySnapshot, ReadinessGate, BacktestRun)
│   └── backfill.py               One-time JSON-to-DB migration script
├── frontend/           Modern Interactive Dashboard (HTML5 / JS / Glassmorphism UI)
│   └── index.html                Real-time metrics, trade history table, execution controls
├── research/           Relocated Validation & Strategy Core (Walk-Forward, Deflated Sharpe, Monte Carlo)
│   ├── strategies/               strategy.py, strategy_registry.py
│   ├── validation/               walk_forward.py, monte_carlo.py, robustness_test.py, readiness_gate.py
│   └── bots/                     bot_runner.py (Unified parameterized runner for V1 & V4)
├── infra/              Docker Compose & Infrastructure Setup
│   └── docker-compose.yml        TimescaleDB & FastAPI service definition
└── docs/               Consolidated documentation & historical guides
```

## 🚀 Quick Start

### 1. Database & Migration
```bash
# Run one-time migration to backfill historical JSON trades into SQLite
python backend/backfill.py
```

### 2. Run Unified Bot Runner
```bash
# Run V4 Fast Strategy (1H) — single cycle, checks the readiness gate first
python -m research.bots.bot_runner --strategy v4

# Run V1 Turtle Strategy (4H)
python -m research.bots.bot_runner --strategy v1

# Run all registered strategies
python -m research.bots.bot_runner --strategy all
```
In production this runs on a GitHub Actions schedule (`.github/workflows/bot.yml`),
not as a local always-on loop — see [BOT_GUIDE.md](docs/BOT_GUIDE.md).

### 3. Launch FastAPI Backend & Dashboard
```bash
# Start backend server
uvicorn backend.main:app --reload --port 8000

# Open frontend/index.html (or http://localhost:8000 if the backend serves it) for the dashboard UI
```


### Run Liquidation Hunter From This Repo

The project now includes a bridge to run the attached `liquidation-hunter` stack from the same entrypoint.

```bash
# Run robustness research via bridge
python main.py --engine liquidation-hunter --lh-mode research --candles 8760 --oos-ratio 0.4 --folds 4 --mc-iterations 2000

# Backtest mode via bridge
python main.py --engine liquidation-hunter --lh-mode backtest --candles 8760

# Dry-run to verify command/path without execution
python main.py --engine liquidation-hunter --lh-mode research --dry-run
```

Path resolution order for `liquidation-hunter`:

1. `--lh-dir` argument.
2. `LIQUIDATION_HUNTER_DIR` environment variable.
3. Default sibling path: `../Trading Bot/liquidation-hunter`.

When a bridge run completes, key artifacts are synced into `results/liquidation_hunter/`.

## 📁 Project Structure

This is the actual current layout — earlier revisions of this README described
a flat root-level layout (`paper_bot.py`, `dashboard.py`, `github_bot.py`,
`pages/`) that was consolidated away; those files no longer exist.

```
Paper-Trading-Bot/
├── main.py                        # Legacy 7-phase in-sample pipeline (NOT the validation pipeline — see docs/TESTING_AND_OPTIMIZATION_GUIDE.md)
│
├── research/
│   ├── strategies/
│   │   ├── strategy.py            # V1: canonical Turtle-Donchian implementation (4H) — the one module
│   │   │                          #     walk_forward.py, monte_carlo.py, cross_market_validation.py all run against
│   │   ├── strategy_registry.py
│   │   └── config.py              # DEFAULT_PARAMS, DataSplitConfig, GateThresholds, WalkForwardConfig, etc.
│   │
│   ├── data/
│   │   ├── data_fetcher.py        # Binance BTCUSDT data
│   │   ├── data_fetcher_kraken.py # Kraken data (used by cross-market validation & live bots)
│   │   ├── data_splits.py         # Enforces the one frozen in-sample/validation/holdout boundary
│   │   └── metrics_utils.py
│   │
│   ├── validation/                 # ⭐ The corrected validation pipeline — see docs/TESTING_AND_OPTIMIZATION_GUIDE.md
│   │   ├── readiness_gate.py       # Phase 0: fail-closed live-trading gate
│   │   ├── walk_forward.py / walk_forward_test.py   # Phase 2: rolling walk-forward OOS
│   │   ├── monte_carlo.py          # Phase 3: OOS block-bootstrap (+ legacy in-sample sequence-risk MC)
│   │   ├── deflated_sharpe.py      # Phase 3: Probabilistic/Deflated Sharpe Ratio
│   │   ├── cross_market_validation.py  # Phase 4: same rules on ETH/1h/Kraken
│   │   ├── build_readiness_gates.py    # Phase 5: generates readiness_<strategy>.json from real paper trades
│   │   ├── final_holdout_validation.py # Only script allowed to touch 2025-01-01→latest
│   │   ├── robustness_test.py      # In-sample plateau check only — not a performance claim
│   │   └── validate_all.py         # Regression test for backtest code, not a validity check
│   │
│   ├── analysis/                   # regime_analysis.py, regime_simulation.py, survivability.py, sp500_reinvest.py
│   └── bots/
│       ├── bot_runner.py           # Unified V1/V4 runner — checks readiness_gate before trading
│       └── telegram_bot.py
│
├── backend/                        # FastAPI app (main.py), SQLAlchemy models (models.py), JSON→DB migration (backfill.py)
├── frontend/                       # index.html dashboard
├── infra/                          # docker-compose.yml
│
├── docs/
│   ├── README.md                   # This file
│   ├── TESTING_AND_OPTIMIZATION_GUIDE.md  # ⭐ Authoritative validation workflow + current pass/fail status
│   ├── SYSTEM_SUMMARY.md
│   ├── BOT_GUIDE.md
│   ├── CRITICAL_WARNINGS.md
│   ├── PAPER_TRADING_CHECKLIST.md
│   └── archive/backups/            # Superseded V2/V3/old-V4 strategy variants — see backups/README.md
│
├── data/
│   ├── BTCUSDT_4h.csv / .parquet, BTCUSDT_1h.csv / .parquet
│   ├── readiness_v1.json, readiness_v4.json           # Live-trading gate state (currently both false)
│   ├── walk_forward_results_v1.json, _v4.json          # Out-of-sample walk-forward summaries
│   ├── trades.json, trades_v4.json                     # Real paper-trading trade logs
│   └── bot_state.json, bot_state_v4.json
│
├── tests/                          # pytest suite (readiness gates, no-negative-equity, no-same-bar-pyramid, etc.)
└── requirements.txt / .env         # Dependencies / API keys (not committed)
```

## 🎯 Strategy Versions

### V1: 4-Hour Optimized Strategy (855% in-sample backtest)
**Best for:** Swing trading, lower trade frequency, easier to manage

| Parameter | Value | Why |
|-----------|-------|-----|
| Timeframe | 4H | Filters noise, captures major trends |
| Entry Length | 40 | 160-hour breakout (fewer false signals) |
| Exit Length | 16 | 64-hour channel (good trend capture) |
| Trail Multiplier | 4.0 | Wide stops for volatile crypto markets |
| Risk % | 1.0% | Conservative position sizing |
| Direction | Long Only | Shorts historically unprofitable in BTC |

**In-sample backtest results (Aug 2017 - Dec 2023, optimizer's own data):**
- Total Return: **+855%** ($1,000 → $9,550)
- Max Drawdown: **-39.3%**
- Win Rate: **45.5%**
- Profit Factor: **1.60**
- Total Trades: **442** (~0.14 trades/day)

**Walk-forward out-of-sample (2020-05 → 2024-11, 9 folds, data the optimizer never saw):**
- Total Return: **+120.5%** — Sharpe **0.69** — Max Drawdown **-61.7%** — 321 OOS trades
- **Paper-trading gate: NOT READY** (0 completed real paper trades logged yet)

### V4: 1-Hour High-Frequency Strategy (1572% in-sample backtest)
**Best for:** Active traders, higher capital efficiency, more opportunities

| Parameter | Value | Difference from V1 |
|-----------|-------|-------------------|
| Timeframe | 1H | 4x more granular |
| Entry Length | 8 | 8-hour breakout (faster entries) |
| Exit Length | 16 | 16-hour channel |
| Trail Multiplier | 3.5 | Slightly tighter for faster timeframe |
| Risk % | 1.0% | Same conservative sizing |
| Direction | Long Only | Same as V1 |

**In-sample backtest results (optimizer's own data):**
- Total Return: **+1572%** ($1,000 → $16,720)
- Max Drawdown: **-65%** (higher volatility)
- Win Rate: **44%**
- Profit Factor: **1.27**
- Total Trades: **~3,900** (~1.33 trades/day)

**Walk-forward out-of-sample (2020-05 → 2024-11, 9 folds):**
- Sharpe **-0.39 (negative)** — Max Drawdown **-121.3%** (equity curve went negative in at least one fold)
- **Real paper-trading track record: confirmed loss** — -28.7% return, Sharpe -1.36, 39.8% max drawdown, 20.2% win rate over 104 trades (Dec 2025–Jul 2026)
- **Paper-trading gate: NOT READY**

**Bottom line:** the "1572%" headline is not representative of V4's validated
performance. See [TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md)
for full numbers and methodology before drawing conclusions from either table above.

## 🔬 Research & Development Journey

> **This section documents the original in-sample optimization history** (how
> V1/V4's parameters were arrived at). It predates, and does not replace, the
> out-of-sample validation pipeline. The "phases" below are `main.py`'s
> phases, numbered separately from the corrected `research/validation/`
> pipeline's Phase 0–7 in [TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md) —
> don't conflate the two. Everything here is in-sample unless stated otherwise.

### Phase 1: Original TradingView Parameters (FAILED ❌)
```
Entry: 20, Exit: 10, Trail: 2.5, Risk: 1.5%
Result: -84.4% return, -90.9% max drawdown
```
The original parameters from TradingView **lost money in every market regime**.

### Phase 2: Regime Decomposition ✓
Tested strategy across 6 distinct market regimes:
- **2017 Bull** (+198% BTC): Strategy +48%
- **2018 Bear** (-72% BTC): Strategy -37%  
- **2019 Chop** (+96% BTC): Strategy +24%
- **2020-2021 Bull** (+553% BTC): Strategy -42%
- **2022 Bear** (-65% BTC): Strategy -51%
- **2023-2025 Recovery** (+435% BTC): Strategy -41%

**Key Finding:** Original parameters failed in ALL regimes!

### Phase 3: Parameter Optimization ✓
Tested **2,800 parameter combinations**:
- **56%** of combinations were profitable
- **24.5%** achieved >100% returns
- Identified robust parameter ranges (plateaus vs cliffs)

**Optimization Results:**
| Parameter | Original | Optimized | Impact |
|-----------|----------|-----------|--------|
| Entry Length | 20 | 40 | ✅ Fewer false breakouts |
| Exit Length | 10 | 16 | ✅ Better trend capture |
| Trail Multiplier | 2.5 | 4.0 | ✅ Avoids noise volatility |
| Risk % | 1.5% | 1.0% | ✅ More sustainable |

### Phase 4: Monte Carlo Simulation (20-Year Forward Projection) ✓
Simulated **1,000 paths** over 20 years with realistic market conditions:

**Realistic Scenario** (with costs):
- Median Outcome: **$220,210** (22,021% return)
- 5th Percentile: **$17,846** (worst case still profitable)
- 95th Percentile: **$4.3M** (best case)

**Probability Table:**
- 100% chance of profit (no paths lost money)
- 97.9% chance of exceeding $10,000
- 66.9% chance of exceeding $100,000  
- 18.3% chance of exceeding $1,000,000

### Phase 5: Cost Modeling ✓
Included realistic trading costs:
- Exchange fees: **0.15%** per trade
- Slippage: **0.05%** per trade
- Annual execution errors: **1%**
- Network fees and spread costs

**Impact:** Realistic simulation shows ~40% lower returns than optimistic scenario.

## 🤖 Trading Bot Features

> `paper_bot.py` and `github_bot.py` (referenced in older versions of this
> README) no longer exist. Both were consolidated into a single unified
> runner. Full details: [BOT_GUIDE.md](docs/BOT_GUIDE.md).

### Unified Bot Runner (`research/bots/bot_runner.py`)
- Connects to **Kraken's public API** (candle data, no keys required to run)
- One cycle per invocation, scheduled hourly (V4) / every 4H (V1) via
  `.github/workflows/bot.yml` — not a local always-on loop
- **Checks `readiness_gate.check_gate()` before every run.** If the gate is
  blocked, it still logs price/ATR/signal state and saves state, but takes no
  trading action ("log-only mode")
- State persists to `data/bot_state.json` / `data/bot_state_v4.json`, trades
  to `data/trades.json` / `data/trades_v4.json`
- Telegram notifications via `research/bots/telegram_bot.py`

```bash
# Run one cycle for either strategy (or both)
python -m research.bots.bot_runner --strategy v1
python -m research.bots.bot_runner --strategy v4
python -m research.bots.bot_runner --strategy all
```

⚠️ Both strategies currently sit behind a blocked gate
(`ready_for_live: false` in `data/readiness_v1.json` / `readiness_v4.json`) —
every scheduled run is currently log-only. See
[TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md)
for why and what has to change first.

Required setup (Telegram notifications only — no exchange keys needed for
paper/log-only operation):
```bash
# Repo secrets (GitHub Actions) or .env for local runs:
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DATABASE_URL=your_db_url   # optional, for backend/ persistence
```

## 📊 Interactive Dashboard

The Streamlit dashboard (`dashboard.py`) referenced in older docs was
replaced by a FastAPI backend + static frontend:

```bash
uvicorn backend.main:app --reload --port 8000
# then open frontend/index.html
```

Features:
- 📈 Live BTC price charts with Donchian channels
- 📊 Strategy performance metrics
- 🔍 Individual trade analysis
- 📉 Drawdown visualization
- 🎯 Parameter sensitivity analysis
- 📱 Multi-page interface for different analyses

## ⚠️ Critical Warnings

### 🚨 READ BEFORE TRADING REAL MONEY

This simulation is **optimistic**. Real trading will face:

1. **Trading Costs** (reduces returns by 20-40%)
   - Exchange fees: 0.1-0.15% per trade
   - Slippage: 0.05-0.2% per execution
   - Network fees and spread costs

2. **Execution Problems**
   - Order delays and partial fills
   - Price gaps through stop losses
   - Exchange downtime and API failures

3. **Psychological Failures** (90% of traders fail here)
   - Panic selling during drawdowns
   - Revenge trading after losses
   - Parameter tweaking mid-strategy
   - FOMO and emotional overrides

4. **Black Swan Events**
   - Exchange collapses (FTX, Mt.Gox)
   - Flash crashes and liquidation cascades
   - Regulatory changes
   - Market structure shifts

5. **Historical Bias**
   - BTC had unprecedented bull market 2017-2021
   - Past performance ≠ future results
   - Strategy optimized on limited data

**See [CRITICAL_WARNINGS.md](docs/CRITICAL_WARNINGS.md) for full risk disclosure.**

## 🎓 What You'll Learn

This project demonstrates:
- ✅ Professional backtesting methodology
- ✅ Parameter optimization techniques
- ✅ Monte Carlo simulation for risk assessment
- ✅ Real-time data integration (Binance API)
- ✅ Position sizing and risk management
- ✅ Trade execution and order management
- ✅ State persistence and error handling
- ✅ Telegram bot integration
- ✅ Interactive data visualization
- ✅ Performance metrics and analytics

## 🔧 Dependencies

### Core Libraries
```
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
vectorbt>=0.26.0        # Vectorized backtesting
matplotlib>=3.7.0       # Plotting
seaborn>=0.12.0         # Statistical visualization
scipy>=1.10.0           # Scientific computing
```

### Trading & Data
```
python-binance>=1.0.19  # Binance API wrapper
requests>=2.28.0        # HTTP requests
python-dotenv>=1.0.0    # Environment variables
```

### Dashboard & Utilities
```
streamlit>=1.28.0       # Interactive dashboard
plotly>=5.18.0          # Interactive plots
tqdm>=4.65.0            # Progress bars
pyarrow>=14.0.0         # Fast data serialization
```

## 📚 Key Files Explained

| File | Purpose |
|------|---------|
| `research/strategies/strategy.py` | Core V1 strategy class with Donchian logic |
| `research/strategies/config.py` | `DEFAULT_PARAMS`, `DataSplitConfig`, `GateThresholds` |
| `research/data/data_fetcher.py` | Downloads historical data from Binance |
| `research/validation/robustness_test.py` | In-sample grid search over 2,800 combos — plateau check only, not a performance claim |
| `research/validation/walk_forward.py` / `walk_forward_test.py` | Rolling out-of-sample validation — the actual "is this real" test |
| `research/validation/readiness_gate.py` | Fail-closed check that blocks live trading by default |
| `research/validation/build_readiness_gates.py` | Computes real readiness from paper-trading trade logs |
| `research/bots/bot_runner.py` | Unified V1/V4 bot — gated by `readiness_gate.py` |
| `backend/main.py` + `frontend/index.html` | FastAPI backend + dashboard (replaced the old Streamlit `dashboard.py`) |
| `research/validation/validate_all.py` | Regression test that a backtest reproduces expected in-sample numbers — **not** a validity check for the strategy |

## 🔥 Performance Highlights

⚠️ These are the **in-sample** backtest numbers. See
[TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md)
for the out-of-sample numbers that actually matter for judging whether the
edge is real.

### V1 Strategy (4H Timeframe) — In-Sample
- **Initial Capital:** $1,000
- **Final Equity:** $9,550
- **Return:** **+855%**
- **CAGR:** ~30% over 8 years
- **Max Drawdown:** -39.3%
- **Sharpe Ratio:** 1.98
- **Profit Factor:** 1.60
- **Win Rate:** 45.5%

### Walk-Forward Out-of-Sample (validated)
- **Return:** +120.5% | **Sharpe:** 0.69 | **Max Drawdown:** -61.7% | 321 OOS trades across 9 folds

### Yearly Breakdown (in-sample)
| Year | Return | Status |
|------|--------|--------|
| 2018 | -2.9% | 📉 Bear market |
| 2019 | +101.4% | ✅ Best year |
| 2020 | +81.7% | ✅ Strong bull |
| 2021 | -7.7% | 📉 Choppy top |
| 2022 | -22.6% | 📉 Bear market |
| 2023 | +74.7% | ✅ Recovery |
| 2024 | +32.2% | ✅ Solid growth |
| 2025 | +6.9% | ⏳ Partial year |

**Winning Years:** 5 | **Losing Years:** 3

## 🎯 Usage Examples

### Run a Quick In-Sample Backtest
```bash
# Legacy 7-phase pipeline — in-sample only, useful as a code sanity check
python main.py
```

### Walk-Forward Out-of-Sample Validation (the number that matters)
```bash
cd research/validation
python walk_forward_test.py --fast     # quick sanity run
python walk_forward_test.py --jobs 8   # full grid per fold
```

### Parameter Robustness Check (in-sample plateau, not a performance claim)
```bash
python research/validation/robustness_test.py
# Results saved to: results/robustness_results.csv
```

### Cross-Market Validation
```bash
python research/validation/cross_market_validation.py
```

### Rebuild Live-Trading Readiness Gates
```bash
python research/validation/build_readiness_gates.py --all
```

### Run the Bot (Single Cycle, Gated)
```bash
python -m research.bots.bot_runner --strategy v1
```

### Compare vs S&P 500
```bash
python research/analysis/sp500_reinvest.py
```

## 🧪 Testing & Validation

**Full methodology, current pass/fail status, and every script's role:**
[TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md) —
this is the doc to read, not the summary below.

### Code regression check (not a strategy-validity check)
```bash
python research/validation/validate_all.py
```
This verifies a backtest reproduces expected in-sample numbers (e.g. "Total
Return > 800%"). It's a regression test for the backtest engine — a strategy
that's badly overfit would still pass every check in this file.

### Actual strategy-validity pipeline
```bash
python research/validation/walk_forward_test.py --fast   # Phase 2: OOS folds
python research/validation/edge_validation_test.py        # Phase 3: bootstrap + deflated Sharpe
python research/validation/cross_market_validation.py     # Phase 4: does it hold on ETH/1h/Kraken?
python research/validation/build_readiness_gates.py --all # Phase 5: real paper-trading gate
```

**Current status for both strategies: `ready_for_live: false`.** V1 has zero
recorded paper trades; V4 has a confirmed losing paper-trading track record.
Neither is authorized to trade live right now.

## 📈 Comparison: Strategy vs Buy-and-Hold Bitcoin

### The Uncomfortable Truth

**This is the most important comparison you need to see** — though note the
V1/V4 columns below are the in-sample backtest, which walk-forward testing
shows is optimistic (V1's validated OOS return is +120.5%, not 855%; V4's
validated OOS Sharpe is negative). The underperformance-vs-buy-and-hold
conclusion below holds regardless — it gets *more* true with the honest
numbers, not less.

| Metric | BTC Buy-Hold | V1 Strategy (in-sample) | V4 Strategy (in-sample) |
|--------|--------------|-------------|-------------|
| Period | Jan 2017 - Dec 2025 | Same | Same |
| Starting Price | $998 | $1,000 capital | $1,000 capital |
| Ending Price | $87,450 | N/A | N/A |
| **CAGR** | **74.0%** | **32.2%** | **41.7%** |
| Total Return | **8,663%** | **855%** | **1,572%** |
| **vs Buy-Hold** | Baseline | 43.6% | 56.4% |
| Max Drawdown | ~-65% (2018) | -39.3% | -65% |
| Active Management | None | Required | Intensive |

### What This Means

**If you had $1,000 and did NOTHING:**
- Buy-and-hold: **$87,450** (74% CAGR)
- V1 Strategy: **$9,550** (32.2% CAGR)
- **Difference: $77,900 forgone**

**The Hard Truth:**
- ✓ Both strategies are profitable
- ✗ Both massively underperform simple buy-and-hold on BTC
- ✗ V1 captures only **44%** of buy-and-hold returns
- ✗ V4 captures only **56%** of buy-and-hold returns
- ✓ But V1 has **lower psychological stress** (-39% vs -65% drawdown)

### Why This Happened

Trend-following systems like the Turtle-Donchian work best in:
- Choppy/sideways markets (missed 2017-2021 mega-bull)
- Ranging periods (lock in small gains)
- Volatile corrections (protect downside)

They struggle in:
- Strong bull markets (miss early rallies)
- Parabolic moves (whipsaws on corrections)
- Bitcoin's historical 74% CAGR trend (too good to beat!)

### The Real Value Proposition

**This system isn't about beating buy-and-hold.** It's about:
- 🛡️ **Lower drawdown** (39% vs 65%+ swings)
- 😴 **Sleep better at night** (less volatility stress)
- 📊 **Active income** (something to actively manage)
- 🎓 **Learning framework** (understand algo trading)
- 🔍 **Diverse portfolio** (use alongside other strategies)

### Strategy vs S&P 500 (Alternative Comparison)

**Scenario:** $1,000 initial + $300/month for 8 years

| Metric | BTC Strategy (V1) | S&P 500 Index |
|--------|-------------------|---------------|
| Initial Capital | $1,000 | $1,000 |
| Monthly Addition | $0 | $300 |
| Total Invested | $1,000 | $29,800 |
| Final Value | $9,550 | ~$45,000 |
| Return on Initial | +855% | +4,400% |
| Risk (Max DD) | -39.3% | ~-25% |
| Active Management | Required | Passive |

**Key Insight:** S&P 500 wins with disciplined monthly contributions, but V1 wins on pure initial capital efficiency (no ongoing deposits needed).

## 🛡️ Risk Management

### Position Sizing
- **Risk per trade:** 1% of equity
- **Max units:** 4 (pyramiding on breakouts)
- **Stop loss:** 4.0 × ATR trailing stop

### Portfolio Protection
- **Max drawdown experienced:** -39.3% (V1), -65% (V4)
- **Diversification:** Single asset (BTC) - consider adding other assets
- **Capital preservation:** Never risk more than can afford to lose

### Recommended Safety Measures
1. Start with paper trading (testnet)
2. Use only 20-30% of total capital for strategy
3. Keep emergency fund outside of trading
4. Set maximum drawdown threshold (e.g., -50%)
5. Review and adjust parameters quarterly

## 🔒 Security & API Setup

> Older versions of this README described Binance API keys as required for
> paper trading. The current `bot_runner.py` uses **Kraken's public API** for
> candle data — no exchange API key or secret is needed to run either bot in
> its current (log-only/gated) form. Binance keys would only become relevant
> if live order execution against Binance were added later; nothing in this
> repo currently places live orders.

### Environment Variables (.env / GitHub Actions secrets)
```bash
# Telegram Bot (get from @BotFather) — matches bot_runner.py's actual variable names
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional: backend/ database persistence
DATABASE_URL=your_db_url
```

### Security Best Practices
- ✅ Never commit `.env` file to GitHub
- ✅ If/when exchange keys are added, use trade-only permissions (no withdrawals)
- ✅ Enable IP whitelisting on any exchange account used
- ✅ Use 2FA on exchange account
- ✅ Regularly rotate any keys in use
- ✅ Monitor unusual activity via Telegram alerts

## 📊 Performance Metrics Explained

### Return Metrics
- **Total Return:** (Final Equity - Initial Capital) / Initial Capital
- **CAGR:** Compound Annual Growth Rate
- **Sharpe Ratio:** Risk-adjusted returns (>1.0 is good)
- **Profit Factor:** Gross Profit / Gross Loss

### Risk Metrics
- **Max Drawdown:** Largest peak-to-trough decline
- **Win Rate:** % of profitable trades
- **Average Win:** Average profit per winning trade
- **Average Loss:** Average loss per losing trade
- **Expectancy:** (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

### Trade Metrics
- **Total Trades:** Number of completed round trips
- **Avg Trade Duration:** Time from entry to exit
- **Trade Frequency:** Trades per day/week/month

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError` for a missing package
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue:** Bot shows `GATE BLOCKED — log-only mode` and isn't trading
```bash
# This is expected right now — both strategies are gated NOT READY.
# Check why:
cat data/readiness_v1.json   # or readiness_v4.json
# See docs/TESTING_AND_OPTIMIZATION_GUIDE.md for what has to change first.
```

**Issue:** Bot run fails with no candle data
```bash
# Kraken's public API may be temporarily unavailable, or the runner's
# network access failed. Check the GitHub Actions run log for bot.yml.
# Verify data fetching independently:
python -c "from research.data.data_fetcher_kraken import *"
```

**Issue:** Dashboard won't load
```bash
# The dashboard is now FastAPI + static HTML, not Streamlit.
uvicorn backend.main:app --reload --port 8000
# then open frontend/index.html directly, or check the backend logs
# for errors if it's meant to serve the frontend itself
```

## 🤝 Contributing

This is a personal research project, but suggestions are welcome!

### Areas for Improvement
- [ ] Add more timeframes (15m, 1D, 1W)
- [ ] Multi-asset portfolio version
- [ ] Machine learning for parameter adaptation
- [ ] Alternative entry/exit signals
- [ ] Integration with other exchanges
- [ ] Mobile app for monitoring
- [ ] Advanced risk management (Kelly Criterion)
- [ ] Tax reporting automation

## 📜 License

This project is for educational purposes only. Use at your own risk.

**Disclaimer:** Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. This software is provided "as is" without warranty. The author is not responsible for any financial losses.

---

## 💡 Honest Assessment for Buyers/Users

### What This System Actually Is

✅ **A professional backtesting and paper trading framework** for learning algorithmic trading
✅ **A risk management approach** that prioritizes consistent returns over maximum gains
✅ **A research tool** for understanding Donchian breakout strategies
✅ **A lower-volatility alternative** to simple buy-and-hold (39% vs 65%+ drawdowns)

### What This System Is NOT

❌ **A "get rich quick" scheme** (it underperforms buy-and-hold on Bitcoin)
❌ **A replacement for passive investing** (if you want max returns, just HODL)
❌ **A guaranteed money-maker** (historical results ≠ future profits)
❌ **Better than buying Bitcoin** (74% CAGR vs 32% for V1, 41.7% for V4)

### Who Should Use This

**Perfect for:**
- Quant traders learning algorithmic systems
- Traders who prefer active management over passive holding
- People with high stress/anxiety from market volatility
- Portfolio allocation (e.g., 20% in this strategy, 80% in index funds)
- Risk management practitioners studying position sizing
- Developers wanting to learn trading bot architecture

**NOT recommended for:**
- People wanting maximum returns (just buy BTC/ETH)
- Passive "set-and-forget" investors
- Beginners expecting consistent 10%+ monthly returns
- Anyone who can't afford to lose their capital
- Traders uncomfortable with 39%+ temporary drawdowns

### The Core Truth

**The Bitcoin market from 2017-2025 was a 74% CAGR bull market.** Trend-following systems aren't designed to maximize bull market returns—they're designed to:
1. Survive bear markets (this system did)
2. Avoid psychological panic (lower drawdown helps)
3. Provide steady income streams (active trading)
4. Offer learning and adaptability (parameters can evolve)

If maximizing returns is your goal, this system finished second. But if managing risk and staying sane is your goal, it succeeded.

## 🙏 Acknowledgments

- **Turtle Trading System:** Original methodology by Richard Dennis
- **Donchian Channels:** Indicator by Richard Donchian  
- **VectorBT:** Fast backtesting library
- **Binance API:** Data and execution infrastructure
- **Community:** TradingView, QuantConnect, Reddit r/algotrading

## 📞 Support & Contact

- **Issues:** Open a GitHub issue
- **Questions:** Check documentation in `/docs`
- **Updates:** Watch this repository for improvements

## 🗺️ Roadmap

### ✅ Completed
- [x] V1 strategy optimization (4H timeframe)
- [x] V4 high-frequency strategy (1H timeframe)
- [x] Monte Carlo simulations
- [x] Paper trading bot (testnet)
- [x] Live trading bot with Telegram
- [x] Interactive Streamlit dashboard
- [x] Comprehensive documentation

### 🚧 In Progress
- [ ] Multi-exchange support (Coinbase, Kraken)
- [ ] Portfolio mode (BTC, ETH, SOL)
- [ ] Machine learning parameter optimization

### 📋 Planned
- [ ] Mobile app for iOS/Android
- [ ] Web-based dashboard (React)
- [ ] Advanced analytics (regime detection)
- [ ] Social trading features
- [ ] Automated tax reporting

---

## 📚 Additional Resources

- [TESTING_AND_OPTIMIZATION_GUIDE.md](docs/TESTING_AND_OPTIMIZATION_GUIDE.md) - **Start here** for validation methodology and current pass/fail status
- [SYSTEM_SUMMARY.md](docs/SYSTEM_SUMMARY.md) - Complete technical overview
- [BOT_GUIDE.md](docs/BOT_GUIDE.md) - How to run trading bots
- [CRITICAL_WARNINGS.md](docs/CRITICAL_WARNINGS.md) - Must-read before trading
- [PAPER_TRADING_CHECKLIST.md](docs/PAPER_TRADING_CHECKLIST.md) - Pre-flight checklist

---


**Last Updated:** July 24, 2026 (validation-workflow documentation corrected — see TESTING_AND_OPTIMIZATION_GUIDE.md)

**Built with ❤️ for algorithmic trading education**


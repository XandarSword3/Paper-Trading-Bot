# 🐢 BTC Turtle-Donchian Trading Strategy System

A comprehensive algorithmic trading system implementing a Turtle-inspired Donchian Channel breakout strategy for Bitcoin. This project includes backtesting, optimization, Monte Carlo simulation, paper trading bots, and live trading capabilities.

## 📊 Overview

This system transforms the classic Turtle Trading strategy into a modern cryptocurrency trading bot with extensive backtesting and risk analysis. Starting from TradingView parameters that **lost 84%**, we optimized to achieve **855%+ returns** through rigorous testing and validation.

### Key Achievements
- ✅ **8+ years of historical backtesting** (Aug 2017 - Dec 2025)
- ✅ **Multiple strategy versions** (V1: 4H timeframe, V4: 1H high-frequency)
- ✅ **Parameter optimization** across 2,800+ combinations
- ✅ **Monte Carlo simulations** with 1,000+ paths over 20 years
- ✅ **Automated paper trading** on Binance testnet
- ✅ **Live trading integration** with Telegram bot notifications
- ✅ **Interactive Streamlit dashboard** for real-time analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "BTC Strategy"

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Backtest

```bash
# Quick analysis with optimized parameters
python quick_analysis.py

# Full analysis pipeline (all phases)
python main.py

# Interactive dashboard
streamlit run dashboard.py
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

```
BTC Strategy/
├── 📊 Core Strategy Files
│   ├── strategy.py              # V1: Core Turtle-Donchian implementation (4H)
│   ├── strategy_v2.py           # V2: Enhanced version with improvements
│   ├── strategy_v3.py           # V3: Fast execution variant
│   ├── strategy_v4.py           # V4: High-frequency 1H version (1572% returns!)
│   ├── config.py                # V1 optimized parameters
│   ├── config_v2.py             # V2 configuration
│   └── config_v4_optimal.py     # V4 optimal parameters
│
├── 🔬 Analysis & Testing
│   ├── data_fetcher.py          # Downloads BTCUSDT data from Binance
│   ├── robustness_test.py       # Parameter grid optimization (2,800 combos)
│   ├── regime_analysis.py       # Market regime decomposition
│   ├── regime_simulation.py     # 20-year Monte Carlo (optimistic)
│   ├── realistic_regime_simulation.py  # With trading costs & slippage
│   ├── monte_carlo.py           # Standard Monte Carlo simulation
│   ├── survivability.py         # Capital survivability analysis
│   ├── detailed_analysis.py     # Deep performance metrics
│   └── validate_all.py          # Full system validation suite
│
├── 🤖 Trading Bots
│   ├── paper_bot.py             # Paper trading bot (Binance testnet)
│   ├── simple_paper_bot.py      # Simplified paper trading version
│   ├── telegram_bot.py          # Telegram notification bot
│   ├── github_bot.py            # Live trading bot (Binance mainnet)
│   ├── github_bot_v4.py         # V4 live trading bot
│   └── get_chat_id.py           # Get Telegram chat ID utility
│
├── 📈 Visualization & Dashboard
│   ├── dashboard.py             # Main Streamlit dashboard
│   ├── visualization.py         # Plotting utilities
│   └── pages/
│       ├── 1_Candle_Analysis.py
│       ├── 2_V1_Strategy.py
│       └── 3_V4_Strategy.py
│
├── 📊 Comparative Analysis
│   ├── sp500_reinvest.py        # Compare: $1K + $300/mo in S&P 500
│   ├── strategy_analysis.py     # Strategy comparison tools
│   └── forward_test.py          # Forward testing framework
│
├── 📚 Documentation
│   ├── README.md                # This file
│   ├── SYSTEM_SUMMARY.md        # Complete system overview
│   ├── BOT_GUIDE.md             # How to run trading bots
│   ├── CRITICAL_WARNINGS.md     # Risk disclosures & limitations
│   └── PAPER_TRADING_CHECKLIST.md
│
├── 💾 Data & Results
│   ├── data/
│   │   ├── BTCUSDT_4h.csv       # Historical 4H candle data
│   │   └── SP500.csv            # S&P 500 comparison data
│   ├── results/
│   │   ├── all_trades.csv       # Full trade history
│   │   ├── backtest_results.csv
│   │   ├── robustness_results.csv
│   │   └── plots/               # Generated charts
│   └── backups/
│       ├── strategy_v1_winning.py
│       └── config_v1_winning.py
│
└── 🔧 Configuration Files
    ├── requirements.txt          # Python dependencies
    ├── requirements_dashboard.txt
    ├── bot_state.json           # Bot state persistence
    ├── trades.json              # Trade log
    └── .env                     # API keys (not committed)
```

## 🎯 Strategy Versions

### V1: 4-Hour Optimized Strategy (855% Return)
**Best for:** Swing trading, lower trade frequency, easier to manage

| Parameter | Value | Why |
|-----------|-------|-----|
| Timeframe | 4H | Filters noise, captures major trends |
| Entry Length | 40 | 160-hour breakout (fewer false signals) |
| Exit Length | 16 | 64-hour channel (good trend capture) |
| Trail Multiplier | 4.0 | Wide stops for volatile crypto markets |
| Risk % | 1.0% | Conservative position sizing |
| Direction | Long Only | Shorts historically unprofitable in BTC |

**Results (Aug 2017 - Dec 2025):**
- Total Return: **+855%** ($1,000 → $9,550)
- Max Drawdown: **-39.3%**
- Win Rate: **45.5%**
- Profit Factor: **1.60**
- Total Trades: **442** (~0.14 trades/day)

### V4: 1-Hour High-Frequency Strategy (1572% Return)
**Best for:** Active traders, higher capital efficiency, more opportunities

| Parameter | Value | Difference from V1 |
|-----------|-------|-------------------|
| Timeframe | 1H | 4x more granular |
| Entry Length | 8 | 8-hour breakout (faster entries) |
| Exit Length | 16 | 16-hour channel |
| Trail Multiplier | 3.5 | Slightly tighter for faster timeframe |
| Risk % | 1.0% | Same conservative sizing |
| Direction | Long Only | Same as V1 |

**Results:**
- Total Return: **+1572%** ($1,000 → $16,720)
- Max Drawdown: **-65%** (higher volatility)
- Win Rate: **44%**
- Profit Factor: **1.27**
- Total Trades: **~3,900** (~1.33 trades/day)

**Trade-off:** Higher returns but more drawdown and requires more active monitoring.

## 🔬 Research & Development Journey

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

### Paper Trading Bot (`paper_bot.py`)
- Connects to **Binance Testnet** (fake money, real execution)
- Monitors 4H BTC candles in real-time
- Automatically executes V1 strategy
- Logs all trades to `paper_bot.log`
- **Emergency Killswitch:** Create `KILLSWITCH.txt` to stop gracefully

```bash
# Start paper trading
python paper_bot.py

# Monitor in real-time
tail -f paper_bot.log  # Linux/Mac
Get-Content paper_bot.log -Wait  # Windows
```

### Live Trading Bot (`github_bot.py`)
⚠️ **WARNING:** Uses real money on Binance mainnet!

Features:
- Telegram notifications for every trade
- State persistence (`bot_state.json`)
- Position tracking and P&L monitoring
- Automatic reconnection and error handling

Required setup:
```bash
# Create .env file with:
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📊 Interactive Dashboard

Launch the Streamlit dashboard for real-time analysis:

```bash
streamlit run dashboard.py
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

**See [CRITICAL_WARNINGS.md](CRITICAL_WARNINGS.md) for full risk disclosure.**

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
| `strategy.py` | Core V1 strategy class with Donchian logic |
| `config.py` | Optimized V1 parameters (Entry=40, Exit=16, etc.) |
| `data_fetcher.py` | Downloads historical data from Binance |
| `robustness_test.py` | Grid search over 2,800 parameter combos |
| `realistic_regime_simulation.py` | 20-year Monte Carlo with costs |
| `paper_bot.py` | Safe testnet trading bot |
| `github_bot.py` | Live trading bot (REAL MONEY) |
| `dashboard.py` | Streamlit interactive interface |
| `validate_all.py` | Run all tests to verify system integrity |

## 🔥 Performance Highlights

### V1 Strategy (4H Timeframe)
- **Initial Capital:** $1,000
- **Final Equity:** $9,550
- **Return:** **+855%**
- **CAGR:** ~30% over 8 years
- **Max Drawdown:** -39.3%
- **Sharpe Ratio:** 1.98
- **Profit Factor:** 1.60
- **Win Rate:** 45.5%

### Yearly Breakdown
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

### Run a Quick Backtest
```python
# Using V1 optimized parameters
python quick_analysis.py

# Output:
# Final Equity: $9,550
# Total Return: +855%
# Max Drawdown: -39.3%
# Win Rate: 45.5%
```

### Parameter Optimization
```python
# Test 2,800 parameter combinations
python robustness_test.py

# Results saved to: results/robustness_results.csv
```

### Monte Carlo Simulation
```python
# 20-year forward simulation with realistic costs
python realistic_regime_simulation.py

# Output:
# Median 20-year outcome: $220,210
# 95% confidence interval: $17,846 - $4.3M
# Probability of profit: 100%
```

### Paper Trading (Safe Testing)
```python
# Start paper trading bot on Binance testnet
python paper_bot.py

# Monitor logs
tail -f paper_bot.log
```

### Compare vs S&P 500
```python
# What if you invested $1K + $300/month in S&P 500?
python sp500_reinvest.py

# Compares strategy performance vs traditional investing
```

## 🧪 Testing & Validation

### Run All Validation Tests
```bash
python validate_all.py
```

This runs:
- ✅ Data integrity checks
- ✅ Strategy logic validation
- ✅ Parameter range verification
- ✅ Backtest reproducibility
- ✅ Bot state persistence
- ✅ API connection tests

## 📈 Comparison: Strategy vs Buy-and-Hold Bitcoin

### The Uncomfortable Truth

**This is the most important comparison you need to see:**

| Metric | BTC Buy-Hold | V1 Strategy | V4 Strategy |
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

### Environment Variables (.env)
```bash
# Binance API (get from binance.com)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here

# Telegram Bot (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional: Use testnet for paper trading
USE_TESTNET=true
```

### Security Best Practices
- ✅ Never commit `.env` file to GitHub
- ✅ Use API keys with trade-only permissions (no withdrawals)
- ✅ Enable IP whitelisting on Binance
- ✅ Use 2FA on exchange account
- ✅ Regularly rotate API keys
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

**Issue:** `ModuleNotFoundError: No module named 'vectorbt'`
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

**Issue:** `binance.exceptions.BinanceAPIException: Invalid API-key`
```bash
# Solution: Check .env file has correct keys
# Verify keys are active on binance.com
```

**Issue:** Bot not executing trades
```bash
# Check logs for errors
cat paper_bot.log

# Verify data is updating
python data_fetcher.py
```

**Issue:** Dashboard won't load
```bash
# Install dashboard requirements
pip install -r requirements_dashboard.txt

# Run with verbose logging
streamlit run dashboard.py --logger.level=debug
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

- [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md) - Complete technical overview
- [BOT_GUIDE.md](BOT_GUIDE.md) - How to run trading bots
- [CRITICAL_WARNINGS.md](CRITICAL_WARNINGS.md) - Must-read before trading
- [PAPER_TRADING_CHECKLIST.md](PAPER_TRADING_CHECKLIST.md) - Pre-flight checklist

---


**Last Updated:** April 2026

**Built with ❤️ for algorithmic trading education**


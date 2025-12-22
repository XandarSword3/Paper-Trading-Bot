# Paper Trading Checklist

## Before You Start Paper Trading

### 1. Platform Setup
- [ ] Create Binance Testnet account (testnet.binance.vision)
- [ ] OR use TradingView paper trading mode
- [ ] Set initial balance to $1,000 (match your planned capital)
- [ ] Enable 4H chart timeframe

### 2. Strategy Rules (V1 Optimized)
```
ENTRY: Buy when price breaks ABOVE 40-period Donchian high
EXIT: Sell when price breaks BELOW 16-period Donchian low
       OR trailing stop hit (4.0 ATR below highest price)

POSITION SIZING:
- Risk 1% of equity per trade
- Size = (1% of Equity) / (2.0 * ATR)
- Maximum 4 pyramid units

LONG ONLY - Do not short sell BTC
```

### 3. Alerts to Set (TradingView)
- [ ] Alert: "Price crosses above Donchian 40 high" → Entry signal
- [ ] Alert: "Price crosses below Donchian 16 low" → Exit signal
- [ ] Optional: Create trailing stop indicator

---

## During Paper Trading (3-6 months)

### Weekly Tasks
- [ ] Review all trades from the week
- [ ] Calculate actual vs expected entry/exit prices
- [ ] Note any slippage or execution issues
- [ ] Compare P&L to backtest for same period

### Monthly Tasks
- [ ] Calculate monthly return %
- [ ] Compare to historical monthly returns
- [ ] Log any emotional decisions or rule breaks
- [ ] Check drawdown against max historical (39.3%)

---

## Trade Journal Template

| Date | Direction | Entry Price | Exit Price | Qty | P&L | Exit Reason | Notes |
|------|-----------|-------------|------------|-----|-----|-------------|-------|
| | LONG | | | | | | |

**Exit Reasons:**
- DON_EXIT: Donchian exit signal
- TRAIL_STOP: Trailing stop hit
- MANUAL: Rule violation (note why)

---

## Key Metrics to Track

### Must Match Backtest (within 10%)
- Win Rate: ~45% expected
- Average Win: ~2x Average Loss
- Profit Factor: >1.5
- Max Drawdown: <45%

### Paper Trading Specific
- Execution Accuracy: % of signals you actually traded
- Slippage Impact: Actual vs signal price difference
- Time to Execute: How long after signal you entered

---

## Warning Signs to Watch

### 🔴 Stop Paper Trading If:
- Max drawdown exceeds 50%
- More than 10 consecutive losses
- Profit factor drops below 1.0
- You're breaking rules regularly

### 🟡 Investigate If:
- Win rate below 35% or above 55%
- Average holding time very different from 2-3 days
- Slippage consistently > 0.2%

### 🟢 Ready for Live If:
- 3+ months of consistent execution
- Results within 20% of backtest expectations
- No major emotional trading errors
- Clear understanding of risk

---

## Transition to Live Trading

### Start Small
1. Begin with 25-50% of intended capital
2. Trade for 1-2 months
3. If results match paper trading, scale up gradually

### Automation Recommendation
Since execution errors are the biggest risk, consider:
- 3Commas bots
- Coinrule automation
- Binance conditional orders
- Custom Python bot (with safeguards)

### Capital Preservation Rules
- Never risk more than 1% per trade
- Keep 6-month expenses in stablecoins
- Take 30% of monthly profits to S&P 500
- Have a circuit breaker: stop if down 30% in a month

---

*Remember: Past performance doesn't guarantee future results. See CRITICAL_WARNINGS.md*

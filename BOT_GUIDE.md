# Paper Trading Bot - Quick Start Guide

## What It Does
- Monitors BTC/USDT 4H candles in real-time
- Executes V1 Turtle-Donchian strategy automatically
- Places real orders on Binance **TESTNET** (fake money)
- Logs all trades and performance to `paper_bot.log`

## Starting Capital
- $1,000 USDT (simulated)
- Uses testnet funds (no real money risk)

## How to Run

### Start the Bot
```powershell
.venv\Scripts\python.exe paper_bot.py
```

### Stop the Bot
- **Option 1:** Press `Ctrl+C` (graceful shutdown, closes positions)
- **Option 2:** Create a file named `KILLSWITCH.txt` in the project folder

### Monitor Progress
- Watch the terminal output for real-time updates
- Check `paper_bot.log` for full trade history

## What to Expect

### Checking Interval
- Bot checks every **1 hour**
- 4H candles update every 4 hours
- Will execute trades when signals trigger

### Typical Output
```
======================================================================
🤖 PAPER TRADING BOT STARTED
======================================================================
Initial Capital: $1,000.00
Check Interval: 3600s (1.0h)
Testnet: True
Killswitch: Create 'KILLSWITCH.txt' to stop
======================================================================

======================================================================
Checking signals - 2025-12-22 15:30:00
BTC Price: $89,416.32
ATR: $2,145.50
Entry High: $90,500.00
Exit Low: $85,200.00
Current Equity: $1,000.00
Position: 0.00000 BTC (0 units)
Total Trades: 0
======================================================================
Sleeping 3600s until next check...
```

### When a Trade Triggers
```
🟢 LONG ENTRY #1
   Price: $89,500.00
   Quantity: 0.01234 BTC
   Total Position: 0.01234 BTC
   Trailing Stop: $81,918.00

🔴 LONG EXIT - Donchian Exit
   Exit Price: $91,200.00
   Quantity: 0.01234 BTC
   P&L: $20.96 (+2.35%)
   New Equity: $1,020.96
```

## Safety Features

### Killswitch
Create a file to emergency stop:
```powershell
New-Item -Path KILLSWITCH.txt -ItemType File
```

Bot will:
1. Detect the file
2. Close all positions
3. Shut down gracefully

### Testnet Protection
- All trades use **fake money** on testnet
- No real funds at risk
- Can't withdraw or lose actual money

### Logging
- Every action logged to `paper_bot.log`
- Includes timestamps, prices, P&L
- Can review trade history anytime

## Expected Performance

Based on backtest (2017-2025):
- Win rate: ~45%
- Profit factor: ~1.6
- Avg trade: ~2-3 days
- Max drawdown: ~40%

Paper trading may differ from backtest due to:
- Real-time execution
- Market conditions
- Slippage on testnet

## Monitoring Checklist

### Daily
- [ ] Check bot is still running
- [ ] Review any new trades in log
- [ ] Verify equity is tracking correctly

### Weekly
- [ ] Calculate weekly return
- [ ] Compare to backtest expectations
- [ ] Note any execution issues

### Monthly
- [ ] Full performance review
- [ ] Update tracking spreadsheet
- [ ] Decide if ready for live trading

## Troubleshooting

### Bot Not Starting
- Check .env file has testnet keys
- Run `python test_binance_keys.py` first
- Check internet connection

### No Trades Executing
- BTC may not be in a trending phase
- Strategy waits for clear breakouts
- Check signals match TradingView chart

### API Errors
- Testnet may have temporary outages
- Check binance testnet status
- Restart bot after connection restored

## Next Steps

After 3-6 months of successful paper trading:
1. Review all trades and compare to backtest
2. If results match expectations (±20%)
3. Consider transitioning to live with small capital
4. See `PAPER_TRADING_CHECKLIST.md` for criteria

---

**Remember:** This is practice. Take it seriously but don't stress about losses - it's fake money!

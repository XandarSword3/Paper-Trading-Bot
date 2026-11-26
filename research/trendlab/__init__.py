"""TrendLab: long-history, net-of-cost research for crypto perpetual futures.

This package is deliberately independent from the live bot.  It downloads
public Binance USDT-M futures data, builds a clean panel, runs vectorised
portfolio backtests with explicit costs/funding, and reports out-of-sample
statistics with multiple-testing controls.
"""

__all__ = ["data", "engine", "strategies", "stats"]

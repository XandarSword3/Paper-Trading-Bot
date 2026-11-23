"""Regression tests for the RGMT regime gate.

The production runners compute targets from a live price panel. If the regime
reference (BTCUSDT) is missing, the gate must fail closed instead of silently
trading the ungated trend book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.strategies.trend_portfolio import TrendConfig, TrendPortfolio, target_weights


def _panel_with_btc(days: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    btc = np.linspace(50000.0, 30000.0, days)  # below its own long EMA at the end
    eth = np.linspace(3000.0, 1500.0, days)
    return pd.DataFrame({"BTCUSDT": btc, "ETHUSDT": eth}, index=dates)


def test_regime_gate_closes_when_btc_below_ma() -> None:
    panel = _panel_with_btc()
    cfg = TrendConfig(lookbacks=(60, 120), regime_enabled=True, regime_ma=200)
    portfolio = TrendPortfolio(cfg)

    assert portfolio.regime_active(panel) is False
    assert (target_weights(panel, cfg).iloc[-1] == 0.0).all()


def test_regime_gate_fails_closed_when_btc_missing() -> None:
    panel = _panel_with_btc().drop(columns=["BTCUSDT"])
    cfg = TrendConfig(lookbacks=(60, 120), regime_enabled=True)

    portfolio = TrendPortfolio(cfg)
    assert portfolio.regime_active(panel) is False

    weights = target_weights(panel, cfg)
    assert (weights == 0.0).all().all()

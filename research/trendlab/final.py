"""The frozen, final strategy definition used by the report and production code.

Do not tune these values on the holdout period.  They were selected using the
train (2020-2023) and validation (2024 - mid-2025) windows only:

* multi-horizon trend: sign of 60-day and 120-day returns, averaged;
* regime gate: BTC must be above its 200-day EMA;
* inverse-volatility sizing, 25% annualised portfolio vol target;
* daily rebalance, long and short symmetric.
"""

from __future__ import annotations

import pandas as pd

from research.strategies.trend_portfolio import TrendConfig, momentum_score

FINAL_CONFIG = TrendConfig(
    lookbacks=(60, 120),
    mode="sign",
    target_vol=0.25,
    max_gross=3.0,
    corr=0.6,
    vol_halflife=20,
    max_asset_weight=0.15,
    min_asset_vol=0.15,
    regime_enabled=True,
    regime_symbol="BTCUSDT",
    regime_ma=200,
)

NAME = "Regime-Gated Multi-Horizon Trend (RGMT)"
SHORT_NAME = "rgmt"


def regime_mask(close: pd.DataFrame, cfg: TrendConfig = FINAL_CONFIG) -> pd.Series:
    if not cfg.regime_enabled:
        return pd.Series(1.0, index=close.index)
    if cfg.regime_symbol not in close.columns:
        # Fail closed: no regime reference means no verified risk budget.
        return pd.Series(0.0, index=close.index)
    ref = close[cfg.regime_symbol]
    ma = ref.ewm(span=cfg.regime_ma, min_periods=cfg.regime_ma, adjust=False).mean()
    return (ref > ma).astype(float).fillna(0.0)


def final_signal(panel: dict[str, pd.DataFrame], cfg: TrendConfig = FINAL_CONFIG) -> pd.DataFrame:
    close = panel["close"]
    score = momentum_score(close, cfg.lookbacks, cfg.mode)
    return score.mul(regime_mask(close, cfg), axis=0)

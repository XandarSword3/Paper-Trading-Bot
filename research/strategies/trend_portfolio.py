"""Multi-asset trend-following portfolio for USDT perpetual futures.

This is the production implementation of the strategy validated in
``research/trendlab``.  It is deliberately pure: it takes a close-price panel
and returns target portfolio weights.  Execution lives in
``scripts/run_trend_portfolio.py``.

Strategy in one paragraph
------------------------
For each liquid perpetual, compute the sign of trailing returns over several
horizons (2, 3 and 6 months by default) and average them into a trend score in
[-1, 1].  Size every position inversely to its recent realised volatility so
each contributes similar risk, then scale the whole book so that its ex-ante
annualised volatility equals ``target_vol``.  Rebalance once per day at the UTC
close.  Long and short are symmetric; funding and fees are charged explicitly
in research.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIODS = 365


@dataclass
class TrendConfig:
    lookbacks: tuple[int, ...] = (60, 120)
    mode: str = "sign"                 # "sign" or "tanh"
    target_vol: float = 0.25           # annualised portfolio volatility target
    max_gross: float = 3.0             # hard leverage cap
    corr: float = 0.6                  # constant cross-asset correlation
    vol_halflife: int = 20             # EWMA halflife (days)
    min_vol_obs: int = 20
    max_asset_weight: float = 0.15     # per-name cap as fraction of equity
    min_asset_vol: float = 0.0         # floor on annualised vol used for sizing
    regime_enabled: bool = True        # BTC long-term trend gate
    regime_symbol: str = "BTCUSDT"
    regime_ma: int = 200
    dd_throttle: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # dd_throttle: ((0.15, 0.5), (0.25, 0.0)) means halve risk at 15% DD, flat at 25%.

    def __post_init__(self) -> None:
        self.lookbacks = tuple(int(x) for x in self.lookbacks)
        if not self.lookbacks:
            raise ValueError("at least one lookback is required")
        if self.mode not in {"sign", "tanh"}:
            raise ValueError("mode must be 'sign' or 'tanh'")


def _pct(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def momentum_score(close: pd.DataFrame, lookbacks: tuple[int, ...], mode: str = "sign") -> pd.DataFrame:
    """Average multi-horizon momentum into a score in [-1, 1]."""
    scores = []
    for lb in lookbacks:
        past = close / close.shift(lb) - 1.0
        if mode == "sign":
            s = np.sign(past)
        else:
            vol = _pct(close).rolling(max(lb // 2, 20), min_periods=20).std()
            z = past / (vol * np.sqrt(lb))
            s = np.tanh(z / 1.5)
        scores.append(s.where(past.notna()))
    out = sum(scores) / float(len(scores))
    return out.clip(-1.0, 1.0)


def realised_vol(close: pd.DataFrame, halflife: int = 20, min_obs: int = 20) -> pd.DataFrame:
    vol = _pct(close).ewm(halflife=halflife, min_periods=min_obs).std() * np.sqrt(PERIODS)
    return vol.where(vol > 0)


def ex_ante_portfolio_vol(unit: pd.DataFrame, sigma: pd.DataFrame, corr: float) -> pd.Series:
    s = sigma.reindex_like(unit).fillna(0.0)
    u = unit.fillna(0.0)
    weighted = u * s
    diag = (weighted**2).sum(axis=1)
    total = weighted.sum(axis=1)
    var = diag + corr * (total**2 - diag)
    return np.sqrt(var.clip(lower=0.0))


def target_weights(
    close: pd.DataFrame,
    cfg: TrendConfig | None = None,
    equity_curve: pd.Series | None = None,
) -> pd.DataFrame:
    """Return target weights (fraction of equity) for every close-date."""
    cfg = cfg or TrendConfig()
    score = momentum_score(close, cfg.lookbacks, cfg.mode)

    if cfg.regime_enabled and cfg.regime_symbol in close.columns:
        ref = close[cfg.regime_symbol]
        regime_ma = ref.ewm(span=cfg.regime_ma, min_periods=cfg.regime_ma, adjust=False).mean()
        regime = (ref > regime_ma).astype(float).fillna(0.0)
        score = score.mul(regime, axis=0)

    sigma = realised_vol(close, cfg.vol_halflife, cfg.min_vol_obs)
    if cfg.min_asset_vol > 0:
        sigma = sigma.clip(lower=cfg.min_asset_vol)
    unit = (score / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ex_vol = ex_ante_portfolio_vol(unit, sigma, cfg.corr)
    scale = (cfg.target_vol / ex_vol.replace(0.0, np.nan)).fillna(0.0)
    raw = unit.mul(scale, axis=0)

    gross = raw.abs().sum(axis=1)
    cap = np.minimum(1.0, cfg.max_gross / gross.replace(0.0, np.nan)).fillna(0.0)
    w = raw.mul(cap, axis=0)
    w = w.clip(-cfg.max_asset_weight, cfg.max_asset_weight)

    if cfg.dd_throttle and equity_curve is not None:
        eq = equity_curve.reindex(w.index).ffill()
        peak = eq.cummax()
        dd = (eq / peak - 1.0).abs()
        mult = pd.Series(1.0, index=w.index)
        for level, factor in cfg.dd_throttle:
            mult = mult.where(dd < level, np.minimum(mult, factor))
        w = w.mul(mult, axis=0)

    return w.where(close.notna(), 0.0).fillna(0.0)


def latest_targets(close: pd.DataFrame, cfg: TrendConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper returning only the most recent target row as a frame."""
    w = target_weights(close, cfg)
    return w.tail(1).T.rename(columns={w.index[-1]: "target_weight"})


class TrendPortfolio:
    """Stateful wrapper used by the execution script."""

    def __init__(self, cfg: TrendConfig | None = None) -> None:
        self.cfg = cfg or TrendConfig()

    def regime_active(self, close: pd.DataFrame) -> bool:
        if not self.cfg.regime_enabled or self.cfg.regime_symbol not in close.columns:
            return True
        ref = close[self.cfg.regime_symbol]
        ma = ref.ewm(span=self.cfg.regime_ma, min_periods=self.cfg.regime_ma, adjust=False).mean()
        val = (ref > ma).iloc[-1]
        return bool(val) if pd.notna(val) else False

    def signals(self, close: pd.DataFrame) -> pd.DataFrame:
        return momentum_score(close, self.cfg.lookbacks, self.cfg.mode)

    def weights(self, close: pd.DataFrame, equity_curve: pd.Series | None = None) -> pd.DataFrame:
        return target_weights(close, self.cfg, equity_curve)

    def describe(self, close: pd.DataFrame) -> pd.DataFrame:
        score = self.signals(close).iloc[-1]
        sigma = realised_vol(close, self.cfg.vol_halflife, self.cfg.min_vol_obs).iloc[-1]
        w = self.weights(close).iloc[-1]
        out = pd.DataFrame({"trend_score": score, "ann_vol": sigma, "target_weight": w})
        out["notional_at_100k"] = out["target_weight"] * 100_000
        return out.sort_values("target_weight", ascending=False)

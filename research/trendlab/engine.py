"""Vectorised daily portfolio backtester for perpetual futures.

Design choices that matter for honesty:

* Signals are computed from a bar's close and the resulting position earns the
  *next* bar's close-to-close return (``signal.shift(1) * ret``), so there is no
  same-bar look-ahead.
* Transaction costs are charged on realised turnover at ``cost_bps`` per unit
  of notional traded (one-way).
* Perpetual funding is charged using the historical Binance funding series:
  a long position pays positive funding, a short receives it.
* Position sizing is volatility-targeted with a constant-correlation
  approximation (crypto cross-correlations are high and stable enough that a
  noisy full covariance matrix hurts more than it helps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIODS = 365


@dataclass
class BacktestConfig:
    target_vol: float = 0.25
    max_gross: float = 3.0
    cost_bps: float = 8.0          # one-way, per unit turnover
    corr: float = 0.6              # constant cross-asset correlation
    vol_halflife: int = 20         # EWMA halflife for per-asset vol (days)
    min_vol_obs: int = 20
    max_asset_weight: float = 0.15  # per-name cap, fraction of equity
    min_asset_vol: float = 0.0      # floor on annualised vol used for sizing
    funding_multiplier: float = 1.0
    borrow_bps_per_day: float = 0.0  # extra cost for short notional (none on perps)


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    gross: pd.Series
    turnover: pd.Series
    costs: pd.Series
    funding: pd.Series
    cfg: BacktestConfig
    meta: dict = field(default_factory=dict)

    @property
    def net_returns(self) -> pd.Series:
        return self.returns


def funding_daily(funding: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Aggregate irregular funding prints into the daily grid (sum per day)."""
    if funding is None or funding.empty:
        return pd.Series(0.0, index=index)
    daily = funding.resample("1D").sum()
    daily.index = daily.index.tz_convert(index.tz) if index.tz is not None else daily.index
    return daily.reindex(index).fillna(0.0)


def _ex_ante_vol(unit: pd.DataFrame, sigma: pd.DataFrame, corr: float) -> pd.Series:
    """Portfolio vol of raw weights ``unit`` given per-asset vols ``sigma``."""
    s = sigma.reindex_like(unit).fillna(0.0)
    u = unit.fillna(0.0)
    weighted = u * s
    var = (weighted**2).sum(axis=1)
    total = weighted.sum(axis=1)
    # sum_{i != j} w_i w_j s_i s_j = total^2 - sum(w_i^2 s_i^2)
    var = var + corr * (total**2 - var)
    return np.sqrt(var.clip(lower=0.0))


def run_backtest(
    panel: dict[str, pd.DataFrame],
    signal: pd.DataFrame,
    funding: dict[str, pd.Series] | None = None,
    cfg: BacktestConfig | None = None,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    close = panel["close"]
    idx = close.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start, tz=idx.tz)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end, tz=idx.tz)]

    close = close.loc[idx]
    ret = close.pct_change(fill_method=None)
    ret = ret.where(close.notna())

    sig = signal.reindex(index=idx, columns=close.columns)
    # A signal for an asset with no price is meaningless.
    sig = sig.where(close.notna())

    sigma = (
        ret.ewm(halflife=cfg.vol_halflife, min_periods=cfg.min_vol_obs).std()
        * np.sqrt(PERIODS)
    )
    sigma = sigma.where(sigma > 0)
    if cfg.min_asset_vol > 0:
        sigma = sigma.clip(lower=cfg.min_asset_vol)

    unit = sig / sigma
    unit = unit.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ex_vol = _ex_ante_vol(unit, sigma, cfg.corr)
    scale = (cfg.target_vol / ex_vol.replace(0.0, np.nan)).fillna(0.0)
    raw_w = unit.mul(scale, axis=0)
    gross_raw = raw_w.abs().sum(axis=1)
    # Hard leverage cap, applied proportionally.
    cap = np.minimum(1.0, cfg.max_gross / gross_raw.replace(0.0, np.nan)).fillna(0.0)
    w = raw_w.mul(cap, axis=0)
    if cfg.max_asset_weight > 0:
        w = w.clip(-cfg.max_asset_weight, cfg.max_asset_weight)

    # ---- PnL ------------------------------------------------------------
    w_held = w.shift(1).fillna(0.0)
    gross_ret = (w_held * ret.fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * cfg.cost_bps / 1e4

    funding_cost = pd.Series(0.0, index=idx)
    if funding:
        for sym in w.columns:
            if sym not in funding:
                continue
            f = funding_daily(funding[sym], idx)
            funding_cost += (w_held[sym].fillna(0.0) * f) * cfg.funding_multiplier

    short_notional = w_held.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * cfg.borrow_bps_per_day / 1e4

    net = gross_ret - cost - funding_cost - borrow
    net = net.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        returns=net,
        equity=equity,
        weights=w,
        gross=w.abs().sum(axis=1),
        turnover=turnover,
        costs=cost,
        funding=funding_cost,
        cfg=cfg,
        meta={
            "gross_ret": gross_ret,
            "short_notional": short_notional,
            "signal": sig,
        },
    )


def buy_and_hold(
    panel: dict[str, pd.DataFrame],
    symbols: list[str] | None = None,
    cfg: BacktestConfig | None = None,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    close = panel["close"]
    syms = symbols or list(close.columns)
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig[syms] = 1.0
    return run_backtest(panel, sig, cfg=cfg, start=start, end=end)

"""Signal generators.

Every function returns a DataFrame of target *risk units* on the daily grid:

* columns = symbols, index = decision times (bar close),
* values are in ``[-1, 1]`` (0 = flat), interpreted by the engine as a
  directional position before volatility scaling,
* all calculations use only information available at the bar close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import funding_daily


def _pct(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def _vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return _pct(close).rolling(window, min_periods=max(window // 2, 10)).std()


def ema(s: pd.DataFrame, span: int) -> pd.DataFrame:
    return s.ewm(span=span, min_periods=span, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Time-series trend
# --------------------------------------------------------------------------- #
def tsmom_sign(panel: dict[str, pd.DataFrame], lookback: int = 90) -> pd.DataFrame:
    """Classic time-series momentum: sign of the trailing return."""
    close = panel["close"]
    past = close / close.shift(lookback) - 1.0
    sig = np.sign(past)
    return sig.where(past.notna())


def tsmom_tanh(panel: dict[str, pd.DataFrame], lookback: int = 90) -> pd.DataFrame:
    """Volatility-normalised momentum, squashed to [-1, 1]."""
    close = panel["close"]
    past = close / close.shift(lookback) - 1.0
    vol = _vol(close, max(lookback // 2, 20))
    z = past / (vol * np.sqrt(lookback))
    sig = np.tanh(z / 1.5)
    return sig.where(past.notna())


def ma_cross(panel: dict[str, pd.DataFrame], fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long when the fast EMA is above the slow EMA, short otherwise."""
    close = panel["close"]
    f = ema(close, fast)
    s = ema(close, slow)
    raw = (f - s) / close
    sig = np.sign(raw)
    return sig.where(s.notna() & f.notna())


def ma_slope(panel: dict[str, pd.DataFrame], span: int = 100, slope_window: int = 20) -> pd.DataFrame:
    """Long when the trend MA is rising, short when falling."""
    close = panel["close"]
    m = ema(close, span)
    slope = m - m.shift(slope_window)
    return np.sign(slope).where(m.notna())


def donchian(
    panel: dict[str, pd.DataFrame],
    entry: int = 55,
    exit_n: int = 20,
    long_only: bool = False,
) -> pd.DataFrame:
    """Stateful Donchian breakout with a trailing exit channel."""
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    up = high.rolling(entry, min_periods=entry).max().shift(1)
    dn = low.rolling(entry, min_periods=entry).min().shift(1)
    exit_lo = low.rolling(exit_n, min_periods=exit_n).min().shift(1)
    exit_hi = high.rolling(exit_n, min_periods=exit_n).max().shift(1)

    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        u = up[sym].to_numpy(dtype=float)
        d = dn[sym].to_numpy(dtype=float)
        xl = exit_lo[sym].to_numpy(dtype=float)
        xh = exit_hi[sym].to_numpy(dtype=float)
        state = 0
        out = np.zeros(len(c))
        for i in range(len(c)):
            if np.isnan(c[i]) or np.isnan(u[i]) or np.isnan(d[i]):
                out[i] = 0.0
                continue
            if state == 0:
                if c[i] > u[i]:
                    state = 1
                elif c[i] < d[i] and not long_only:
                    state = -1
            elif state == 1:
                if c[i] < xl[i]:
                    state = 0
            elif state == -1:
                if c[i] > xh[i]:
                    state = 0
            out[i] = state
        sig[sym] = out
    return sig


# --------------------------------------------------------------------------- #
# Cross-sectional momentum
# --------------------------------------------------------------------------- #
def xsmom(
    panel: dict[str, pd.DataFrame],
    lookback: int = 90,
    skip: int = 0,
    quantile: float = 0.3,
    vol_adj: bool = True,
    rebalance: int = 7,
    long_short: bool = True,
) -> pd.DataFrame:
    close = panel["close"]
    past = close.shift(skip) / close.shift(lookback + skip) - 1.0
    if vol_adj:
        past = past / (_vol(close, lookback) * np.sqrt(lookback))
    ranks = past.rank(axis=1, pct=True)
    n_active = past.notna().sum(axis=1)
    # Require a minimum cross-section before trading.
    valid = n_active >= 6
    long_mask = ranks >= (1.0 - quantile)
    short_mask = (ranks <= quantile) if long_short else pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)
    w = long_mask.astype(float).div(n_long, axis=0)
    if long_short:
        w = w - short_mask.astype(float).div(n_short, axis=0)
    w = w.where(valid, 0.0).fillna(0.0)
    # Hold between rebalances.
    if rebalance > 1:
        keep = pd.Series(False, index=w.index)
        keep.iloc[::rebalance] = True
        w = w.where(keep, np.nan).ffill().fillna(0.0)
    return w.clip(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Short-horizon mean reversion
# --------------------------------------------------------------------------- #
def meanrev(
    panel: dict[str, pd.DataFrame],
    lookback: int = 5,
    z_entry: float = 2.0,
    max_hold: int = 5,
    trend_lookback: int = 200,
    long_only: bool = False,
) -> pd.DataFrame:
    close = panel["close"]
    mean = close.rolling(lookback, min_periods=lookback).mean()
    std = close.rolling(lookback, min_periods=lookback).std()
    z = (close - mean) / std
    trend = close > close.rolling(trend_lookback, min_periods=trend_lookback).mean()

    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for sym in close.columns:
        zz = z[sym].to_numpy(dtype=float)
        tt = trend[sym].to_numpy(dtype=float)
        out = np.zeros(len(zz))
        state = 0
        hold = 0
        for i in range(len(zz)):
            if np.isnan(zz[i]):
                state, hold = 0, 0
                out[i] = 0.0
                continue
            if state == 0:
                if zz[i] < -z_entry and (tt[i] == 1 or np.isnan(tt[i])):
                    state, hold = 1, 0
                elif zz[i] > z_entry and not long_only and tt[i] == 0:
                    state, hold = -1, 0
            else:
                hold += 1
                exit_long = state == 1 and (zz[i] >= 0 or hold >= max_hold)
                exit_short = state == -1 and (zz[i] <= 0 or hold >= max_hold)
                if exit_long or exit_short:
                    state, hold = 0, 0
            out[i] = state
        sig[sym] = out
    return sig


# --------------------------------------------------------------------------- #
# Funding carry (cross-sectional, perp-only approximation)
# --------------------------------------------------------------------------- #
def funding_carry(
    panel: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
    lookback: int = 3,
    quantile: float = 0.3,
    rebalance: int = 3,
    long_short: bool = True,
) -> pd.DataFrame:
    close = panel["close"]
    idx = close.index
    cols = {}
    for sym in close.columns:
        f = funding.get(sym)
        if f is None or f.empty:
            cols[sym] = pd.Series(0.0, index=idx)
        else:
            cols[sym] = funding_daily(f, idx)
    fmat = pd.DataFrame(cols)[close.columns]
    score = fmat.rolling(lookback, min_periods=1).mean()
    ranks = score.rank(axis=1, pct=True)
    long_mask = ranks <= quantile          # low/negative funding -> long
    short_mask = ranks >= (1 - quantile)   # high funding -> short
    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)
    w = long_mask.astype(float).div(n_long, axis=0)
    if long_short:
        w = w - short_mask.astype(float).div(n_short, axis=0)
    w = w.where(close.notna(), 0.0).fillna(0.0)
    if rebalance > 1:
        keep = pd.Series(False, index=w.index)
        keep.iloc[::rebalance] = True
        w = w.where(keep, np.nan).ffill().fillna(0.0)
    return w.clip(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Combinations / masks
# --------------------------------------------------------------------------- #
def ensemble(
    panel: dict[str, pd.DataFrame],
    lookbacks: list[int],
    mode: str = "sign",
) -> pd.DataFrame:
    """Average several momentum horizons (multi-scale trend following)."""
    fn = tsmom_sign if mode == "sign" else tsmom_tanh
    sigs = [fn(panel, lb) for lb in lookbacks]
    out = sum(sigs) / float(len(sigs))
    return out.clip(-1.0, 1.0)


def rebalance(sig: pd.DataFrame, every: int) -> pd.DataFrame:
    """Hold the signal for ``every`` bars, refreshing at the sampled dates."""
    if every <= 1:
        return sig
    keep = pd.Series(False, index=sig.index)
    keep.iloc[::every] = True
    return sig.where(keep, np.nan).ffill().fillna(0.0)


def long_only(sig: pd.DataFrame) -> pd.DataFrame:
    return sig.clip(lower=0.0)


def mask_by_regime(
    sig: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    symbol: str = "BTCUSDT",
    span: int = 200,
) -> pd.DataFrame:
    """Turn the book off when BTC is below its long MA (risk-off filter)."""
    close = panel["close"]
    if symbol not in close:
        return sig
    regime = (close[symbol] > ema(close[[symbol]], span)[symbol]).astype(float)
    return sig.mul(regime, axis=0)


def inverse_vol_weight(sig: pd.DataFrame) -> pd.DataFrame:
    """Normalise cross-sectional gross exposure to 1 while preserving signs."""
    gross = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(gross, axis=0).fillna(0.0)

"""FROZEN strategy definitions — S1..S6.

These rules were fixed BEFORE any backtest was run. Do not tune.
All signals are computed on daily bar CLOSES and are executed at the NEXT
day's OPEN (no look-ahead). All strategies are long/flat or delta-neutral;
no leverage in the base run.

Universe: BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, DOT, LTC, TRX
(current top-liquidity OKX spot majors — NOT point-in-time, see README).

Choices that the spec left open (fixed here, pre-results):
- Evaluation timeframe: DAILY bars for all strategies (4h data is downloaded
  per spec but the strategy rules are daily-scale).
- S3 Donchian 55 is applied to BTC (single-asset trend, parallel to S1).
- S4 universe = the 10 alt majors (excludes BTC and ETH), lookback = 90 days
  (consistent with S2), hold top 3 equal-weight, rebalance every 7 days.
- S2 vol target = 50% annualized, per-asset weight = min(1, target/realized_vol),
  realized vol = 30-day std of daily log returns * sqrt(365), long/flat;
  gross exposure capped at 100% of equity.
- S5 = BTC spot long + BTC-USDT-SWAP perp short, constant coin quantity,
  held for the full available funding-history window.
- S6 = buy-and-hold BTC.
"""
import numpy as np
import pandas as pd

DATA = __import__("pathlib").Path(__file__).parent / "data"

FEE_PER_SIDE = 0.0010   # 0.10% taker
SLIP_PER_SIDE = 0.0005  # 0.05% slippage
COST_PER_SIDE = FEE_PER_SIDE + SLIP_PER_SIDE  # 0.15%
MIN_NOTIONAL_USD = 1.0  # conservative floor; OKX spot minimums are ~cents
START_EQUITY = 1000.0

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX"]
ALTS = [s for s in SYMBOLS if s not in ("BTC", "ETH")]


def load_bars(symbol, tf="1D"):
    # Coinbase product ids are USD pairs (BTC-USD etc.)
    df = pd.read_csv(DATA / f"{symbol}-USD_{tf}.csv", parse_dates=["dt"])
    df = df.set_index("dt").sort_index()
    return df


def load_panel(symbols, tf="1D"):
    """Return dict symbol -> DataFrame with open/close, plus union date index."""
    out = {}
    for s in symbols:
        try:
            out[s] = load_bars(s, tf)
        except FileNotFoundError:
            pass
    return out


# ---------------------------------------------------------------- signals
def s1_weights(btc):
    """BTC 50/200-day MA trend, long/flat. Signal at close t -> weight for t."""
    close = btc["close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    w = (close > ma50).astype(float) * (ma50 > ma200).astype(float)
    # long only when price > MA50 AND MA50 > MA200 (classic trend filter)
    return w.to_frame("BTC")


def s2_weights(bars):
    """TS momentum 90d, BTC+ETH, vol-targeted, long/flat."""
    target_vol = 0.50
    cols = {}
    for s in ("BTC", "ETH"):
        if s not in bars:
            continue
        close = bars[s]["close"]
        mom = close / close.shift(90) - 1.0
        vol = close.pct_change().rolling(30).std() * np.sqrt(365)
        w = np.where(mom > 0, np.minimum(1.0, target_vol / vol.replace(0, np.nan)), 0.0)
        cols[s] = pd.Series(w, index=close.index).fillna(0.0)
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    # cap gross exposure at 100% of equity (no hidden leverage): if the two
    # vol-targeted weights sum above 1, scale both down proportionally.
    gross = df.sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
    return df.mul(scale, axis=0)


def s3_weights(btc):
    """Donchian 55-day breakout on BTC, long/flat (classic channel rule).

    Flat until close > highest HIGH of the prior 55 bars (excluding today) -> long.
    Long until close < lowest LOW of the prior 55 bars -> flat.
    State machine evaluated on daily closes; executed next open by the engine.
    """
    high = btc["high"]
    low = btc["low"]
    close = btc["close"]
    upper = high.shift(1).rolling(55).max()
    lower = low.shift(1).rolling(55).min()
    w = pd.Series(np.nan, index=close.index)
    state = 0.0
    for i, c in enumerate(close):
        u, l = upper.iloc[i], lower.iloc[i]
        if np.isnan(u) or np.isnan(l):
            w.iloc[i] = state
            continue
        if state == 0.0 and c > u:
            state = 1.0
        elif state == 1.0 and c < l:
            state = 0.0
        w.iloc[i] = state
    return w.fillna(0.0).to_frame("BTC")


def s4_weights(bars, alts=ALTS, lookback=90, top_n=3, rebalance_days=7):
    """Cross-sectional momentum: weekly rebalance, top 3 of 10 alts, equal weight.

    Momentum = close/close.shift(lookback)-1 computed at rebalance dates only;
    between rebalances weights are held (engine rebalances anyway only when the
    target weight series changes — we emit weights only on rebalance dates and
    forward-fill).
    """
    rets = {}
    for s in alts:
        if s in bars:
            c = bars[s]["close"]
            rets[s] = c / c.shift(lookback) - 1.0
    if not rets:
        return pd.DataFrame()
    mom = pd.DataFrame(rets)
    # candidate rebalance dates: every 7th calendar day of the union index
    idx = mom.index
    rebal_dates = idx[(idx.dayofyear % rebalance_days) == 0]
    weights = pd.DataFrame(0.0, index=idx, columns=alts)
    current = pd.Series(0.0, index=alts)
    ri = 0
    for d in idx:
        if ri < len(rebal_dates) and d == rebal_dates[ri]:
            row = mom.loc[d]
            valid = row.dropna()
            if len(valid) >= top_n:
                top = valid.nlargest(top_n).index
                current = pd.Series(0.0, index=alts)
                current[top] = 1.0 / top_n
            ri += 1
        weights.loc[d] = current
    return weights


def s6_weights(btc):
    """Buy-and-hold BTC."""
    return pd.Series(1.0, index=btc.index).to_frame("BTC")


STRATEGIES = {
    "S1_ma_trend": s1_weights,
    "S2_tsmom_voltarget": s2_weights,
    "S3_donchian55": s3_weights,
    "S4_xsmom_weekly": s4_weights,
    "S6_buyhold": s6_weights,
}

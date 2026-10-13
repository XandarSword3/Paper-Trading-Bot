"""
Pin the indicator code to Binance's own published chart numbers.

Ground truth: a SOLUSDT-perp 3m chart screenshot showing the FORMING candle at
2025-05-01 12:27 UTC (= 15:27 on a UTC+3 clock). The bar is identified from the data,
not from the reported axis label, by the two volume moving averages, which are exact
identities in the bucket's partial volume:

    MA5  = (vol[B-4] + vol[B-3] + vol[B-2] + vol[B-1] + F) / 5
    MA10 = (vol[B-9] + ... + vol[B-1] + F) / 10

Exactly one bucket in the search window satisfies both. See
quant_research/indicator_validation.py and REPORT section 17.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quant_research.core.indicators import compute_indicators  # noqa: E402

DATA = os.path.join(ROOT, "quant_research", "data", "processed", "SOLUSDT_1m.parquet")
ANCHOR = pd.Timestamp("2025-05-01 12:27:00", tz="UTC")
GROUND = {"bb_up": (152.12, 0.02), "bb_mb": (151.69, 0.02), "bb_dn": (151.26, 0.02),
          "rsi6": (36.80, 1.0), "stoch_k": (28.30, 2.0), "stoch_d": (21.83, 2.0),
          "macd_dif": (-0.08, 0.01), "macd_dea": (-0.03, 0.01), "macd_hist": (-0.05, 0.01)}
GROUND_VOLUME = 5048.36
GROUND_MA5, GROUND_MA10 = 34324.31, 32900.47


@pytest.fixture(scope="module")
def bars():
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    # up to and including the 12:27 1m bar: exactly one completed 1m bar inside the
    # forming 3m bucket that opens at 12:27
    cut = df.loc[:ANCHOR]
    b = cut.resample("3min", label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return b.iloc[-2200:]


def test_the_bar_is_identified_by_the_two_volume_moving_averages(bars):
    """Only the 12:27 UTC bucket reproduces both published volume averages."""
    v = bars["volume"]
    f = GROUND_VOLUME
    residual_ma5 = abs(5.0 * GROUND_MA5 - f - float(v.iloc[-5:-1].sum()))
    residual_ma10 = abs(10.0 * GROUND_MA10 - f - float(v.iloc[-10:-1].sum()))
    assert residual_ma5 < 0.06, residual_ma5      # 2dp rounding of the published average
    assert residual_ma10 < 0.06, residual_ma10
    # the previous bucket must NOT satisfy them, i.e. the identification is a real test
    v2 = v.iloc[:-1]
    assert abs(5.0 * GROUND_MA5 - f - float(v2.iloc[-5:-1].sum())) > 0.06


def test_indicators_reproduce_binances_published_values(bars):
    r = compute_indicators(bars[["open", "high", "low", "close", "volume"]]).iloc[-1]
    got = {"bb_up": r["bb_upper"], "bb_mb": r["bb_basis"], "bb_dn": r["bb_lower"],
           "rsi6": r["rsi6"], "stoch_k": r["stoch_k"], "stoch_d": r["stoch_d"],
           "macd_dif": r["macd_line"], "macd_dea": r["macd_signal"], "macd_hist": r["macd_hist"]}
    misses = {k: (float(got[k]), g, float(got[k]) - g) for k, (g, tol) in GROUND.items()
              if abs(got[k] - g) > tol}
    assert not misses, f"outside the screenshot tolerances: {misses}"


def test_bollinger_uses_the_population_standard_deviation(bars):
    """ddof=0 is required to reproduce Binance's bands; ddof=1 is measurably worse."""
    c = bars["close"]
    mb = c.rolling(20).mean().iloc[-1]
    pop = mb + 2 * c.rolling(20).std(ddof=0).iloc[-1]
    samp = mb + 2 * c.rolling(20).std(ddof=1).iloc[-1]
    assert abs(pop - GROUND["bb_up"][0]) < abs(samp - GROUND["bb_up"][0])
    assert abs(c.rolling(20).std(ddof=0).iloc[-1] / c.rolling(20).std(ddof=1).iloc[-1] - (19 / 20) ** 0.5) < 1e-9

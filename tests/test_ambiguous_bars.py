import pytest
import pandas as pd
import numpy as np

from quant_research.core.engine import QuantEngine, CostModel
from quant_research.strategies.parameter_grid_strategy import run_fast_grid_simulation
from quant_research.strategies.turtle_strategy import (
    prepare_turtle_features,
    run_turtle_simulation,
)


def test_quant_engine_ambiguous_bar_resolution():
    """
    Verifies that when a single bar's range hits both stop and target:
    1. Without sub-bars, QuantEngine resolves conservatively to Stop Loss (Pessimistic fallback).
    2. With 1m sub-bars showing target hit first, QuantEngine resolves to Take Profit (1m resolved).
    3. With 1m sub-bars showing stop hit first, QuantEngine resolves to Stop Loss (1m resolved).
    """
    engine = QuantEngine(CostModel())

    # Long position: Entry at 100, Stop at 90, Target at 110.
    # Bar has Low = 85 (below stop) and High = 115 (above target).
    exit_p, reason = engine.resolve_intrabar_exit(
        stop_price=90.0,
        target_price=110.0,
        direction="long",
        bar_high=115.0,
        bar_low=85.0,
        sub_bars_df=None
    )
    assert exit_p == 90.0
    assert "Stop Loss" in reason

    # With sub-bars: sub-bar 1 hits target (high 112, low 98), sub-bar 2 hits stop (low 88)
    sub_bars_target_first = pd.DataFrame([
        {"open": 100.0, "high": 112.0, "low": 98.0, "close": 111.0},
        {"open": 111.0, "high": 111.0, "low": 88.0, "close": 89.0},
    ])
    exit_p, reason = engine.resolve_intrabar_exit(
        stop_price=90.0,
        target_price=110.0,
        direction="long",
        bar_high=115.0,
        bar_low=85.0,
        sub_bars_df=sub_bars_target_first
    )
    assert exit_p == 110.0
    assert reason == "Take Profit (1m resolved)"

    # With sub-bars: sub-bar 1 hits stop (high 101, low 88), sub-bar 2 hits target (high 112)
    sub_bars_stop_first = pd.DataFrame([
        {"open": 100.0, "high": 101.0, "low": 88.0, "close": 89.0},
        {"open": 89.0, "high": 112.0, "low": 89.0, "close": 111.0},
    ])
    exit_p, reason = engine.resolve_intrabar_exit(
        stop_price=90.0,
        target_price=110.0,
        direction="long",
        bar_high=115.0,
        bar_low=85.0,
        sub_bars_df=sub_bars_stop_first
    )
    assert exit_p == 90.0
    assert reason == "Stop Loss (1m resolved)"


def test_fast_simulator_ambiguous_bar_counter_synthetic():
    """
    Constructs a synthetic dataset where a position enters, and subsequent candle
    hits BOTH the stop loss and target price within the same bar.
    Verifies that the ambiguous-bar counter strictly increments by 1.
    """
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0),
        "atr": np.full(n, 2.0),
        "bb_basis": np.full(n, 100.0),
        "bb_std": np.full(n, 1.0),
        "rsi6": np.full(n, 50.0),
        "adx": np.full(n, 35.0),
        "funding": np.zeros(n),
    })

    # Trigger long entry at bar 60: close < bb_basis - 2.5*bb_std (e.g. close=96) and rsi < 20
    df.loc[60, "close"] = 96.0
    df.loc[60, "rsi6"] = 15.0

    # Entry fills on next bar open (bar 61 at 97.0).
    df.loc[61, "open"] = 97.0
    df.loc[61, "high"] = 98.0
    df.loc[61, "low"] = 96.0
    df.loc[61, "close"] = 97.0
    # Active stop = 97 - 3.0 * 2.0 = 91.0.
    # Target is bb_basis = 100.0.
    # At bar 62, create an ambiguous bar: Low hits 90 (<= stop 91), High hits 105 (>= target 100).
    df.loc[62, "low"] = 90.0
    df.loc[62, "high"] = 105.0

    funding_df = pd.DataFrame({"timestamp": timestamps, "funding_rate": np.zeros(n)})

    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)
    cfg = {
        "config_id": "TEST_AMBIGUOUS",
        "bb_mult": 2.0,
        "rsi_thresh": (20.0, 80.0),
        "adx_thresh": 30.0,
        "stop_atr_mult": 3.0,
        "tp_type": "basis_reversion",
        "max_hold": 20
    }

    res = run_fast_grid_simulation(
        open_arr=df["open"].to_numpy(dtype=np.float64),
        high_arr=df["high"].to_numpy(dtype=np.float64),
        low_arr=df["low"].to_numpy(dtype=np.float64),
        close_arr=df["close"].to_numpy(dtype=np.float64),
        atr_arr=df["atr"].to_numpy(dtype=np.float64),
        bb_basis_arr=df["bb_basis"].to_numpy(dtype=np.float64),
        bb_std_arr=df["bb_std"].to_numpy(dtype=np.float64),
        rsi6_arr=df["rsi6"].to_numpy(dtype=np.float64),
        adx_arr=df["adx"].to_numpy(dtype=np.float64),
        funding_arr=df["funding"].to_numpy(dtype=np.float64),
        params=cfg,
        cost_model=cost,
    )
    assert res["ambiguous_bars"] >= 1, f"Expected ambiguous_bars >= 1, got {res['ambiguous_bars']}"


def _build_ambiguous_turtle_frame():
    """
    Synthetic 1h path (N=20, M=10, long) engineered so that ONE bar (index 60)
    simultaneously satisfies all three of:
      (a) pyramid-add trigger  : close[60] >= E2 + 0.5*ATR[60]
      (b) ratcheted stop touched: low[60]  <= E2 - 2*ATR[40]
      (c) M-bar channel exit    : close[60] < exit_low[60]

    Upper wicks on bars 42-59 inflate Wilders ATR(N) from ~1.17 to ~35.2, which
    is what makes (a) and (c) compatible: the channel line (min low of the last
    M bars) has to sit above the pyramid trigger, while the pyramid trigger has
    to stay below it.  Returns (df, features, meta).
    """
    n = 70
    n_bars = 20
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    cl = np.full(n, 100.0)

    # bar 30: Donchian breakout (close 110 > 20-bar entry high 100.5)
    o[30], h[30], lo[30], cl[30] = 100.0, 111.0, 109.0, 110.0
    # bar 31: Unit 1 fills at the open, 110.0
    o[31], h[31], lo[31], cl[31] = 110.0, 110.5, 109.5, 110.0
    # bars 32-39: quiet drift, pyramid trigger NOT met
    for j in range(32, 40):
        o[j], h[j], lo[j], cl[j] = 110.3, 110.6, 110.0, 110.3
    # bar 40: pyramid-add trigger met -> Unit 2 fills at open[41], stop ratchets
    o[40], h[40], lo[40], cl[40] = 110.3, 111.0, 110.1, 110.7
    # bar 41: Unit 2 fill bar (last_action_bar -> pyramid blocked on this bar)
    o[41], h[41], lo[41], cl[41] = 110.8, 111.2, 110.5, 111.0
    # bars 42-47: upper wicks only.  close stays far below E2 + 0.5*ATR so no
    # further pyramid fires, low stays above the ratcheted stop, ATR balloons.
    wicks = [140.0, 170.0, 200.0, 240.0, 280.0, 320.0]
    for k, j in enumerate(range(42, 48)):
        o[j] = 111.0 + 0.2 * k
        cl[j] = 111.2 + 0.2 * k
        lo[j] = o[j] - 0.5
        h[j] = wicks[k]
    # bars 48-49: step up towards the band
    for j in (48, 49):
        o[j], cl[j] = 112.4, 112.6
        lo[j] = 112.0
        h[j] = 200.0

    # bars 50-59: the M=10 channel window.  Every low sits above the pyramid
    # trigger of bar 60, every close sits below its own pyramid trigger.
    band = 128.15
    for j in range(50, 60):
        o[j] = band + 0.02
        h[j] = band + 30.0
        lo[j] = band
        cl[j] = band + 0.05

    # bar 60: THE ambiguous bar
    o[60] = band + 0.03
    h[60] = band + 0.08
    cl[60] = band - 0.07
    lo[60] = 107.5  # below the ratcheted stop

    # tail: plain continuation
    for j in range(61, n):
        o[j] = cl[j - 1]
        h[j] = cl[j - 1] + 0.5
        lo[j] = cl[j - 1] - 0.5
        cl[j] = cl[j - 1]

    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": cl}, index=ts)
    features = prepare_turtle_features(df, N=20, M=10)
    meta = {
        "entry_price": o[31],
        "pyramid_entry": o[41],
        "atr_entry_signal": float(features["atr"][30]),
        "atr_pyramid_signal": float(features["atr"][40]),
        "atr_ambiguous_bar": float(features["atr"][60]),
        "atr_prev_bar": float(features["atr"][59]),
        "initial_stop": o[31] - 2.0 * float(features["atr"][30]),
        "ratcheted_stop": o[41] - 2.0 * float(features["atr"][40]),
        "channel_level": float(features["exit_low"][60]),
        "ambiguous_bar": 60,
        "next_open": o[61],
        "timestamps": ts,
        "n_bars_param": n_bars,
    }
    return df, features, meta


def _build_1m_subbars(meta):
    """1m sub-bars for the ambiguous 1h bar: channel level reached FIRST."""
    bar_ts = meta["timestamps"][meta["ambiguous_bar"]]
    sub_ts = pd.date_range(bar_ts, periods=60, freq="1min")
    level = meta["channel_level"]
    stop = meta["ratcheted_stop"]

    s_open = np.empty(60)
    s_high = np.empty(60)
    s_low = np.empty(60)
    s_close = np.empty(60)

    # minutes 0-29: drift down through the channel level, never near the stop
    for k in range(30):
        s_open[k] = 128.20 if k == 0 else s_close[k - 1]
        s_high[k] = max(s_open[k], level + 0.03)
        s_low[k] = level - 0.02 if k == 0 else level - 0.05
        s_close[k] = s_low[k] + 0.01
    # minutes 30-59: collapse through the stop
    for k in range(30, 60):
        s_open[k] = s_close[k - 1]
        s_high[k] = s_open[k]
        s_low[k] = stop - 0.5
        s_close[k] = stop - 0.6

    return pd.DataFrame(
        {"open": s_open, "high": s_high, "low": s_low, "close": s_close},
        index=sub_ts,
    )


def test_turtle_ambiguous_bar_pyramid_stop_and_subbar_resolution():
    """
    One synthetic 1h bar touches the pyramid-add trigger AND the ratcheted stop
    AND the M-bar channel line.  Then:
      * without 1m sub-bars the pessimistic fallback fires the stop;
      * with 1m sub-bars showing the channel line reached first, the exit changes.
    That difference only exists if the sub-bars really reach the exit logic.
    """
    df, features, meta = _build_ambiguous_turtle_frame()
    b = meta["ambiguous_bar"]

    # --- the bar really does touch all three levels ---
    pyramid_trigger = meta["pyramid_entry"] + 0.5 * meta["atr_ambiguous_bar"]
    assert meta["atr_pyramid_signal"] > 0
    assert df["close"].iloc[b] >= pyramid_trigger, "pyramid-add trigger not met"
    assert df["low"].iloc[b] <= meta["ratcheted_stop"], "ratcheted stop not touched"
    assert df["close"].iloc[b] < meta["channel_level"], "channel exit not triggered"
    # and the ORIGINAL (pre-ratchet) stop was NOT touched -> only the ratchet fires
    assert df["low"].iloc[b] > meta["initial_stop"]

    cost = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=False)
    params = {"N": 20, "M": 10, "config_id": "TEST_AMBIGUOUS_TURTLE"}

    # ---------- (1) no sub-bars: pessimistic stop-first fallback ----------
    m_off, trades_off = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost,
        initial_capital=100_000.0,
        return_trades=True,
    )
    assert m_off["ambiguous_bars"] == 1, m_off["ambiguous_bars"]
    # the pyramid-add trigger was present but the exit wins -> still 2 units
    assert len(trades_off) == 2
    assert sorted(t.unit_number for t in trades_off) == [1, 2]
    for t in trades_off:
        assert t.exit_reason == "2N Stop Loss"
        assert t.exit_price == pytest.approx(meta["ratcheted_stop"], abs=1e-12)
        assert t.exit_time == df.index[b]

    # ---------- (2) with 1m sub-bars: exit must change ----------
    sub_bars = _build_1m_subbars(meta)
    assert len(sub_bars) == 60
    assert sub_bars.index[0] == df.index[b]
    assert sub_bars["low"].min() <= meta["ratcheted_stop"]
    assert (sub_bars["low"].iloc[:30] <= meta["channel_level"]).all()
    assert (sub_bars["low"].iloc[:30] > meta["ratcheted_stop"]).all()

    m_on, trades_on = run_turtle_simulation(
        features=features,
        funding_series=None,
        params=params,
        cost_model=cost,
        initial_capital=100_000.0,
        df_1m=sub_bars,
        return_trades=True,
    )
    assert m_on["ambiguous_bars"] == 1, m_on["ambiguous_bars"]
    assert len(trades_on) == 2
    for t in trades_on:
        assert t.exit_reason == "10-Bar Donchian Exit (1m resolved)", t.exit_reason
        assert t.exit_price == pytest.approx(meta["next_open"], abs=1e-12)
        assert t.exit_time == df.index[b + 1]

    # ---------- the exit genuinely changed ----------
    assert trades_on[0].exit_price != trades_off[0].exit_price
    assert trades_on[0].exit_reason != trades_off[0].exit_reason
    assert trades_on[0].exit_time != trades_off[0].exit_time

    # ---------- sub-bars are handed to the engine's own resolver ----------
    engine = QuantEngine(cost)
    price, side, reason = engine.resolve_intrabar_exit_levels(
        stop_price=meta["ratcheted_stop"],
        level_price=meta["channel_level"],
        direction="long",
        bar_high=float(df["high"].iloc[b]),
        bar_low=float(df["low"].iloc[b]),
        sub_bars_df=sub_bars,
    )
    assert side == "level"
    assert price == pytest.approx(meta["channel_level"], abs=1e-12)
    assert reason == "Level (1m resolved)"

    # ...and without sub-bars the very same bar falls back to the stop
    price, side, reason = engine.resolve_intrabar_exit_levels(
        stop_price=meta["ratcheted_stop"],
        level_price=meta["channel_level"],
        direction="long",
        bar_high=float(df["high"].iloc[b]),
        bar_low=float(df["low"].iloc[b]),
        sub_bars_df=None,
    )
    assert side == "stop"
    assert price == pytest.approx(meta["ratcheted_stop"], abs=1e-12)

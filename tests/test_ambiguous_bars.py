import pytest
import pandas as pd
import numpy as np

from quant_research.core.engine import QuantEngine, CostModel
from quant_research.strategies.parameter_grid_strategy import run_fast_grid_simulation


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

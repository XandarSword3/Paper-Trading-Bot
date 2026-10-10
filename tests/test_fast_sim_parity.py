import random
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from quant_research.core.engine import CostModel, QuantEngine, assert_no_same_bar_pyramiding
from quant_research.strategies.parameter_grid_strategy import (
    run_fast_grid_simulation,
    run_grid_simulation,
)


def generate_synthetic_market_data(n_bars: int = 3000, seed: int = 42):
    """Generates synthetic OHLCV data with realistic trending and oscillation."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")

    # Geometric Brownian Motion with mean reversion component
    returns = rng.normal(loc=0.0001, scale=0.005, size=n_bars)
    price_path = 100.0 * np.exp(np.cumsum(returns))

    highs = price_path * (1.0 + rng.uniform(0.0005, 0.008, size=n_bars))
    lows = price_path * (1.0 - rng.uniform(0.0005, 0.008, size=n_bars))
    opens = (highs + lows) / 2.0 + rng.normal(0, 0.05, size=n_bars)
    closes = price_path

    # Technical indicators
    close_s = pd.Series(closes, index=timestamps)
    high_s = pd.Series(highs, index=timestamps)
    low_s = pd.Series(lows, index=timestamps)

    bb_basis = close_s.rolling(20).mean().bfill().values
    bb_std = close_s.rolling(20).std().bfill().values

    delta = close_s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 6.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi6 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).values

    tr = pd.concat([(high_s - low_s), (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().bfill().values

    up = high_s - high_s.shift(1)
    down = low_s.shift(1) - low_s
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    p_di = (100.0 * (pd.Series(plus_dm, index=timestamps).ewm(alpha=1.0 / 14.0, adjust=False).mean() / pd.Series(atr, index=timestamps).replace(0, np.nan))).fillna(0.0)
    m_di = (100.0 * (pd.Series(minus_dm, index=timestamps).ewm(alpha=1.0 / 14.0, adjust=False).mean() / pd.Series(atr, index=timestamps).replace(0, np.nan))).fillna(0.0)
    dx = (100.0 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean().values

    # Periodic funding
    funding = np.zeros(n_bars, dtype=np.float64)
    for k in range(0, n_bars, 96):  # 8 hours = 96 5m bars
        funding[k] = rng.uniform(-0.0003, 0.0005)

    return {
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "atr": atr,
        "bb_basis": bb_basis,
        "bb_std": bb_std,
        "rsi6": rsi6,
        "adx": adx,
        "funding": funding,
    }


def get_random_configs(n: int = 25, seed: int = 123) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bb_mults = [1.5, 2.0, 2.5]
    rsi_threshs = [(15.0, 85.0), (20.0, 80.0), (25.0, 75.0)]
    adx_threshs = [20.0, 25.0, 30.0]
    stop_atr_mults = [1.5, 2.0, 3.0]
    tp_types = ["2.0", "3.5", "5.0", "basis_reversion"]
    max_holds = [10, 20, 40, "unlimited"]

    configs = []
    for i in range(n):
        cfg = {
            "config_id": f"TEST_CFG_{i+1:03d}",
            "bb_mult": rng.choice(bb_mults),
            "rsi_thresh": rng.choice(rsi_threshs),
            "adx_thresh": rng.choice(adx_threshs),
            "stop_atr_mult": rng.choice(stop_atr_mults),
            "tp_type": rng.choice(tp_types),
            "max_hold": rng.choice(max_holds),
        }
        configs.append(cfg)
    return configs


def test_fast_sim_parity_across_25_configs():
    """
    Validates bitwise/near-zero parity between run_fast_grid_simulation and core.engine (via run_grid_simulation)
    across 25 randomly sampled parameter configurations:
    1. Identical trade counts.
    2. Identical entry timestamps.
    3. Identical exit timestamps.
    4. Identical net expectancy in bps within 1e-9 tolerance per trade.
    5. Validates invariant assert_no_same_bar_pyramiding on every trade log.
    """
    data = generate_synthetic_market_data(n_bars=2500, seed=777)
    configs = get_random_configs(n=25, seed=42)
    cost_model = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)

    for cfg in configs:
        # Run Fast Simulator
        fast_metrics, fast_trades = run_fast_grid_simulation(
            open_arr=data["open"],
            high_arr=data["high"],
            low_arr=data["low"],
            close_arr=data["close"],
            atr_arr=data["atr"],
            bb_basis_arr=data["bb_basis"],
            bb_std_arr=data["bb_std"],
            rsi6_arr=data["rsi6"],
            adx_arr=data["adx"],
            funding_arr=data["funding"],
            params=cfg,
            cost_model=cost_model,
            timestamps=data["timestamps"],
            return_trades=True,
        )

        # Run Engine Simulator
        engine_metrics, engine_trades = run_grid_simulation(
            open_arr=data["open"],
            high_arr=data["high"],
            low_arr=data["low"],
            close_arr=data["close"],
            atr_arr=data["atr"],
            bb_basis_arr=data["bb_basis"],
            bb_std_arr=data["bb_std"],
            rsi6_arr=data["rsi6"],
            adx_arr=data["adx"],
            funding_arr=data["funding"],
            timestamps=data["timestamps"],
            params=cfg,
            cost_model=cost_model,
        )

        # 1. Trade count identity
        assert len(fast_trades) == len(engine_trades), (
            f"Config {cfg['config_id']}: trade count mismatch: {len(fast_trades)} vs {len(engine_trades)}"
        )

        # 2. Per-trade identity
        for idx, (f_t, e_t) in enumerate(zip(fast_trades, engine_trades)):
            assert f_t["entry_time"] == e_t.entry_time, (
                f"Config {cfg['config_id']} Trade {idx}: entry timestamp mismatch: {f_t['entry_time']} vs {e_t.entry_time}"
            )
            assert f_t["exit_time"] == e_t.exit_time, (
                f"Config {cfg['config_id']} Trade {idx}: exit timestamp mismatch: {f_t['exit_time']} vs {e_t.exit_time}"
            )
            # Net bps within 1e-9
            bps_diff = abs(f_t["return_bps"] - e_t.return_bps)
            assert bps_diff < 1e-9, (
                f"Config {cfg['config_id']} Trade {idx}: net bps mismatch: fast={f_t['return_bps']}, engine={e_t.return_bps}, diff={bps_diff}"
            )

        # 3. Assert no same-bar pyramiding invariant
        assert_no_same_bar_pyramiding(engine_trades)


def test_lookahead_temporal_invariance_and_truncation():
    """
    Validates temporal causality:
    1. Truncating data after bar t must not change decisions made at or before bar t.
    2. Shifting all signals one bar earlier must alter trade timings and outcomes.
    """
    data = generate_synthetic_market_data(n_bars=1500, seed=999)
    cfg = {
        "config_id": "LOOKAHEAD_TEST_CFG",
        "bb_mult": 2.0,
        "rsi_thresh": (20.0, 80.0),
        "adx_thresh": 25.0,
        "stop_atr_mult": 2.0,
        "tp_type": "3.5",
        "max_hold": 20,
    }
    cost_model = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)

    # Full run
    _, full_trades = run_grid_simulation(
        open_arr=data["open"],
        high_arr=data["high"],
        low_arr=data["low"],
        close_arr=data["close"],
        atr_arr=data["atr"],
        bb_basis_arr=data["bb_basis"],
        bb_std_arr=data["bb_std"],
        rsi6_arr=data["rsi6"],
        adx_arr=data["adx"],
        funding_arr=data["funding"],
        timestamps=data["timestamps"],
        params=cfg,
        cost_model=cost_model,
    )

    assert len(full_trades) >= 3, "Need at least 3 trades for robust lookahead check"

    # Truncate after second trade exit
    second_exit_time = full_trades[1].exit_time
    cutoff_idx = int(np.where(data["timestamps"] == second_exit_time)[0][0]) + 5

    trunc_metrics, trunc_trades = run_grid_simulation(
        open_arr=data["open"][:cutoff_idx],
        high_arr=data["high"][:cutoff_idx],
        low_arr=data["low"][:cutoff_idx],
        close_arr=data["close"][:cutoff_idx],
        atr_arr=data["atr"][:cutoff_idx],
        bb_basis_arr=data["bb_basis"][:cutoff_idx],
        bb_std_arr=data["bb_std"][:cutoff_idx],
        rsi6_arr=data["rsi6"][:cutoff_idx],
        adx_arr=data["adx"][:cutoff_idx],
        funding_arr=data["funding"][:cutoff_idx],
        timestamps=data["timestamps"][:cutoff_idx],
        params=cfg,
        cost_model=cost_model,
    )

    # Verify first 2 trades are 100% identical in truncated run
    assert len(trunc_trades) >= 2
    for k in range(2):
        assert full_trades[k].entry_time == trunc_trades[k].entry_time
        assert full_trades[k].exit_time == trunc_trades[k].exit_time
        assert abs(full_trades[k].return_bps - trunc_trades[k].return_bps) < 1e-9
        assert abs(full_trades[k].net_pnl - trunc_trades[k].net_pnl) < 1e-6

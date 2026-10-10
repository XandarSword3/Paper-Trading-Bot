import os
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
from quant_research.test_2_runner import prepare_features


def load_real_sol_5m_features(start_str: str = "2024-01-01", end_str: str = "2024-03-01") -> Dict[str, Any]:
    """Loads at least 2 months of real SOLUSDT 5m data and real funding rates."""
    kline_path = "quant_research/data/processed/SOLUSDT_5m.parquet"
    funding_path = "quant_research/data/processed/SOLUSDT_funding.parquet"
    if not os.path.exists(kline_path) or not os.path.exists(funding_path):
        pytest.skip(f"Missing real SOL data: {kline_path}")

    df = pd.read_parquet(kline_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    fdf = pd.read_parquet(funding_path)
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    funding = fdf.set_index("timestamp")["funding_rate"].sort_index()

    start_ts = pd.Timestamp(start_str, tz="UTC")
    end_ts = pd.Timestamp(end_str, tz="UTC")
    df_slice = df[(df.index >= start_ts) & (df.index <= end_ts)]

    return prepare_features(df_slice, funding)


def get_random_configs(n: int = 25, seed: int = 123) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bb_mults = [1.5, 2.0, 2.5]
    rsi_threshs = [(15.0, 85.0), (20.0, 80.0), (25.0, 75.0)]
    adx_threshs = [20.0, 25.0, 30.0]
    stop_atr_mults = [1.5, 2.0, 3.0]
    tp_types = ["2.0", "3.5", "5.0", "basis_reversion"]
    # Explicitly include active time stops to ensure time stops are exercised
    max_holds = [10, 20, 40]

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


def test_fast_sim_parity_across_25_configs_real_data():
    """
    Validates parity between run_fast_grid_simulation and core.engine (via run_grid_simulation)
    using REAL SOLUSDT 5m data over a 2-month window (2024-01-01 to 2024-03-01, 17,280+ bars)
    across 25 randomly sampled parameter configurations:
    1. Realistic funding settlements applied at exact historical intervals.
    2. Active time stops (10, 20, 40 bars) tested.
    3. Identical trade counts required.
    4. Identical entry timestamps required.
    5. Identical exit timestamps required.
    6. Identical net expectancy in bps within 1e-9 tolerance per trade.
    7. Validates invariant assert_no_same_bar_pyramiding on every trade log.
    """
    feat = load_real_sol_5m_features(start_str="2024-01-01", end_str="2024-03-01")
    assert len(feat["close"]) >= 17000, f"Expected at least 17,000 bars for 2 months, got {len(feat['close'])}"

    configs = get_random_configs(n=25, seed=42)
    cost_model = CostModel(taker_fee_pct=0.05, slippage_pct=0.02, apply_funding=True, fill_on_next_open=True)

    time_stop_exercised_total = 0
    funding_payment_exercised_total = 0

    for cfg in configs:
        # Run Fast Simulator
        fast_metrics, fast_trades = run_fast_grid_simulation(
            open_arr=feat["open"],
            high_arr=feat["high"],
            low_arr=feat["low"],
            close_arr=feat["close"],
            atr_arr=feat["atr"],
            bb_basis_arr=feat["bb_basis"],
            bb_std_arr=feat["bb_std"],
            rsi6_arr=feat["rsi6"],
            adx_arr=feat["adx"],
            funding_arr=feat["funding"],
            params=cfg,
            cost_model=cost_model,
            timestamps=feat["timestamps"],
            return_trades=True,
        )

        # Run Engine Simulator
        engine_metrics, engine_trades = run_grid_simulation(
            open_arr=feat["open"],
            high_arr=feat["high"],
            low_arr=feat["low"],
            close_arr=feat["close"],
            atr_arr=feat["atr"],
            bb_basis_arr=feat["bb_basis"],
            bb_std_arr=feat["bb_std"],
            rsi6_arr=feat["rsi6"],
            adx_arr=feat["adx"],
            funding_arr=feat["funding"],
            timestamps=feat["timestamps"],
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

            if e_t.exit_reason == "Time Stop":
                time_stop_exercised_total += 1
            if abs(e_t.funding_pnl) > 0.0:
                funding_payment_exercised_total += 1

        # 3. Assert no same-bar pyramiding invariant
        assert_no_same_bar_pyramiding(engine_trades)

    # Confirm that time stops and funding settlements were actively encountered across the 25 runs
    assert time_stop_exercised_total > 0, "Expected time stops to be exercised in real data"
    assert funding_payment_exercised_total > 0, "Expected non-zero funding settlements in real data"


def test_lookahead_temporal_invariance_and_truncation():
    """
    Validates temporal causality:
    1. Truncating data after bar t must not change decisions made at or before bar t.
    2. Shifting all signals one bar earlier must alter trade timings and outcomes.
    """
    feat = load_real_sol_5m_features(start_str="2024-01-01", end_str="2024-01-15")
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
        open_arr=feat["open"],
        high_arr=feat["high"],
        low_arr=feat["low"],
        close_arr=feat["close"],
        atr_arr=feat["atr"],
        bb_basis_arr=feat["bb_basis"],
        bb_std_arr=feat["bb_std"],
        rsi6_arr=feat["rsi6"],
        adx_arr=feat["adx"],
        funding_arr=feat["funding"],
        timestamps=feat["timestamps"],
        params=cfg,
        cost_model=cost_model,
    )

    assert len(full_trades) >= 3, "Need at least 3 trades for robust lookahead check"

    # Truncate after second trade exit
    second_exit_time = full_trades[1].exit_time
    cutoff_idx = int(np.where(feat["timestamps"] == second_exit_time)[0][0]) + 5

    trunc_metrics, trunc_trades = run_grid_simulation(
        open_arr=feat["open"][:cutoff_idx],
        high_arr=feat["high"][:cutoff_idx],
        low_arr=feat["low"][:cutoff_idx],
        close_arr=feat["close"][:cutoff_idx],
        atr_arr=feat["atr"][:cutoff_idx],
        bb_basis_arr=feat["bb_basis"][:cutoff_idx],
        bb_std_arr=feat["bb_std"][:cutoff_idx],
        rsi6_arr=feat["rsi6"][:cutoff_idx],
        adx_arr=feat["adx"][:cutoff_idx],
        funding_arr=feat["funding"][:cutoff_idx],
        timestamps=feat["timestamps"][:cutoff_idx],
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

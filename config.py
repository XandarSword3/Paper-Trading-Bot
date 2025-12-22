"""
Configuration file for the Turtle-Inspired Donchian Strategy Backtester
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


@dataclass
class StrategyParams:
    """Strategy parameters - OPTIMIZED based on robustness testing"""
    # Original TradingView params in comments for reference:
    # entry_len=20, exit_len=10, trail_mult=2.5, risk_percent=1.5, pyramid_spacing_n=0.5
    
    entry_len: int = 40          # Was 20 - longer reduces false breakouts
    exit_len: int = 16           # Was 10 - balanced exit
    atr_len: int = 20
    trail_mult: float = 4.0      # Was 2.5 - wider trailing stop
    size_stop_mult: float = 2.0
    risk_percent: float = 1.0    # Was 1.5 - more conservative
    max_units: int = 4
    pyramid_spacing_n: float = 1.5  # Was 0.5 - wider pyramid spacing
    long_only: bool = True
    use_regime_filter: bool = True
    lot_step: float = 0.001
    
    # Trading costs
    commission_pct: float = 0.08  # 0.08%
    slippage_pct: float = 0.05   # 0.05% slippage model


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100_000.0
    timeframe: str = "4h"  # 1h or 4h recommended
    start_date: str = "2017-01-01"
    end_date: str = "2025-12-20"


@dataclass
class RobustnessRanges:
    """Parameter ranges for robustness testing - NO OPTIMIZATION"""
    entry_len: Tuple[int, int, int] = (15, 40, 5)  # min, max, step
    exit_len: Tuple[int, int, int] = (7, 20, 3)
    trail_mult: Tuple[float, float, float] = (2.0, 4.0, 0.5)
    risk_percent: Tuple[float, float, float] = (0.25, 1.0, 0.25)
    pyramid_spacing_n: Tuple[float, float, float] = (0.5, 1.5, 0.25)


@dataclass
class RegimeDefinition:
    """Market regime definitions for BTC"""
    regimes: List[dict] = field(default_factory=lambda: [
        {"name": "2017 Bull", "start": "2017-01-01", "end": "2017-12-31", "type": "bull"},
        {"name": "2018 Bear", "start": "2018-01-01", "end": "2018-12-31", "type": "bear"},
        {"name": "2019 Chop", "start": "2019-01-01", "end": "2019-12-31", "type": "chop"},
        {"name": "2020-2021 Bull", "start": "2020-01-01", "end": "2021-12-31", "type": "bull"},
        {"name": "2022 Bear", "start": "2022-01-01", "end": "2022-12-31", "type": "bear"},
        {"name": "2023-2025 Recovery", "start": "2023-01-01", "end": "2025-12-31", "type": "recovery"},
    ])


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters"""
    n_simulations: int = 1000
    random_seed: int = 42
    slippage_std: float = 0.02  # Standard deviation for slippage variation
    execution_noise_std: float = 0.001  # Price execution noise


# Default instances
DEFAULT_PARAMS = StrategyParams()
DEFAULT_BACKTEST = BacktestConfig()
DEFAULT_ROBUSTNESS = RobustnessRanges()
DEFAULT_REGIMES = RegimeDefinition()
DEFAULT_MONTE_CARLO = MonteCarloConfig()

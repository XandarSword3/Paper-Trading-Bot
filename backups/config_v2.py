"""
Configuration file for V2 High-Frequency Turtle Strategy
Target: ~1 trade per day (vs V1's ~1 trade per week)

Key differences from V1:
- 1H timeframe (vs 4H)
- Shorter lookback periods
- Tighter stops and faster exits
- Same core Turtle-Donchian principles
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
class StrategyParamsV2:
    """
    V2 Strategy parameters - HIGH FREQUENCY variant
    
    Design rationale:
    - V1 uses 4H with entry_len=40 (160 hours = 6.7 days lookback)
    - V2 uses 1H with entry_len=10-20 (10-20 hours lookback)
    - This should produce ~7x more signals
    """
    # Donchian Channel lengths (shorter for more signals)
    entry_len: int = 12          # V1=40 on 4H. 12 on 1H = 12 hours
    exit_len: int = 6            # V1=16 on 4H. 6 on 1H = 6 hours
    atr_len: int = 14            # Slightly shorter ATR
    
    # Trailing stop (tighter for faster exits)
    trail_mult: float = 2.5      # V1=4.0. Tighter for HF trading
    size_stop_mult: float = 2.0
    
    # Risk management (same principles)
    risk_percent: float = 0.5    # V1=1.0. Lower risk per trade due to higher frequency
    max_units: int = 3           # V1=4. Fewer pyramids for HF
    pyramid_spacing_n: float = 1.0  # V1=1.5. Tighter pyramids
    
    # Direction filter
    long_only: bool = True
    use_regime_filter: bool = True
    
    # Position sizing
    lot_step: float = 0.001
    
    # Trading costs (same as V1)
    commission_pct: float = 0.08  # 0.08%
    slippage_pct: float = 0.05   # 0.05%


@dataclass
class BacktestConfigV2:
    """Backtest configuration for V2"""
    initial_capital: float = 100_000.0
    timeframe: str = "1h"  # KEY CHANGE: 1 hour candles
    start_date: str = "2017-08-01"  # Binance BTC data starts here
    end_date: str = "2025-12-26"


@dataclass
class OptimizationRangesV2:
    """Parameter ranges for V2 optimization"""
    # Shorter lookbacks for higher frequency
    entry_len: Tuple[int, int, int] = (8, 24, 2)    # 8-24 hours
    exit_len: Tuple[int, int, int] = (4, 12, 2)     # 4-12 hours
    trail_mult: Tuple[float, float, float] = (1.5, 3.5, 0.5)
    risk_percent: Tuple[float, float, float] = (0.25, 1.0, 0.25)
    pyramid_spacing_n: Tuple[float, float, float] = (0.5, 1.5, 0.5)


# Default V2 parameters (will be optimized)
DEFAULT_PARAMS_V2 = StrategyParamsV2()
DEFAULT_BACKTEST_V2 = BacktestConfigV2()

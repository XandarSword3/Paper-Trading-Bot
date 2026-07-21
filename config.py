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
        # Was "2023-2025 Recovery" through 2025-12-31 — that range reached into
        # 2025, which Phase 1 of the remediation plan froze as the true
        # holdout (DataSplitConfig.holdout_start). Trimmed to stop at the end
        # of the validation window so regime_simulation.py never scores on
        # holdout data. If you need a regime covering 2025+, that belongs in
        # final_holdout_validation.py, not here.
        {"name": "2023-2024 Recovery", "start": "2023-01-01", "end": "2024-12-31", "type": "recovery"},
    ])


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters"""
    n_simulations: int = 1000
    random_seed: int = 42
    slippage_std: float = 0.02  # Standard deviation for slippage variation
    execution_noise_std: float = 0.001  # Price execution noise


@dataclass
class BlockBootstrapConfig:
    """
    Block-bootstrap parameters — Phase 3 of VALIDATION_REMEDIATION_PLAN.md.

    Resamples contiguous blocks of the walk-forward OOS return series
    (walk_forward.py / walk_forward_test.py output) rather than individual
    in-sample trades, and rather than shuffling returns independently — a
    block preserves the short-run autocorrelation/volatility-clustering
    real bar-to-bar returns have, which an iid shuffle destroys.

    block_size_bars: default 42 bars ≈ one week at the 4h timeframe — long
        enough to span a handful of consecutive trades and typical regime
        persistence, short enough to give many blocks per resample. Override
        for other timeframes.
    """
    block_size_bars: int = 42
    n_simulations: int = 1000
    random_seed: int = 42


@dataclass
class DataSplitConfig:
    """
    Canonical chronological data split — frozen per Phase 1 of
    VALIDATION_REMEDIATION_PLAN.md. This is the ONE split definition for the
    whole repo; nothing else should hand-roll its own date filter.

    in_sample:  free to use for feature/rule development and robustness
                testing (Phase 3 of the 16-phase pipeline).
    validation: used for walk-forward OOS folds (Phase 2 of the remediation
                plan) and stress testing (Monte Carlo/regime/survivability) —
                still "seen" during development, never used to pick final
                parameters.
    holdout:    touched by exactly ONE script in this repo —
                final_holdout_validation.py. Scored once, at the very end,
                after Phases 2-4 of the remediation plan are done. See
                data_splits.py for the enforcement mechanism and its
                acceptance check.
    """
    in_sample_start: str = "2017-01-01"
    in_sample_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"
    holdout_end: str = None  # None = through whatever is the latest available candle


@dataclass
class WalkForwardConfig:
    """
    Rolling walk-forward parameters — Phase 2 of VALIDATION_REMEDIATION_PLAN.md.

    Each fold optimizes on `train_years` of data and scores the result on the
    following `test_months`, never touched during that fold's optimization.
    Folds roll forward by `step_months` and are generated only inside the
    development window (DataSplitConfig.in_sample_start -> holdout_start) —
    see walk_forward.generate_folds() for the enforcement.

    train_years:      length of each fold's optimization (in-sample) window.
    test_months:      length of each fold's out-of-sample scoring window.
    step_months:      how far the window rolls between folds. Equal to
                       test_months by default so OOS windows tile the
                       development period with no gap or overlap.
    expanding:        if True, train_start stays fixed at the development
                       start and the window grows each fold instead of
                       sliding (anchored walk-forward). Default is the
                       sliding window the remediation plan describes.
    selection_metric: field on ParameterResult used to pick each fold's
                       winning parameter set from the in-sample grid.
    min_trades_for_selection: in-sample combos with fewer trades than this
                       are excluded from fold-winner selection so a lucky
                       near-zero-trade combo can't top the ranking on a
                       near-zero, near-riskless "sharpe".
    """
    train_years: float = 2.0
    test_months: int = 6
    step_months: int = 6
    expanding: bool = False
    selection_metric: str = "sharpe_ratio"
    min_trades_for_selection: int = 5


@dataclass
class CrossMarketLeg:
    """One market to run the frozen rule set against, unchanged — Phase 4 of
    VALIDATION_REMEDIATION_PLAN.md. 'source' is 'binance' or 'kraken';
    'symbol' is that source's own ticker (data_fetcher.py's Binance symbols
    vs. data_fetcher_kraken.py's PAIR_ALIASES keys)."""
    label: str
    source: str
    symbol: str
    timeframe: str


@dataclass
class CrossMarketConfig:
    """
    Phase 4 — cross-market validation. Goal: check whether V1's edge is
    structural or a BTC-2017-2025-shaped coincidence, by running the exact
    same StrategyParams (DEFAULT_PARAMS below, no re-fitting per market) on
    a few other liquid assets/timeframes/sources and comparing.

    reference: the strategy's own primary market (BTC/USDT 4h, Binance),
        run through this same script for an apples-to-apples baseline
        instead of quoting an old headline number computed a different way.
    legs: one leg per axis the plan names as worth varying — a different
        asset (ETH/USDT), a different timeframe (BTC/USDT 1h), and a
        different data source (BTC/USD via Kraken, the source both live
        bots already trade against).
    """
    reference: CrossMarketLeg = field(
        default_factory=lambda: CrossMarketLeg(
            "BTC/USDT 4h (Binance) - reference", "binance", "BTCUSDT", "4h"
        )
    )
    legs: List[CrossMarketLeg] = field(default_factory=lambda: [
        CrossMarketLeg("ETH/USDT 4h (Binance) - different asset", "binance", "ETHUSDT", "4h"),
        CrossMarketLeg("BTC/USDT 1h (Binance) - different timeframe", "binance", "BTCUSDT", "1h"),
        CrossMarketLeg("BTC/USD 4h (Kraken) - different data source", "kraken", "BTCUSD", "4h"),
    ])


# Default instances
DEFAULT_PARAMS = StrategyParams()
DEFAULT_BACKTEST = BacktestConfig()
DEFAULT_ROBUSTNESS = RobustnessRanges()
DEFAULT_REGIMES = RegimeDefinition()
DEFAULT_MONTE_CARLO = MonteCarloConfig()
DEFAULT_BLOCK_BOOTSTRAP = BlockBootstrapConfig()
DEFAULT_SPLIT = DataSplitConfig()
DEFAULT_WALK_FORWARD = WalkForwardConfig()
DEFAULT_CROSS_MARKET = CrossMarketConfig()

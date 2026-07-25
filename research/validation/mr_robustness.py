"""
Parameter grid + single-combo scorer for BollingerRSIMeanReversion, mirroring
robustness_test.py's ParameterResult/RobustnessTester shape so mr_walk_forward.py
can plug into the SAME fold/stitch/PSR machinery walk_forward.py and
deflated_sharpe.py already provide, without touching TurtleDonchianStrategy at
all.

IN-SAMPLE ONLY, same caveat as robustness_test.py: this file just scores
whatever window it's handed. Honest OOS numbers come from mr_walk_forward.py.
"""
from dataclasses import dataclass
from itertools import product
from typing import List

import pandas as pd

from mean_reversion_strategy import BollingerRSIMeanReversion, MeanReversionParams


@dataclass
class MRParameterResult:
    ma_len: int
    band_mult: float
    rsi_oversold: float
    rsi_overbought: float
    stop_atr_mult: float
    max_hold_bars: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    win_rate: float
    profit_factor: float
    cagr_pct: float
    calmar_ratio: float


@dataclass
class MRRobustnessRanges:
    """Deliberately modest grid — this is a fresh strategy, not a tuned one,
    so the point is to see if ANY region of this space holds up OOS, not to
    chase a single best backtest."""
    ma_len: List[int] = None
    band_mult: List[float] = None
    rsi_pairs: List[tuple] = None       # (oversold, overbought)
    stop_atr_mult: List[float] = None
    max_hold_bars: List[int] = None

    def __post_init__(self):
        self.ma_len = self.ma_len or [15, 20, 30]
        self.band_mult = self.band_mult or [1.5, 2.0, 2.5]
        self.rsi_pairs = self.rsi_pairs or [(25.0, 75.0), (30.0, 70.0)]
        self.stop_atr_mult = self.stop_atr_mult or [1.5, 2.5, 3.5]
        self.max_hold_bars = self.max_hold_bars or [10, 20, 40]


DEFAULT_MR_ROBUSTNESS = MRRobustnessRanges()

# Mirrors walk_forward_test.py's FAST_RANGES: a small grid for a quick CI
# sanity check (does the wiring work at all), not a real validation result.
FAST_MR_RANGES = MRRobustnessRanges(
    ma_len=[20],
    band_mult=[2.0],
    rsi_pairs=[(30.0, 70.0)],
    stop_atr_mult=[2.5],
    max_hold_bars=[20],
)


class MRRobustnessTester:
    def __init__(self, ranges: MRRobustnessRanges = None):
        self.ranges = ranges or DEFAULT_MR_ROBUSTNESS

    def generate_parameter_grid(self) -> List[dict]:
        combos = list(product(
            self.ranges.ma_len,
            self.ranges.band_mult,
            self.ranges.rsi_pairs,
            self.ranges.stop_atr_mult,
            self.ranges.max_hold_bars,
        ))
        return [
            {
                "ma_len": c[0],
                "band_mult": c[1],
                "rsi_oversold": c[2][0],
                "rsi_overbought": c[2][1],
                "stop_atr_mult": c[3],
                "max_hold_bars": c[4],
            }
            for c in combos
        ]

    def test_parameters(self, df: pd.DataFrame, params_dict: dict,
                         initial_capital: float = 100_000.0) -> MRParameterResult:
        params = MeanReversionParams(
            ma_len=params_dict["ma_len"],
            band_mult=params_dict["band_mult"],
            rsi_oversold=params_dict["rsi_oversold"],
            rsi_overbought=params_dict["rsi_overbought"],
            stop_atr_mult=params_dict["stop_atr_mult"],
            max_hold_bars=params_dict["max_hold_bars"],
            # repo-standard defaults for everything else, same as V1/V4 use:
            rsi_len=14,
            atr_len=14,
            risk_percent=1.0,
            long_only=False,
            lot_step=0.001,
            commission_pct=0.08,
            slippage_pct=0.05,
        )
        strategy = BollingerRSIMeanReversion(params)
        strategy.run_backtest(df, initial_capital=initial_capital, verbose=False)
        trade_stats = strategy.get_trade_stats()
        equity_stats = strategy.get_equity_stats(initial_capital)

        return MRParameterResult(
            ma_len=params_dict["ma_len"],
            band_mult=params_dict["band_mult"],
            rsi_oversold=params_dict["rsi_oversold"],
            rsi_overbought=params_dict["rsi_overbought"],
            stop_atr_mult=params_dict["stop_atr_mult"],
            max_hold_bars=params_dict["max_hold_bars"],
            total_return_pct=equity_stats.get("total_return_pct", 0.0),
            max_drawdown_pct=abs(equity_stats.get("max_drawdown_pct", 0.0)),
            sharpe_ratio=equity_stats.get("sharpe_ratio", 0.0),
            num_trades=trade_stats.get("total_trades", 0),
            win_rate=trade_stats.get("win_rate", 0.0),
            profit_factor=trade_stats.get("profit_factor", 0.0),
            cagr_pct=equity_stats.get("cagr_pct", 0.0),
            calmar_ratio=equity_stats.get("calmar_ratio", 0.0),
        )

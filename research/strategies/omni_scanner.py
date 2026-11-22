"""OmniScanner: High-speed cross-sectional momentum and breakout anomaly detector.

Analyzes hundreds of cryptocurrency perpetual contracts concurrently every second,
detecting volume surges, volatility breakouts, and statistical velocity outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import numpy as np

from data.omni_feed import OmniFeed, TickerSnapshot

logger = logging.getLogger("omni_scanner")


@dataclass
class OmniSignal:
    symbol: str
    side: str  # "Buy" or "Sell"
    price: float
    velocity_60s: float
    velocity_zscore: float
    turnover_surge: float
    spread_bps: float
    score: float
    timestamp: float
    reason: str


@dataclass
class ScannerConfig:
    min_turnover_24h: float = 5_000_000.0  # Only trade liquid contracts
    max_spread_bps: float = 12.0          # Reject wide-spread coins (liquidity filter)
    min_velocity_60s: float = 0.006       # Min 0.6% price move in 60 seconds
    min_velocity_zscore: float = 2.2      # Statistical outlier threshold (>2.2 std dev)
    min_turnover_surge: float = 2.0       # Volume must be at least 2.0x normal baseline
    btc_regime_guard: bool = True         # Don't buy if BTC is dumping; don't short if BTC is pumping
    cooldown_seconds: float = 180.0       # Cooldown between signals on the same coin


class OmniScanner:
    """Scans hundreds of coins in real time and detects explosive momentum candidates."""

    def __init__(self, feed: OmniFeed, config: Optional[ScannerConfig] = None) -> None:
        self.feed = feed
        self.config = config or ScannerConfig()
        self.last_signal_time: Dict[str, float] = {}

    def scan(self) -> List[OmniSignal]:
        """Runs cross-sectional scan across all buffered tokens and returns qualified signals."""
        latest = self.feed.latest
        if len(latest) < 10:
            return []

        # 1. Collect 60s velocities and spreads across all eligible symbols
        symbols: List[str] = []
        velocities_60s: List[float] = []

        for sym, snap in latest.items():
            if snap.turnover_24h < self.config.min_turnover_24h:
                continue
            if snap.spread_bps > self.config.max_spread_bps:
                continue
            v = self.feed.get_velocity(sym, lookback_seconds=60.0)
            symbols.append(sym)
            velocities_60s.append(v)

        if len(symbols) < 10:
            return []

        vel_arr = np.array(velocities_60s)
        med = float(np.median(vel_arr))
        # Robust standard deviation via interquartile range (IQR / 1.349)
        q75, q25 = np.percentile(vel_arr, [75, 25])
        iqr = q75 - q25
        std = max(float(iqr / 1.349), 0.001)

        # 2. Check BTC state for cross-market direction guard
        btc_v60 = self.feed.get_velocity("BTCUSDT", lookback_seconds=60.0)
        btc_dumping = btc_v60 < -0.004  # BTC dropping >0.4% in 60s
        btc_pumping = btc_v60 > +0.004  # BTC rising >0.4% in 60s

        signals: List[OmniSignal] = []
        now = self.feed.last_poll_time

        for sym, v60 in zip(symbols, velocities_60s):
            # Check symbol cooldown
            if now - self.last_signal_time.get(sym, 0.0) < self.config.cooldown_seconds:
                continue

            zscore = (v60 - med) / std
            surge = self.feed.get_turnover_surge(sym, lookback_seconds=60.0)
            snap = latest[sym]

            # Long breakout condition:
            # - Strong upward velocity (> +0.6% and > +2.2 z-score)
            # - Volume surging > 2.0x normal
            # - Tight spread
            # - BTC is not crashing
            if (
                v60 >= self.config.min_velocity_60s
                and zscore >= self.config.min_velocity_zscore
                and surge >= self.config.min_turnover_surge
                and not (self.config.btc_regime_guard and btc_dumping)
            ):
                score = min(100.0, 50.0 + zscore * 10.0 + surge * 5.0)
                reason = f"Bullish velocity surge (+{v60*100:.2f}%, z={zscore:.1f}, vol={surge:.1f}x)"
                sig = OmniSignal(
                    symbol=sym,
                    side="Buy",
                    price=snap.last_price,
                    velocity_60s=v60,
                    velocity_zscore=zscore,
                    turnover_surge=surge,
                    spread_bps=snap.spread_bps,
                    score=score,
                    timestamp=now,
                    reason=reason,
                )
                signals.append(sig)
                self.last_signal_time[sym] = now
                logger.info("SIGNAL: %s %s @ %.4f | %s", sig.side, sig.symbol, sig.price, sig.reason)

            # Short breakdown condition:
            # - Strong downward velocity (< -0.6% and < -2.2 z-score)
            # - Volume surging > 2.0x normal
            # - BTC is not ripping upward
            elif (
                v60 <= -self.config.min_velocity_60s
                and zscore <= -self.config.min_velocity_zscore
                and surge >= self.config.min_turnover_surge
                and not (self.config.btc_regime_guard and btc_pumping)
            ):
                score = min(100.0, 50.0 + abs(zscore) * 10.0 + surge * 5.0)
                reason = f"Bearish breakdown surge ({v60*100:.2f}%, z={zscore:.1f}, vol={surge:.1f}x)"
                sig = OmniSignal(
                    symbol=sym,
                    side="Sell",
                    price=snap.last_price,
                    velocity_60s=v60,
                    velocity_zscore=zscore,
                    turnover_surge=surge,
                    spread_bps=snap.spread_bps,
                    score=score,
                    timestamp=now,
                    reason=reason,
                )
                signals.append(sig)
                self.last_signal_time[sym] = now
                logger.info("SIGNAL: %s %s @ %.4f | %s", sig.side, sig.symbol, sig.price, sig.reason)

        return sorted(signals, key=lambda s: s.score, reverse=True)

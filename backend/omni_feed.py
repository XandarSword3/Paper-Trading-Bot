"""OmniFeed: High-throughput real-time market data feed for hundreds of crypto perps.

Ingests all Bybit USDT Linear Perpetual contracts simultaneously using batch ticker snapshots,
maintaining in-memory rolling time-series buffers for price, spread, and volume velocity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger("omni_feed")


@dataclass
class TickerSnapshot:
    symbol: str
    timestamp: float
    last_price: float
    bid_price: float
    ask_price: float
    turnover_24h: float
    volume_24h: float
    price_1h_change: float
    funding_rate: float
    open_interest: float

    @property
    def spread_bps(self) -> float:
        if self.bid_price <= 0:
            return 999.0
        return ((self.ask_price - self.bid_price) / self.bid_price) * 10000.0


class OmniFeed:
    """Ingests and buffers real-time market metrics across hundreds of USDT perpetual contracts."""

    def __init__(
        self,
        min_turnover_usd: float = 3_000_000.0,
        max_history_seconds: int = 600,
        testnet: bool = False,
    ) -> None:
        self.min_turnover_usd = min_turnover_usd
        self.max_history_seconds = max_history_seconds
        self.client = HTTP(testnet=testnet, recv_window=5000)
        self._lock = threading.Lock()
        
        # symbol -> deque of TickerSnapshot
        self.history: Dict[str, deque[TickerSnapshot]] = {}
        # symbol -> latest TickerSnapshot
        self.latest: Dict[str, TickerSnapshot] = {}
        
        self.last_poll_time: float = 0.0
        self.poll_count: int = 0
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None

    def fetch_snapshot(self) -> Dict[str, TickerSnapshot]:
        """Fetch real-time snapshot of all active linear perpetuals in a single API call."""
        now = time.time()
        try:
            resp = self.client.get_tickers(category="linear")
        except Exception as exc:
            logger.warning("Bybit batch ticker fetch failed: %s", exc)
            return self.latest

        raw_list = resp.get("result", {}).get("list", [])
        updated: Dict[str, TickerSnapshot] = {}

        with self._lock:
            for item in raw_list:
                sym = str(item.get("symbol", ""))
                if not sym.endswith("USDT"):
                    continue

                try:
                    last_px = float(item.get("lastPrice", 0.0) or 0.0)
                    turnover = float(item.get("turnover24h", 0.0) or 0.0)
                    if last_px <= 0 or turnover < self.min_turnover_usd:
                        continue

                    bid_px = float(item.get("bid1Price", last_px) or last_px)
                    ask_px = float(item.get("ask1Price", last_px) or last_px)
                    vol = float(item.get("volume24h", 0.0) or 0.0)
                    prev_1h = float(item.get("prevPrice1h", last_px) or last_px)
                    pct_1h = (last_px - prev_1h) / prev_1h if prev_1h > 0 else 0.0
                    funding = float(item.get("fundingRate", 0.0) or 0.0)
                    oi = float(item.get("openInterest", 0.0) or 0.0)

                    snap = TickerSnapshot(
                        symbol=sym,
                        timestamp=now,
                        last_price=last_px,
                        bid_price=bid_px,
                        ask_price=ask_px,
                        turnover_24h=turnover,
                        volume_24h=vol,
                        price_1h_change=pct_1h,
                        funding_rate=funding,
                        open_interest=oi,
                    )

                    updated[sym] = snap
                    self.latest[sym] = snap

                    if sym not in self.history:
                        self.history[sym] = deque(maxlen=600)
                    buf = self.history[sym]
                    buf.append(snap)

                    # Prune entries older than max_history_seconds
                    cutoff = now - self.max_history_seconds
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

                except (ValueError, TypeError):
                    continue

            self.last_poll_time = now
            self.poll_count += 1

        return updated

    def get_velocity(self, symbol: str, lookback_seconds: float = 60.0) -> float:
        """Returns price return over the past N seconds for symbol."""
        with self._lock:
            buf = self.history.get(symbol)
            if not buf or len(buf) < 2:
                return 0.0
            latest = buf[-1]
            target_ts = latest.timestamp - lookback_seconds
            
            # Find earliest snapshot >= target_ts or closest
            chosen = buf[0]
            for s in buf:
                if s.timestamp >= target_ts:
                    chosen = s
                    break
            
            if chosen.last_price <= 0:
                return 0.0
            return (latest.last_price - chosen.last_price) / chosen.last_price

    def get_turnover_surge(self, symbol: str, lookback_seconds: float = 60.0) -> float:
        """Ratio of turnover gained over lookback compared to average expected rate."""
        with self._lock:
            buf = self.history.get(symbol)
            if not buf or len(buf) < 2:
                return 1.0
            latest = buf[-1]
            target_ts = latest.timestamp - lookback_seconds
            
            chosen = buf[0]
            for s in buf:
                if s.timestamp >= target_ts:
                    chosen = s
                    break
            
            dt = latest.timestamp - chosen.timestamp
            if dt <= 0:
                return 1.0
            
            # Actual turnover delta in $
            turnover_delta = max(0.0, latest.turnover_24h - chosen.turnover_24h)
            actual_rate_per_sec = turnover_delta / dt
            
            # 24h baseline turnover rate per second
            baseline_rate_per_sec = latest.turnover_24h / 86400.0
            if baseline_rate_per_sec <= 0:
                return 1.0
            
            return actual_rate_per_sec / baseline_rate_per_sec

    def start_background_poller(self, interval_seconds: float = 2.0, callback: Optional[Callable[[], None]] = None) -> None:
        """Runs the poller loop in a background daemon thread."""
        if self.is_running:
            return

        self.is_running = True

        def _loop() -> None:
            while self.is_running:
                t0 = time.time()
                self.fetch_snapshot()
                if callback:
                    try:
                        callback()
                    except Exception as err:
                        logger.error("Callback error: %s", err)
                elapsed = time.time() - t0
                sleep_time = max(0.1, interval_seconds - elapsed)
                time.sleep(sleep_time)

        self._thread = threading.Thread(target=_loop, name="OmniFeedPoller", daemon=True)
        self._thread.start()
        logger.info("OmniFeed background poller started (interval: %.1fs)", interval_seconds)

    def stop(self) -> None:
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("OmniFeed background poller stopped")

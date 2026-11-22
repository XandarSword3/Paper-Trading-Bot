"""Market Tape Recorder (Institutional Grade).

Records raw Bybit v5 top-of-book quotes (orderbook.1) and execution trades (publicTrade)
into compressed, partitioned Parquet files for high-speed offline simulation and replay.

Storage structure:
    logs/market_tape/
        quotes/
            quotes_YYYYMMDD_HH.parquet
        trades/
            trades_YYYYMMDD_HH.parquet
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("market_recorder")

QUOTE_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("symbol", pa.string()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("mid", pa.float64()),
    ("spread_bps", pa.float64()),
])

TRADE_SCHEMA = pa.schema([
    ("ts", pa.float64()),
    ("symbol", pa.string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("side", pa.string()),
])


class MarketTapeRecorder:
    """Buffers live streaming quotes and trade prints and writes them to compressed Parquet."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        flush_interval_seconds: float = 20.0,
        max_batch_size: int = 10000,
    ) -> None:
        self.base_dir = base_dir or Path("logs/market_tape")
        self.quotes_dir = self.base_dir / "quotes"
        self.trades_dir = self.base_dir / "trades"
        self.quotes_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        self.flush_interval = flush_interval_seconds
        self.max_batch_size = max_batch_size

        self.quote_buffer: List[dict] = []
        self.trade_buffer: List[dict] = []
        self.lock = threading.Lock()
        self.last_flush = time.time()
        self.total_quotes_saved = 0
        self.total_trades_saved = 0

        self._active_writers: dict = {}
        self._running = True
        self._flush_thread = threading.Thread(target=self._background_flush_loop, daemon=True)
        self._flush_thread.start()

    def _background_flush_loop(self) -> None:
        """Periodic background flush so data persists smoothly without blocking WS ticks."""
        while self._running:
            time.sleep(self.flush_interval)
            try:
                with self.lock:
                    self._flush_locked()
            except Exception as e:
                logger.error("Error in background parquet flush: %s", e)

    def record_quote(self, symbol: str, exchange_ts: float, bid: float, ask: float, mid: float, spread_bps: float) -> None:
        """Buffers an orderbook.1 quote tick."""
        with self.lock:
            self.quote_buffer.append({
                "ts": exchange_ts,
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_bps": spread_bps,
            })
            if len(self.quote_buffer) >= self.max_batch_size:
                self._flush_locked()

    def record_trade(self, symbol: str, exchange_ts: float, price: float, size: float, side: str) -> None:
        """Buffers a publicTrade execution print."""
        with self.lock:
            self.trade_buffer.append({
                "ts": exchange_ts,
                "symbol": symbol,
                "price": price,
                "size": size,
                "side": side,
            })
            if len(self.trade_buffer) >= self.max_batch_size:
                self._flush_locked()

    def check_periodic_flush(self) -> None:
        """Flushes buffers if interval has elapsed."""
        now = time.time()
        if now - self.last_flush >= self.flush_interval:
            with self.lock:
                self._flush_locked()

    def _get_writer(self, file_path: Path, schema: pa.Schema) -> pq.ParquetWriter:
        key = str(file_path)
        if key not in self._active_writers:
            self._active_writers[key] = pq.ParquetWriter(file_path, schema, compression="zstd")
        return self._active_writers[key]

    def _flush_locked(self) -> None:
        """Internal flush writing parquet chunks with PyArrow streaming (called under lock)."""
        now_dt = datetime.now(timezone.utc)
        hour_tag = now_dt.strftime("%Y%m%d_%H")

        # 1. Flush quotes
        if self.quote_buffer:
            q_file = self.quotes_dir / f"quotes_{hour_tag}.parquet"
            batch = self.quote_buffer
            self.quote_buffer = []
            self.total_quotes_saved += len(batch)
            try:
                table = pa.Table.from_pylist(batch, schema=QUOTE_SCHEMA)
                writer = self._get_writer(q_file, QUOTE_SCHEMA)
                writer.write_table(table)
            except Exception as e:
                logger.error("Failed flushing quotes to parquet: %s", e)

        # 2. Flush trades
        if self.trade_buffer:
            t_file = self.trades_dir / f"trades_{hour_tag}.parquet"
            batch = self.trade_buffer
            self.trade_buffer = []
            self.total_trades_saved += len(batch)
            try:
                table = pa.Table.from_pylist(batch, schema=TRADE_SCHEMA)
                writer = self._get_writer(t_file, TRADE_SCHEMA)
                writer.write_table(table)
            except Exception as e:
                logger.error("Failed flushing trades to parquet: %s", e)

        self.last_flush = time.time()

    def close(self) -> None:
        """Final flush on shutdown and close all open parquet writers."""
        self._running = False
        with self.lock:
            self._flush_locked()
            for key, writer in list(self._active_writers.items()):
                try:
                    writer.close()
                except Exception as e:
                    logger.error("Error closing parquet writer %s: %s", key, e)
            self._active_writers.clear()
            logger.info(
                "Market recorder closed cleanly. Total quotes saved: %d, Total trades saved: %d",
                self.total_quotes_saved,
                self.total_trades_saved,
            )

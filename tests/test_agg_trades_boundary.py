"""Regression tests for aggTrades end-time boundary enforcement.

The downloader must never return or store trades outside
[start_ms, end_ms] (end INCLUSIVE), regardless of whether the exchange
API respects its startTime/endTime parameters — fromId pagination pages
carry no time filter at all. All tests use a mocked client; no network.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from v4_system.data_engine import agg_trades, store

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _trade(a: int, t_ms: int) -> dict:
    return {"a": a, "p": "100.0", "q": "1.0", "f": a, "l": a,
            "T": t_ms, "m": False}


class FakeAggClient:
    """Mock /fapi/v1/aggTrades.

    mode="honest":      applies startTime/endTime like a well-behaved API.
    mode="ignore_times": ignores startTime/endTime entirely (returns every
                        trade, or every trade after fromId) — simulates an
                        API that does not respect the boundary, which is
                        exactly what the local filter must defend against.
    """

    def __init__(self, trades, mode="honest"):
        self.trades = sorted(trades, key=lambda t: t["a"])
        self.mode = mode
        self.calls = []

    def get(self, path, params=None):
        self.calls.append(dict(params or {}))
        limit = int(params.get("limit", 1000))
        if "fromId" in params:
            fid = int(params["fromId"])
            rows = [t for t in self.trades if t["a"] >= fid]
        elif self.mode == "ignore_times":
            rows = list(self.trades)
        else:
            start = int(params["startTime"])
            end = int(params["endTime"])
            rows = [t for t in self.trades
                    if start <= int(t["T"]) <= end]
        return rows[:limit]


def _fetch(trades, start, end, mode="honest", max_rows=200_000):
    client = FakeAggClient(trades, mode=mode)
    return agg_trades.fetch_agg_trades("BTCUSDT", start=start, end=end,
                                       max_rows=max_rows, client=client), client


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------

def test_trade_exactly_at_end_boundary_included():
    end_ms = _ms(T0) + 60_000
    trades = [_trade(1, _ms(T0)), _trade(2, end_ms)]
    df, _ = _fetch(trades, T0, T0.replace(minute=1), mode="ignore_times")
    assert list(df.index) == [pd.Timestamp(_ms(T0), unit="ms", tz="UTC"),
                              pd.Timestamp(end_ms, unit="ms", tz="UTC")]


def test_trade_immediately_after_end_excluded():
    end_ms = _ms(T0) + 60_000
    trades = [_trade(1, _ms(T0)), _trade(2, end_ms + 1)]
    df, _ = _fetch(trades, T0, T0.replace(minute=1), mode="ignore_times")
    assert len(df) == 1
    assert df.index[-1] == pd.Timestamp(_ms(T0), unit="ms", tz="UTC")


def test_trade_before_start_excluded():
    start_ms = _ms(T0)
    trades = [_trade(1, start_ms - 1), _trade(2, start_ms)]
    df, _ = _fetch(trades, T0, T0.replace(minute=5), mode="ignore_times")
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp(start_ms, unit="ms", tz="UTC")


def test_out_of_order_trades_sorted_and_bounded():
    start_ms, end_ms = _ms(T0), _ms(T0) + 60_000
    trades = [_trade(3, end_ms), _trade(1, start_ms), _trade(2, end_ms + 5),
              _trade(4, start_ms + 1000)]
    df, _ = _fetch(trades, T0, T0.replace(minute=1), mode="ignore_times")
    assert list(df.index) == sorted(df.index)
    assert df.index.max() <= pd.Timestamp(end_ms, unit="ms", tz="UTC")
    assert len(df) == 3  # trades 1, 3, 4 in range; trade 2 beyond end


def test_beyond_boundary_on_later_page_excluded():
    """Page 1 fills the limit with in-range trades; page 2 (fromId) returns
    trades beyond end — they must be dropped and pagination must stop."""
    start_ms = _ms(T0)
    end_ms = start_ms + 2000  # window spans two "pages" of trades
    end = T0 + pd.Timedelta(milliseconds=2000)
    page1 = [_trade(i + 1, start_ms + i) for i in range(1000)]       # T < end
    page2_in = [_trade(1500, start_ms + 1500)]                        # in range, page 2
    page2_out = [_trade(2001 + i, end_ms + 1 + i) for i in range(10)]  # beyond end
    df, client = _fetch(page1 + page2_in + page2_out, T0, end,
                        mode="ignore_times")
    assert len(df) == 1001
    assert df.index.max() <= pd.Timestamp(end_ms, unit="ms", tz="UTC")
    # Pagination used fromId after the first full page.
    assert any("fromId" in c for c in client.calls)


def test_page_with_only_out_of_range_trades_terminates():
    start_ms = _ms(T0)
    trades = [_trade(1, start_ms - 5000), _trade(2, start_ms + 999_999)]
    with pytest.raises(ValueError, match="no aggTrades"):
        _fetch(trades, T0, T0.replace(minute=1), mode="ignore_times")


def test_empty_page_raises():
    with pytest.raises(ValueError, match="no aggTrades"):
        _fetch([], T0, T0.replace(minute=1))


def test_millisecond_precision():
    end = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    end_ms = _ms(end)
    trades = [_trade(1, end_ms), _trade(2, end_ms + 1)]
    df, _ = _fetch(trades, T0, end, mode="ignore_times")
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp(end_ms, unit="ms", tz="UTC")


def test_start_equals_end_window():
    """start == end -> only trades at exactly that millisecond survive."""
    exact = _ms(T0)
    trades = [_trade(1, exact - 1), _trade(2, exact), _trade(3, exact + 1)]
    df, _ = _fetch(trades, T0, T0, mode="ignore_times")
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp(exact, unit="ms", tz="UTC")


def test_honest_api_still_bounded():
    """Even a well-behaved API result is re-checked locally."""
    start_ms, end_ms = _ms(T0), _ms(T0) + 60_000
    trades = [_trade(1, start_ms), _trade(2, end_ms), _trade(3, end_ms + 1)]
    df, _ = _fetch(trades, T0, T0.replace(minute=1), mode="honest")
    assert len(df) == 2
    assert df.index.max() == pd.Timestamp(end_ms, unit="ms", tz="UTC")


# ---------------------------------------------------------------------------
# Persisted dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    from v4_system.data_engine import manifest as manifest_mod
    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", str(tmp_path / "MANIFEST.json"))
    return tmp_path


def test_no_out_of_range_trade_reaches_saved_dataset(store_dir):
    start_ms, end_ms = _ms(T0), _ms(T0) + 60_000
    trades = ([_trade(i + 1, start_ms + i * 10) for i in range(50)]
              + [_trade(100 + i, end_ms + 1 + i) for i in range(20)])
    df, _ = _fetch(trades, T0, T0.replace(minute=1), mode="ignore_times")
    store.save(df, "BTCUSDT_agg_1h", source="test", symbol="BTCUSDT",
               timeframe=None)
    loaded = store.load("BTCUSDT_agg_1h")
    assert len(loaded) == 50
    assert loaded.index.min() >= pd.Timestamp(start_ms, unit="ms", tz="UTC")
    assert loaded.index.max() <= pd.Timestamp(end_ms, unit="ms", tz="UTC")

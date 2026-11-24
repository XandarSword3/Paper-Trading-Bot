"""Safety tests for the RGMT live/testnet execution adapter.

The executor is the only path that can lose real money, so its guards are
tested independently of the network: order planning, exchange-filter parsing,
notional limits, and the explicit opt-in gates.
"""

from __future__ import annotations

import os

import pytest

from scripts.run_rgmt_live import (
    check_execution_gates,
    load_dotenv,
    parse_filters,
    plan_orders,
    round_down,
    safety_check,
)


FILTERS = {
    "BTCUSDT": {"step": 0.001, "min_qty": 0.001, "min_notional": 5.0},
    "ETHUSDT": {"step": 0.01, "min_qty": 0.01, "min_notional": 5.0},
}


def test_round_down_never_rounds_up() -> None:
    assert round_down(1.2349, 0.001) == pytest.approx(1.234)
    assert round_down(1.2399, 0.01) == pytest.approx(1.23)
    assert round_down(5.0, 1.0) == pytest.approx(5.0)
    # Binary float floor may drop one extra step; that is safe (never over-trade).
    assert round_down(0.3, 0.1) <= 0.3
    assert round_down(1.234, 0.0) == pytest.approx(1.234)


def test_plan_opens_long_from_flat() -> None:
    orders = plan_orders(
        {"BTCUSDT": 0.5},
        equity=10_000.0,
        positions={},
        marks={"BTCUSDT": 100.0},
        filters=FILTERS,
    )
    assert len(orders) == 1
    order = orders[0]
    assert order["side"] == "BUY"
    assert order["quantity"] == pytest.approx(50.0)
    assert order["reduce_only"] is False
    assert order["reason"] == "rebalance"


def test_plan_opens_short_from_flat() -> None:
    orders = plan_orders(
        {"BTCUSDT": -0.5},
        equity=10_000.0,
        positions={},
        marks={"BTCUSDT": 100.0},
        filters=FILTERS,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == "SELL"
    assert orders[0]["quantity"] == pytest.approx(50.0)
    assert orders[0]["reduce_only"] is False


def test_plan_flattens_a_position_missing_from_targets() -> None:
    # A stale position must be closable even when it is absent from the targets.
    orders = plan_orders(
        {},
        equity=10_000.0,
        positions={"ETHUSDT": 2.0},
        marks={"ETHUSDT": 50.0},
        filters=FILTERS,
    )
    assert len(orders) == 1
    assert orders[0]["symbol"] == "ETHUSDT"
    assert orders[0]["side"] == "SELL"
    assert orders[0]["quantity"] == pytest.approx(2.0)
    assert orders[0]["reduce_only"] is True


def test_plan_reduces_with_reduce_only() -> None:
    orders = plan_orders(
        {"BTCUSDT": 0.25},
        equity=10_000.0,
        positions={"BTCUSDT": 50.0},
        marks={"BTCUSDT": 100.0},
        filters=FILTERS,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == "SELL"
    assert orders[0]["reduce_only"] is True
    assert orders[0]["quantity"] == pytest.approx(25.0)


def test_plan_flip_closes_then_opens() -> None:
    orders = plan_orders(
        {"BTCUSDT": -0.5},
        equity=10_000.0,
        positions={"BTCUSDT": 50.0},
        marks={"BTCUSDT": 100.0},
        filters=FILTERS,
    )
    assert [o["reason"] for o in orders] == ["flip_close", "flip_open"]
    assert orders[0]["side"] == "SELL"
    assert orders[0]["reduce_only"] is True
    assert orders[1]["side"] == "SELL"
    assert orders[1]["reduce_only"] is False
    assert orders[0]["quantity"] == pytest.approx(50.0)
    assert orders[1]["quantity"] == pytest.approx(50.0)


def test_plan_skips_dust_and_missing_marks() -> None:
    orders = plan_orders(
        {"BTCUSDT": 0.5, "ETHUSDT": 0.5},
        equity=100.0,
        positions={},
        marks={"BTCUSDT": 100.0},  # ETH has no mark
        filters=FILTERS,
    )
    # BTC target is only $50 at 0.5 weight -> 0.5 units, $50 notional, but a
    # 0.5 weight is not the concern here; the missing ETH mark must be skipped.
    assert all(order["symbol"] != "ETHUSDT" for order in orders)

    dust = plan_orders(
        {"BTCUSDT": 0.0001},
        equity=10_000.0,
        positions={},
        marks={"BTCUSDT": 100.0},
        filters=FILTERS,
    )
    assert dust == []


def test_plan_rejects_too_many_orders() -> None:
    targets = {f"S{i}USDT": 0.01 for i in range(5)}
    marks = {symbol: 100.0 for symbol in targets}
    filters = {symbol: {"step": 0.001, "min_qty": 0.001, "min_notional": 5.0} for symbol in targets}
    with pytest.raises(RuntimeError):
        plan_orders(targets, 10_000.0, {}, marks, filters, max_orders=3)


def test_parse_filters_reads_lot_size_and_min_notional() -> None:
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "100"},
                ],
            },
            {
                "symbol": "ETHUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"},
                    {"filterType": "NOTIONAL", "minNotional": "20"},
                ],
            },
            {"symbol": "NOFILTERS", "filters": []},
        ]
    }
    filters = parse_filters(exchange_info)
    assert filters["BTCUSDT"] == {"step": 0.001, "min_qty": 0.001, "min_notional": 100.0}
    assert filters["ETHUSDT"] == {"step": 0.01, "min_qty": 0.01, "min_notional": 20.0}
    assert filters["NOFILTERS"] == {"step": 0.001, "min_qty": 0.001, "min_notional": 5.0}


def test_safety_check_accepts_a_legal_plan() -> None:
    targets = {"BTCUSDT": 0.15, "ETHUSDT": -0.15}
    equity = 10_000.0
    marks = {"BTCUSDT": 100.0, "ETHUSDT": 100.0}
    orders = plan_orders(targets, equity, {}, marks, FILTERS)
    safety_check(targets, equity, orders, marks)


def test_safety_check_rejects_oversized_asset_weight() -> None:
    targets = {"BTCUSDT": 0.16}
    marks = {"BTCUSDT": 100.0}
    orders = plan_orders(targets, 10_000.0, {}, marks, FILTERS)
    with pytest.raises(ValueError):
        safety_check(targets, 10_000.0, orders, marks)


def test_safety_check_rejects_gross_above_frozen_limit() -> None:
    targets = {f"S{i}USDT": 0.15 for i in range(21)}
    with pytest.raises(ValueError):
        safety_check(targets, 10_000.0, [], {})


def test_safety_check_rejects_a_single_order_over_fraction_of_equity() -> None:
    targets = {"BTCUSDT": 0.15}
    equity = 10_000.0
    marks = {"BTCUSDT": 100.0}
    oversized = [{"symbol": "BTCUSDT", "quantity": 50.0, "reduce_only": False}]
    with pytest.raises(RuntimeError):
        safety_check(targets, equity, oversized, marks)


def test_safety_check_never_blocks_reduce_only_de_risking() -> None:
    # A small or nearly empty account must still be able to flatten.
    targets = {"BTCUSDT": 0.0}
    marks = {"BTCUSDT": 100.0}
    closing = [{"symbol": "BTCUSDT", "quantity": 50.0, "reduce_only": True}]
    safety_check(targets, 100.0, closing, marks)


def test_execution_gates_require_explicit_opt_in() -> None:
    # Testnet dry run is always allowed.
    check_execution_gates(testnet=True, allow_live=False, allow_mainnet=False, execute=False)
    # Testnet execution needs ALLOW_LIVE.
    with pytest.raises(SystemExit):
        check_execution_gates(testnet=True, allow_live=False, allow_mainnet=False, execute=True)
    check_execution_gates(testnet=True, allow_live=True, allow_mainnet=False, execute=True)
    # Mainnet is refused even for a dry run without ALLOW_MAINNET_LIVE.
    with pytest.raises(SystemExit):
        check_execution_gates(testnet=False, allow_live=True, allow_mainnet=False, execute=False)
    check_execution_gates(testnet=False, allow_live=True, allow_mainnet=True, execute=True)


def test_load_dotenv_reads_values_without_overwriting(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "RGMT_TEST_A=1\n"
        "# a comment\n"
        "\n"
        "RGMT_TEST_B='two words'\n"
        'RGMT_TEST_C="quoted"\n'
        "RGMT_TEST_D=keep\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RGMT_TEST_D", "existing")
    for key in ("RGMT_TEST_A", "RGMT_TEST_B", "RGMT_TEST_C"):
        monkeypatch.delenv(key, raising=False)
    try:
        load_dotenv(path)
        assert os.environ["RGMT_TEST_A"] == "1"
        assert os.environ["RGMT_TEST_B"] == "two words"
        assert os.environ["RGMT_TEST_C"] == "quoted"
        assert os.environ["RGMT_TEST_D"] == "existing"
    finally:
        for key in ("RGMT_TEST_A", "RGMT_TEST_B", "RGMT_TEST_C"):
            os.environ.pop(key, None)


def test_load_dotenv_ignores_a_missing_file(tmp_path) -> None:
    load_dotenv(tmp_path / "does-not-exist.env")

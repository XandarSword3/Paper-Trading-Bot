"""Accounting tests for the RGMT paper trader.

These tests are the contract for the money math.  They cover the bugs that
were live in the runner: missing funding, no drawdown enforcement, wrong
average entry on adds, and flips that failed to realise the old position.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research.rgmt_paper import (
    COST_RATE,
    accrue_funding,
    already_ran_today,
    flatten_targets,
    mark_to_market,
    new_state,
    rebalance,
    update_risk,
    validate_targets,
)


def test_open_long_from_flat() -> None:
    state = new_state(10_000.0)
    new, orders = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == "BUY"
    assert order.qty == pytest.approx(50.0)
    assert order.notional == pytest.approx(5_000.0)
    assert order.fee == pytest.approx(5_000.0 * COST_RATE)
    assert order.realized_pnl == 0.0

    assert new["positions"]["BTCUSDT"] == pytest.approx(50.0)
    assert new["entry_prices"]["BTCUSDT"] == pytest.approx(100.0)
    assert new["cash"] == pytest.approx(10_000.0 - 5_000.0 * COST_RATE)
    assert new["total_fees_paid"] == pytest.approx(5_000.0 * COST_RATE)


def test_mark_to_market_uses_signed_quantity() -> None:
    state = new_state(10_000.0)
    state, _ = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)

    equity, unrealised, marks = mark_to_market(state, {"BTCUSDT": 110.0})
    assert marks["BTCUSDT"] == pytest.approx(110.0)
    assert unrealised == pytest.approx(50.0 * 10.0)
    assert equity == pytest.approx(state["cash"] + 500.0)

    short = new_state(10_000.0)
    short["positions"] = {"BTCUSDT": -50.0}
    short["entry_prices"] = {"BTCUSDT": 100.0}
    equity, unrealised, _ = mark_to_market(short, {"BTCUSDT": 90.0})
    assert unrealised == pytest.approx(500.0)
    assert equity == pytest.approx(10_500.0)


def test_partial_close_realises_only_the_closed_slice() -> None:
    state = new_state(10_000.0)
    state, _ = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)
    entry_before = state["entry_prices"]["BTCUSDT"]

    # Target 0.25 of equity at 110 -> 22.727 units, so ~27.27 units closed.
    new, orders = rebalance(state, {"BTCUSDT": 0.25}, {"BTCUSDT": 110.0}, 10_500.0)
    order = orders[0]
    assert order.side == "SELL"
    assert new["entry_prices"]["BTCUSDT"] == pytest.approx(entry_before)
    assert order.realized_pnl == pytest.approx(order.qty * (110.0 - 100.0))
    assert new["positions"]["BTCUSDT"] < 50.0


def test_full_close_removes_position_and_realises_pnl() -> None:
    state = new_state(10_000.0)
    state, _ = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)

    new, orders = rebalance(state, {"BTCUSDT": 0.0}, {"BTCUSDT": 90.0}, 9_500.0)
    assert len(orders) == 1
    assert orders[0].realized_pnl == pytest.approx(-500.0)
    assert "BTCUSDT" not in new["positions"]
    assert "BTCUSDT" not in new["entry_prices"]


def test_flip_realises_old_position_and_resets_entry() -> None:
    state = new_state(10_000.0)
    state, _ = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)

    new, orders = rebalance(state, {"BTCUSDT": -0.5}, {"BTCUSDT": 90.0}, 10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert orders[0].realized_pnl == pytest.approx(-500.0)
    assert new["entry_prices"]["BTCUSDT"] == pytest.approx(90.0)
    assert new["positions"]["BTCUSDT"] < 0.0


def test_increase_uses_notional_weighted_average_entry() -> None:
    state = new_state(10_000.0)
    state, _ = rebalance(state, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 10_000.0)
    old_qty = state["positions"]["BTCUSDT"]
    old_entry = state["entry_prices"]["BTCUSDT"]

    new, orders = rebalance(state, {"BTCUSDT": 1.0}, {"BTCUSDT": 110.0}, 10_000.0)
    added = orders[0].qty
    expected = (old_qty * old_entry + added * 110.0) / (old_qty + added)
    assert new["entry_prices"]["BTCUSDT"] == pytest.approx(expected)
    assert new["positions"]["BTCUSDT"] > old_qty


def test_funding_is_charged_on_signed_notional() -> None:
    long_state = new_state(10_000.0)
    long_state["positions"] = {"BTCUSDT": 50.0}
    long_state["entry_prices"] = {"BTCUSDT": 100.0}
    charged, cost = accrue_funding(long_state, {"BTCUSDT": 0.0001}, {"BTCUSDT": 100.0})
    assert cost == pytest.approx(0.5)
    assert charged["cash"] == pytest.approx(10_000.0 - 0.5)

    short_state = new_state(10_000.0)
    short_state["positions"] = {"BTCUSDT": -50.0}
    short_state["entry_prices"] = {"BTCUSDT": 100.0}
    credited, cost = accrue_funding(short_state, {"BTCUSDT": 0.0001}, {"BTCUSDT": 100.0})
    assert cost == pytest.approx(-0.5)
    assert credited["cash"] == pytest.approx(10_000.0 + 0.5)


def test_drawdown_guard_halts_once() -> None:
    state = new_state(10_000.0)
    state["equity"] = 6_900.0
    state["peak_equity"] = 10_000.0

    new, drawdown, breached = update_risk(state)
    assert drawdown == pytest.approx(0.31)
    assert breached is True
    assert new["halted"] is True
    assert "max_drawdown_guard" in new["halt_reason"]

    again, _, breached_again = update_risk(new)
    assert breached_again is False


def test_validate_targets_fails_closed() -> None:
    validate_targets({"BTCUSDT": 0.15, "ETHUSDT": -0.15})

    with pytest.raises(ValueError):
        validate_targets({"BTCUSDT": 0.16})
    with pytest.raises(ValueError):
        validate_targets({f"S{i}": 0.15 for i in range(21)})


def test_flatten_targets_covers_every_symbol() -> None:
    assert flatten_targets(["BTCUSDT", "ETHUSDT"]) == {
        "BTCUSDT": 0.0,
        "ETHUSDT": 0.0,
    }


def test_already_ran_today() -> None:
    now = datetime(2026, 11, 24, 0, 5, tzinfo=timezone.utc)
    assert already_ran_today("2026-11-24T00:05:00+00:00", now) is True
    assert already_ran_today("2026-11-23T00:05:00+00:00", now) is False
    assert already_ran_today(None, now) is False
    assert already_ran_today("not-a-date", now) is False

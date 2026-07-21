"""
Phase 5 of VALIDATION_REMEDIATION_PLAN.md — build a real readiness_<strategy>.json
gate file per strategy.

readiness_gate.py (Phase 0) only reads a gate file and fails closed if it's
missing, stale, or says not ready — it deliberately does not decide whether a
strategy IS ready. This module is the generator that produces the file it
reads, computed from that strategy's OWN recorded paper-trading track record
(trades.json for V1, trades_v4.json for V4), not from a backtest or an
in-sample number, and not from the other strategy's file.

Design notes:
- Per-trade returns are used (pnl / equity_before_trade) rather than a
  bar-indexed equity curve, because completed trades land at irregular times
  — there's no fixed bars/year to borrow from metrics_utils.infer_bars_per_year
  the way strategy.py/walk_forward.py do. Annualization instead uses the
  actual observed trade rate (completed trades / elapsed years) over the
  logged window. This is a rougher measure than a bar-based Sharpe and is
  documented as such in the output JSON.
- A walk-forward OOS summary (Phase 2's walk_forward_test.py output) is
  consulted if present at walk_forward_results_<strategy>.json, to flag
  paper-vs-backtest degradation. Its absence does not block the gate — it's
  recorded as "not evaluated," not silently treated as a pass.
- Fail-closed throughout: any exception while loading/parsing trades produces
  ready_for_live=False with the exception message as the reason, never a
  crash that could leave a stale (or absent-then-defaulted) file behind.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import GateThresholds, DEFAULT_GATE_THRESHOLDS


def compute_paper_metrics(trades: list, initial_capital: float = 1000.0) -> dict:
    """
    Compute paper-trading track record metrics from a trades.json-style list
    (ENTRY/EXIT dicts as written by paper_bot.py / paper_bot_v4.py /
    github_bot.py / github_bot_v4.py).

    Only EXIT records carry realized pnl; ENTRY records are ignored here.
    Returns a dict of metrics. All numeric fields are None if there are zero
    completed trades (nothing to compute), so callers must handle that rather
    than dividing by zero.
    """
    exits = sorted(
        (t for t in trades if t.get("type") == "EXIT"),
        key=lambda t: t["time"],
    )
    trade_count = len(exits)

    if trade_count == 0:
        return {
            "trade_count": 0,
            "win_rate": None,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": None,
            "first_trade_time": None,
            "last_trade_time": None,
            "span_days": 0.0,
        }

    per_trade_returns = []
    equity_curve = [initial_capital]
    wins = 0
    for t in exits:
        pnl = t["pnl"]
        equity_after = t["equity"]
        equity_before = equity_after - pnl
        if equity_before > 0:
            per_trade_returns.append(pnl / equity_before)
        if pnl > 0:
            wins += 1
        equity_curve.append(equity_after)

    win_rate = wins / trade_count

    # Max drawdown, peak-to-trough, over the realized paper equity curve.
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

    final_equity = equity_curve[-1]
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0

    first_time = datetime.fromisoformat(exits[0]["time"])
    last_time = datetime.fromisoformat(exits[-1]["time"])
    span_days = max((last_time - first_time).total_seconds() / 86400.0, 0.0)
    span_years = span_days / 365.25

    sharpe = None
    if len(per_trade_returns) >= 2 and span_years > 0:
        mean_r = sum(per_trade_returns) / len(per_trade_returns)
        variance = sum((r - mean_r) ** 2 for r in per_trade_returns) / (len(per_trade_returns) - 1)
        std_r = variance ** 0.5
        trades_per_year = trade_count / span_years
        if std_r > 0:
            sharpe = (mean_r / std_r) * (trades_per_year ** 0.5)

    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "first_trade_time": exits[0]["time"],
        "last_trade_time": exits[-1]["time"],
        "span_days": span_days,
    }


def load_walk_forward_summary(strategy_name: str) -> dict | None:
    """
    Optional: walk_forward_test.py (Phase 2) can drop a summary at
    walk_forward_results_<strategy>.json. If present and it has an
    'oos_sharpe' field, degradation vs the paper Sharpe is recorded. Absence
    is NOT an error and does NOT block the gate — it's reported as
    unevaluated so the gate file is honest about what it did and didn't check.
    """
    path = Path(f"walk_forward_results_{strategy_name}.json")
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def evaluate_gate(metrics: dict, thresholds: GateThresholds, walk_forward: dict | None) -> tuple[bool, list]:
    """
    Fail-closed evaluation: starts from ready=True and every unmet condition
    both flips it to False and adds a specific, human-readable reason. No
    single check can flip it back to True.
    """
    reasons = []
    ready = True

    if metrics["trade_count"] < thresholds.min_trades:
        ready = False
        reasons.append(
            f"only {metrics['trade_count']} completed paper trades recorded "
            f"(need >= {thresholds.min_trades})"
        )

    if metrics["sharpe"] is None:
        ready = False
        reasons.append("Sharpe could not be computed (insufficient trades or zero variance)")
    elif metrics["sharpe"] < thresholds.min_sharpe:
        ready = False
        reasons.append(
            f"annualized paper Sharpe {metrics['sharpe']:.2f} < required {thresholds.min_sharpe}"
        )

    if metrics["total_return_pct"] < thresholds.min_total_return_pct:
        ready = False
        reasons.append(
            f"total paper return {metrics['total_return_pct']:.1f}% is below the "
            f"{thresholds.min_total_return_pct:.1f}% floor (i.e. net negative)"
        )

    if metrics["max_drawdown_pct"] > thresholds.max_drawdown_pct:
        ready = False
        reasons.append(
            f"max paper drawdown {metrics['max_drawdown_pct']:.1f}% exceeds "
            f"{thresholds.max_drawdown_pct:.1f}% limit"
        )

    if walk_forward is not None and "oos_sharpe" in walk_forward and metrics["sharpe"] is not None:
        oos_sharpe = walk_forward["oos_sharpe"]
        if oos_sharpe > 0 and metrics["sharpe"] < 0.5 * oos_sharpe:
            ready = False
            reasons.append(
                f"paper Sharpe {metrics['sharpe']:.2f} degrades more than 50% vs "
                f"walk-forward OOS Sharpe {oos_sharpe:.2f}"
            )
    else:
        reasons.append("walk-forward OOS comparison not evaluated (no walk_forward_results file found)")

    if ready:
        reasons.insert(0, "all gate thresholds met")

    return ready, reasons


def build_gate_file(strategy_name: str, trades_path: str, output_path: str,
                     thresholds: GateThresholds = DEFAULT_GATE_THRESHOLDS,
                     initial_capital: float = 1000.0) -> dict:
    """
    Load trades_path, compute metrics, evaluate against thresholds, and write
    output_path. Returns the gate dict that was written. Fail-closed: any
    exception while loading/parsing trades yields ready_for_live=False rather
    than propagating and potentially leaving a stale file in place.
    """
    try:
        with open(trades_path, "r") as f:
            trades = json.load(f)
        metrics = compute_paper_metrics(trades, initial_capital=initial_capital)
        walk_forward = load_walk_forward_summary(strategy_name)
        ready, reasons = evaluate_gate(metrics, thresholds, walk_forward)
    except Exception as e:
        metrics = {}
        ready = False
        reasons = [f"failed to load/evaluate {trades_path}: {e}"]

    gate = {
        "ready_for_live": ready,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": {
            "min_trades": thresholds.min_trades,
            "min_sharpe": thresholds.min_sharpe,
            "min_total_return_pct": thresholds.min_total_return_pct,
            "max_drawdown_pct": thresholds.max_drawdown_pct,
        },
    }

    with open(output_path, "w") as f:
        json.dump(gate, f, indent=2)

    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["v1", "v4"], help="Build a single strategy's gate")
    parser.add_argument("--trades", help="Path to that strategy's trades JSON (required with --strategy)")
    parser.add_argument("--all", action="store_true", help="Build both v1 (trades.json) and v4 (trades_v4.json)")
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = [("v1", "trades.json", "readiness_v1.json"),
                   ("v4", "trades_v4.json", "readiness_v4.json")]
    elif args.strategy:
        if not args.trades:
            parser.error("--trades is required when using --strategy")
        targets = [(args.strategy, args.trades, f"readiness_{args.strategy}.json")]
    else:
        parser.error("pass --all or --strategy/--trades")

    exit_code = 0
    for strategy_name, trades_path, output_path in targets:
        gate = build_gate_file(strategy_name, trades_path, output_path)
        status = "READY" if gate["ready_for_live"] else "NOT READY"
        print(f"[{strategy_name}] {status} -> {output_path}")
        for r in gate["reasons"]:
            print(f"    - {r}")
        if not gate["ready_for_live"]:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

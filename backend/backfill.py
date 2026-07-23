"""
One-Time Backfill Migration Script: JSON Files -> SQLite Database
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import sys

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"

from backend.models import init_db, SessionLocal, Strategy, Trade, EquitySnapshot, ReadinessGate, BacktestRun

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill")


def parse_timestamp(ts_val):
    if not ts_val:
        return datetime.now(timezone.utc)
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc) if ts_val > 1e11 else datetime.fromtimestamp(ts_val, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def resolve_file(file_name: str) -> Path:
    p1 = DATA_DIR / file_name
    if p1.exists():
        return p1
    return ROOT_DIR / file_name


def backfill():
    init_db()
    db = SessionLocal()

    logger.info("Seeding strategies...")
    strategies_data = [
        {"id": "v1", "name": "V1 Turtle-Donchian Strategy", "timeframe": "4h"},
        {"id": "v4", "name": "V4 High-Frequency Strategy", "timeframe": "1h"}
    ]
    for s_data in strategies_data:
        existing = db.query(Strategy).filter(Strategy.id == s_data["id"]).first()
        if not existing:
            db.add(Strategy(id=s_data["id"], name=s_data["name"], timeframe=s_data["timeframe"]))
    db.commit()

    # 1. Backfill Trades
    trade_sources = [
        ("v1", resolve_file("trades.json")),
        ("v4", resolve_file("trades_v4.json"))
    ]
    for strat_id, file_path in trade_sources:
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    trades_list = json.load(f)
                logger.info(f"Importing {len(trades_list)} trades for strategy {strat_id} from {file_path}")
                for t in trades_list:
                    t_type = t.get("type", "BUY_ENTRY" if "BUY" in str(t).upper() else "SELL_EXIT")
                    db.add(Trade(
                        strategy_id=strat_id,
                        trade_type=t_type,
                        price=float(t.get("price", 0.0)),
                        quantity=float(t.get("size", t.get("quantity", 0.0))),
                        pnl=float(t["pnl"]) if "pnl" in t else None,
                        reason=t.get("reason", "Backfilled from JSON"),
                        timestamp=parse_timestamp(t.get("time", t.get("timestamp")))
                    ))
            except Exception as e:
                logger.error(f"Error backfilling {file_path}: {e}")
    db.commit()

    # 2. Backfill Bot States (Equity Snapshots)
    state_sources = [
        ("v1", resolve_file("bot_state.json")),
        ("v4", resolve_file("bot_state_v4.json"))
    ]
    for strat_id, file_path in state_sources:
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    state_data = json.load(f)
                logger.info(f"Importing equity snapshot for strategy {strat_id} from {file_path}")
                db.add(EquitySnapshot(
                    strategy_id=strat_id,
                    equity=float(state_data.get("equity", 1000.0)),
                    position_size=float(state_data.get("position_size", 0.0)),
                    position_units_json=json.dumps(state_data.get("position_units", [])),
                    timestamp=parse_timestamp(state_data.get("last_run"))
                ))
            except Exception as e:
                logger.error(f"Error backfilling {file_path}: {e}")
    db.commit()

    # 3. Backfill Readiness Gates
    gate_sources = [
        ("v1", resolve_file("readiness_v1.json")),
        ("v4", resolve_file("readiness_v4.json"))
    ]
    for strat_id, file_path in gate_sources:
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    gate_data = json.load(f)
                logger.info(f"Importing readiness gate for strategy {strat_id} from {file_path}")
                db.add(ReadinessGate(
                    strategy_id=strat_id,
                    ready_for_live=bool(gate_data.get("ready_for_live", False)),
                    sharpe_ratio=float(gate_data.get("paper_sharpe", 0.0)) if gate_data.get("paper_sharpe") else None,
                    win_rate=float(gate_data.get("paper_win_rate", 0.0)) if gate_data.get("paper_win_rate") else None,
                    checks_json=json.dumps(gate_data.get("checks", [])),
                    timestamp=parse_timestamp(gate_data.get("timestamp"))
                ))
            except Exception as e:
                logger.error(f"Error backfilling {file_path}: {e}")
    db.commit()

    # 4. Backfill Walk-Forward / Backtest Runs
    wf_sources = [
        ("v1", resolve_file("walk_forward_results_v1.json")),
        ("v4", resolve_file("walk_forward_results_v4.json"))
    ]
    for strat_id, file_path in wf_sources:
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    wf_data = json.load(f)
                logger.info(f"Importing backtest run for strategy {strat_id} from {file_path}")
                summary = wf_data.get("summary", wf_data)
                db.add(BacktestRun(
                    strategy_id=strat_id,
                    total_return=float(summary.get("total_return", 0.0)) if isinstance(summary, dict) and "total_return" in summary else None,
                    sharpe_ratio=float(summary.get("sharpe", 0.0)) if isinstance(summary, dict) and "sharpe" in summary else None,
                    max_drawdown=float(summary.get("max_drawdown", 0.0)) if isinstance(summary, dict) and "max_drawdown" in summary else None,
                    results_json=json.dumps(wf_data),
                    timestamp=datetime.now(timezone.utc)
                ))
            except Exception as e:
                logger.error(f"Error backfilling {file_path}: {e}")
    db.commit()
    db.close()
    logger.info("Backfill migration completed successfully!")


if __name__ == "__main__":
    backfill()

"""
Database Models and Session Management for Paper-Trading-Bot
Supports SQLite (default local) and PostgreSQL / TimescaleDB.
"""
import os
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from pathlib import Path

# Data directory path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "paper_trading.db"

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(String, primary_key=True, index=True)  # e.g., 'v1', 'v4'
    name = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    config_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    trades = relationship("Trade", back_populates="strategy", cascade="all, delete-orphan")
    equity_snapshots = relationship("EquitySnapshot", back_populates="strategy", cascade="all, delete-orphan")
    readiness_gates = relationship("ReadinessGate", back_populates="strategy", cascade="all, delete-orphan")
    backtest_runs = relationship("BacktestRun", back_populates="strategy", cascade="all, delete-orphan")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    trade_type = Column(String, nullable=False)  # BUY_ENTRY, SELL_EXIT
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    strategy = relationship("Strategy", back_populates="trades")


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    position_units_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    strategy = relationship("Strategy", back_populates="equity_snapshots")


class ReadinessGate(Base):
    __tablename__ = "readiness_gates"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    ready_for_live = Column(Boolean, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    checks_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    strategy = relationship("Strategy", back_populates="readiness_gates")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    total_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    deflated_sharpe = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    results_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    strategy = relationship("Strategy", back_populates="backtest_runs")


class SystemEventLog(Base):
    __tablename__ = "system_event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True, index=True)
    category = Column(String, nullable=False, index=True)  # SYSTEM, EXECUTION, RISK, MARKET
    message = Column(Text, nullable=False)
    level = Column(String, default="INFO")  # INFO, WARNING, SUCCESS, CRITICAL
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ExecutionPipelineStep(Base):
    __tablename__ = "execution_pipeline_steps"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    signal_generated_at = Column(DateTime, nullable=True)
    risk_passed_at = Column(DateTime, nullable=True)
    size_calculated_at = Column(DateTime, nullable=True)
    order_submitted_at = Column(DateTime, nullable=True)
    order_filled_at = Column(DateTime, nullable=True)
    position_opened_at = Column(DateTime, nullable=True)
    sl_armed_at = Column(DateTime, nullable=True)
    tp_set_at = Column(DateTime, nullable=True)
    trailing_active_at = Column(DateTime, nullable=True)
    current_stage = Column(String, default="IDLE")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class StrategyHealthSnapshot(Base):
    """
    NOTE: nothing currently computes these values — no job populates this
    table yet. Columns are nullable with no default on purpose: a missing
    row (or a null field) should read as "not yet computed" in the API/UI,
    not silently show a plausible-looking placeholder number. Previously
    these had hardcoded defaults (core_stability_pct=98.0, confidence_level_pct=81.0,
    market_regime="TRENDING", etc.) that had no real computation behind them.
    """
    __tablename__ = "strategy_health_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    core_stability_pct = Column(Float, nullable=True)
    confidence_level_pct = Column(Float, nullable=True)
    market_regime = Column(String, nullable=True)
    volatility_regime = Column(String, nullable=True)
    liquidity_regime = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class CapitalAllocation(Base):
    """
    NOTE: same caveat as StrategyHealthSnapshot — nothing populates this
    table yet. total_equity should come from a real EquitySnapshot, and the
    per-asset percentages need a real computation once/if the bot trades
    more than one asset. No defaults on purpose.
    """
    __tablename__ = "capital_allocations"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False, index=True)
    btc_pct = Column(Float, nullable=True)
    eth_pct = Column(Float, nullable=True)
    sol_pct = Column(Float, nullable=True)
    usdt_pct = Column(Float, nullable=True)
    others_pct = Column(Float, nullable=True)
    total_equity = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


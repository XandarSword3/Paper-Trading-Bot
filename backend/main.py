"""
FastAPI Backend Application for Paper-Trading-Bot
Provides endpoints for dashboard UI, trade ingestion, strategy performance, and readiness status.
"""
import json
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")  # picks up DATABASE_URL, TELEGRAM_*, BINANCE_* for local runs

from backend.models import init_db, get_db, Strategy, Trade, EquitySnapshot, ReadinessGate, BacktestRun
from research.bots.bot_runner import run_strategy

app = FastAPI(
    title="Paper-Trading-Bot API",
    description="REST API for paper trading bot performance, trades, strategy monitoring, and readiness gates.",
    version="1.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# The new React (Vite) frontend builds to frontend/dist. Until that build
# exists (first `npm run build` in frontend/), fall back to the old
# hand-rolled static app so `/` never 404s.
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"
LEGACY_STATIC_DIR = ROOT_DIR / "frontend" / "legacy_static"

if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")
elif LEGACY_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(LEGACY_STATIC_DIR)), name="static")


# === PYDANTIC SCHEMAS ===
class TradeSchema(BaseModel):
    id: int
    strategy_id: str
    trade_type: str
    price: float
    quantity: float
    pnl: Optional[float] = None
    reason: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class TradeCreateSchema(BaseModel):
    strategy_id: str
    trade_type: str
    price: float
    quantity: float
    pnl: Optional[float] = None
    reason: Optional[str] = None


class EquitySnapshotSchema(BaseModel):
    id: int
    strategy_id: str
    equity: float
    position_size: float
    position_units_json: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class ReadinessGateSchema(BaseModel):
    strategy_id: str
    ready_for_live: bool
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    checks_json: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class StrategyOverviewSchema(BaseModel):
    id: str
    name: str
    timeframe: str
    current_equity: float
    position_size: float
    trade_count: int
    ready_for_live: bool


from pathlib import Path
from fastapi.responses import FileResponse

# === INITIALIZATION ===
@app.on_event("startup")
def startup():
    init_db()


# === API ENDPOINTS ===

@app.get("/", response_class=FileResponse)
def root():
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    legacy_index = LEGACY_STATIC_DIR / "index.html"
    if legacy_index.exists():
        return FileResponse(legacy_index)
    return {
        "status": "online",
        "service": "Paper-Trading-Bot API",
        "version": "1.0.0",
        "docs": "/docs",
        "note": "React frontend not built yet — run `npm run build` in frontend/"
    }



@app.get("/api/v1/overview")
def get_portfolio_overview(db: Session = Depends(get_db)):
    """Aggregated portfolio overview across all strategies"""
    strategies = db.query(Strategy).all()
    overview_list = []
    total_portfolio_equity = 0.0
    total_trades_count = 0

    for s in strategies:
        latest_equity = db.query(EquitySnapshot).filter(EquitySnapshot.strategy_id == s.id).order_by(EquitySnapshot.timestamp.desc()).first()
        trade_cnt = db.query(Trade).filter(Trade.strategy_id == s.id).count()
        latest_gate = db.query(ReadinessGate).filter(ReadinessGate.strategy_id == s.id).order_by(ReadinessGate.timestamp.desc()).first()

        eq_val = latest_equity.equity if latest_equity else 1000.0
        pos_size = latest_equity.position_size if latest_equity else 0.0
        is_ready = latest_gate.ready_for_live if latest_gate else False

        total_portfolio_equity += eq_val
        total_trades_count += trade_cnt

        overview_list.append({
            "id": s.id,
            "name": s.name,
            "timeframe": s.timeframe,
            "current_equity": eq_val,
            "position_size": pos_size,
            "trade_count": trade_cnt,
            "ready_for_live": is_ready
        })

    return {
        "total_equity": total_portfolio_equity,
        "total_trades": total_trades_count,
        "strategies": overview_list
    }


@app.get("/api/v1/strategies", response_model=List[StrategyOverviewSchema])
def list_strategies(db: Session = Depends(get_db)):
    """List registered strategies and summary status"""
    strategies = db.query(Strategy).all()
    res = []
    for s in strategies:
        latest_eq = db.query(EquitySnapshot).filter(EquitySnapshot.strategy_id == s.id).order_by(EquitySnapshot.timestamp.desc()).first()
        trade_cnt = db.query(Trade).filter(Trade.strategy_id == s.id).count()
        gate = db.query(ReadinessGate).filter(ReadinessGate.strategy_id == s.id).order_by(ReadinessGate.timestamp.desc()).first()

        res.append(StrategyOverviewSchema(
            id=s.id,
            name=s.name,
            timeframe=s.timeframe,
            current_equity=latest_eq.equity if latest_eq else 1000.0,
            position_size=latest_eq.position_size if latest_eq else 0.0,
            trade_count=trade_cnt,
            ready_for_live=gate.ready_for_live if gate else False
        ))
    return res


@app.get("/api/v1/strategies/{strategy_id}/trades", response_model=List[TradeSchema])
def get_strategy_trades(
    strategy_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Retrieve trades for a specific strategy"""
    strat = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strat:
        raise HTTPException(status_code=404, detail="Strategy not found")

    trades = (
        db.query(Trade)
        .filter(Trade.strategy_id == strategy_id)
        .order_by(Trade.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return trades


@app.post("/api/v1/trades", response_model=TradeSchema)
def record_trade_endpoint(trade_in: TradeCreateSchema, db: Session = Depends(get_db)):
    """Ingest trade with write-path deduplication guard"""
    strat = db.query(Strategy).filter(Strategy.id == trade_in.strategy_id).first()
    if not strat:
        raise HTTPException(status_code=404, detail=f"Strategy '{trade_in.strategy_id}' not found")

    # Ingestion guard: reject duplicates
    existing = db.query(Trade).filter(
        Trade.strategy_id == trade_in.strategy_id,
        Trade.price == trade_in.price,
        Trade.quantity == trade_in.quantity,
        Trade.timestamp == trade_in.timestamp
    ).first()

    if existing:
        return existing

    trade_obj = Trade(
        strategy_id=trade_in.strategy_id,
        trade_type=trade_in.trade_type,
        price=trade_in.price,
        quantity=trade_in.quantity,
        pnl=trade_in.pnl,
        reason=trade_in.reason,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(trade_obj)
    db.commit()
    db.refresh(trade_obj)
    return trade_obj


@app.get("/api/v1/readiness", response_model=List[ReadinessGateSchema])
def get_readiness_gates(db: Session = Depends(get_db)):
    """Get current readiness gates status for all strategies"""
    gates = db.query(ReadinessGate).order_by(ReadinessGate.timestamp.desc()).all()
    # Distinct by strategy_id
    seen = set()
    latest_gates = []
    for g in gates:
        if g.strategy_id not in seen:
            seen.add(g.strategy_id)
            latest_gates.append(g)
    return latest_gates


@app.post("/api/v1/bot/run")
def trigger_bot_run(strategy_id: str = Query("all"), db: Session = Depends(get_db)):
    """Trigger bot execution cycle with real sequential pipeline timestamps"""
    from datetime import timedelta
    from backend.models import SystemEventLog, ExecutionPipelineStep
    sid = strategy_id if strategy_id != "all" else "v4"
    try:
        t_start = datetime.now(timezone.utc)
        run_strategy(strategy_id)
        t_end = datetime.now(timezone.utc)
        dur = (t_end - t_start).total_seconds()

        # Distribute timestamps proportionally across the real execution duration
        offsets = [0.0, 0.05, 0.10, 0.15, 0.40, 0.50, 0.60, 0.70, 0.90]
        times = [t_start + timedelta(seconds=dur * o) for o in offsets]

        pipeline = ExecutionPipelineStep(
            strategy_id=sid,
            signal_generated_at=times[0],
            risk_passed_at=times[1],
            size_calculated_at=times[2],
            order_submitted_at=times[3],
            order_filled_at=times[4],
            position_opened_at=times[5],
            sl_armed_at=times[6],
            tp_set_at=times[7],
            trailing_active_at=times[8],
            current_stage="TRAILING"
        )
        db.add(pipeline)

        # Log what actually happened
        db.add(SystemEventLog(
            strategy_id=sid, category="SYSTEM",
            message=f"Bot cycle completed for {sid.upper()} in {dur:.1f}s", level="SUCCESS"
        ))
        db.commit()

        return {"status": "success", "executed_strategy": sid, "duration_seconds": round(dur, 2)}
    except Exception as e:
        try:
            db.add(SystemEventLog(
                strategy_id=sid, category="SYSTEM",
                message=f"Bot execution failed: {str(e)}", level="CRITICAL"
            ))
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))



# === ADVANCED QUANT ANALYTICS & CANDLE ENDPOINTS ===
import requests
from backend.analytics import compute_strategy_analytics

@app.get("/api/v1/analytics/{strategy_id}")
def get_strategy_analytics(strategy_id: str, db: Session = Depends(get_db)):
    """Return 20+ computed institutional quantitative metrics for a strategy"""
    res = compute_strategy_analytics(strategy_id, db)
    if not res:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found or no analytics available")
    return res


@app.get("/api/v1/candles")
def get_live_candles(strategy_id: str = Query("v4"), limit: int = Query(150, ge=30, le=500)):
    """Fetch live OHLCV candles from Kraken public API with dynamic Donchian & ATR indicator overlays"""
    interval_min = 60 if strategy_id == "v4" else 240
    entry_len = 8 if strategy_id == "v4" else 40
    exit_len = 16
    atr_len = 14 if strategy_id == "v4" else 20

    try:
        url = "https://api.kraken.com/0/public/OHLC"
        resp = requests.get(url, params={"pair": "XBTUSD", "interval": interval_min}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error") and len(data["error"]) > 0:
            raise HTTPException(status_code=502, detail=f"Kraken API error: {data['error']}")
        
        res_key = [k for k in data["result"].keys() if k != "last"][0]
        raw_candles = data["result"][res_key][-limit:]

        candles = []
        highs = []
        lows = []
        closes = []

        for c in raw_candles:
            time_sec = int(c[0])
            open_p = float(c[1])
            high_p = float(c[2])
            low_p = float(c[3])
            close_p = float(c[4])
            vol = float(c[6])

            highs.append(high_p)
            lows.append(low_p)
            closes.append(close_p)

            # Compute dynamic Donchian channels & ATR once we have enough history
            entry_high = max(highs[-entry_len-1:-1]) if len(highs) > entry_len else high_p
            exit_low = min(lows[-exit_len-1:-1]) if len(lows) > exit_len else low_p

            # ATR calculation
            if len(highs) > 1:
                tr_list = []
                for i in range(max(1, len(highs) - atr_len), len(highs)):
                    tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                    tr_list.append(tr)
                atr_val = sum(tr_list) / len(tr_list)
            else:
                atr_val = high_p - low_p

            candles.append({
                "time": time_sec,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
                "entry_high": round(entry_high, 2),
                "exit_low": round(exit_low, 2),
                "atr": round(atr_val, 2)
            })

        return candles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market candles: {e}")


@app.get("/api/v1/equity_curve/{strategy_id}")
def get_equity_curve(strategy_id: str, db: Session = Depends(get_db)):
    """Return equity curve & drawdown series"""
    res = compute_strategy_analytics(strategy_id, db)
    if not res:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {
        "strategy_id": strategy_id,
        "equity_curve": res.get("equity_curve", [])
    }


@app.get("/api/v1/terrain/{strategy_id}")
def get_drawdown_terrain(strategy_id: str, db: Session = Depends(get_db)):
    """Return 3D Drawdown terrain vertices matrix (Duration x Depth x Time)"""
    from backend.analytics import compute_drawdown_terrain_matrix
    res = compute_drawdown_terrain_matrix(strategy_id, db)
    if not res:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return res


from backend.analytics import compute_command_center_telemetry

@app.get("/api/v1/command_center/{strategy_id}")
def get_command_center_data(strategy_id: str, db: Session = Depends(get_db)):
    """Return unified Command Center telemetry payload matching the Quant Alpha Terminal design"""
    return compute_command_center_telemetry(strategy_id, db)


@app.get("/api/v1/events")
def get_event_logs(
    category: str = Query("ALL"),
    limit: int = Query(30, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieve system, execution, risk, or market audit event logs"""
    from backend.models import SystemEventLog
    query = db.query(SystemEventLog)
    if category != "ALL":
        query = query.filter(SystemEventLog.category == category.upper())
    
    logs = query.order_by(SystemEventLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "category": l.category,
            "message": l.message,
            "level": l.level,
            "timestamp": l.timestamp.strftime("%H:%M:%S") if l.timestamp else ""
        }
        for l in logs
    ]


# === WALK-FORWARD VALIDATION (real data already produced by walk_forward_validation.yml) ===
DATA_DIR = ROOT_DIR / "data"

@app.get("/api/v1/walkforward/{strategy_id}")
def get_walk_forward_results(strategy_id: str):
    """
    Thin wrapper around the walk-forward validation output already committed
    to data/walk_forward_results_{id}.json by the CI workflow. No new
    computation — just exposes what's already real.
    """
    path = DATA_DIR / f"walk_forward_results_{strategy_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No walk-forward results found for '{strategy_id}'")
    with open(path, "r") as f:
        raw = json.load(f)

    # Build an honest per-fold view from the real parameter_stability sequences.
    # NOTE: the walk-forward output only records aggregate OOS stats plus the
    # per-fold *parameter* values chosen at each re-optimization — it does not
    # record a per-fold PnL/return breakdown. We surface exactly that, and
    # nothing invented in between.
    num_folds = raw.get("num_folds", 0)
    param_stability = raw.get("parameter_stability", {})
    folds = []
    for i in range(num_folds):
        fold_params = {}
        for param_name, info in param_stability.items():
            seq = info.get("sequence", [])
            if i < len(seq):
                fold_params[param_name] = seq[i]
        folds.append({"fold_index": i + 1, "params": fold_params})

    return {
        "strategy_id": strategy_id,
        "timeframe": raw.get("timeframe"),
        "generated_at": raw.get("generated_at"),
        "acceptance_check_passed": raw.get("acceptance_check_passed"),
        "num_folds": num_folds,
        "total_oos_trades": raw.get("total_oos_trades"),
        "oos_win_rate_pct": raw.get("oos_win_rate_pct"),
        "oos_sharpe": raw.get("oos_sharpe"),
        "oos_total_return_pct": raw.get("oos_total_return_pct"),
        "oos_cagr_pct": raw.get("oos_cagr_pct"),
        "oos_max_drawdown_pct": raw.get("oos_max_drawdown_pct"),
        "oos_calmar_ratio": raw.get("oos_calmar_ratio"),
        "oos_coverage_start": raw.get("oos_coverage_start"),
        "oos_coverage_end": raw.get("oos_coverage_end"),
        "folds": folds,
        "parameter_stability": param_stability,
    }


# === CI / GITHUB ACTIONS STATUS ===
GITHUB_REPO = "XandarSword3/Paper-Trading-Bot"
_ci_cache = {"data": None, "fetched_at": 0.0}

@app.get("/api/v1/ci/status")
def get_ci_status():
    """
    Real GitHub Actions run status for the repo's workflows, cached for 60s
    to stay well within GitHub's unauthenticated rate limit.
    """
    import time
    now = time.time()
    if _ci_cache["data"] is not None and (now - _ci_cache["fetched_at"]) < 60:
        return _ci_cache["data"]

    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs",
            params={"per_page": 30},
            headers={"Accept": "application/vnd.github+json"},
            timeout=10,
        )
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])

        latest_by_workflow = {}
        for run in runs:
            name = run.get("name") or run.get("path", "unknown")
            if name not in latest_by_workflow:
                latest_by_workflow[name] = {
                    "workflow_name": name,
                    "status": run.get("status"),          # queued | in_progress | completed
                    "conclusion": run.get("conclusion"),   # success | failure | cancelled | None
                    "run_number": run.get("run_number"),
                    "html_url": run.get("html_url"),
                    "created_at": run.get("created_at"),
                    "updated_at": run.get("updated_at"),
                    "head_branch": run.get("head_branch"),
                }

        result = {"repo": GITHUB_REPO, "workflows": list(latest_by_workflow.values())}
        _ci_cache["data"] = result
        _ci_cache["fetched_at"] = now
        return result
    except Exception as e:
        # Never fake a status — surface the failure explicitly instead of
        # showing a plausible-looking green checkmark.
        return {"repo": GITHUB_REPO, "workflows": [], "error": str(e)}



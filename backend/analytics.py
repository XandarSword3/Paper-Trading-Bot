"""
Quantitative Analytics Engine for Paper-Trading-Bot
Computes institutional diagnostic metrics, bootstrap confidence intervals,
underwater drawdown time-series, rolling metrics, and duration scatter data directly from database records.
"""
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from backend.models import Strategy, Trade, EquitySnapshot, ReadinessGate


def compute_strategy_analytics(strategy_id: str, db: Session) -> Dict[str, Any]:
    """
    Compute comprehensive institutional quantitative metrics and diagnostic arrays for a given strategy.
    """
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        return {}

    # Fetch unique trades ordered by timestamp
    trades = (
        db.query(Trade)
        .filter(Trade.strategy_id == strategy_id)
        .order_by(Trade.timestamp.asc())
        .all()
    )

    # Deduplicate trades in Python in case uncommitted memory objects are queried
    seen_keys = set()
    unique_trades = []
    for t in trades:
        key = (t.price, t.quantity, t.timestamp.isoformat() if t.timestamp else '')
        if key not in seen_keys:
            seen_keys.add(key)
            unique_trades.append(t)

    # Fetch equity snapshots
    snapshots = (
        db.query(EquitySnapshot)
        .filter(EquitySnapshot.strategy_id == strategy_id)
        .order_by(EquitySnapshot.timestamp.asc())
        .all()
    )

    latest_gate = (
        db.query(ReadinessGate)
        .filter(ReadinessGate.strategy_id == strategy_id)
        .order_by(ReadinessGate.timestamp.desc())
        .first()
    )

    initial_capital = 1000.0
    current_equity = snapshots[-1].equity if snapshots else initial_capital
    position_size = snapshots[-1].position_size if snapshots else 0.0

    # 1. Closed Trades & PnL Metrics
    closed_trades = [t for t in unique_trades if t.pnl is not None]
    total_closed_count = len(closed_trades)

    pnls = [t.pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_closed_count * 100.0) if total_closed_count > 0 else 0.0
    loss_rate = (loss_count / total_closed_count * 100.0) if total_closed_count > 0 else 0.0

    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (gross_wins if gross_wins > 0 else 1.0)

    avg_win = (sum(wins) / win_count) if win_count > 0 else 0.0
    avg_loss = (abs(sum(losses)) / loss_count) if loss_count > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else avg_win

    expectancy = ((win_rate / 100.0) * avg_win) - ((loss_rate / 100.0) * avg_loss)
    net_realized_pnl = sum(pnls) if pnls else 0.0

    # Streak counts
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    curr_w = 0
    curr_l = 0
    for p in pnls:
        if p > 0:
            curr_w += 1
            curr_l = 0
            max_consecutive_wins = max(max_consecutive_wins, curr_w)
        elif p < 0:
            curr_l += 1
            curr_w = 0
            max_consecutive_losses = max(max_consecutive_losses, curr_l)

    # 2. Equity Curve & Underwater Drawdown Series
    running_equity = initial_capital
    running_peak = initial_capital
    equity_curve = []
    max_drawdown_amount = 0.0
    max_drawdown_pct = 0.0
    drawdown_durations = []
    dd_start_time = None

    if closed_trades:
        first_time = closed_trades[0].timestamp
        last_time = closed_trades[-1].timestamp
        total_days = max((last_time - first_time).total_seconds() / 86400.0, 1.0) if (first_time and last_time) else 365.0
    else:
        total_days = 365.0

    total_return_pct = ((current_equity - initial_capital) / initial_capital) * 100.0
    cagr = (((current_equity / initial_capital) ** (365.0 / total_days)) - 1.0) * 100.0 if (current_equity > 0 and total_days > 0) else 0.0

    # Build underwater equity series from closed trade timestamps
    for t in closed_trades:
        running_equity += t.pnl
        if running_equity > running_peak:
            running_peak = running_equity
            if dd_start_time and t.timestamp:
                dur_hrs = (t.timestamp - dd_start_time).total_seconds() / 3600.0
                drawdown_durations.append(dur_hrs)
                dd_start_time = None
        else:
            if dd_start_time is None and t.timestamp:
                dd_start_time = t.timestamp

        dd_amt = running_peak - running_equity
        dd_pct = (dd_amt / running_peak * 100.0) if running_peak > 0 else 0.0
        max_drawdown_amount = max(max_drawdown_amount, dd_amt)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)

        equity_curve.append({
            "timestamp": t.timestamp.isoformat() if t.timestamp else "",
            "equity": round(running_equity, 2),
            "peak": round(running_peak, 2),
            "drawdown_pct": round(-dd_pct, 2)  # negative representation for underwater shading
        })

    max_drawdown_duration_hrs = max(drawdown_durations) if drawdown_durations else 0.0

    # 3. Statistical Risk Metrics (Sharpe, Sortino, VaR, CVaR, Bootstrap CI)
    trade_pct_returns = [p / initial_capital for p in pnls]

    if trade_pct_returns:
        rets_arr = np.array(trade_pct_returns)
        mean_ret = np.mean(rets_arr)
        std_ret = np.std(rets_arr, ddof=1) if len(rets_arr) > 1 else 0.01

        ann_factor = math.sqrt(len(rets_arr))
        sharpe_val = float(mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0
        sharpe_ratio = round(sharpe_val, 2)

        downside_returns = rets_arr[rets_arr < 0]
        downside_std = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else (abs(float(np.mean(downside_returns))) if len(downside_returns) > 0 else 0.01)
        sortino_ratio = round((float(mean_ret) / downside_std * ann_factor) if downside_std > 0 else sharpe_val, 2)

        calmar_ratio = round(float((total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0), 2)

        # Bootstrap 95% Confidence Interval for Sharpe (1,000 resamples)
        boot_sharpes = []
        for _ in range(1000):
            sample = np.random.choice(rets_arr, size=len(rets_arr), replace=True)
            s_std = np.std(sample, ddof=1)
            if s_std > 0:
                boot_sharpes.append((np.mean(sample) / s_std) * ann_factor)

        if boot_sharpes:
            ci_low = float(np.percentile(boot_sharpes, 5))
            ci_high = float(np.percentile(boot_sharpes, 95))
            sharpe_ci = [round(ci_low, 2), round(ci_high, 2)]
        else:
            sharpe_ci = [sharpe_ratio, sharpe_ratio]

        # VaR 95% & CVaR 95%
        perc_5 = float(np.percentile(rets_arr, 5))
        var_95 = round(abs(perc_5) * 100.0, 2)
        var_99 = round(abs(float(np.percentile(rets_arr, 1))) * 100.0, 2)

        tail_5 = rets_arr[rets_arr <= perc_5]
        cvar_95 = round(abs(float(np.mean(tail_5))) * 100.0, 2) if len(tail_5) > 0 else var_95
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        calmar_ratio = 0.0
        sharpe_ci = [0.0, 0.0]
        var_95 = 0.0
        var_99 = 0.0
        cvar_95 = 0.0

    # 4. Diagnostic Series Generation
    # Rolling 30-Trade Win Rate & Sharpe
    rolling_metrics = []
    window = 30
    for i in range(len(pnls)):
        if i >= window - 1:
            w_pnls = pnls[i - window + 1 : i + 1]
            w_rets = rets_arr[i - window + 1 : i + 1]
            w_wins = sum(1 for p in w_pnls if p > 0)
            w_win_rate = (w_wins / window) * 100.0

            w_std = np.std(w_rets, ddof=1)
            w_sharpe = (np.mean(w_rets) / w_std * math.sqrt(window)) if w_std > 0 else 0.0

            rolling_metrics.append({
                "trade_index": i + 1,
                "win_rate": round(w_win_rate, 1),
                "sharpe": round(w_sharpe, 2)
            })

    # Trade PnL Distribution Buckets (10 bins)
    pnl_histogram = []
    if pnls:
        counts, bin_edges = np.histogram(pnls, bins=10)
        for i in range(len(counts)):
            pnl_histogram.append({
                "bin_label": f"${float(bin_edges[i]):.1f} to ${float(bin_edges[i+1]):.1f}",
                "count": int(counts[i]),
                "is_win": True if float(bin_edges[i+1]) > 0 else False
            })

    # Trade Duration vs Realized PnL Scatter Array
    duration_scatter = []
    for idx in range(1, len(unique_trades)):
        t_curr = unique_trades[idx]
        t_prev = unique_trades[idx - 1]
        if t_curr.pnl is not None and t_curr.timestamp and t_prev.timestamp:
            dur_hrs = round((t_curr.timestamp - t_prev.timestamp).total_seconds() / 3600.0, 2)
            duration_scatter.append({
                "duration_hrs": max(dur_hrs, 0.1),
                "pnl": round(t_curr.pnl, 2),
                "trade_type": t_curr.trade_type
            })

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "timeframe": strategy.timeframe,
        "current_equity": round(current_equity, 2),
        "initial_capital": round(initial_capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr": round(cagr, 2),
        "net_realized_pnl": round(net_realized_pnl, 2),
        "sharpe_ratio": sharpe_ratio,
        "sharpe_ci": sharpe_ci,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "profit_factor": round(profit_factor, 2),
        "win_rate": round(win_rate, 1),
        "loss_rate": round(loss_rate, 1),
        "total_trades": total_closed_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "expectancy": round(expectancy, 2),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_drawdown_amount": round(max_drawdown_amount, 2),
        "max_drawdown_duration_hrs": round(max_drawdown_duration_hrs, 1),
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "position_size": position_size,
        "ready_for_live": latest_gate.ready_for_live if latest_gate else False,
        "equity_curve": equity_curve,
        "rolling_metrics": rolling_metrics,
        "pnl_histogram": pnl_histogram,
        "duration_scatter": duration_scatter
    }


# ---------------------------------------------------------------------------
# Live Kraken market data fetcher — returns real values or None, never fakes
# ---------------------------------------------------------------------------
def _fetch_kraken_market_data(strategy_id: str) -> Dict[str, Any]:
    """
    Fetch REAL market data from Kraken public API.
    Returns actual computed values or None when unavailable.
    Never returns hardcoded placeholder numbers.
    """
    result: Dict[str, Any] = {
        "btc_price": None,
        "btc_price_delta_pct": None,
        "eth_price": None,
        "eth_price_delta_pct": None,
        "regime": "UNAVAILABLE",
        "volatility": "UNAVAILABLE",
        "liquidity": "UNAVAILABLE",
        "btc_sparkline": [],
        "eth_sparkline": [],
    }

    interval = 60 if strategy_id == "v4" else 240
    url = "https://api.kraken.com/0/public/OHLC"

    # ---- BTC/USD candles ----
    try:
        resp = requests.get(url, params={"pair": "XBTUSD", "interval": interval}, timeout=8)
        data = resp.json()
        if data.get("error") and len(data["error"]) > 0:
            return result

        res_key = [k for k in data["result"].keys() if k != "last"][0]
        candles = data["result"][res_key]

        if len(candles) < 5:
            return result

        recent = candles[-30:]
        closes = [float(c[4]) for c in recent]
        highs = [float(c[2]) for c in recent]
        lows = [float(c[3]) for c in recent]
        vols = [float(c[6]) for c in recent]

        result["btc_price"] = closes[-1]
        result["btc_sparkline"] = closes[-10:]

        # Real 24h delta (24 hourly candles or 6 four-hour candles)
        lookback = 24 if interval == 60 else 6
        if len(closes) >= lookback:
            old = closes[-lookback]
            result["btc_price_delta_pct"] = round((closes[-1] - old) / old * 100, 2)

        # Regime: directional price change over window
        pct_change = abs(closes[-1] - closes[0]) / closes[0] * 100
        result["regime"] = (
            "STRONG TREND" if pct_change > 3.0 else
            "TRENDING" if pct_change > 1.5 else
            "WEAK TREND" if pct_change > 0.6 else
            "RANGING"
        )

        # Volatility: ATR as percentage of price
        atr_vals = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            atr_vals.append(tr)
        if atr_vals:
            atr_pct = (sum(atr_vals) / len(atr_vals)) / closes[-1] * 100
            result["volatility"] = "HIGH" if atr_pct > 1.8 else ("MEDIUM" if atr_pct > 0.8 else "LOW")

        # Liquidity: average volume bucket
        if vols:
            avg_vol = sum(vols) / len(vols)
            result["liquidity"] = "HIGH" if avg_vol > 80 else ("MEDIUM" if avg_vol > 15 else "LOW")

    except Exception:
        pass  # values stay None / UNAVAILABLE

    # ---- ETH/USD candles (supplementary, failures are fine) ----
    try:
        resp_eth = requests.get(url, params={"pair": "XETHZUSD", "interval": interval}, timeout=5)
        d_eth = resp_eth.json()
        if not (d_eth.get("error") and len(d_eth["error"]) > 0):
            rk = [k for k in d_eth["result"].keys() if k != "last"][0]
            eth_candles = d_eth["result"][rk]
            if len(eth_candles) >= 5:
                eth_closes = [float(c[4]) for c in eth_candles[-30:]]
                result["eth_price"] = eth_closes[-1]
                result["eth_sparkline"] = eth_closes[-10:]
                lookback = 24 if interval == 60 else 6
                if len(eth_closes) >= lookback:
                    result["eth_price_delta_pct"] = round(
                        (eth_closes[-1] - eth_closes[-lookback]) / eth_closes[-lookback] * 100, 2
                    )
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Monte Carlo Ruin Survival — vectorized moving-block bootstrap
# ---------------------------------------------------------------------------
def compute_monte_carlo_ruin_survival(
    pnls: List[float],
    current_equity: float,
    n_simulations: int = 3000,
) -> Dict[str, Any]:
    """
    Sequence-risk Monte Carlo: resamples the REAL closed-trade PnL sequence
    in contiguous blocks (moving block bootstrap) rather than shuffling
    trades independently. An i.i.d. shuffle (the previous implementation)
    destroys win/loss streaks; real strategies have autocorrelated streaks
    (a losing regime tends to cluster), so block resampling gives a more
    honest picture of how bad a real bad stretch can look.

    Like MonteCarloSimulator in research/validation/monte_carlo.py, this is
    SEQUENCE-RISK ONLY — it answers "given these trades happen, in what
    order, how bad can the ride get", not "is the edge real" (every trade
    here was already selected by the strategy's own live/paper performance).

    Returns survival probability at four ruin thresholds, the drawdown
    distribution, median trades-to-ruin, and a percentile fan chart of
    simulated equity paths for the frontend to plot.
    """
    n = len(pnls)
    thresholds_pct = [20, 35, 50, 75]

    if n < 5:
        return {
            "n_simulations": 0,
            "n_trades": n,
            "block_size": 0,
            "thresholds": [{"loss_pct": t, "survival_pct": 0.0} for t in thresholds_pct],
            "primary_confidence_pct": 0.0,
            "primary_threshold_pct": 50,
            "median_max_drawdown_pct": 0.0,
            "p95_max_drawdown_pct": 0.0,
            "median_trades_to_ruin": None,
            "fan_chart": [],
            "methodology": "insufficient trade history (need 5+ closed trades)",
        }

    arr = np.array(pnls, dtype=float)
    block_size = max(2, min(10, n // 4))
    ruin_levels = {t: current_equity * (t / 100.0) for t in thresholds_pct}

    rng = np.random.default_rng()
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size  # inclusive upper bound for a valid block start

    # Vectorized moving-block bootstrap: draw all block starts for all sims
    # at once, gather the blocks via fancy indexing, trim to n trades/sim.
    starts = rng.integers(0, max_start + 1, size=(n_simulations, n_blocks_needed))
    offsets = np.arange(block_size)
    idx_matrix = (starts[:, :, None] + offsets[None, None, :]).reshape(n_simulations, -1)[:, :n]
    resampled = arr[idx_matrix]  # (n_simulations, n)

    equity_paths = current_equity + np.cumsum(resampled, axis=1)  # (n_simulations, n)

    # Survival = the path never dips to/below a given ruin level at any point
    path_min = equity_paths.min(axis=1)
    thresholds = [
        {"loss_pct": t, "survival_pct": round(float(np.mean(path_min > ruin_levels[t])) * 100, 1)}
        for t in thresholds_pct
    ]
    primary = next(r for r in thresholds if r["loss_pct"] == 50)

    # Drawdown distribution across all simulated paths
    running_peak = np.maximum.accumulate(equity_paths, axis=1)
    drawdown_pct = (running_peak - equity_paths) / running_peak * 100
    max_dd_per_sim = drawdown_pct.max(axis=1)

    # Median trades-to-ruin at the primary (50%) threshold, among paths that breached it
    breach_mask = equity_paths <= ruin_levels[50]
    has_breach = breach_mask.any(axis=1)
    if has_breach.any():
        first_breach = np.argmax(breach_mask, axis=1)[has_breach]
        median_trades_to_ruin = int(np.median(first_breach)) + 1
    else:
        median_trades_to_ruin = None

    # Percentile fan chart — 20 checkpoints across the trade sequence
    n_checkpoints = min(20, n)
    checkpoint_idxs = np.unique(np.linspace(0, n - 1, n_checkpoints).astype(int))
    fan_chart = [
        {
            "step": int(checkpoint_idxs[i]) + 1,
            "p5": round(float(np.percentile(equity_paths[:, checkpoint_idxs[i]], 5)), 2),
            "p25": round(float(np.percentile(equity_paths[:, checkpoint_idxs[i]], 25)), 2),
            "p50": round(float(np.percentile(equity_paths[:, checkpoint_idxs[i]], 50)), 2),
            "p75": round(float(np.percentile(equity_paths[:, checkpoint_idxs[i]], 75)), 2),
            "p95": round(float(np.percentile(equity_paths[:, checkpoint_idxs[i]], 95)), 2),
        }
        for i in range(len(checkpoint_idxs))
    ]

    return {
        "n_simulations": n_simulations,
        "n_trades": n,
        "block_size": block_size,
        "thresholds": thresholds,
        "primary_confidence_pct": primary["survival_pct"],
        "primary_threshold_pct": 50,
        "median_max_drawdown_pct": round(float(np.median(max_dd_per_sim)), 1),
        "p95_max_drawdown_pct": round(float(np.percentile(max_dd_per_sim, 95)), 1),
        "median_trades_to_ruin": median_trades_to_ruin,
        "fan_chart": fan_chart,
        "methodology": f"{n_simulations:,}-path block bootstrap (block={block_size} trades) \u2014 sequence-risk only",
    }


# ---------------------------------------------------------------------------
# Command Center unified telemetry — zero stubs, zero hardcoded values
# ---------------------------------------------------------------------------
def compute_command_center_telemetry(strategy_id: str, db: Session) -> Dict[str, Any]:
    """
    Build the full Command Center payload from REAL data only.
    Every number is computed from database records or live Kraken data.
    When data is missing, values are 0 / None / 'NO DATA' — never faked.
    """
    from backend.models import SystemEventLog, ExecutionPipelineStep

    # ---------- foundation ----------
    base = compute_strategy_analytics(strategy_id, db)
    market = _fetch_kraken_market_data(strategy_id)

    # ---------- 1. strategy health (real computation) ----------
    pnls = [t.pnl for t in db.query(Trade).filter(Trade.strategy_id == strategy_id).all() if t.pnl is not None]

    if pnls and len(pnls) >= 3:
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        sharpe = base.get("sharpe_ratio", 0.0)
        dd = base.get("max_drawdown_pct", 0.0)
        pf = base.get("profit_factor", 1.0)

        raw = 50.0 + (wr * 30.0) + (min(sharpe, 3.0) * 5.0) - (min(dd, 30.0) * 0.5) + (min(max(pf - 1.0, 0), 2.0) * 5.0)
        core_stability = round(min(max(raw, 5.0), 99.0), 1)

        mc = compute_monte_carlo_ruin_survival(pnls, base.get("current_equity", 1000.0))
        confidence = mc["primary_confidence_pct"]

        status = ("EXCELLENT" if core_stability >= 80 else "GOOD" if core_stability >= 65
                  else "NORMAL" if core_stability >= 50 else "CAUTION" if core_stability >= 35 else "BREACH")
    else:
        core_stability = 0.0
        mc = compute_monte_carlo_ruin_survival(pnls, base.get("current_equity", 1000.0))
        confidence = mc["primary_confidence_pct"]
        status = "NO DATA" if not pnls else "INSUFFICIENT"

    # ---------- 2. equity & position ----------
    snaps = (db.query(EquitySnapshot).filter(EquitySnapshot.strategy_id == strategy_id)
             .order_by(EquitySnapshot.timestamp.desc()).limit(50).all())

    current_eq = snaps[0].equity if snaps else base.get("current_equity", 1000.0)
    init_cap = base.get("initial_capital", 1000.0)
    pos_size = snaps[0].position_size if snaps else 0.0

    # real equity delta
    if len(snaps) >= 2:
        eq_delta = round((snaps[0].equity - snaps[-1].equity) / max(snaps[-1].equity, 1) * 100, 2)
    else:
        eq_delta = round((current_eq - init_cap) / init_cap * 100, 2) if init_cap else 0.0

    eq_sparkline = [s.equity for s in reversed(snaps[-10:])] if snaps else []

    # ---------- 3. real capital allocation ----------
    btc_px = market["btc_price"] or 0.0
    if current_eq > 0 and btc_px > 0 and pos_size > 0:
        btc_val = max(pos_size * btc_px, 0.0)
        cash_val = max(current_eq - btc_val, 0.0)
        total = btc_val + cash_val or 1
        btc_pct = round(btc_val / total * 100, 1)
        cash_pct = round(100 - btc_pct, 1)
    else:
        btc_val = 0.0
        cash_val = current_eq
        btc_pct = 0.0
        cash_pct = 100.0

    assets = [
        {"name": "BTC", "pct": btc_pct, "color": "#00f5a0", "value": round(btc_val, 2)},
        {"name": "USDT", "pct": cash_pct, "color": "#ffb800", "value": round(cash_val, 2)},
    ]

    # ---------- 4. real exposure & leverage ----------
    exposure = round(min((pos_size * btc_px) / max(current_eq, 1) * 100, 100), 1) if (pos_size > 0 and btc_px > 0) else 0.0
    lev = 1.0 if pos_size > 0 else 0.0  # paper trading = no borrowed funds

    max_dd = base.get("max_drawdown_pct", 0.0)
    v95 = base.get("var_95", 0.0)

    risk_radar = {
        "max_drawdown_pct": f"{max_dd:.2f}%",
        "var_95_pct": f"{v95:.2f}%",
        "exposure_pct": f"{int(exposure)}%",
        "leverage": f"{lev:.2f}x",
        "radar_scores": {
            "drawdown": min(int(max_dd * 3), 100) if max_dd > 0 else 0,
            "volatility": {"HIGH": 80, "MEDIUM": 50, "LOW": 20}.get(market["volatility"], 0),
            "liquidity": {"HIGH": 85, "MEDIUM": 50, "LOW": 20}.get(market["liquidity"], 0),
            "leverage": int(lev * 50),
            "exposure": int(exposure),
        },
    }

    # ---------- 5. pipeline steps (DB only, no faking) ----------
    pipe = (db.query(ExecutionPipelineStep).filter(ExecutionPipelineStep.strategy_id == strategy_id)
            .order_by(ExecutionPipelineStep.timestamp.desc()).first())

    def _fmt(dt):
        return dt.strftime("%H:%M:%S") if dt else "—"

    if pipe:
        pipeline_steps = [
            {"key": k, "label": lbl, "status": "DONE" if getattr(pipe, attr) else "PENDING", "time": _fmt(getattr(pipe, attr))}
            for k, lbl, attr in [
                ("SIGNAL", "SIGNAL GENERATED", "signal_generated_at"),
                ("RISK", "RISK PASSED", "risk_passed_at"),
                ("SIZE", "SIZE CALCULATED", "size_calculated_at"),
                ("SUBMITTED", "ORDER SUBMITTED", "order_submitted_at"),
                ("FILLED", "ORDER FILLED", "order_filled_at"),
                ("POSITION", "POSITION OPENED", "position_opened_at"),
                ("ARMED", "SL ARMED", "sl_armed_at"),
                ("TP", "TP SET", "tp_set_at"),
                ("TRAILING", "TRAILING ACTIVE", "trailing_active_at"),
            ]
        ]
    else:
        pipeline_steps = []

    # ---------- 6. event feed (DB only — no seed faking) ----------
    logs = db.query(SystemEventLog).order_by(SystemEventLog.timestamp.desc()).limit(30).all()
    event_feed = [
        {"time": _fmt(l.timestamp), "category": l.category, "message": l.message, "level": l.level}
        for l in logs
    ]

    # ---------- 7. AI copilot insights (all computed) ----------
    now_t = datetime.now(timezone.utc).strftime("%H:%M")
    insights: List[Dict] = []

    if pnls:
        n = len(pnls)
        wr_pct = sum(1 for p in pnls if p > 0) / n * 100
        insights.append({"time": now_t, "text": f"Analyzed {n} closed trades. Win rate: {wr_pct:.1f}%.", "highlight": wr_pct >= 55})

        pf_val = base.get("profit_factor", 1.0)
        if pf_val >= 1.5:
            insights.append({"time": now_t, "text": f"Profit factor strong at {pf_val:.2f}x.", "highlight": True})
        elif pf_val >= 1.0:
            insights.append({"time": now_t, "text": f"Profit factor positive ({pf_val:.2f}x). Room to optimize.", "highlight": False})
        else:
            insights.append({"time": now_t, "text": f"Profit factor below 1.0 ({pf_val:.2f}x). Strategy is losing.", "highlight": True})

        if max_dd > 15:
            insights.append({"time": now_t, "text": f"Drawdown elevated at {max_dd:.1f}%. Consider reducing size.", "highlight": True})
        elif max_dd > 0:
            insights.append({"time": now_t, "text": f"Max drawdown: {max_dd:.1f}%. Within tolerance.", "highlight": False})
    else:
        insights.append({"time": now_t, "text": "No closed trades yet. Waiting for first signal.", "highlight": False})

    if market["regime"] != "UNAVAILABLE":
        insights.append({"time": now_t, "text": f"Market: {market['regime']}. Vol: {market['volatility']}.", "highlight": market["volatility"] == "HIGH"})

    if pos_size > 0:
        insights.append({"time": now_t, "text": f"Active BTC position: {pos_size:.6f} BTC ({exposure:.0f}% exposure).", "highlight": False})
    else:
        insights.append({"time": now_t, "text": "No active position. Bot is flat.", "highlight": False})

    # ---------- 8. sparkline data (real) ----------
    pnl_cum = []
    if pnls:
        cs = 0
        step = max(1, len(pnls) // 10)
        for i in range(0, len(pnls), step):
            cs += sum(pnls[i:i + step])
            pnl_cum.append(round(cs, 2))
        pnl_cum = pnl_cum[-10:]

    wr_spark = []
    if len(pnls) >= 5:
        w = min(10, len(pnls))
        for i in range(w, len(pnls) + 1):
            chunk = pnls[i - w:i]
            wr_spark.append(round(sum(1 for p in chunk if p > 0) / len(chunk) * 100, 1))
        wr_spark = wr_spark[-10:]

    # ---------- 9. satellite globe node telemetry (real computed fields) ----------
    btc_delta = market["btc_price_delta_pct"] or 0.0
    critical_logs_cnt = sum(1 for l in logs if l.level in ["WARNING", "CRITICAL"])
    
    globe_nodes = {
        "execution": "ONLINE" if status != "BREACH" else "PAUSED",
        "sentiment": "BULLISH" if btc_delta > 0 else ("BEARISH" if btc_delta < 0 else "NEUTRAL"),
        "risk_engine": "OPTIMAL" if max_dd < 15 and core_stability >= 50 else ("ELEVATED" if max_dd < 25 else "BREACH"),
        "trend": "STRONG" if market["regime"] in ["STRONG TREND", "TRENDING"] else "WEAK",
        "momentum": "ACTIVE" if abs(btc_delta) > 0.5 else "FLAT",
        "liquidity": market["liquidity"],
        "volume": "ELEVATED" if market["liquidity"] == "HIGH" else "NORMAL",
        "news_feed": "NOMINAL" if critical_logs_cnt == 0 else "ALERT"
    }

    # ---------- assemble ----------
    return {
        "strategy_id": strategy_id,
        "btc_price": market["btc_price"],
        "btc_price_delta_pct": market["btc_price_delta_pct"],
        "eth_price": market["eth_price"],
        "eth_price_delta_pct": market["eth_price_delta_pct"],
        "strategy_health": {
            "core_stability_pct": core_stability,
            "confidence_level_pct": confidence,
            "status": status,
        },
        "monte_carlo_ruin_survival": mc,
        "market_regime": {
            "regime": market["regime"],
            "volatility": market["volatility"],
            "liquidity": market["liquidity"],
        },
        "analytics_summary": {
            "current_equity": round(current_eq, 2),
            "initial_capital": init_cap,
            "equity_delta_pct": eq_delta,
            "net_realized_pnl": base.get("net_realized_pnl", 0.0),
            "win_rate": base.get("win_rate", 0.0),
            "total_trades": base.get("total_trades", 0),
            "sharpe_ratio": base.get("sharpe_ratio", 0.0),
            "profit_factor": base.get("profit_factor", 0.0),
            "max_drawdown_pct": max_dd,
            "position_size": pos_size,
        },
        "pipeline_steps": pipeline_steps,
        "ai_copilot_insights": insights,
        "risk_radar": risk_radar,
        "capital_allocation": {"total_equity": round(current_eq, 2), "assets": assets},
        "event_feed": event_feed,
        "globe_nodes": globe_nodes,
        "sparkline_data": {
            "btc_prices": market["btc_sparkline"],
            "eth_prices": market["eth_sparkline"],
            "equity_history": eq_sparkline,
            "pnl_cumulative": pnl_cum,
            "win_rate_rolling": wr_spark,
        },
    }


def compute_drawdown_terrain_matrix(strategy_id: str, db: Session) -> Dict[str, Any]:
    """
    Generate a 3D vertex mesh (Duration_hrs x Depth_pct x Time_step) representing spatial underwater drawdown dynamics.
    Zero stubs — calculated from database trade & equity snapshot series.
    """
    trades = (
        db.query(Trade)
        .filter(Trade.strategy_id == strategy_id, Trade.pnl.isnot(None))
        .order_by(Trade.timestamp.asc())
        .all()
    )

    if not trades:
        return {
            "strategy_id": strategy_id,
            "terrain_matrix": [],
            "max_depth_pct": 0.0,
            "max_duration_hrs": 0.0,
            "sample_count": 0
        }

    running_equity = 1000.0
    running_peak = 1000.0
    dd_start = None
    terrain_matrix = []
    max_depth = 0.0
    max_duration = 0.0

    for idx, t in enumerate(trades):
        running_equity += t.pnl
        t_time = t.timestamp or datetime.now(timezone.utc)
        
        if running_equity >= running_peak:
            running_peak = running_equity
            dd_start = None
            dd_depth = 0.0
            dd_dur_hrs = 0.0
        else:
            if dd_start is None:
                dd_start = t_time
            dd_amt = running_peak - running_equity
            dd_depth = round((dd_amt / running_peak * 100.0), 2)
            dd_dur_hrs = round((t_time - dd_start).total_seconds() / 3600.0, 2)

        max_depth = max(max_depth, dd_depth)
        max_duration = max(max_duration, dd_dur_hrs)

        terrain_matrix.append({
            "step": idx + 1,
            "timestamp": t_time.isoformat(),
            "time_norm": round((idx + 1) / len(trades), 3),
            "depth_pct": dd_depth,
            "duration_hrs": dd_dur_hrs,
            "equity": round(running_equity, 2),
            "peak": round(running_peak, 2)
        })

    return {
        "strategy_id": strategy_id,
        "terrain_matrix": terrain_matrix,
        "max_depth_pct": max_depth,
        "max_duration_hrs": max_duration,
        "sample_count": len(trades)
    }


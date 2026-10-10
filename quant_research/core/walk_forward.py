"""
Walk-Forward Validation Engine and Statistical Testing Tools.
Features:
- Rolling fold generator: 12-month train / 3-month test / 3-month step.
- Selection on train folds strictly by net expectancy in bps (min 100 trades filter).
- Walk-forward out-of-sample trade stitching.
- Deflated Sharpe Ratio (DSR) computation accounting for trial count and skew/kurtosis.
"""
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def generate_walk_forward_folds(
    start_date: str = "2024-01-01",
    end_oos_date: str = "2026-03-31",
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> List[Dict[str, str]]:
    """
    Generate rolling walk-forward fold date boundaries.
    Fold 1: Train 2024-01-01 to 2024-12-31, Test 2025-01-01 to 2025-03-31
    Fold 2: Train 2024-04-01 to 2025-03-31, Test 2025-04-01 to 2025-06-30
    Fold 3: Train 2024-07-01 to 2025-06-30, Test 2025-07-01 to 2025-09-30
    Fold 4: Train 2024-10-01 to 2025-09-30, Test 2025-10-01 to 2025-12-31
    Fold 5: Train 2025-01-01 to 2025-12-31, Test 2026-01-01 to 2026-03-31
    """
    folds = []
    current_train_start = pd.Timestamp(start_date, tz="UTC")
    final_limit = pd.Timestamp(end_oos_date, tz="UTC")

    fold_idx = 1
    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(seconds=1)
        test_start = current_train_start + pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(seconds=1)

        if test_end > final_limit + pd.Timedelta(days=1):
            break

        folds.append({
            "fold_id": f"Fold_{fold_idx}",
            "train_start": str(current_train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
        })

        current_train_start = current_train_start + pd.DateOffset(months=step_months)
        fold_idx += 1

    return folds


def calculate_deflated_sharpe_ratio(
    sr_annual: float,
    n_trials: int,
    t_bars: int,
    skewness: float = 0.0,
    kurt: float = 3.0,
    var_sr: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute Deflated Sharpe Ratio (DSR) and its p-value based on Bailey and López de Prado (2014).
    Accounts for:
    - Multiple testing / selection bias across n_trials.
    - Non-normality (skewness and kurtosis of returns).
    - Sample length (t_bars).

    Returns:
    - dsr_stat: The DSR test statistic.
    - p_value: The probability of obtaining this Sharpe or higher under H0 (zero true Sharpe).
    """
    if n_trials <= 1:
        n_trials = 1

    # Euler-Mascheroni constant
    gamma_em = 0.57721566490153286

    # Expected maximum Sharpe under null hypothesis of 0 true Sharpe across n independent trials
    if n_trials > 1:
        z_1 = norm.ppf(1.0 - 1.0 / n_trials)
        z_2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr_benchmark = math.sqrt(var_sr) * ((1.0 - gamma_em) * z_1 + gamma_em * z_2)
    else:
        sr_benchmark = 0.0

    # Denominator adjustment for non-normality
    # Var(SR) = (1 - skew * SR + (kurt - 1) / 4 * SR^2) / (T - 1)
    denom_factor = 1.0 - skewness * sr_annual + ((kurt - 1.0) / 4.0) * (sr_annual ** 2)
    denom_factor = max(1e-6, denom_factor)
    sr_std_error = math.sqrt(denom_factor / max(2, t_bars - 1))

    dsr_stat = (sr_annual - sr_benchmark) / sr_std_error
    p_value = 1.0 - norm.cdf(dsr_stat)

    return float(dsr_stat), float(p_value)


def stitch_walk_forward_trades(
    fold_test_trades: List[pd.DataFrame],
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Stitch out-of-sample trades from all walk-forward folds into a continuous equity stream.
    """
    if not fold_test_trades:
        return {
            "total_trades": 0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "trades_df": pd.DataFrame(),
        }

    valid_dfs = [df for df in fold_test_trades if df is not None and not df.empty]
    if not valid_dfs:
        return {
            "total_trades": 0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "trades_df": pd.DataFrame(),
        }

    all_trades = pd.concat(valid_dfs, ignore_index=True)
    all_trades = all_trades.sort_values("exit_time").reset_index(drop=True)

    equity = initial_capital
    equity_curve = [initial_capital]
    returns_pct = []

    gross_profit = 0.0
    gross_loss = 0.0

    for _, tr in all_trades.iterrows():
        pnl = tr["net_pnl"]
        equity += pnl
        equity_curve.append(equity)
        returns_pct.append(tr["return_pct"])
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    total_return_pct = ((equity - initial_capital) / initial_capital) * 100.0

    # Max Drawdown
    eq_series = pd.Series(equity_curve)
    peaks = eq_series.cummax()
    drawdowns = (peaks - eq_series) / peaks * 100.0
    max_dd = float(drawdowns.max())

    # Expectancy in bps
    exp_bps = float(np.mean(returns_pct) * 10000.0) if returns_pct else 0.0

    # Sharpe of trade returns
    if len(returns_pct) > 1 and np.std(returns_pct) > 1e-8:
        # Annualized assuming typical trade frequency
        sharpe = float(np.mean(returns_pct) / np.std(returns_pct) * np.sqrt(len(returns_pct)))
    else:
        sharpe = 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "total_trades": len(all_trades),
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "expectancy_bps": exp_bps,
        "profit_factor": profit_factor,
        "trades_df": all_trades,
    }

"""Portfolio backtest engine for the frozen slow strategies.

Execution model (no look-ahead):
- Signals are target WEIGHTS indexed by signal date = daily bar CLOSE date.
- At the NEXT bar's OPEN the portfolio trades to the new target weights.
- Between trades, coin holdings are fixed; equity is marked at each close.
- Cost per side = taker 0.10% + slippage 0.05% = 0.15% of traded notional.
- Orders below MIN_NOTIONAL_USD are skipped (documented $1k-account constraint).
- Target weights may sum to <= 1; remainder is cash (0% interest, conservative).

S5 (funding carry) is simulated separately in s5_carry.py because it needs a
short perpetual leg and funding cashflows.
"""
import numpy as np
import pandas as pd

from strategies import COST_PER_SIDE, MIN_NOTIONAL_USD, START_EQUITY


def run_backtest(weight_df, bars, start_equity=START_EQUITY, cost_per_side=COST_PER_SIDE,
                 min_notional=MIN_NOTIONAL_USD):
    """weight_df: DataFrame (dates x assets) of target weights, NaN -> 0.
    bars: dict asset -> DataFrame with 'open','close' indexed by date.
    Returns (equity_series, trades_log_DataFrame).
    """
    assets = [c for c in weight_df.columns if c in bars]
    weight_df = weight_df[assets].fillna(0.0)
    # union calendar = weight index intersected with common bar dates
    idx = weight_df.index
    opens = pd.DataFrame({a: bars[a]["open"] for a in assets}).reindex(idx)
    closes = pd.DataFrame({a: bars[a]["close"] for a in assets}).reindex(idx)

    cash = start_equity
    holdings = {a: 0.0 for a in assets}  # coin units
    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []

    for i, d in enumerate(idx):
        o = opens.loc[d]
        c = closes.loc[d]
        # --- mark-to-market at previous close already done; trade at today's open
        # using target weights from YESTERDAY's signal (weight_df.shift(1)).
        if i > 0:
            sig_date = idx[i - 1]
            target_w = weight_df.loc[sig_date].reindex(assets).fillna(0.0)
            # current equity estimate at today's open
            eq_open = cash + sum(holdings[a] * (o[a] if not np.isnan(o[a]) else np.nan)
                                 for a in assets)
            # value holdings at open; NaN open -> use previous close
            eq_open = cash
            for a in assets:
                px = o[a]
                if np.isnan(px):
                    px = closes[a].iloc[i - 1]
                eq_open += holdings[a] * px
            if not np.isnan(eq_open) and eq_open > 0:
                for a in assets:
                    px = o[a]
                    if np.isnan(px):
                        px = closes[a].iloc[i - 1]
                    desired_notional = target_w[a] * eq_open
                    desired_coins = desired_notional / px if px and px > 0 else 0.0
                    delta_coins = desired_coins - holdings[a]
                    delta_notional = delta_coins * px
                    if abs(delta_notional) >= min_notional:
                        cost = abs(delta_notional) * cost_per_side
                        cash -= delta_notional + cost
                        holdings[a] += delta_coins
                        trades.append({
                            "dt": d, "asset": a, "price": px,
                            "notional": delta_notional, "cost": cost,
                            "weight_after": target_w[a],
                        })
                    else:
                        # below min size: leave holdings as-is
                        pass
        # --- mark equity at today's close
        eq = cash
        for a in assets:
            px = c[a]
            if np.isnan(px):
                # carry last known price
                hist = closes[a].iloc[: i + 1].dropna()
                px = hist.iloc[-1] if len(hist) else np.nan
            eq += holdings[a] * (px if not np.isnan(px) else 0.0)
        equity.loc[d] = eq
    eq = equity.dropna()
    return eq, pd.DataFrame(trades)


def run_buy_and_hold(symbol_bars, start_equity=START_EQUITY):
    """Convenience: S6 uses the same engine with constant weight 1."""
    return None  # handled via weight_df path

"""Strategy A — multi-asset funding-rate arbitrage.

FROZEN RULES (fixed before any run; do not tune after seeing results):
- Universe: 12 OKX USDT spot majors with Binance USD-M perp funding history.
- Signal: trailing 30-day sum of daily funding rates per asset (only data
  <= decision day). Annualized carry = mean daily sum * 365.
- Selection: top K=3 assets by trailing funding, requiring trailing funding
  sum > 0 (negative-funding avoidance). Equal weight.
- Rebalance: every 7 days; enter/exit/resize at that day's close.
- Per-position leverage L (run at 1 and 2); alloc = 0.90/K each (sum 0.90,
  leaving >=10% reserve buffer for Rule A top-ups).
- Margin/liquidation/rebalance rules: engine defaults (S5 v2.1 inherited).
"""
import numpy as np
import pandas as pd

K = 3
LOOKBACK_DAYS = 30
MIN_FUNDING_SUM = 0.0      # trailing 30d funding sum must be > 0
ALLOC_TOTAL = 0.90
REBAL_DAYS = 7
LEVERAGES = (1, 2)


def trailing_funding(fund, d, days=LOOKBACK_DAYS):
    """Sum of funding rates over the `days` days ending at d (inclusive)."""
    f = fund[fund.index <= d]
    if f.empty:
        return np.nan
    start = d - pd.Timedelta(days=days)
    f = f[f.index > start]
    return float(f.sum()) if len(f) else np.nan


def make_policy(funds, leverage):
    """Return policy(d, ctx) -> {sym: alloc}."""
    lev = {s: leverage for s in funds}

    def policy(d, ctx):
        ctx["leverage"] = lev
        scores = {}
        for s, f in funds.items():
            scores[s] = trailing_funding(f, d)
        valid = {s: v for s, v in scores.items() if v == v and v > MIN_FUNDING_SUM}
        if not valid:
            return {}
        top = sorted(valid, key=valid.get, reverse=True)[:K]
        return {s: ALLOC_TOTAL / len(top) for s in top}

    return policy

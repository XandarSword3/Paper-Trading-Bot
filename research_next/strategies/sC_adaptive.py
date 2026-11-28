"""Strategy C — adaptive funding carry (BTC, exposure management).

FROZEN RULES (fixed before any run; do not tune after seeing results):
Base = BTC delta-neutral carry (same structure as S5 v2.1), but the capital
allocation is scaled weekly by a two-input regime rule:

- f30 = trailing 30-day sum of BTC funding rates (data <= decision day).
- ann_f = f30_daily_mean * 365  (annualized funding carry estimate)
- vol30 = 30-day realized vol of daily spot log returns * sqrt(365)

Exposure multiplier m (alloc = 0.90 * m):
- m = 1.00  if ann_f > 3%  and vol30 < 60%
- m = 0.50  if ann_f > 0%  and vol30 < 80%
- m = 0.25  otherwise (negative funding or extreme vol -> minimal exposure)

Rationale: cut exposure when carry is negative or vol threatens liquidation
distance; full exposure only when carry is clearly positive and vol moderate.
Leverage L=1 primary (S5-approved benchmark); L=2 reported for information.
Rebalance weekly; engine margin rules unchanged.
"""
import numpy as np
import pandas as pd

LOOKBACK_DAYS = 30
ALLOC_BASE = 0.90
REBAL_DAYS = 7
LEVERAGES = (1, 2)
SYM = "BTC"


def _metrics(spots, funds, d):
    f = funds[SYM]
    f = f[f.index <= d]
    start = d - pd.Timedelta(days=LOOKBACK_DAYS)
    f30 = f[f.index > start]
    ann_f = float(f30.sum() / LOOKBACK_DAYS * 365) if len(f30) else np.nan
    px = spots[SYM]
    px = px[px.index <= d]
    px = px[px.index > start]
    vol30 = float(px["close"].pct_change().std() * np.sqrt(365)) if len(px) > 10 else np.nan
    return ann_f, vol30


def make_policy(spots, funds, leverage):
    lev = {SYM: leverage}

    def policy(d, ctx):
        ctx["leverage"] = lev
        ann_f, vol30 = _metrics(spots, funds, d)
        if ann_f != ann_f or vol30 != vol30:
            m = 0.25
        elif ann_f > 0.03 and vol30 < 0.60:
            m = 1.00
        elif ann_f > 0.0 and vol30 < 0.80:
            m = 0.50
        else:
            m = 0.25
        return {SYM: ALLOC_BASE * m}

    return policy

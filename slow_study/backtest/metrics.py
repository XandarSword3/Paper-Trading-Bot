"""Metrics for the frozen-strategy study.

- Monthly return distribution from daily equity (month-end to month-end).
- CAGR, max drawdown, worst month, longest time underwater (days below HWM).
- Circular block bootstrap over MONTHLY returns (block=6 months, 10,000 reps)
  -> percentile CI for mean monthly net return. Reported at 95% and at the
  Bonferroni-corrected 99% level (5 strategies -> family alpha 0.05/5 each side
  uses 0.5%/99.5% tails => 99% CI).
- Period splits: FULL, 2019-2022, 2023+ (the spec's out-of-sample window).
"""
import numpy as np
import pandas as pd

BOOT_REPS = 10_000
BLOCK = 6
RNG = np.random.default_rng(20261126)


def monthly_returns(equity):
    eq = equity.dropna()
    me = eq.resample("ME").last().dropna()
    return me.pct_change().dropna()


def max_drawdown(equity):
    eq = equity.dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def longest_underwater_days(equity):
    eq = equity.dropna()
    peak = eq.cummax()
    under = eq < peak
    # longest run of True
    maxrun = cur = 0
    for v in under.values:
        cur = cur + 1 if v else 0
        maxrun = max(maxrun, cur)
    return int(maxrun)


def perf_stats(equity, label=""):
    eq = equity.dropna()
    if len(eq) < 2:
        return {"label": label, "n_days": len(eq)}
    mr = monthly_returns(eq)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
    return {
        "label": label,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "n_days": int(len(eq)),
        "n_months": int(len(mr)),
        "mean_monthly": float(mr.mean()),
        "median_monthly": float(mr.median()),
        "pct_months_ge_6": float((mr >= 0.06).mean()),
        "worst_month": float(mr.min()),
        "best_month": float(mr.max()),
        "cagr": cagr,
        "max_drawdown": max_drawdown(eq),
        "longest_underwater_days": longest_underwater_days(eq),
    }


def block_bootstrap_ci(monthly, block=BLOCK, reps=BOOT_REPS, alpha=0.05):
    """Circular block bootstrap of the mean monthly return.
    Returns dict with ci bounds for mean monthly return."""
    r = np.asarray(monthly.dropna().values, dtype=float)
    n = len(r)
    if n < block * 2:
        return {"mean": float(np.mean(r)) if n else np.nan, "lo": np.nan, "hi": np.nan,
                "n": n, "reps": 0}
    nblocks = int(np.ceil(n / block))
    # vectorized circular block bootstrap: (reps, nblocks, block) index array
    starts = RNG.integers(0, n, size=(reps, nblocks))
    offsets = np.arange(block)
    idx_mat = (starts[:, :, None] + offsets[None, None, :]) % n  # (reps, nblocks, block)
    sample = r[idx_mat].reshape(reps, nblocks * block)[:, :n]
    means = sample.mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "lo": float(lo), "hi": float(hi),
            "n": n, "reps": reps}


def evaluate(equity, name):
    """Full evaluation across period splits. Returns list of row dicts.

    Monthly returns are computed on the FULL equity series first, then filtered
    to month-ends inside each split — otherwise the first month of a split is
    silently dropped (its return needs the prior month-end as base)."""
    eq = equity.dropna()
    mr_full = monthly_returns(eq)  # index = month-end dates
    splits = {
        "FULL": eq,
        "2019_2022": eq[(eq.index.year >= 2019) & (eq.index.year <= 2022)],
        "2023_plus": eq[eq.index.year >= 2023],
    }
    rows = []
    for sname, se in splits.items():
        st = perf_stats(se, f"{name}|{sname}")
        if sname == "FULL":
            mr = mr_full
        else:
            y0 = 2019 if sname == "2019_2022" else 2023
            y1 = 2022 if sname == "2019_2022" else 9999
            mr = mr_full[(mr_full.index.year >= y0) & (mr_full.index.year <= y1)]
        # override n_months/mean/median/pct/worst/best with the correctly-scoped mr
        st["n_months"] = int(len(mr))
        if len(mr):
            st["mean_monthly"] = float(mr.mean())
            st["median_monthly"] = float(mr.median())
            st["pct_months_ge_6"] = float((mr >= 0.06).mean())
            st["worst_month"] = float(mr.min())
            st["best_month"] = float(mr.max())
        ci95 = block_bootstrap_ci(mr, alpha=0.05)
        ci99 = block_bootstrap_ci(mr, alpha=0.01)  # Bonferroni 5 strategies
        st.update({
            "split": sname,
            "boot_mean": ci95["mean"],
            "boot_lo95": ci95["lo"], "boot_hi95": ci95["hi"],
            "boot_lo99": ci99["lo"], "boot_hi99": ci99["hi"],
        })
        rows.append(st)
    return rows

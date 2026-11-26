"""Research orchestration: run the strategy grid, log every trial, report splits.

Usage (from ``liquidation-hunter/``)::

    python -m research.trendlab.run_research

Artifacts land in ``logs/trendlab/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, strategies as S
from .engine import BacktestConfig, run_backtest
from .stats import (
    block_bootstrap_sharpe,
    compute_metrics,
    deflated_sharpe_ratio,
    yearly_table,
)

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[2] / "logs" / "trendlab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed, declared once.  Train is for design, validation for selection,
# holdout is touched only by the final report.
SPLITS = {
    "train": ("2020-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "holdout": ("2025-07-01", "2026-10-31"),
}

DEFAULT_CFG = BacktestConfig(target_vol=0.25, max_gross=3.0, cost_bps=8.0, corr=0.6)


def _funding_dict(panel: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    out = {}
    for sym in panel["close"].columns:
        f = data.load_funding(sym)
        if not f.empty:
            out[sym] = f
    return out


def evaluate(
    name: str,
    family: str,
    params: dict,
    sig: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
    cfg: BacktestConfig = DEFAULT_CFG,
) -> tuple[list[dict], object]:
    res = run_backtest(panel, sig, funding, cfg=cfg)
    rows = []
    for period, (a, b) in SPLITS.items():
        r = res.returns.loc[a:b]
        if r.empty:
            continue
        m = compute_metrics(
            r,
            res.gross.loc[a:b],
            res.turnover.loc[a:b],
            res.costs.loc[a:b],
            res.funding.loc[a:b],
        )
        row = {
            "family": family,
            "name": name,
            "params": json.dumps(params, sort_keys=True),
            "period": period,
            **{k: v for k, v in m.as_dict().items() if k not in ("start", "end")},
        }
        rows.append(row)
    return rows, res


def build_grid(panel, funding):
    """Yield (family, name, params, signal) for every declared trial."""
    close = panel["close"]

    # --- time-series trend -------------------------------------------------
    for lb in [20, 40, 60, 90, 120, 180, 250]:
        yield "tsmom_sign", f"tsmom_sign_{lb}", {"lookback": lb}, S.tsmom_sign(panel, lb)
    for lb in [20, 40, 60, 90, 120, 180, 250]:
        yield "tsmom_tanh", f"tsmom_tanh_{lb}", {"lookback": lb}, S.tsmom_tanh(panel, lb)
    for fast, slow in [(10, 50), (20, 100), (25, 100), (50, 200), (100, 400)]:
        yield "ma_cross", f"ma_cross_{fast}_{slow}", {"fast": fast, "slow": slow}, S.ma_cross(panel, fast, slow)
    for span in [50, 100, 200]:
        for sl in [10, 20, 40]:
            yield "ma_slope", f"ma_slope_{span}_{sl}", {"span": span, "slope": sl}, S.ma_slope(panel, span, sl)
    for entry, exit_n in [(20, 10), (55, 20), (100, 50)]:
        yield "donchian", f"donchian_{entry}_{exit_n}", {"entry": entry, "exit": exit_n}, S.donchian(panel, entry, exit_n)

    # --- long-only / regime variants of the core trend ---------------------
    for lb in [60, 90, 120, 180, 250]:
        base = S.tsmom_tanh(panel, lb)
        yield "tsmom_longonly", f"tsmom_lo_{lb}", {"lookback": lb}, S.long_only(base)
    for lb in [90, 120, 180]:
        base = S.tsmom_tanh(panel, lb)
        yield "tsmom_regime", f"tsmom_regime_{lb}", {"lookback": lb}, S.mask_by_regime(base, panel, "BTCUSDT", 200)

    # --- multi-horizon ensembles (robust CTA construction) -----------------
    ensemble_sets = [
        (20, 60, 120, 250),
        (40, 60, 120),
        (60, 120),
        (20, 60, 120),
        (40, 80, 160),
        (10, 20, 40, 80, 160),
    ]
    for lbs in ensemble_sets:
        tag = "_".join(str(x) for x in lbs)
        for mode in ("sign", "tanh"):
            base = S.ensemble(panel, list(lbs), mode)
            yield "ensemble", f"ens_{mode}_{tag}", {"lookbacks": list(lbs), "mode": mode}, base
            yield (
                "ensemble_longonly",
                f"ens_lo_{mode}_{tag}",
                {"lookbacks": list(lbs), "mode": mode},
                S.long_only(base),
            )
            yield (
                "ensemble_regime",
                f"ens_reg_{mode}_{tag}",
                {"lookbacks": list(lbs), "mode": mode},
                S.mask_by_regime(base, panel, "BTCUSDT", 200),
            )
            for reb in (2, 7):
                yield (
                    "ensemble_reb",
                    f"ens_reb{reb}_{mode}_{tag}",
                    {"lookbacks": list(lbs), "mode": mode, "rebalance": reb},
                    S.rebalance(base, reb),
                )

    # --- cross-sectional momentum -----------------------------------------
    for lb in [30, 60, 90, 120]:
        for reb in [3, 7, 14]:
            for q in [0.2, 0.3, 0.4]:
                for ls in [True, False]:
                    name = f"xsmom_{lb}_{reb}_{int(q*100)}_{'ls' if ls else 'lo'}"
                    yield (
                        "xsmom",
                        name,
                        {"lookback": lb, "rebalance": reb, "quantile": q, "long_short": ls},
                        S.xsmom(panel, lb, 0, q, True, reb, ls),
                    )

    # --- short-horizon mean reversion -------------------------------------
    for lb in [3, 5, 10]:
        for z in [1.5, 2.0, 2.5]:
            for hold in [3, 5, 10]:
                name = f"meanrev_{lb}_{int(z*10)}_{hold}"
                yield (
                    "meanrev",
                    name,
                    {"lookback": lb, "z": z, "hold": hold},
                    S.meanrev(panel, lb, z, hold, 200, False),
                )

    # --- funding carry (directional perp approximation) -------------------
    for lb in [3, 7]:
        for reb in [1, 3, 7]:
            for ls in [True, False]:
                name = f"funding_{lb}_{reb}_{'ls' if ls else 'lo'}"
                yield (
                    "funding_carry",
                    name,
                    {"lookback": lb, "rebalance": reb, "long_short": ls},
                    S.funding_carry(panel, funding, lb, 0.3, reb, ls),
                )


def benchmark(panel, funding):
    close = panel["close"]
    rows = []
    # Equal-weight, daily-rebalanced buy & hold.
    ew = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    ew = ew.where(close.notna(), 0.0)
    rows += evaluate("equal_weight_bh", "benchmark", {}, ew, panel, funding)[0]
    # BTC only.
    btc = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    btc["BTCUSDT"] = 1.0
    rows += evaluate("btc_bh", "benchmark", {}, btc, panel, funding)[0]
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    panel = data.load_panel(freq="1d")
    print(f"panel: {panel['close'].shape[1]} symbols, {len(panel['close'])} days, "
          f"{panel['close'].index.min().date()} -> {panel['close'].index.max().date()}")
    funding = _funding_dict(panel)

    all_rows = []
    n = 0
    for family, name, params, sig in build_grid(panel, funding):
        rows, _ = evaluate(name, family, params, sig, panel, funding)
        all_rows.extend(rows)
        n += 1
        if n % 20 == 0:
            print(f"  ran {n} configurations...")

    all_rows += benchmark(panel, funding)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "research_results.csv", index=False)

    train = df[df.period == "train"].copy()
    val = df[df.period == "validation"].copy()
    hold = df[df.period == "holdout"].copy()
    merged = train.merge(val, on=["family", "name", "params"], suffixes=("_tr", "_va"))
    merged = merged.merge(hold, on=["family", "name", "params"], suffixes=("", "_ho"))
    merged.to_csv(OUT_DIR / "research_merged.csv", index=False)

    print("\n=== TOP TRAIN BY SHARPE (net of 8bps + funding) ===")
    cols = ["family", "name", "sharpe", "cagr", "ann_vol", "max_dd", "turnover_annual"]
    print(train.sort_values("sharpe", ascending=False)[cols].head(25).to_string(index=False))

    print("\n=== TRAIN -> VALIDATION (top 25 train, validation shown) ===")
    show = merged.sort_values("sharpe_tr", ascending=False).head(25)
    print(show[["family", "name", "sharpe_tr", "sharpe_va", "sharpe", "cagr_tr", "cagr_va"]].to_string(index=False))

    print("\n=== BENCHMARKS ===")
    print(df[df.family == "benchmark"][["name", "period", "sharpe", "cagr", "max_dd"]].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'research_results.csv'} ({len(df)} rows, {n} configurations)")


if __name__ == "__main__":
    main()

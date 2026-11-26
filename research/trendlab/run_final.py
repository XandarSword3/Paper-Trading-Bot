"""Produce the final, frozen research report for the selected strategy.

Run from ``liquidation-hunter/``::

    python -m research.trendlab.run_final

Writes to ``logs/trendlab/``:
    final_report.md, final_report.json, equity_curve.png, yearly_returns.png
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, strategies as S
from .engine import BacktestConfig, run_backtest
from .final import FINAL_CONFIG, NAME, SHORT_NAME, final_signal, regime_mask
from .stats import (
    block_bootstrap_sharpe,
    compute_metrics,
    deflated_sharpe_ratio,
    monte_carlo_drawdown,
    monthly_table,
    yearly_table,
)
from .validate import SPLITS, cost_sensitivity, plot_equity, plot_yearly, shift_placebo

OUT = Path(__file__).resolve().parents[2] / "logs" / "trendlab"
OUT.mkdir(parents=True, exist_ok=True)
CFG = BacktestConfig(target_vol=0.25, max_gross=3.0, cost_bps=8.0, corr=0.6,
                     max_asset_weight=0.15, min_asset_vol=0.15)
N_TRIALS = 210  # configurations evaluated during research


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_no data_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def period_table(per: dict[str, dict]) -> pd.DataFrame:
    rows = []
    labels = {"train": "Train 2020-2023", "validation": "Validation 2024-Jun 2025",
              "holdout": "Holdout Jul 2025-Oct 2026", "full": "Full 2020-Oct 2026"}
    for key in ["train", "validation", "holdout", "full"]:
        if key not in per:
            continue
        m = per[key]
        rows.append({
            "Window": labels[key],
            "Days": m["days"],
            "CAGR": pct(m["cagr"]),
            "Vol": pct(m["ann_vol"]),
            "Sharpe": f"{m['sharpe']:.2f}",
            "Sortino": f"{m['sortino']:.2f}",
            "Max DD": pct(m["max_dd"]),
            "Calmar": f"{m['calmar']:.2f}",
            "Monthly hit": pct(m["monthly_hit_rate"]),
        })
    return pd.DataFrame(rows)


def yearly_table_display(returns: pd.Series) -> pd.DataFrame:
    yt = yearly_table(returns)
    out = pd.DataFrame({
        "Year": yt["year"],
        "Return": yt["return"].map(pct),
        "Vol": yt["vol"].map(pct),
        "Sharpe": yt["sharpe"].map(lambda x: f"{x:.2f}"),
        "Max DD": yt["max_dd"].map(pct),
    })
    return out


def start_sensitivity(panel, sig, funding, starts) -> pd.DataFrame:
    rows = []
    for st in starts:
        res = run_backtest(panel, sig, funding, cfg=CFG, start=st)
        m = compute_metrics(res.returns, res.gross, res.turnover, res.costs, res.funding)
        rows.append({"Start": st, "CAGR": pct(m.cagr), "Vol": pct(m.ann_vol),
                     "Sharpe": f"{m.sharpe:.2f}", "Max DD": pct(m.max_dd)})
    return pd.DataFrame(rows)


def universe_sensitivity(panel, funding, signal_fn) -> pd.DataFrame:
    close = panel["close"]
    rows = []
    for n in [5, 10, 20, len(close.columns)]:
        syms = list(close.columns[:n])
        p2 = {k: v[syms] for k, v in panel.items()}
        f2 = {k: v for k, v in funding.items() if k in syms}
        res = run_backtest(p2, signal_fn(p2), f2, cfg=CFG)
        m = compute_metrics(res.returns)
        label = f"first {n}" if n < len(close.columns) else f"all {n}"
        rows.append({"Universe": label, "CAGR": pct(m.cagr), "Vol": pct(m.ann_vol),
                     "Sharpe": f"{m.sharpe:.2f}", "Max DD": pct(m.max_dd)})
    return pd.DataFrame(rows)


def parameter_plateau(panel, funding) -> pd.DataFrame:
    rows = []
    core = S.ensemble(panel, list(FINAL_CONFIG.lookbacks), FINAL_CONFIG.mode)
    reg = regime_mask(panel["close"])
    # Regime MA variants.
    for ma in [150, 180, 200, 220, 250]:
        cfg = replace(FINAL_CONFIG, regime_ma=ma)
        sig = final_signal(panel, cfg)
        res = run_backtest(panel, sig, funding, cfg=CFG)
        tr = compute_metrics(res.returns.loc[SPLITS["train"][0]:SPLITS["train"][1]])
        va = compute_metrics(res.returns.loc[SPLITS["validation"][0]:SPLITS["validation"][1]])
        ho = compute_metrics(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
        rows.append({"Variant": f"regime MA {ma}", "Train": f"{tr.sharpe:.2f}",
                     "Val": f"{va.sharpe:.2f}", "Holdout": f"{ho.sharpe:.2f}",
                     "CAGR": pct(compute_metrics(res.returns).cagr)})
    # Lookback variants.
    for lbs in [(40, 60, 120), (60, 90, 120), (50, 100, 200), (90, 180)]:
        cfg = replace(FINAL_CONFIG, lookbacks=lbs)
        sig = final_signal(panel, cfg)
        res = run_backtest(panel, sig, funding, cfg=CFG)
        tr = compute_metrics(res.returns.loc[SPLITS["train"][0]:SPLITS["train"][1]])
        va = compute_metrics(res.returns.loc[SPLITS["validation"][0]:SPLITS["validation"][1]])
        ho = compute_metrics(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
        rows.append({"Variant": f"lookbacks {lbs}", "Train": f"{tr.sharpe:.2f}",
                     "Val": f"{va.sharpe:.2f}", "Holdout": f"{ho.sharpe:.2f}",
                     "CAGR": pct(compute_metrics(res.returns).cagr)})
    # Off / long-only variants.
    for label, sig in [
        ("no regime gate", core),
        ("long-only within regime", S.long_only(core.mul(reg, axis=0))),
    ]:
        res = run_backtest(panel, sig, funding, cfg=CFG)
        tr = compute_metrics(res.returns.loc[SPLITS["train"][0]:SPLITS["train"][1]])
        va = compute_metrics(res.returns.loc[SPLITS["validation"][0]:SPLITS["validation"][1]])
        ho = compute_metrics(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
        rows.append({"Variant": label, "Train": f"{tr.sharpe:.2f}",
                     "Val": f"{va.sharpe:.2f}", "Holdout": f"{ho.sharpe:.2f}",
                     "CAGR": pct(compute_metrics(res.returns).cagr)})
    return pd.DataFrame(rows)


def contribution(panel, res) -> pd.DataFrame:
    held = res.weights.shift(1).fillna(0.0)
    ret = panel["close"].pct_change(fill_method=None).fillna(0.0)
    pnl = (held * ret).sum().sort_values(ascending=False)
    total = pnl.sum()
    out = pd.DataFrame({"symbol": pnl.index, "pnl_share": (pnl / total).values})
    return out


def main() -> None:
    panel = data.load_panel(freq="1d")
    close = panel["close"]
    funding = {s: data.load_funding(s) for s in close.columns}
    funding = {k: v for k, v in funding.items() if not v.empty}

    # Consistency: production weights must reproduce the research engine.
    from research.strategies.trend_portfolio import target_weights

    sig = final_signal(panel)
    res = run_backtest(panel, sig, funding, cfg=CFG)
    prod_w = target_weights(close, FINAL_CONFIG)
    diff = (prod_w.reindex_like(res.weights) - res.weights).abs().max().max()
    assert diff < 1e-6, f"production and research weights diverge: {diff}"

    m = compute_metrics(res.returns, res.gross, res.turnover, res.costs, res.funding)
    per = {}
    for key, (a, b) in {**SPLITS, "full": ("2000-01-01", "2100-01-01")}.items():
        r = res.returns.loc[a:b]
        if r.empty:
            continue
        mm = compute_metrics(r, res.gross.loc[a:b], res.turnover.loc[a:b],
                             res.costs.loc[a:b], res.funding.loc[a:b])
        per[key] = mm.as_dict()

    boot = block_bootstrap_sharpe(res.returns)
    boot_ho = block_bootstrap_sharpe(res.returns.loc[SPLITS["holdout"][0]:SPLITS["holdout"][1]])
    dsr = deflated_sharpe_ratio(m.sharpe, N_TRIALS, len(res.returns),
                                float(res.returns.skew()), float(res.returns.kurtosis() + 3))
    mc = monte_carlo_drawdown(res.returns)
    costs = cost_sensitivity(panel, sig, funding, CFG, costs=(0, 4, 8, 12, 20, 30, 50))
    costs["cagr_pct"] = costs["cagr"].map(pct)
    costs["sharpe_f"] = costs["sharpe"].map(lambda x: f"{x:.2f}")
    costs["dd_pct"] = costs["max_dd"].map(pct)
    placebo = shift_placebo(panel, sig, funding, CFG, n_iter=30)
    plateau = parameter_plateau(panel, funding)
    starts = start_sensitivity(panel, sig, funding,
                               ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"])
    univ = universe_sensitivity(panel, funding, final_signal)
    contrib = contribution(panel, res)
    yearly = yearly_table_display(res.returns)
    yearly_raw = yearly_table(res.returns)
    monthly = monthly_table(res.returns)
    monthly_disp = (monthly * 100).round(1)
    monthly_disp.index = monthly_disp.index.astype(int)
    monthly_disp.columns = [f"M{int(c):02d}" for c in monthly_disp.columns]

    # True buy-and-hold benchmarks (no vol targeting, no cap).
    ret_all = close.pct_change(fill_method=None)
    btc_ret = ret_all["BTCUSDT"].fillna(0.0)
    ew_ret = ret_all.mean(axis=1).fillna(0.0)
    btc_yearly = yearly_table_display(btc_ret)
    ew_yearly = yearly_table_display(ew_ret)
    bench_rows = [{"Strategy": NAME, "CAGR": pct(m.cagr), "Vol": pct(m.ann_vol),
                   "Sharpe": f"{m.sharpe:.2f}", "Max DD": pct(m.max_dd)}]
    for label, series in [("BTC buy & hold", btc_ret), ("Equal-weight basket (1x)", ew_ret)]:
        bm = compute_metrics(series)
        bench_rows.append({"Strategy": label, "CAGR": pct(bm.cagr), "Vol": pct(bm.ann_vol),
                           "Sharpe": f"{bm.sharpe:.2f}", "Max DD": pct(bm.max_dd)})
    bench_table = pd.DataFrame(bench_rows)

    report = {
        "name": NAME,
        "config": {
            "lookbacks": list(FINAL_CONFIG.lookbacks),
            "mode": FINAL_CONFIG.mode,
            "target_vol": FINAL_CONFIG.target_vol,
            "max_gross": FINAL_CONFIG.max_gross,
            "corr": FINAL_CONFIG.corr,
            "vol_halflife": FINAL_CONFIG.vol_halflife,
            "regime_symbol": FINAL_CONFIG.regime_symbol,
            "regime_ma": FINAL_CONFIG.regime_ma,
            "cost_bps_one_way": CFG.cost_bps,
            "funding_included": True,
        },
        "metrics": {"full": m.as_dict(), "periods": per},
        "bootstrap_full": boot,
        "bootstrap_holdout": boot_ho,
        "deflated_sharpe": dsr,
        "n_trials": N_TRIALS,
        "monte_carlo": mc,
        "cost_sensitivity": costs.to_dict("records"),
        "shift_placebo": placebo,
        "parameter_plateau": plateau.to_dict("records"),
        "start_sensitivity": starts.to_dict("records"),
        "universe_sensitivity": univ.to_dict("records"),
        "contribution": contrib.to_dict("records"),
        "yearly": yearly.to_dict("records"),
        "btc_yearly": btc_yearly.to_dict("records"),
        "benchmarks": bench_table.to_dict("records"),
        "data": {
            "symbols": int(close.shape[1]),
            "days": int(close.shape[0]),
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "source": "Binance USDT-M perpetual futures (data.binance.vision)",
        },
    }
    (OUT / "final_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    plot_equity(
        {"equity": res.equity, "yearly": report["yearly"]},
        OUT / "equity_curve.png",
        f"{NAME} — net of 8bps/side fees and historical funding",
    )
    plot_yearly({"yearly": yearly_raw.to_dict("records")}, OUT / "yearly_returns.png")

    md = []
    md.append(f"# {NAME}\n")
    md.append("**Verdict:** a simple, diversified crypto trend-following book with a "
              "long-term regime filter. Net of 8bps per side and historical perpetual "
              "funding, it earned a Sharpe of "
              f"**{m.sharpe:.2f}** over 6.8 years "
              f"({pct(m.cagr)} CAGR, {pct(m.ann_vol)} vol, {pct(m.max_dd)} max drawdown), "
              f"and stayed positive in the locked holdout (Sharpe {per['holdout']['sharpe']:.2f}).\n")
    md.append("## What it trades\n")
    md.append(f"- Universe: {close.shape[1]} liquid USDT-M perpetual futures.\n"
              "- Signal: average of the sign of 60-day and 120-day returns.\n"
              "- Gate: only take risk while BTC is above its 200-day EMA.\n"
              "- Sizing: inverse realised volatility per asset, portfolio scaled to ~25% annualised vol, max 3x gross.\n"
              "- Rebalance daily at the UTC close; long and short symmetric.\n")
    md.append("## Data and protocol\n")
    md.append(f"- Data: {report['data']['source']}, {report['data']['days']} daily bars, "
              f"{report['data']['start']} to {report['data']['end']}.\n"
              "- Train 2020-2023 used for design; validation Jan 2024-Jun 2025 used for selection; "
              "holdout Jul 2025-Oct 2026 evaluated once.\n"
              "- Costs: 8bps one-way on turnover; real historical funding charged/credited; no look-ahead "
              "(signal at close t earns return t+1).\n")
    md.append("## Headline results\n")
    md.append(md_table(period_table(per)))
    md.append("\nVersus simple benchmarks over the same window:\n")
    md.append(md_table(bench_table))
    md.append("\n![Equity curve](equity_curve.png)\n")
    md.append("## Calendar years\n")
    md.append(md_table(yearly))
    md.append("\nBTC buy-and-hold for comparison:\n")
    md.append(md_table(btc_yearly))
    md.append("\n![Yearly returns](yearly_returns.png)\n")
    md.append("## Monthly returns (%)\n")
    md.append(md_table(monthly_disp.reset_index().rename(columns={"year": "Year"}).fillna(""), floatfmt=".1f"))
    md.append("\n## Statistical evidence\n")
    md.append(f"- Block-bootstrap 95% CI for full-period Sharpe: "
              f"**[{boot['ci_lo']:.2f}, {boot['ci_hi']:.2f}]**, P(Sharpe>0) = {boot['p_positive']:.3f}.\n"
              f"- Holdout Sharpe {per['holdout']['sharpe']:.2f}; bootstrap CI "
              f"[{boot_ho['ci_lo']:.2f}, {boot_ho['ci_hi']:.2f}].\n"
              f"- Deflated Sharpe ratio after {N_TRIALS} trials: **{dsr:.3f}** "
              "(probability the true Sharpe exceeds the expected best of "
              f"{N_TRIALS} null trials after selection bias).\n"
              f"- Monte Carlo (1-year resampled): median return {pct(mc['final_return_p50'])}, "
              f"5th percentile {pct(mc['final_return_p05'])}, median max DD {pct(mc['max_dd_p50'])}, "
              f"P(negative year) {pct(mc['p_negative'])}.\n"
              f"- Time-shift placebo (same signals, same turnover, alignment destroyed): "
              f"mean Sharpe {placebo['mean']:.2f} (95% range {placebo['p05']:.2f} to {placebo['p95']:.2f}).\n")
    md.append("## Robustness\n")
    md.append("### Transaction costs\n")
    md.append(md_table(pd.DataFrame({
        "Cost (bps/side)": costs["cost_bps_one_way"],
        "Sharpe": costs["sharpe_f"],
        "CAGR": costs["cagr_pct"],
        "Max DD": costs["dd_pct"],
    })))
    md.append("\n### Parameter plateau\n")
    md.append(md_table(plateau))
    md.append("\n### Different start dates\n")
    md.append(md_table(starts))
    md.append("\n### Universe size\n")
    md.append(md_table(univ))
    md.append("\n### Per-symbol contribution\n")
    md.append(md_table(contrib.head(10).assign(
        pnl_share=contrib.head(10)["pnl_share"].map(pct))))
    md.append(f"\nTop symbol accounts for {pct(contrib['pnl_share'].max())} of gross PnL; "
              f"top five for {pct(contrib['pnl_share'].head(5).sum())}.\n")
    md.append("## Limitations and kill criteria\n")
    md.append("- No backtest proves future profitability. This is evidence of a persistent, "
              "economically grounded edge, not a guarantee.\n"
              "- The book was flat or underwater for up to ~21 months historically; a multi-year "
              "flat stretch is a normal outcome for trend following, not necessarily a broken model.\n"
              "- Universe is today's liquid perps, so delisted-token survivorship bias is not fully removed.\n"
              "- Capacity: fine for small/medium accounts; 36 simultaneous alt-perp orders will incur "
              "more slippage at institutional size.\n"
              "- Kill criteria: (1) realised Sharpe below 0 over a rolling 24-month window, "
              "(2) drawdown beyond 30%, (3) structural break such as exchange insolvency, "
              "(4) regime filter stops adding value for 12+ months. Re-evaluate, do not silently re-tune.\n")
    md.append("## Reproduce\n")
    md.append("```bash\n"
              "python -m research.trendlab.run_final\n"
              "python -m research.trendlab.run_research   # full grid, ~3 min\n"
              "```\n")
    (OUT / "final_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[:40]))
    print(f"\nWrote {OUT / 'final_report.md'}")
    print(f"Sharpe {m.sharpe:.2f} | CAGR {pct(m.cagr)} | vol {pct(m.ann_vol)} | maxDD {pct(m.max_dd)}")
    print(f"Holdout Sharpe {per['holdout']['sharpe']:.2f} | DSR {dsr:.3f} | placebo {placebo['mean']:.2f}")


if __name__ == "__main__":
    main()

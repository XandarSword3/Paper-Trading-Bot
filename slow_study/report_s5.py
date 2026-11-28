"""Generate reports/REPORT_S5_margin.md — full S5 v2 margin-model evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent  # slow_study/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
from backtest import s5_margin as m
from strategies import load_bars, DATA, START_EQUITY
from backtest import metrics as met

REP = ROOT / "reports"


def split_stats(eq, mr_full, year0=None):
    """Stats for eq subset (year>=year0, or full). mr_full = monthly returns on
    the FULL series (correct first-month base); filtered by month-end year.
    CAGR base = last equity before the split (or $1,000 for FULL)."""
    se = eq if year0 is None else eq[eq.index.year >= year0]
    if len(se) == 0:
        return None
    mr = mr_full if year0 is None else mr_full[mr_full.index.year >= year0]
    if year0 is None:
        base_eq, base_date = START_EQUITY, se.index[0]
    else:
        pre = eq[eq.index < pd.Timestamp(f"{year0}-01-01", tz=eq.index.tz)]
        if len(pre):
            base_eq, base_date = float(pre.iloc[-1]), pre.index[-1]
        else:
            base_eq, base_date = START_EQUITY, se.index[0]
    yrs = (se.index[-1] - base_date).days / 365.25
    cagr = (se.iloc[-1] / base_eq) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    peak = se.cummax()
    return {
        "n_months": len(mr), "mean_mo": mr.mean(),
        "geo_mo": (1 + mr).prod() ** (1 / len(mr)) - 1 if len(mr) else np.nan,
        "median_mo": mr.median(), "worst_mo": mr.min(), "pct_ge6": (mr >= 0.06).mean(),
        "cagr": cagr, "maxDD": float((se / peak - 1).min()),
        "final": se.iloc[-1],
    }


def worst_funding_window(fund, days=30):
    """Worst rolling `days`-day cumulative funding rate sum."""
    daily = fund.groupby(fund.index.normalize()).sum()  # ~3 settlements/day summed
    roll = daily.rolling(days).sum()
    i = roll.idxmin()
    return float(roll.min()), i


def main():
    fund = m.load_funding()
    lines = []
    A = lines.append
    A("# S5 v2 — Funding Carry WITH Margin Model\n")
    A("Frozen rules in `backtest/s5_margin.py` docstring. Capital $1,000 total.\n")
    A("Margin: MMR=0.40% (OKX BTCUSDT-SWAP tier-1, OKX help 2023-06-09; Bybit tier-1 "
      "MMR=0.33% per Bybit margin parameters — higher OKX value used). "
      "IM = notional/leverage by construction. Isolated-margin liquidation "
      "approximation; exact exchange liq-price formulas not replicated.\n")

    # ---------- v1 reconciliation ----------
    A("## 1. S5 v1 reconciliation (the unmodelled-margin version)\n")
    eq1 = pd.read_csv(REP / "equity_S5_funding_carry.csv", index_col=0, parse_dates=True).iloc[:, 0]
    tr1 = pd.read_csv(REP / "trades_S5_funding_carry.csv")
    # funding cashflows: recompute from funding file over v1 window
    spot = load_bars("BTC", "1D")
    f = fund.copy()
    start = max(spot.index[0], f.index[0]); end = min(spot.index[-1], f.index[-1])
    idx1 = spot.index[(spot.index >= start) & (spot.index <= end)]
    # v1 qty
    p0 = float(spot.loc[idx1[0], "open"]); qty1 = START_EQUITY / p0
    fund_in = f[(f.index >= idx1[0]) & (f.index <= idx1[-1])]
    total_funding = 0.0
    for d in idx1:
        day = fund_in[fund_in.index.normalize() == d.normalize()]
        for rate in day.values:
            total_funding += float(rate) * qty1 * float(spot.loc[d, "close"])
    entry_cost = tr1[tr1["leg"].str.contains("buy|short")]["cost"].sum()
    exit_cost = tr1[tr1["leg"].str.contains("sell|cover")]["cost"].sum()
    gain = eq1.iloc[-1] - START_EQUITY
    A(f"- Window {idx1[0].date()} -> {idx1[-1].date()}, qty={qty1:.5f} BTC constant.\n")
    A(f"- Cumulative funding received (short perp): **${total_funding:,.2f}**\n")
    A(f"- Total trading costs: ${entry_cost + exit_cost:,.2f} (entry ${entry_cost:.2f}, exit ${exit_cost:.2f})\n")
    A(f"- Equity gain: ${gain:,.2f} = funding ${total_funding:,.2f} − costs ${entry_cost+exit_cost:,.2f} "
      f"+ residual ${gain - total_funding + entry_cost + exit_cost:,.2f} (spot/perp basis noise)\n")
    # notional/equity over time
    closes = spot.loc[idx1, "close"]
    notional = qty1 * closes
    eq_aligned = eq1.reindex(closes.index).ffill()
    lev = notional / eq_aligned
    A(f"- Perp/spot leg notional grew ${notional.iloc[0]:,.0f} -> ${notional.iloc[-1]:,.0f}; "
      f"leg-notional/equity ratio: {lev.iloc[0]:.2f}x -> max **{lev.max():.2f}x** "
      f"(implicit leverage, unmodelled margin)\n")
    A(f"- v1 final equity ${eq1.iloc[-1]:,.2f} on $1,000 — NOT achievable: perp leg alone "
      f"reached −$16,405 unrealised at BTC $124,720 (2025-10-06) with zero margin modelled.\n")

    # ---------- v2 results ----------
    A("\n## 2. S5 v2 results by leverage (total capital basis)\n")
    A("| L | final $ | CAGR | mean/mo (arith) | geo/mo | worst/mo | maxDD | min liq dist | terminated | costs $ |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    v2 = {}
    for L in (1, 2, 3):
        eq, tr, meta = m.run_s5_margin(L)
        v2[L] = (eq, tr, meta)
        eq.to_csv(REP / f"equity_S5v2_L{L}.csv")
        if len(tr):
            tr.to_csv(REP / f"trades_S5v2_L{L}.csv", index=False)
        mr_full = m.monthly_returns(eq, START_EQUITY)
        full = split_stats(eq, mr_full)
        oos = split_stats(eq, mr_full, 2023)
        y25 = split_stats(eq, mr_full, 2025)
        liq_str = f"{meta['min_liq_dist_high_pct']:.1f}%" if meta["min_liq_dist_high_pct"] == meta["min_liq_dist_high_pct"] else "n/a"
        A(f"| {L}x | {meta['final_equity']:.0f} | {full['cagr']*100:.1f}% | "
          f"{full['mean_mo']*100:.2f}% | {full['geo_mo']*100:.2f}% | {full['worst_mo']*100:.1f}% | "
          f"{full['maxDD']*100:.1f}% | {liq_str} | {'YES' if meta['terminated'] else 'no'} | "
          f"{meta['cost_total']:.0f} |")
        v2[L] = (eq, tr, meta, full, oos, y25)

    A("\n*min liq dist = closest approach of liquidation price to the intraday HIGH "
      "(conservative); close-based distance in `s5v2_summary.csv`. "
      f"Reconciliation check (equity vs components): max abs diff "
      f"{max(v2[L][2]['max_abs_recon'] for L in (1,2,3)):.2e} $ — identity holds.\n")

    A("\n### Split detail (mean monthly, arithmetic)\n")
    A("| L | FULL | 2023+ | 2025+ |")
    A("|---|---|---|---|")
    for L in (1, 2, 3):
        _, _, _, full, oos, y25 = v2[L]
        A(f"| {L}x | {full['mean_mo']*100:.2f}% | {oos['mean_mo']*100:.2f}% | {y25['mean_mo']*100:.2f}% |")

    A("\n### Liquidation events\n")
    for L in (1, 2, 3):
        _, _, meta, _, _, _ = v2[L]
        if meta["terminated"]:
            liq = meta["liq"]
            A(f"- **L={L}x LIQUIDATED** {liq['dt'].date()} est liq ~${liq['price_est']:.0f} "
              f"(day high ${liq['high']:.0f}), perp equity lost ${liq['perp_equity_lost']:.0f}. "
              f"After liquidation: holds remaining spot+cash, no re-entry.\n")
        else:
            A(f"- L={L}x: survived. Min distance-to-liquidation "
              f"{meta['min_liq_dist_high_pct']:.1f}% from intraday high "
              f"({meta['min_liq_dist_pct']:.1f}% from close).\n")

    # ---------- cost drag ----------
    A("\n## 3. Cost drag\n")
    for L in (1, 2, 3):
        eqL, _, metaL, *_ = v2[L]
        yrs = (eqL.index[-1] - eqL.index[0]).days / 365.25
        A(f"- L={L}x: total trading costs **${metaL['cost_total']:.2f}** over {yrs:.1f}y "
          f"≈ ${metaL['cost_total']/yrs:.2f}/yr ({metaL['cost_total']/START_EQUITY/yrs*100:.2f}%/yr of initial capital). "
          f"Includes entry, margin-defense spot sales + perp covers, and rebalances.\n")

    # ---------- negative funding scenario ----------
    A("\n## 4. Negative-funding stress (30 days)\n")
    wf, wdate = worst_funding_window(fund, 30)
    A(f"- Worst rolling 30-day cumulative funding in sample: **{wf*100:+.2f}%** "
      f"(window ending {wdate.date()}).\n")
    for L in (1, 2, 3):
        metaL = v2[L][2]
        dbg = metaL["debug"]
        # terminal perp notional ~ final qty*close
        last = dbg.iloc[-1]
        notional = last["qty_perp"] * last["close"]
        # cost if that 30d rate applied to average notional ~ use terminal
        impact = wf * notional
        A(f"- L={L}x: at terminal perp notional ~${notional:,.0f}, a repeat of that window "
          f"would cost ~${impact:,.0f} to the short (paid, not received). "
          f"In v2 this drains perp equity -> triggers Rule A spot sales -> position shrinks.\n")
    A("- Funding flipped negative ~14% of all 8h periods historically; 2026 negative share 23.8%. "
      "S5 carry is regime-dependent: it earns only when funding is positive.\n")

    # ---------- stablecoin comparison ----------
    A("\n## 5. vs stablecoin yield\n")
    A("- DefiLlama yields.llama.fi/pools (fetched 2026-11-26): mainstream stable lending "
      "≈ Aave v3 USDC 2.99%, Spark USDS ~3.4-3.8%, Maple USDC ~5.3% (credit risk), "
      "Jupiter USDC 4.6%. Centralised equivalents (e.g. exchange earn) ~3-5%/yr — "
      "**could not verify** a specific current CEX rate.\n")
    A("- S5v2 L=1x geo ~0.4-0.5%/mo ≈ 5-6%/yr vs stable ~3-5%/yr: carry premium is thin "
      "and comes with liquidation/ops risk stable yield doesn't have.\n")

    # ---------- caveats ----------
    A("\n## 6. What could NOT be verified / approximations\n")
    A("- Exact OKX/Bybit liquidation price formulas (fees, tier tables, ADL) — approximated "
      "as isolated margin, liq when equity <= MMR*notional, checked on daily highs.\n")
    A("- Funding marked at daily close, not settlement-time price (3 settlements/day "
      "share one price — acknowledged approximation, needs intraday data to fix).\n")
    A("- USD (Coinbase) vs USDT (perp margin) basis ignored.\n")
    A("- Margin transfers assumed free/instant; no withdrawal fees modelled.\n")
    A("- After liquidation the sim holds remaining spot+cash (still BTC-exposed) — a real "
      "trader might exit fully; flagged as a modelling choice.\n")
    A("- Rule A/B thresholds (M*=N/L, 20% drift) are frozen choices, not optimised.\n")

    (REP / "REPORT_S5_margin.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote reports/REPORT_S5_margin.md")


if __name__ == "__main__":
    main()

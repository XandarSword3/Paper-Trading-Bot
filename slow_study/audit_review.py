"""Audit computations for the 10 review questions. Read-only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
import strategies as st
from backtest import metrics as met

DATA = ROOT / "data"
REP = ROOT / "reports"


def geo_monthly(equity):
    """Compounded (geometric) mean monthly return."""
    eq = equity.dropna()
    mr = met.monthly_returns(eq)
    if len(mr) == 0:
        return np.nan
    return float((1 + mr).prod() ** (1 / len(mr)) - 1)


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------- 1+8: geometric monthly + splits ----------
section("1/8: geometric monthly returns per strategy per split")
res = pd.read_csv(REP / "results.csv")
for name in res["label"].str.split("|").str[0].unique():
    eq = pd.read_csv(REP / f"equity_{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    for sname, mask in [("FULL", slice(None)),
                        ("2019_2022", (eq.index.year >= 2019) & (eq.index.year <= 2022)),
                        ("2023+", eq.index.year >= 2023)]:
        sub = eq.loc[mask] if sname != "FULL" else eq
        print(f"{name:24s} {sname:9s} geo/mo={geo_monthly(sub)*100:7.3f}%  "
              f"arith/mo={met.monthly_returns(sub).mean()*100:7.3f}%  "
              f"n_months={len(met.monthly_returns(sub))}")

# ---------- 2: funding stats ----------
section("2: funding annualised by year + negative share")
fund = pd.read_csv(DATA / "funding_BTCUSDT_binance.csv")
fund["dt"] = pd.to_datetime(fund["dt"], utc=True, format="mixed")
fund["year"] = fund["dt"].dt.year
# periods per year ~ 3/day*365; annualise mean rate * periods/year
for y, g in fund.groupby("year"):
    ann = g["fundingRate"].mean() * 3 * 365 * 100
    print(f"{y}: n={len(g):5d} mean_rate={g['fundingRate'].mean()*100:+.4f}%/8h "
          f"ann~{ann:+.1f}%  neg_share={(g['fundingRate']<0).mean()*100:.1f}%")
print(f"ALL: neg_share={(fund['fundingRate']<0).mean()*100:.2f}%  "
      f"min={fund['fundingRate'].min()*100:.3f}%  max={fund['fundingRate'].max()*100:.3f}%")

# ---------- 3: S5 perp leg worst excursion + full-period stats ----------
section("3: S5 perp-leg worst adverse excursion & full-period stats")
spot = st.load_bars("BTC", "1D")
# replicate entry: first common date
f = fund.copy().set_index("dt")
start = max(spot.index[0], f.index[0])
px_entry = float(spot.loc[start, "open"])
qty = st.START_EQUITY / px_entry
max_px = float(spot["close"].max())
min_perp_pnl = qty * (px_entry - max_px)
print(f"entry {start.date()} @ open {px_entry:.2f}, qty={qty:.5f}")
print(f"max close after entry: {max_px:.2f} on {spot['close'].idxmax().date()}")
print(f"worst perp-leg unrealised PnL: {min_perp_pnl:.2f} USD "
      f"(offset by spot; equity unaffected if margin existed)")
print(f"BTC max drawdown in window: {(spot['close']/spot['close'].cummax()-1).min()*100:.1f}%")
s5 = res[res["label"].str.startswith("S5")]
print(s5[["label", "mean_monthly", "worst_month", "max_drawdown",
          "longest_underwater_days", "cagr"]].to_string(index=False))

# ---------- 4: S5 cost lines ----------
section("4: S5 cost lines")
tr = pd.read_csv(REP / "trades_S5_funding_carry.csv")
print(tr.to_string(index=False))
entry = tr[tr["leg"].str.contains("buy|short")]
exit_ = tr[tr["leg"].str.contains("sell|cover")]
print(f"entry cost total: {entry['cost'].sum():.4f}  exit cost total: {exit_['cost'].sum():.4f}")
print(f"notional per leg ~ {entry['notional'].iloc[0]:.2f}")

# ---------- 6: look-ahead worked examples ----------
section("6: look-ahead worked examples (signal date -> exec date)")
for name in ["S1_ma_trend", "S3_donchian55", "S6_buyhold",
             "S2_tsmom_voltarget", "S4_xsmom_weekly"]:
    t = pd.read_csv(REP / f"trades_{name}.csv")
    if len(t) == 0:
        print(f"{name}: no trades logged")
        continue
    row = t.iloc[len(t) // 2]  # a mid-sample trade
    exec_dt = pd.Timestamp(row["dt"])
    # signal date = previous bar in that asset's index
    asset = row["asset"]
    bars_a = st.load_bars(asset, "1D")
    pos = bars_a.index.searchsorted(exec_dt)
    sig_dt = bars_a.index[pos - 1]
    print(f"{name}: exec {exec_dt.date()} open={bars_a['open'].iloc[pos]:.4g} | "
          f"signal close {sig_dt.date()} close={bars_a['close'].iloc[pos-1]:.4g} "
          f"(weight {row.get('weight_after','?')})")

# ---------- 7: S6 price check ----------
section("7: S6 buy-and-hold BTC price check")
btc = st.load_bars("BTC", "1D")
for d in ["2023-01-01", "2026-11-25"]:
    ts = pd.Timestamp(d, tz="UTC")
    i = btc.index.searchsorted(ts)
    print(f"{d}: close={btc['close'].iloc[i]:.2f}")
p0 = btc["close"].iloc[btc.index.searchsorted(pd.Timestamp("2023-01-01", tz="UTC"))]
p1 = btc["close"].iloc[-1]
print(f"BTC 2023-01-01 -> last: {p0:.2f} -> {p1:.2f} = {p1/p0:.3f}x")
eq6 = pd.read_csv(REP / "equity_S6_buyhold.csv", index_col=0, parse_dates=True).iloc[:, 0]
e23 = eq6[eq6.index >= "2023-01-01"]
print(f"S6 equity same window: {e23.iloc[0]:.2f} -> {e23.iloc[-1]:.2f} = "
      f"{e23.iloc[-1]/e23.iloc[0]:.3f}x")

# ---------- 9: CI table ----------
section("9: 95% vs 99% bootstrap CIs (2023+)")
oos = res[res["split"] == "2023_plus"]
print(oos[["label", "mean_monthly", "boot_lo95", "boot_hi95",
           "boot_lo99", "boot_hi99"]].to_string(index=False))

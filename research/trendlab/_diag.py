import pandas as pd
import numpy as np
from research.trendlab import data
from research.trendlab.engine import BacktestConfig, run_backtest
from research.trendlab.final import final_signal

panel = data.load_panel(freq="1d")
close = panel["close"]
funding = {s: data.load_funding(s) for s in close.columns}
cfg = BacktestConfig(max_asset_weight=0.15, min_asset_vol=0.15)
res = run_backtest(panel, final_signal(panel), funding, cfg=cfg)
ret = close.pct_change(fill_method=None).fillna(0)
held = res.weights.shift(1).fillna(0)
pnl = held * ret
m = pnl.resample("ME").sum()
tot = res.returns.resample("ME").sum()

print("month | strategy | top symbol (abs) | share of month gross pnl")
for d in m.index:
    row = m.loc[d]
    top = row.abs().idxmax()
    topv = row[top]
    share = abs(topv) / row.abs().sum() if row.abs().sum() > 0 else 0
    print("%s %+6.1f%%  top=%12s %+6.1f%% share=%4.1f%%" % (
        d.strftime("%Y-%m"), tot.loc[d] * 100, top, topv * 100, share * 100))

print("\ntop 5 single days by return (%):")
print((res.returns.sort_values(ascending=False).head(5) * 100).round(2).to_string())
print("\nworst 3 days (%):")
print((res.returns.sort_values().head(3) * 100).round(2).to_string())
print("\nOct-2025 top symbols (gross pnl, %):")
print((pnl.loc["2025-10-01":"2025-10-31"].sum().sort_values(ascending=False).head(8) * 100).round(1).to_string())
print("\n2025 top symbols (gross pnl, %):")
print((pnl.loc["2025-01-01":"2025-12-31"].sum().sort_values(ascending=False).head(8) * 100).round(1).to_string())

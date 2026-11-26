import pandas as pd
import numpy as np
from research.trendlab import data
from research.trendlab.engine import BacktestConfig, run_backtest
from research.trendlab.final import final_signal

panel = data.load_panel(freq="1d")
close = panel["close"]
ret = close.pct_change(fill_method=None)

print("=== data sanity ===")
print("rows", len(close), "symbols", close.shape[1])
print("non-positive closes:", int((close <= 0).sum().sum()))
print("duplicate index:", int(close.index.duplicated().sum()))
big = ret.stack()
print("\nlargest 15 absolute daily moves in the panel:")
print((big.reindex(big.abs().sort_values(ascending=False).index).head(15) * 100).round(1).to_string())

print("\n2025-10-10 returns (%), sorted:")
d = ret.loc["2025-10-10"]
print((d * 100).sort_values().round(1).to_string())

funding = {s: data.load_funding(s) for s in close.columns}
cfg = BacktestConfig(max_asset_weight=0.15, min_asset_vol=0.15)
sig = final_signal(panel)
res = run_backtest(panel, sig, funding, cfg=cfg)
print("\nweight on 2025-10-09 (applied to 10-10 return):")
w = res.weights.loc["2025-10-09"]
print(w[w != 0].round(3).to_string())
print("\nstrategy return on 2025-10-10: %.2f%%" % (res.returns.loc["2025-10-10"] * 100))
print("gross exposure %.2f" % res.gross.loc["2025-10-09"])
print("\nsum of daily returns 2025 excluding 10-10: %.1f%%" % (
    (res.returns.loc["2025-01-01":"2025-12-31"].sum() - res.returns.loc["2025-10-10"]) * 100))
print("2025 total return: %.1f%%" % (
    ((1 + res.returns.loc["2025-01-01":"2025-12-31"]).prod() - 1) * 100))

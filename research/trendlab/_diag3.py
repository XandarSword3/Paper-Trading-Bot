import pandas as pd
import numpy as np
from research.trendlab import data
from research.trendlab.engine import BacktestConfig, run_backtest
from research.trendlab.final import final_signal
from research.trendlab.stats import compute_metrics, yearly_table

panel = data.load_panel(freq="1d")
close = panel["close"]
funding = {s: data.load_funding(s) for s in close.columns}
cfg = BacktestConfig(max_asset_weight=0.15, min_asset_vol=0.15)
res = run_backtest(panel, final_signal(panel), funding, cfg=cfg)
r = res.returns

print("=== tail dependence ===")
for k in [0, 1, 3, 5, 10, 20]:
    rr = r.copy()
    if k:
        worst = rr.nlargest(k).index
        rr.loc[worst] = 0.0
    m = compute_metrics(rr)
    print("drop best %2d days: sharpe=%.2f cagr=%5.1f%% vol=%4.1f%% mdd=%5.1f%%" % (
        k, m.sharpe, m.cagr * 100, m.ann_vol * 100, m.max_dd * 100))

top = r.nlargest(20)
print("\ntop 20 days as %% of sum of positive returns: %.1f%%" % (
    top.sum() / r[r > 0].sum() * 100))
print("top 1 day: %.2f%% of total sum of returns" % (r.max() / r.sum() * 100))

print("\n=== beta / correlation ===")
btc = close["BTCUSDT"].pct_change(fill_method=None)
ew = close.pct_change(fill_method=None).mean(axis=1)
for name, x in [("BTC", btc), ("equal-weight alts", ew)]:
    d = pd.concat([r, x], axis=1).dropna()
    beta = np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / np.var(d.iloc[:, 1])
    corr = d.iloc[:, 0].corr(d.iloc[:, 1])
    print("%s: beta=%.2f corr=%.2f" % (name, beta, corr))

print("\n=== 2025 ex-crash ===")
r25 = r.loc["2025-01-01":"2025-12-31"].copy()
print("2025 total %.1f%%" % (((1 + r25).prod() - 1) * 100))
r25.loc["2025-10-10"] = 0
print("2025 ex 10-10 %.1f%%" % (((1 + r25).prod() - 1) * 100))

print("\n=== gross exposure distribution ===")
g = res.gross
print(g.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())
print("days gross > 2x: %d (%.1f%%)" % ((g > 2).sum(), (g > 2).mean() * 100))

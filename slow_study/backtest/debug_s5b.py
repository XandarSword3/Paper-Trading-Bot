import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import s5_margin as m
from strategies import COST_PER_SIDE, START_EQUITY, load_bars, DATA

L = 1
spot = load_bars("BTC", "1D")
fund = m.load_funding()
idx = spot.index
cs = max(idx[0], fund.index[0]); ce = min(idx[-1], fund.index[-1])
idx = idx[(idx >= cs) & (idx <= ce)]
capital = START_EQUITY

d0 = idx[0]; p0 = float(spot.loc[d0, "open"])
est_cost = 2 * COST_PER_SIDE
N0 = (capital - capital * est_cost) / (1 + 1 / L)
qty_spot = N0 / p0; qty_perp = qty_spot
cost_basis = qty_perp * p0
margin_posted = N0 / L
reserve = capital - N0 - margin_posted
entry_cost = N0 * COST_PER_SIDE * 2
reserve -= entry_cost
funding_acc = 0.0; terminated = False
fund_in = fund[(fund.index >= idx[0]) & (fund.index <= idx[-1])]
MMR = m.MMR

rows = []
for i, d in enumerate(idx):
    o = float(spot.loc[d, "open"]); h = float(spot.loc[d, "high"]); c = float(spot.loc[d, "close"])
    day_f = fund_in[fund_in.index.normalize() == d.normalize()]
    for rate in day_f.values:
        funding_acc += float(rate) * qty_perp * c
    def perp_eq(price):
        return margin_posted + funding_acc + (cost_basis - qty_perp * price)
    # --- replicate rules by calling the real function is hard; instead just
    # snapshot BEFORE rules using same order: we re-implement minimal marks
    # Actually simplest: import and monkeypatch not feasible. Print state by
    # re-running m.run_s5_margin is opaque. So we inline-copy is too long.
    break
print("use instrumented copy instead")

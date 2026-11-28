import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import s5_margin as m

# instrumented rerun of L=1 printing state every ~90 days
spot = m.load_bars("BTC", "1D")
fund = m.load_funding()
idx = spot.index
cs = max(idx[0], fund.index[0]); ce = min(idx[-1], fund.index[-1])
idx = idx[(idx >= cs) & (idx <= ce)]

# monkeypatch: copy of run loop with prints is complex; instead inspect outputs
tr = pd.read_csv("reports/trades_S5v2_L1.csv")
print(tr["action"].value_counts())
print(tr.head(10).to_string())
print(tr.tail(10).to_string())
eq = pd.read_csv("reports/equity_S5v2_L1.csv", index_col=0, parse_dates=True).iloc[:,0]
print("equity min:", eq.min(), "at", eq.idxmin())
print(eq.resample("YE").last())

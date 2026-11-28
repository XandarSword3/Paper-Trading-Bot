import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import s5_margin as m

eq, tr, meta = m.run_s5_margin(1)
dbg = meta["debug"]
w = dbg[(dbg["dt"] >= "2022-12-28") & (dbg["dt"] <= "2023-01-20")]
cols = ["dt","close","reserve","qty_spot","qty_perp","margin_posted","funding_acc","cost_basis","E_p","E_total"]
print(w[cols].to_string())
print("\nTRADES in window:")
tt = tr[(tr["dt"] >= "2022-12-28") & (tr["dt"] <= "2023-01-20")]
print(tt.to_string())

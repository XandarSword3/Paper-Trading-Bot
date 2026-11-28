import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import s5_margin as m

eq, tr, meta = m.run_s5_margin(1)
dbg = meta["debug"]
dbg["spot_val"] = dbg["qty_spot"] * dbg["close"]
dbg["perp_notional"] = dbg["qty_perp"] * dbg["close"]
dbg["check"] = dbg["reserve"] + dbg["spot_val"] + dbg["E_p"] - dbg["E_total"]
print("identity check max abs:", dbg["check"].abs().max())
# find first day E_total < 500
bad = dbg[dbg["E_total"] < 500]
print("first E_total<500:", bad["dt"].iloc[0] if len(bad) else "never")
# print around 2022-2023 transition
w = dbg[(dbg["dt"] >= "2022-06-01") & (dbg["dt"] <= "2023-06-01")]
print(w.iloc[::30][["dt","close","reserve","qty_spot","qty_perp","margin_posted","funding_acc","cost_basis","E_p","E_total"]].to_string())

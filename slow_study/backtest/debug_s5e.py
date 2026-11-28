import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
import s5_margin as m

for L in (1, 2, 3):
    eq, tr, meta = m.run_s5_margin(L)
    dbg = meta["debug"]
    # recompute liq distance daily: p_liq vs close
    # p_liq = (margin_posted + funding_acc + cost_basis) / (qty_perp*(1+MMR))
    q = dbg["qty_perp"].replace(0, np.nan)
    p_liq = (dbg["margin_posted"] + dbg["funding_acc"] + dbg["cost_basis"]) / (q * (1 + m.MMR))
    dist = (p_liq - dbg["close"]) / dbg["close"]
    i = dist.idxmin()
    print(f"L={L}: min dist {dist.min()*100:.1f}% on {dbg.loc[i,'dt'].date()} "
          f"close={dbg.loc[i,'close']:.0f} p_liq={p_liq.loc[i]:.0f} "
          f"E_p={dbg.loc[i,'E_p']:.1f} qty_perp={dbg.loc[i,'qty_perp']:.5f} "
          f"margin={dbg.loc[i,'margin_posted']:.1f} fund={dbg.loc[i,'funding_acc']:.1f} cb={dbg.loc[i,'cost_basis']:.1f}")
    print(f"   terminated={meta['terminated']} liq={meta['liq']}")
    # when did qty_perp first get tiny / E_p go negative?
    neg = dbg[dbg["E_p"] < 0]
    print(f"   first E_p<0: {neg['dt'].iloc[0].date() if len(neg) else 'never'}")

import pandas as pd

b = pd.read_csv("data/funding_BTCUSDT_binance.csv")
b["dt"] = pd.to_datetime(b["dt"], utc=True, format="mixed")
b = b.set_index("dt")["fundingRate"]
y = pd.read_csv("data/funding_BTCUSDT_bybit.csv")
y["dt"] = pd.to_datetime(y["dt"], utc=True)
y = y.set_index("dt")["fundingRate"]
o = pd.read_csv("data/BTC-USDT-SWAP_funding.csv")
o["dt"] = pd.to_datetime(o["dt"], utc=True, format="mixed")
o = o.set_index("dt")["fundingRate"]


def align(a, b, name):
    aa = a.reset_index().rename(columns={"dt": "t", "fundingRate": "x"}).sort_values("t")
    bb = b.reset_index().rename(columns={"dt": "t", "fundingRate": "y"}).sort_values("t")
    m = pd.merge_asof(aa, bb, on="t", tolerance=pd.Timedelta("1h")).dropna()
    diff = (m["x"] - m["y"]) * 100
    corr = m["x"].corr(m["y"])
    print(f"{name}: n={len(m)} mean_diff={diff.mean():+.4f}%/8h "
          f"corr={corr:.4f} maxabs={diff.abs().max():.4f}%")


align(b, y, "binance vs bybit")
align(b, o, "binance vs okx  ")
align(y, o, "bybit   vs okx  ")

for nm, s in [("binance", b), ("bybit", y), ("okx", o)]:
    w = s[(s.index >= y.index.min()) & (s.index <= y.index.max())]
    print(f"{nm}: cum funding over bybit window = {w.sum()*100:+.3f}%  n={len(w)}")

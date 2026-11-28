"""S5: delta-neutral funding carry — BTC spot long + BTC-USDT-SWAP perp short.

Frozen rule: at the first common date of spot/funding history, spend ~all
equity on spot BTC (taker) and short the same coin quantity on the perp
(taker). Hold to the end. Funding cashflows are applied at each historical
funding timestamp: fundingRate > 0 means longs pay shorts, so the SHORT perp
RECEIVES rate * perp_notional. If fundingRate < 0 the short PAYS.

Costs: 0.15% per side on spot entry/exit and perp entry/exit notional.
Mark-to-market daily at spot close; perp PnL offsets the spot leg.
Only the overlapping window where BOTH spot bars and funding history exist
is simulated (documented coverage limitation).
"""
import numpy as np
import pandas as pd

from strategies import COST_PER_SIDE, MIN_NOTIONAL_USD, START_EQUITY, load_bars, DATA


def run_s5(start_equity=START_EQUITY):
    spot = load_bars("BTC", "1D")
    # Prefer deep Binance funding history (OKX public endpoint only serves ~3
    # months — verified). Fall back to the shallow OKX file if absent.
    fpath = DATA / "funding_BTCUSDT_binance.csv"
    source = "binance_fapi"
    if not fpath.exists():
        fpath = DATA / "BTC-USDT-SWAP_funding.csv"
        source = "okx_public_shallow"
    if not fpath.exists():
        raise FileNotFoundError("funding history missing — run fetch_data.py / fetch_funding_binance.py")
    fund = pd.read_csv(fpath)
    fund["dt"] = pd.to_datetime(fund["dt"], utc=True, format="mixed", errors="coerce")
    fund = fund.dropna(subset=["dt"]).set_index("dt").sort_index()
    print(f"S5 funding source: {source} ({fpath.name}), {len(fund)} rows "
          f"{fund.index[0]} -> {fund.index[-1]}")

    idx = spot.index
    common_start = max(idx[0], fund.index[0])
    common_end = min(idx[-1], fund.index[-1])
    idx = idx[(idx >= common_start) & (idx <= common_end)]
    if len(idx) < 30:
        raise RuntimeError("insufficient overlap between spot and funding history")

    cash = start_equity
    qty = 0.0  # BTC coins, long spot / short perp
    entered = False
    equity = pd.Series(np.nan, index=idx, dtype=float)
    trades = []
    funding_events = []

    # entry at first day's OPEN
    first = idx[0]
    px_open = float(spot.loc[first, "open"])
    notional = start_equity
    if notional >= MIN_NOTIONAL_USD:
        spot_cost = notional * COST_PER_SIDE
        qty = notional / px_open  # coins bought with full notional; cost is a fee
        cash = start_equity - notional - spot_cost
        perp_notional = qty * px_open
        perp_cost = perp_notional * COST_PER_SIDE
        cash -= perp_cost
        entered = True
        trades.append({"dt": first, "leg": "spot_buy", "price": px_open,
                       "notional": perp_notional, "cost": spot_cost})
        trades.append({"dt": first, "leg": "perp_short", "price": px_open,
                       "notional": perp_notional, "cost": perp_cost})

    fund_in_window = fund[(fund.index >= idx[0]) & (fund.index <= idx[-1])]

    for d in idx:
        day_fund = fund_in_window[fund_in_window.index.normalize() == d.normalize()]
        if entered and len(day_fund):
            mark = float(spot.loc[d, "close"])
            perp_notional = qty * mark
            for _, row in day_fund.iterrows():
                cf = float(row["fundingRate"]) * perp_notional  # short receives if >0
                cash += cf
                funding_events.append({"dt": row.name, "rate": float(row["fundingRate"]),
                                       "notional": perp_notional, "cashflow": cf})
        close_px = float(spot.loc[d, "close"])
        if entered:
            perp_pnl = qty * (px_open - close_px)
            eq = cash + qty * close_px + perp_pnl
        else:
            eq = cash
        equity.loc[d] = eq

    # exit at last day's close (taker both legs)
    last = idx[-1]
    close_px = float(spot.loc[last, "close"])
    if entered:
        spot_notional = qty * close_px
        spot_cost = spot_notional * COST_PER_SIDE
        cash += spot_notional - spot_cost
        perp_pnl = qty * (px_open - close_px)
        cash += perp_pnl
        perp_cost = spot_notional * COST_PER_SIDE
        cash -= perp_cost
        trades.append({"dt": last, "leg": "spot_sell", "price": close_px,
                       "notional": spot_notional, "cost": spot_cost})
        trades.append({"dt": last, "leg": "perp_cover", "price": close_px,
                       "notional": spot_notional, "cost": perp_cost})
        equity.iloc[-1] = cash

    return equity.dropna(), pd.DataFrame(trades), pd.DataFrame(funding_events)

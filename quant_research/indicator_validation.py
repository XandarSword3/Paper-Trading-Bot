"""
Indicator validation against Binance's own chart numbers.

Ground truth is a SOLUSDT-perp 3m chart screenshot showing the *forming* candle. Its
axis label was reported as 2025-05-01 15:06 with an unknown timezone, which does not
identify the bar (see REPORT section 17) - so the bar is identified from the data
instead, using the two volume moving averages, which are exact arithmetic identities:

    MA5  = (vol[B-4] + vol[B-3] + vol[B-2] + vol[B-1] + F) / 5
    MA10 = (vol[B-9] + ... + vol[B-1] + F) / 10

where B is the forming bucket and F is its partial volume. Only one bucket and one
sub-minute offset reproduce both published averages. The indicator match then confirms it.

This script writes NEW files; nothing committed is overwritten:
    quant_research/results/indicator_validation.json
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from quant_research.core.indicators import compute_indicators  # noqa: E402

DATA = "quant_research/data/processed/SOLUSDT_1m.parquet"
OUT = "quant_research/results/indicator_validation.json"

# Transcribed from the screenshot. Tolerance per indicator per the audit request.
GROUND = {
    "last_price": 151.34, "chart_high": 152.38, "chart_low": 150.25,
    "boll_up": 152.12, "boll_mb": 151.69, "boll_dn": 151.26,
    "rsi6": 36.80,
    "stochrsi_k": 28.30, "stochrsi_d": 21.83,
    "macd_dif": -0.08, "macd_dea": -0.03, "macd_hist": -0.05,
    "volume": 5048.36, "vol_ma5": 34324.31, "vol_ma10": 32900.47,
}
TOL = {"boll": 0.02, "rsi": 1.0, "stochrsi": 2.0, "macd": 0.01, "volume_ma_rel": 0.001}
MINUTE_RANGE = ("2025-04-24", "2025-05-08")   # search window for the forming bucket


def load():
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    return df


def bars_3m(df, upto=None):
    """3m bars (label = bucket open, aligned to :00/:03) built from the 1m series."""
    d = df if upto is None else df.loc[:upto]
    b = d.resample("3min", label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return b.dropna()


def identify(vol3):
    """Find the forming bucket B and partial volume F from the two volume MA identities."""
    F = GROUND["volume"]
    v = vol3["volume"]
    hits = []
    idx = v.index
    lo, hi = np.searchsorted(idx, pd.Timestamp(MINUTE_RANGE[0], tz="UTC")), \
        np.searchsorted(idx, pd.Timestamp(MINUTE_RANGE[1], tz="UTC"))
    for i in range(max(9, lo), min(hi, len(idx))):
        s4 = float(v.iloc[i - 4:i].sum())
        s9 = float(v.iloc[i - 9:i].sum())
        e5 = abs(5.0 * GROUND["vol_ma5"] - F - s4)
        e10 = abs(10.0 * GROUND["vol_ma10"] - F - s9)
        # the published averages are rounded to 2dp, so a residual up to 5x0.005 (MA5) and
        # 10x0.005 (MA10) is expected from rounding alone; 0.06 is a safe envelope.
        if e5 < 0.06 and e10 < 0.06:
            hits.append({"bucket_open_utc": str(idx[i]), "vol_ma5_residual": e5,
                         "vol_ma10_residual": e10, "prev4_sum": s4, "prev9_sum": s9})
    return hits


def pre_fix_indicators(b):
    """The repo's indicator module as it stood when this validation first ran.

    REPORT 17.4 fixes the Bollinger bands to `ddof=0` (Binance's convention). This
    function applies that one historical difference on top of the current module so the
    committed artifact stays byte-for-byte reproducible after the fix: re-running this
    script must not silently rewrite the evidence of the defect it documents. Every
    other indicator in `compute_indicators` is unaffected by the fix.
    """
    ind = compute_indicators(b[["open", "high", "low", "close", "volume"]])
    c = b["close"]
    mb = c.rolling(20).mean()
    sd = c.rolling(20).std()          # ddof=1, the convention the repo used pre-fix
    ind["bb_upper"] = mb + 2.0 * sd
    ind["bb_lower"] = mb - 2.0 * sd
    ind["bb_basis"] = mb
    return ind


def features_variants(b):
    """The repo's indicator set plus labelled alternatives, at the last row of b."""
    c = b["close"]
    out = {}
    ind = pre_fix_indicators(b)
    r = ind.iloc[-1]
    out["repo"] = {"bb_up": float(r["bb_upper"]), "bb_mb": float(r["bb_basis"]),
                   "bb_dn": float(r["bb_lower"]), "rsi6": float(r["rsi6"]),
                   "stoch_k": float(r["stoch_k"]), "stoch_d": float(r["stoch_d"]),
                   "macd_dif": float(r["macd_line"]), "macd_dea": float(r["macd_signal"]),
                   "macd_hist": float(r["macd_hist"])}
    mb = c.rolling(20).mean().iloc[-1]
    alt = {}
    for dd in (0, 1):
        sd = c.rolling(20).std(ddof=dd).iloc[-1]
        alt[f"boll_ddof{dd}"] = {"bb_up": float(mb + 2 * sd), "bb_mb": float(mb),
                                 "bb_dn": float(mb - 2 * sd)}
    d = c.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    for name, gp, ls in (
            ("wilder_rma", gain.ewm(alpha=1 / 6, adjust=False).mean(), loss.ewm(alpha=1 / 6, adjust=False).mean()),
            ("ewm_adjust_true", gain.ewm(alpha=1 / 6, adjust=True).mean(), loss.ewm(alpha=1 / 6, adjust=True).mean()),
            ("sma_6", gain.rolling(6).mean(), loss.rolling(6).mean())):
        alt[f"rsi6_{name}"] = {"rsi6": float(100 - 100 / (1 + gp.iloc[-1] / ls.iloc[-1]))}
    for base in (6, 14):
        g = gain.ewm(alpha=1 / base, adjust=False).mean()
        ls_ = loss.ewm(alpha=1 / base, adjust=False).mean()
        r_ = 100 - 100 / (1 + g / ls_)
        raw = (r_ - r_.rolling(14).min()) / (r_.rolling(14).max() - r_.rolling(14).min())
        k1 = raw * 100
        k3 = (k1.rolling(3).mean())
        # both %K=%D variants carry the pair, so the comparison is like-for-like
        alt[f"stochrsi_base{base}_raw_k1_d1"] = {"stoch_k": float(k1.iloc[-1]),
                                                 "stoch_d": float(k1.rolling(3).mean().iloc[-1])}
        alt[f"stochrsi_base{base}_sma3_k3_d3"] = {"stoch_k": float(k3.iloc[-1]),
                                                  "stoch_d": float(k3.rolling(3).mean().iloc[-1])}
    for adj in (False, True):
        e12 = c.ewm(span=12, adjust=adj).mean()
        e26 = c.ewm(span=26, adjust=adj).mean()
        dif = e12 - e26
        dea = dif.ewm(span=9, adjust=adj).mean()
        alt[f"macd_adjust_{adj}"] = {"macd_dif": float(dif.iloc[-1]), "macd_dea": float(dea.iloc[-1]),
                                     "macd_hist": float(dif.iloc[-1] - dea.iloc[-1])}
    out["alternatives"] = alt
    out["close"] = float(c.iloc[-1])
    return out


def check(vals, prefix=""):
    """Tolerance check of a value dict against the ground truth."""
    res = {}
    pairs = [("bb_up", "boll_up", "boll"), ("bb_mb", "boll_mb", "boll"), ("bb_dn", "boll_dn", "boll"),
             ("rsi6", "rsi6", "rsi"), ("stoch_k", "stochrsi_k", "stochrsi"),
             ("stoch_d", "stochrsi_d", "stochrsi"),
             ("macd_dif", "macd_dif", "macd"), ("macd_dea", "macd_dea", "macd"),
             ("macd_hist", "macd_hist", "macd")]
    for k, g, fam in pairs:
        if k not in vals:
            continue
        delta = vals[k] - GROUND[g]
        res[prefix + k] = {"computed": vals[k], "ground": GROUND[g], "delta": delta,
                           "tolerance": TOL[fam], "pass": bool(abs(delta) <= TOL[fam])}
    return res


def main():
    df = load()
    vol3 = bars_3m(df)
    hits = identify(vol3)
    print("=" * 108)
    print("INDICATOR VALIDATION vs Binance's own 3m chart numbers (SOLUSDT perp)")
    print("=" * 108)
    print(f"ground truth: {json.dumps(GROUND)}")
    print(f"tolerances  : {json.dumps(TOL)}")
    print(f"\n1) bar identification from the two volume-MA identities (search {MINUTE_RANGE[0]}..{MINUTE_RANGE[1]}):")
    for h in hits:
        print("   bucket_open_utc={}  residual_ma5={:.6f}  residual_ma10={:.6f}".format(
            h["bucket_open_utc"], h["vol_ma5_residual"], h["vol_ma10_residual"]))
    if len(hits) != 1:
        print(f"   -> {len(hits)} candidates; identification is not unique")
    anchor = pd.Timestamp(hits[0]["bucket_open_utc"])

    results = {"candidates": hits, "ground_truth": GROUND, "tolerances": TOL,
               "anchor_bucket_open_utc": str(anchor), "sub_steps": {}}

    for nsub in (1, 2, 3):
        cut = anchor + pd.Timedelta(minutes=nsub)
        b = bars_3m(df, upto=cut - pd.Timedelta(minutes=1)).iloc[-2200:]
        v = features_variants(b)
        checks = check(v["repo"])
        ok = all(c["pass"] for c in checks.values())
        print(f"\n2) forming bucket {anchor:%Y-%m-%d %H:%M} UTC, sub-step {nsub} "
              f"(partial volume {b['volume'].iloc[-1]:.2f}, close {v['close']:.2f})  ->  {'ALL PASS' if ok else 'MISS'}")
        for k, c in checks.items():
            print("   {:12s} computed {:>9.4f}  ground {:>8.2f}  delta {:+.4f}  tol ±{:.2f}  {}".format(
                k, c["computed"], c["ground"], c["delta"], c["tolerance"], "pass" if c["pass"] else "MISS"))
        results["sub_steps"][f"sub_{nsub}"] = {
            "partial_volume": float(b["volume"].iloc[-1]), "close": v["close"],
            "repo_indicators": v["repo"], "checks": checks, "all_within_tolerance": ok,
            "alternatives": v["alternatives"],
            "alternative_checks": {name: check(vals) for name, vals in v["alternatives"].items()},
        }
    # The last COMPLETED 3m candle - the one before the forming bucket. The screenshot
    # shows the forming candle, so there is no published ground truth for this bar; it is
    # recorded (not checked) because the request asks for the indicator values on it, and
    # because a completed bar has no partial-volume ambiguity.
    prev = bars_3m(df, upto=anchor - pd.Timedelta(minutes=1))
    v_prev = features_variants(prev)
    results["convention_note"] = (
        "`repo` / `indicators` are the repository's compute_indicators with the Bollinger "
        "bands as they were when this validation first ran (sample std, ddof=1); REPORT 17.4 "
        "records the fix to Binance's population std. pre_fix_indicators() applies that one "
        "historical difference explicitly so re-running this script reproduces the committed "
        "artifact after the fix rather than erasing the evidence of the defect.")
    results["last_completed_candle"] = {
        "bucket_open_utc": str(prev.index[-1]),
        "bucket_close_utc": str(prev.index[-1] + pd.Timedelta(minutes=3)),
        "close": v_prev["close"],
        "indicators": v_prev["repo"],
        "bollinger_both_conventions": {
            "ddof0_population": v_prev["alternatives"]["boll_ddof0"],
            "ddof1_sample": v_prev["alternatives"]["boll_ddof1"],
        },
        "ground_truth_for_this_candle": None,
        "note": "the 3m candle that closed immediately before the forming bucket; the "
                "screenshot shows the forming candle, so no published values exist to "
                "compare this bar against and it is reported rather than tolerance-tested",
    }

    # Compare conventions family by family, with the SAME keys in every variant of a
    # family, so that sum|delta| is comparable and cannot be gamed by dropping a key.
    FAMILIES = {"boll": ["bb_up", "bb_mb", "bb_dn"],
                "rsi6": ["rsi6"],
                "stochrsi": ["stoch_k", "stoch_d"],
                "macd": ["macd_dif", "macd_dea", "macd_hist"]}
    win = results["sub_steps"]["sub_1"]
    family_checks = {"repo (current code)": win["checks"]}
    family_checks.update(win["alternative_checks"])
    family_rankings = {}
    for fam, keys in FAMILIES.items():
        rows = []
        for name, checks in family_checks.items():
            if not all(k in checks for k in keys):
                continue
            rows.append({"variant": name,
                         "sum_abs_delta": float(sum(abs(checks[k]["delta"]) for k in keys)),
                         "max_abs_delta": float(max(abs(checks[k]["delta"]) for k in keys)),
                         "all_within_tolerance": bool(all(checks[k]["pass"] for k in keys))})
        rows.sort(key=lambda r: r["sum_abs_delta"])
        family_rankings[fam] = rows
    results["convention_ranking_by_family_at_sub_1"] = family_rankings
    print("\n3) convention ranking at sub-step 1, per family (sum of |delta|, lower = closer to Binance):")
    for fam, rows in family_rankings.items():
        print(f"   [{fam}]")
        print("     {:<34s} {:>10s} {:>10s}  {}".format("variant", "sum|d|", "max|d|", "all within tol"))
        for row in rows:
            print("     {:<34s} {:>10.4f} {:>10.4f}  {}".format(
                row["variant"], row["sum_abs_delta"], row["max_abs_delta"], row["all_within_tolerance"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

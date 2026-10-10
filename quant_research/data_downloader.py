"""
Robust Bulk Data Downloader for Binance Vision.
Features:
- Local disk caching: raw zips saved in quant_research/data/raw/ to prevent re-downloads.
- Independent per-thread requests with appropriate read timeout (180s).
- Detailed progress and error logging.
- Audits: gaps, duplicates, UTC alignment, outliers, funding intervals.
"""
import os
import sys
import io
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

DATA_DIR = "quant_research/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
MONTHS = [f"{y}-{m:02d}" for y in [2024, 2025, 2026] for m in range(1, 13) if not (y == 2026 and m > 9)]
DAYS_OCT_2026 = [f"2026-10-{d:02d}" for d in range(1, 10)]


def download_file_to_disk(url: str, dest_path: str, retries: int = 3) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        if zipfile.is_zipfile(dest_path):
            return True
        else:
            try:
                os.remove(dest_path)
            except OSError:
                pass

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=(10, 180))
            if resp.status_code == 200:
                tmp_path = dest_path + f".tmp_{attempt}"
                with open(tmp_path, "wb") as f:
                    f.write(resp.content)
                if zipfile.is_zipfile(tmp_path):
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(tmp_path, dest_path)
                    return True
                else:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            elif resp.status_code == 404:
                return False
        except Exception as e:
            if attempt == retries - 1:
                print(f"Error downloading {url}: {e}", flush=True)
                return False
    return False


def fetch_one_kline(args: Tuple[str, str, str]) -> Tuple[str, Optional[pd.DataFrame]]:
    symbol, period_type, period_val = args
    if period_type == "monthly":
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{period_val}.zip"
        raw_name = f"{symbol}-1m-{period_val}.zip"
    else:
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{period_val}.zip"
        raw_name = f"{symbol}-1m-daily-{period_val}.zip"

    dest = os.path.join(RAW_DIR, raw_name)
    success = download_file_to_disk(url, dest)
    if not success:
        return f"{period_type}/{period_val}", None

    try:
        with zipfile.ZipFile(dest) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                first_line = f.readline().decode("utf-8", errors="ignore")
                f.seek(0)
                has_header = ("open" in first_line.lower()) or ("timestamp" in first_line.lower())
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    usecols=[0, 1, 2, 3, 4, 5],
                )
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = df["timestamp"].astype("int64")
                return f"{period_type}/{period_val}", df
    except Exception as e:
        print(f"Parse error on {dest}: {e}", flush=True)
        return f"err/{period_val}: {e}", None


def fetch_one_funding(args: Tuple[str, str, str]) -> Tuple[str, Optional[pd.DataFrame]]:
    symbol, period_type, period_val = args
    if period_type == "monthly":
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{period_val}.zip"
        raw_name = f"{symbol}-fundingRate-{period_val}.zip"
    else:
        url = f"https://data.binance.vision/data/futures/um/daily/fundingRate/{symbol}/{symbol}-fundingRate-{period_val}.zip"
        raw_name = f"{symbol}-fundingRate-daily-{period_val}.zip"

    dest = os.path.join(RAW_DIR, raw_name)
    success = download_file_to_disk(url, dest)
    if not success:
        return f"{period_type}/{period_val}", None

    try:
        with zipfile.ZipFile(dest) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                first_line = f.readline().decode("utf-8", errors="ignore")
                f.seek(0)
                has_header = (
                    ("calc_time" in first_line.lower())
                    or ("timestamp" in first_line.lower())
                    or ("funding" in first_line.lower())
                )
                if has_header:
                    fdf = pd.read_csv(f)
                    if "calc_time" in fdf.columns:
                        fdf = fdf.rename(columns={"calc_time": "timestamp", "last_funding_rate": "funding_rate"})
                    elif "funding_rate" not in fdf.columns and len(fdf.columns) >= 3:
                        fdf = fdf.rename(columns={fdf.columns[0]: "timestamp", fdf.columns[2]: "funding_rate"})
                else:
                    fdf = pd.read_csv(f, header=None)
                    fdf = fdf.iloc[:, [0, 2]]
                    fdf.columns = ["timestamp", "funding_rate"]

                fdf["timestamp"] = pd.to_numeric(fdf["timestamp"], errors="coerce")
                fdf["funding_rate"] = pd.to_numeric(fdf["funding_rate"], errors="coerce")
                fdf = fdf.dropna(subset=["timestamp", "funding_rate"])
                fdf["timestamp"] = fdf["timestamp"].astype("int64")
                return f"{period_type}/{period_val}", fdf[["timestamp", "funding_rate"]]
    except Exception as e:
        print(f"Funding parse error on {dest}: {e}", flush=True)
        return f"err/{period_val}: {e}", None


def fetch_symbol_klines(symbol: str) -> Tuple[pd.DataFrame, List[str]]:
    tasks = [((symbol, "monthly", ym)) for ym in MONTHS] + [((symbol, "daily", d)) for d in DAYS_OCT_2026]
    print(f"[{symbol}] Downloading 1m klines ({len(tasks)} files) with 5 threads...", flush=True)

    dfs = []
    missing_items = []

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_one_kline, t): t for t in tasks}
        done_cnt = 0
        for fut in as_completed(futures):
            done_cnt += 1
            label, res_df = fut.result()
            if res_df is not None:
                dfs.append(res_df)
                if done_cnt % 5 == 0 or done_cnt == len(tasks):
                    print(f"[{symbol}] Kline progress: {done_cnt}/{len(tasks)} files processed ({len(dfs)} successful)...", flush=True)
            else:
                missing_items.append(label)

    if not dfs:
        raise ValueError(f"No kline data downloaded for {symbol}")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="ms", utc=True)
    full_df = full_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"[{symbol}] Total ingested: {len(full_df):,} 1m bars. Missing archives: {missing_items}", flush=True)
    return full_df, missing_items


def fetch_symbol_funding(symbol: str) -> Tuple[pd.DataFrame, List[str]]:
    tasks = [((symbol, "monthly", ym)) for ym in MONTHS] + [((symbol, "daily", d)) for d in DAYS_OCT_2026]
    print(f"[{symbol}] Downloading funding rates ({len(tasks)} files) with 5 threads...", flush=True)

    dfs = []
    missing_items = []

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_one_funding, t): t for t in tasks}
        done_cnt = 0
        for fut in as_completed(futures):
            done_cnt += 1
            label, res_df = fut.result()
            if res_df is not None:
                dfs.append(res_df)
            else:
                missing_items.append(label)

    if not dfs:
        raise ValueError(f"No funding data downloaded for {symbol}")

    full_fdf = pd.concat(dfs, ignore_index=True)
    full_fdf["timestamp"] = pd.to_datetime(full_fdf["timestamp"], unit="ms", utc=True)
    full_fdf["funding_rate"] = pd.to_numeric(full_fdf["funding_rate"], errors="coerce")
    full_fdf = full_fdf.dropna().sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"[{symbol}] Ingested {len(full_fdf):,} funding records. Missing archives: {missing_items}", flush=True)
    return full_fdf, missing_items


def audit_dataset(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    audit = {}
    audit["name"] = name
    audit["row_count"] = len(df)
    audit["start"] = str(df["timestamp"].iloc[0])
    audit["end"] = str(df["timestamp"].iloc[-1])
    audit["timezone"] = str(df["timestamp"].dt.tz)

    audit["duplicates"] = int(df.duplicated(subset=["timestamp"]).sum())

    time_diffs = df["timestamp"].diff().dropna()
    expected_delta = pd.Timedelta(minutes=1)
    gap_count = (time_diffs > expected_delta).sum()
    max_gap = str(time_diffs.max()) if len(time_diffs) > 0 else "0"
    audit["gaps_over_1m"] = int(gap_count)
    audit["max_gap"] = max_gap

    audit["zero_or_negative_prices"] = int(((df["low"] <= 0) | (df["high"] <= 0) | (df["close"] <= 0)).sum())
    high_low_ratio = df["high"] / df["low"]
    audit["extreme_wicks_gt_20pct"] = int((high_low_ratio > 1.20).sum())

    return audit


def resample_klines(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df_indexed = df.set_index("timestamp")
    resampled = df_indexed.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def process_symbol(symbol: str) -> Dict[str, Any]:
    df_1m, missing_k = fetch_symbol_klines(symbol)
    df_fund, missing_f = fetch_symbol_funding(symbol)

    audit_res = audit_dataset(df_1m, symbol)
    audit_res["missing_kline_archives"] = missing_k
    audit_res["missing_funding_archives"] = missing_f
    audit_res["funding_settlements_count"] = len(df_fund)

    funding_diffs = df_fund["timestamp"].diff().dropna()
    audit_res["funding_intervals_observed"] = [str(x) for x in funding_diffs.value_counts().head(5).index]

    p_1m = os.path.join(PROCESSED_DIR, f"{symbol}_1m.parquet")
    p_fund = os.path.join(PROCESSED_DIR, f"{symbol}_funding.parquet")
    df_1m.to_parquet(p_1m, index=False)
    df_fund.to_parquet(p_fund, index=False)

    timeframe_rules = {"3m": "3min", "5m": "5min", "15m": "15min", "1h": "1h"}
    for tf_name, rule in timeframe_rules.items():
        print(f"[{symbol}] Resampling to {tf_name} ({rule})...", flush=True)
        res = resample_klines(df_1m, rule)
        res.to_parquet(os.path.join(PROCESSED_DIR, f"{symbol}_{tf_name}.parquet"), index=False)

    return audit_res


def main():
    print("=" * 80, flush=True)
    print("BINANCE VISION BULK DATA ACQUISITION & AUDIT", flush=True)
    print("=" * 80, flush=True)

    target_symbols = [s for s in sys.argv[1:] if not s.startswith("-")] if len(sys.argv) > 1 else SYMBOLS
    print(f"Target symbols to download: {target_symbols}", flush=True)

    audits = []
    for s in target_symbols:
        audit = process_symbol(s)
        audits.append(audit)
        print(f"\n--- AUDIT REPORT FOR {s} ---", flush=True)
        for k, v in audit.items():
            print(f"  {k}: {v}", flush=True)
        print("-" * 50, flush=True)

    report_path = os.path.join(DATA_DIR, "data_audit_report.json")
    with open(report_path, "w") as f:
        import json
        json.dump(audits, f, indent=2)

    print(f"\nSaved all files and {report_path} successfully.", flush=True)


if __name__ == "__main__":
    main()

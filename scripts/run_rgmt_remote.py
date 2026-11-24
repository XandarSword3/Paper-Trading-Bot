"""Remote daily execution runner for Regime-Gated Multi-Horizon Trend (RGMT).

Designed to run in GitHub Actions on a daily cron (00:05 UTC) or locally.

The runner is intentionally boring: it fetches prices, computes the frozen
research signal, marks the book, accrues funding, enforces the drawdown guard,
rebalances the paper state, and reports.  All money math lives in
``research.rgmt_paper`` and is unit-tested.

Safety properties
-----------------
* Idempotent: a second run on the same UTC day is a no-op unless ``--force``.
* Fail-closed: missing BTC, missing prices, or target limits violated => flat.
* Funding is charged on signed notional using the last 24h of funding prints.
* A 30% drawdown from peak flattens the book and latches ``halted``.
* State is written atomically; a lock prevents concurrent runners.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.rgmt_paper import (  # noqa: E402
    MAX_DRAWDOWN_GUARD,
    MAX_GROSS,
    COST_RATE,
    accrue_funding,
    already_ran_today,
    append_history,
    append_trade,
    flatten_targets,
    mark_to_market,
    new_state,
    rebalance,
    update_risk,
    validate_targets,
)
from research.strategies.trend_portfolio import TrendConfig, TrendPortfolio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rgmt_remote")


def _configure_stdout() -> None:
    """Avoid UnicodeEncodeError on Windows consoles when printing the report."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_configure_stdout()

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DATA_DIR / "rgmt_paper_state.json"
TARGETS_PATH = DATA_DIR / "rgmt_targets.csv"
LOCK_PATH = DATA_DIR / ".rgmt_paper.lock"

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT",
    "ICPUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
    "TIAUSDT", "SEIUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT",
    "TONUSDT", "ENAUSDT", "AAVEUSDT", "UNIUSDT", "RENDERUSDT", "FETUSDT",
    "ARUSDT",
]

MIN_TURNOVER_24H = 5_000_000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 3.0
FUNDING_LOOKBACK_HOURS = 24
FUNDING_LIMIT = 100


def _okx_turnover_usd(item: dict) -> float:
    """Convert an OKX ticker's 24h volume to USD notional.

    OKX reports ``volCcy24h`` in base-currency units (BTC, PEPE, ...), not USDT.
    Comparing that raw number to a USD threshold dropped BTC from the universe,
    which in turn silently disabled the regime gate.
    """
    try:
        return float(item.get("volCcy24h") or 0.0) * float(item.get("last") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sum_recent(rows: List[dict], rate_key: str, time_key: str) -> Optional[float]:
    """Sum funding prints inside the lookback window.

    Providers return newest-first.  If timestamps are missing, fall back to the
    three most recent prints (the standard 8-hour cadence).
    """
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(hours=FUNDING_LOOKBACK_HOURS)).timestamp() * 1000)
    total = 0.0
    counted = 0
    for row in rows:
        try:
            rate = float(row.get(rate_key))
        except (TypeError, ValueError):
            continue
        raw_time = row.get(time_key)
        if raw_time in (None, ""):
            if counted >= 3:
                break
            total += rate
            counted += 1
            continue
        try:
            ts = int(raw_time)
        except (TypeError, ValueError):
            continue
        if ts < cutoff_ms:
            continue
        total += rate
        counted += 1
    return total if counted else None


class ResilientMarketClient:
    """Direct HTTP market client fetching from Bybit with automatic OKX fallback for cloud/US runners."""

    def __init__(self, bybit_url: str = "https://api.bybit.com", okx_url: str = "https://www.okx.com") -> None:
        self.bybit_url = bybit_url
        self.okx_url = okx_url
        self.session = requests.Session()
        self.provider = "bybit"
        try:
            r = self.session.get(f"{self.bybit_url}/v5/market/tickers?category=linear&limit=1", timeout=5)
            if r.status_code == 200 and "result" in r.json():
                self.provider = "bybit"
                logger.info("Market data provider: Bybit Linear API")
            else:
                self.provider = "okx"
                logger.info("Market data provider: OKX API (Bybit returned %s)", r.status_code)
        except Exception as e:
            self.provider = "okx"
            logger.info("Market data provider: OKX API (Bybit probe failed: %s)", e)

    def _to_okx_inst(self, symbol: str) -> str:
        base = symbol.replace("USDT", "")
        if base.startswith("1000"):
            base = base[4:]
        return f"{base}-USDT-SWAP"

    def get_tickers(self) -> Dict[str, dict]:
        if self.provider == "bybit":
            try:
                url = f"{self.bybit_url}/v5/market/tickers"
                params = {"category": "linear"}
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json().get("result", {}).get("list", [])
                if data:
                    return {item["symbol"]: item for item in data if "symbol" in item}
            except Exception as e:
                logger.warning("Bybit tickers failed (%s), falling back to OKX...", e)
                self.provider = "okx"

        # OKX fallback
        url = f"{self.okx_url}/api/v5/market/tickers"
        params = {"instType": "SWAP"}
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        result = {}
        for item in data:
            inst = item.get("instId", "")
            if inst.endswith("-USDT-SWAP"):
                base = inst.replace("-USDT-SWAP", "")
                sym = f"{base}USDT"
                result[sym] = {
                    "symbol": sym,
                    "lastPrice": item.get("last", "0.0"),
                    "turnover24h": _okx_turnover_usd(item),
                }
                if base in ("PEPE", "SHIB", "FLOKI", "BONK", "SATS"):
                    result[f"1000{base}USDT"] = result[sym]
        return result

    def get_daily_klines(self, symbol: str, limit: int = 500) -> pd.Series:
        if self.provider == "bybit":
            try:
                url = f"{self.bybit_url}/v5/market/kline"
                params = {"category": "linear", "symbol": symbol, "interval": "D", "limit": limit}
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                raw = resp.json().get("result", {}).get("list", [])
                if raw:
                    df = pd.DataFrame(raw, columns=["start", "open", "high", "low", "close", "volume", "turnover"])
                    df["timestamp"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df = df.dropna(subset=["close"]).sort_values("timestamp")
                    today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
                    df = df[df["timestamp"] < today_utc]
                    if not df.empty:
                        return df.set_index("timestamp")["close"]
            except Exception as e:
                logger.warning("Bybit klines failed for %s (%s), trying OKX...", symbol, e)

        # OKX fallback
        try:
            inst_id = self._to_okx_inst(symbol)
            url = f"{self.okx_url}/api/v5/market/candles"
            params = {"instId": inst_id, "bar": "1D", "limit": min(limit, 300)}
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json().get("data", [])
            if not raw:
                return pd.Series(dtype=float)
            df = pd.DataFrame(raw, columns=["start", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df["timestamp"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp")
            today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
            df = df[df["timestamp"] < today_utc]
            if df.empty:
                return pd.Series(dtype=float)
            return df.set_index("timestamp")["close"]
        except Exception as e:
            logger.warning("OKX klines also failed for %s: %s", symbol, e)
            return pd.Series(dtype=float)

    def _bybit_funding(self, symbol: str) -> Optional[float]:
        try:
            resp = self.session.get(
                f"{self.bybit_url}/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": FUNDING_LIMIT},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json().get("result", {}).get("list", [])
            return _sum_recent(rows, "fundingRate", "fundingRateTimestamp")
        except Exception as exc:
            logger.debug("Bybit funding failed for %s: %s", symbol, exc)
            return None

    def _okx_funding(self, symbol: str) -> Optional[float]:
        try:
            resp = self.session.get(
                f"{self.okx_url}/api/v5/public/funding-rate-history",
                params={"instId": self._to_okx_inst(symbol), "limit": FUNDING_LIMIT},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            return _sum_recent(rows, "fundingRate", "fundingTime")
        except Exception as exc:
            logger.debug("OKX funding failed for %s: %s", symbol, exc)
            return None

    def _binance_funding(self, symbol: str) -> Optional[float]:
        try:
            resp = self.session.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": FUNDING_LIMIT},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json()
            return _sum_recent(rows, "fundingRate", "fundingTime")
        except Exception as exc:
            logger.debug("Binance funding failed for %s: %s", symbol, exc)
            return None

    def get_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Return the sum of funding rates over the last 24h per symbol.

        Missing data is treated as zero rather than aborting the run, but every
        fallback is logged so a silently broken provider is visible.
        """
        out: Dict[str, float] = {}
        for symbol in symbols:
            value = None
            if self.provider == "bybit":
                value = self._bybit_funding(symbol)
                if value is None:
                    value = self._okx_funding(symbol)
            else:
                value = self._okx_funding(symbol)
                if value is None:
                    value = self._bybit_funding(symbol)
            if value is None:
                value = self._binance_funding(symbol)
            if value is None:
                logger.warning("No funding data for %s; assuming 0", symbol)
                value = 0.0
            out[symbol] = float(value)
        return out


def fetch_panel(client: ResilientMarketClient, symbols: List[str]) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            s = client.get_daily_klines(sym)
            if not s.empty:
                series_map[sym] = s
        except Exception as err:
            logger.warning("Failed kline fetch for %s: %s", sym, err)

    if not series_map:
        raise RuntimeError("No candle data fetched from market data client")

    panel = pd.DataFrame(series_map).sort_index()
    logger.info(
        "Panel fetched: %d symbols x %d daily bars (last close %s)",
        panel.shape[1],
        panel.shape[0],
        panel.index[-1].strftime("%Y-%m-%d"),
    )
    return panel


def load_state(initial_capital: float = 10000.0) -> dict:
    base = new_state(initial_capital)
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                base.update(payload)
        except Exception as exc:
            logger.warning("Could not read state (%s); starting fresh", exc)
    base.setdefault("positions", {})
    base.setdefault("entry_prices", {})
    base.setdefault("history", [])
    base.setdefault("trades", [])
    base.setdefault("halted", False)

    # Older state files may contain explicit nulls (e.g. total_funding_cost),
    # which would break float arithmetic on the next run.  Normalise them.
    numeric_defaults = {
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "equity": initial_capital,
        "peak_equity": initial_capital,
        "realized_pnl": 0.0,
        "total_fees_paid": 0.0,
        "total_funding_cost": 0.0,
    }
    for key, default in numeric_defaults.items():
        try:
            base[key] = float(base.get(key, default))
        except (TypeError, ValueError):
            base[key] = float(default)
    try:
        base["runs"] = int(base.get("runs", 0))
    except (TypeError, ValueError):
        base["runs"] = 0

    for key in ("positions", "entry_prices"):
        raw_map = base.get(key) or {}
        if not isinstance(raw_map, dict):
            raw_map = {}
        clean: Dict[str, float] = {}
        for symbol, raw_value in raw_map.items():
            try:
                clean[str(symbol)] = float(raw_value)
            except (TypeError, ValueError):
                logger.warning("Dropping invalid %s value for %s: %r", key, symbol, raw_value)
        base[key] = clean

    return base


def save_state(state: dict) -> None:
    """Atomic write so a killed runner cannot leave a half-written state."""
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


class FileLock:
    def __init__(self, path: Path, stale_seconds: int = 3600) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.acquired = False

    def __enter__(self) -> "FileLock":
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.acquired = True
            return self
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_seconds:
                logger.warning("Removing stale lock (age %.0fs)", age)
                self.path.unlink(missing_ok=True)
                return self.__enter__()
            raise RuntimeError(f"another RGMT run holds {self.path} (age {age:.0f}s)")

    def __exit__(self, *exc: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as exc:
        logger.warning("Telegram notification failed: %s", exc)


def write_github_summary(report_md: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(report_md)
        except Exception as exc:
            logger.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", exc)


def validate_state(state: dict, prices: Dict[str, float]) -> List[str]:
    """Return a list of consistency errors; empty means healthy."""
    errors: List[str] = []
    equity, unrealised, _ = mark_to_market(state, prices)
    cash = float(state.get("cash", 0.0))
    if abs(float(state.get("equity", 0.0)) - equity) > 1e-6:
        errors.append(f"equity mismatch stored={state.get('equity')} recomputed={equity:.6f}")
    if not np.isfinite(cash):
        errors.append("cash is not finite")
    gross_notional = 0.0
    for symbol, qty in state.get("positions", {}).items():
        price = prices.get(symbol)
        if price is None:
            errors.append(f"missing mark price for open position {symbol}")
            continue
        notional = abs(float(qty) * float(price))
        gross_notional += notional
        if notional > 0.16 * equity + 10.0:
            errors.append(f"position {symbol} notional {notional:.2f} exceeds 16% of equity")
    if gross_notional > 3.05 * equity + 10.0:
        errors.append(f"gross notional {gross_notional:.2f} exceeds 3.05x equity {equity:.2f}")
    return errors


def _prices_from_sources(tickers: Dict[str, dict], panel: pd.DataFrame, symbols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (live prices, daily-close fallbacks)."""
    prices: Dict[str, float] = {}
    for symbol in symbols:
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        try:
            value = float(ticker.get("lastPrice", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            prices[symbol] = value

    fallback: Dict[str, float] = {}
    if not panel.empty:
        last = panel.ffill().iloc[-1]
        for symbol in symbols:
            if symbol in last and pd.notna(last[symbol]):
                fallback[symbol] = float(last[symbol])
    return prices, fallback


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Daily RGMT Runner")
    parser.add_argument("--mode", choices=["signal", "paper"], default="paper")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--force", action="store_true", help="Run even if the state already ran today")
    parser.add_argument("--resume", action="store_true", help="Clear a drawdown halt and reset the peak to current equity")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print targets without touching state")
    args = parser.parse_args()

    # Signal mode and dry-run are read-only, so they do not need the lock.
    lock_context = FileLock(LOCK_PATH) if (args.mode == "paper" and not args.dry_run) else _NullLock()
    with lock_context:
        state = load_state(args.capital)

        if args.resume:
            if state.get("halted"):
                logger.warning("Manual resume: clearing halt and resetting peak equity")
                state["halted"] = False
                state["halt_reason"] = None
                state["halted_at"] = None
                state["peak_equity"] = float(state.get("equity", args.capital))
                save_state(state)

        # Position symbols must be fetched even if they fall below the liquidity
        # filter, otherwise the runner can never close them.
        tracked = set(state.get("positions", {}).keys())

        client = ResilientMarketClient()
        all_tickers = client.get_tickers()

        liquid = []
        for symbol in UNIVERSE:
            if symbol == "BTCUSDT" or symbol in tracked:
                liquid.append(symbol)
                continue
            ticker = all_tickers.get(symbol)
            if not ticker:
                continue
            try:
                turnover = float(ticker.get("turnover24h", 0.0) or 0.0)
            except (TypeError, ValueError):
                turnover = 0.0
            if turnover >= MIN_TURNOVER_24H:
                liquid.append(symbol)
        # A tracked symbol outside the configured universe still needs a price.
        for symbol in tracked:
            if symbol not in liquid:
                liquid.append(symbol)

        logger.info("Filtered universe: %d / %d liquid contracts (> $5M turnover)", len(liquid), len(UNIVERSE))

        panel = fetch_panel(client, liquid)
        if "BTCUSDT" not in panel.columns:
            logger.error(
                "BTCUSDT is missing from the fetched panel; the regime gate cannot be "
                "evaluated. Forcing a flat book (no new risk)."
            )
        # These are the frozen research parameters.  ``min_asset_vol`` in
        # particular must match ``FINAL_CONFIG`` in research/trendlab/final.py;
        # omitting it silently changes position sizing for low-vol assets.
        cfg = TrendConfig(
            lookbacks=(60, 120),
            target_vol=0.25,
            max_gross=MAX_GROSS,
            max_asset_weight=0.15,
            min_asset_vol=0.15,
        )
        portfolio = TrendPortfolio(cfg)
        regime_on = portfolio.regime_active(panel)
        detail = portfolio.describe(panel)
        if "BTCUSDT" not in panel.columns and float(detail["target_weight"].abs().sum()) > 1e-12:
            raise RuntimeError(
                "Regime reference missing but non-zero targets were produced; refusing to trade"
            )

        targets = {str(sym): float(val) for sym, val in detail["target_weight"].items()}
        gross_exposure, worst_weight = validate_targets(targets)
        if not regime_on and gross_exposure > 1e-9:
            raise RuntimeError(
                f"Regime gate is OFF but targets gross {gross_exposure:.6f}; refusing to trade"
            )

        detail.to_csv(TARGETS_PATH)

        gross_exposure = float(detail["target_weight"].abs().sum())
        net_exposure = float(detail["target_weight"].sum())
        if "BTCUSDT" not in panel.columns:
            regime_status = "OFF (BTCUSDT unavailable, forced flat)"
        else:
            regime_status = "ON (Risk Active)" if regime_on else "OFF (Cash Preservation)"

        symbols_to_price = sorted(set(liquid) | tracked | set(targets))
        prices, fallback_prices = _prices_from_sources(all_tickers, panel, symbols_to_price)

        if args.mode == "signal" or args.dry_run:
            logger.info("Signal mode: regime=%s gross=%.2fx net=%+.2fx", regime_status, gross_exposure, net_exposure)
            print(detail.to_string())
            return

        if already_ran_today(state.get("last_run")) and not args.force:
            logger.info("RGMT already ran today (%s); use --force to override", state.get("last_run"))
            print("already ran today; no state change")
            return

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # 1. Funding on the positions held over the last 24h.
        open_symbols = [s for s, q in state.get("positions", {}).items() if abs(float(q)) > 1e-12]
        funding_cost = 0.0
        if open_symbols:
            funding_rates = client.get_funding_rates(open_symbols)
            state, funding_cost = accrue_funding(state, funding_rates, prices, fallback_prices)
            logger.info("Funding accrued: %+.4f USDT", funding_cost)

        # 2. Mark the book and enforce the drawdown guard.
        equity, unrealised, marks = mark_to_market(state, prices, fallback_prices)
        state["equity"] = equity
        state, drawdown, breached = update_risk(state)
        if breached:
            state["halted_at"] = now_iso
            logger.error("DRAWDOWN GUARD BREACHED: %.2f%% >= %.2f%%; flattening", drawdown * 100, MAX_DRAWDOWN_GUARD * 100)

        # 3. Rebalance.  A halt flattens everything and stays flat.
        if state.get("halted"):
            targets = flatten_targets(set(state.get("positions", {})) | set(targets))
            logger.warning("Book is halted (%s); flattening all positions", state.get("halt_reason"))

        state, orders = rebalance(
            state,
            targets,
            prices,
            equity,
            cost_rate=COST_RATE,
            fallback_prices=fallback_prices,
        )

        # 4. Re-mark after trading and persist an auditable record.
        final_equity, final_unrealised, final_marks = mark_to_market(state, prices, fallback_prices)
        state["equity"] = final_equity
        state["peak_equity"] = max(float(state.get("peak_equity", final_equity)), final_equity)
        state["runs"] = int(state.get("runs", 0)) + 1
        state["last_run"] = now_iso
        previous_regime = state.get("last_regime")
        state["last_regime"] = bool(regime_on)

        for order in orders:
            append_trade(state, order, now_iso, final_equity)
        append_history(
            state,
            {
                "time": now_iso,
                "equity": final_equity,
                "drawdown_pct": drawdown * 100.0,
                "funding_cost": funding_cost,
                "orders": len(orders),
                "regime": bool(regime_on),
                "gross_target": gross_exposure,
            },
        )

        health_errors = validate_state(state, {**fallback_prices, **prices})
        if health_errors:
            state["health_error"] = "; ".join(health_errors)
            logger.error("State health check failed: %s", state["health_error"])
        else:
            state.pop("health_error", None)

        save_state(state)

        regime_flip = previous_regime is not None and bool(previous_regime) != bool(regime_on)
        flip_text = ""
        if regime_flip:
            flip_text = (
                "🟢 *REGIME FLIPPED ON* — risk budget re-enabled\n"
                if regime_on
                else "🔴 *REGIME FLIPPED OFF* — book moved to cash\n"
            )

        now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        report = f"""
### 📊 RGMT Daily Paper Trading Report ({now_str})

{flip_text}* **BTC Regime Gate:** `{regime_status}`
* **Portfolio Equity:** `{_format_money(final_equity)}` (Initial: `{_format_money(state['initial_capital'])}`)
* **Drawdown from Peak:** `{drawdown * 100:.2f}%`
* **Target Gross Leverage:** `{gross_exposure:.2f}x` | **Net:** `{net_exposure:+.2f}x`
* **Rebalance Orders Placed:** `{len(orders)}`
* **Funding (24h):** `{funding_cost:+.4f} USDT`

| Metric | Value |
| :--- | :--- |
| **Realized PnL** | `{state['realized_pnl']:+,.2f}` |
| **Total Fees Paid** | `{state['total_fees_paid']:,.2f}` |
| **Total Funding Cost** | `{state.get('total_funding_cost', 0.0):+,.2f}` |
| **Active Positions** | `{len([q for q in state['positions'].values() if abs(q) > 1e-6])}` |
| **Halted** | `{state.get('halted', False)}` |
"""
        print(report)
        write_github_summary(report)

        tg_msg = (
            f"📊 *RGMT Daily Rebalance Update*\n"
            f"{flip_text}"
            f"• Status: *{regime_status}*\n"
            f"• Equity: *{_format_money(final_equity)}* (DD: {drawdown * 100:.1f}%)\n"
            f"• Gross Leverage: *{gross_exposure:.2f}x*\n"
            f"• Orders: *{len(orders)}*\n"
            f"• Funding: *{funding_cost:+.4f} USDT*\n"
            f"• Time: {now_str}"
        )
        send_telegram(tg_msg)


class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


if __name__ == "__main__":
    main()

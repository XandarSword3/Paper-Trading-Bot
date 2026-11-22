"""Decoupled Signal Research & Forward Return Evaluation Engine (Institutional Upgrade).

Features:
- Event-level episode debouncing (5-minute quiet gap) to prevent burst duplication.
- Execution realism: Entry at t + latency at Ask (long) / Bid (short); Exit at Bid (long) / Ask (short).
- Round-trip fee deduction (Bybit VIP0 11 bps) for true NET returns.
- Dual evaluation: Momentum Continuation vs. Mean-Reversion Fade.
- Matched Naive-Momentum Control (sign of prior 20s move, no Z or surge filter).
- Crash-proof persistent disk state (atomic pending_queue.json reload on startup).
- Dose-response bucketing across Z-score and surge tiers.
- WebSocket gap invalidation flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional
import uuid

import numpy as np

logger = logging.getLogger("signal_evaluator")

HORIZONS = {
    "10s": 10.0,
    "30s": 30.0,
    "1m": 60.0,
    "5m": 300.0,
    "15m": 900.0,
}

PRIMARY_HORIZON = "5m"
ROUNDTRIP_TAKER_FEE_BPS = 11.0  # Bybit VIP0 taker: 5.5 bps entry + 5.5 bps exit
EPISODE_GAP_SECONDS = 300.0     # 5 minutes quiet gap between independent episodes


@dataclass
class SignalObservation:
    signal_id: str
    timestamp: float
    time_utc: str
    symbol: str
    direction: int                  # +1 for Long (Buy), -1 for Short (Sell)
    z_score: float
    turnover_surge: float
    velocity_pct: float
    detected_mid: float
    detected_bid: float
    detected_ask: float
    detected_spread_bps: float
    is_primary_episode: bool = True  # True if first signal after >= 5m quiet; False for intra-burst ticks
    is_control: bool = False         # True for matched naive-momentum control
    matched_signal_id: Optional[str] = None
    z_bucket: str = "2.2-2.8"

    # Execution realism fields (filled at t + latency)
    latency_seconds: float = 0.75
    entry_fill_price: Optional[float] = None
    entry_fill_ts: Optional[float] = None
    fade_entry_price: Optional[float] = None

    # Maker execution fields (Post-Only limit order at Bid/Ask, 4.0 bps VIP0 fee)
    maker_attempted: bool = True
    maker_filled: bool = False
    maker_fill_price: Optional[float] = None
    maker_fill_ts: Optional[float] = None
    maker_net_returns_bps: Dict[str, Optional[float]] = field(
        default_factory=lambda: {h: None for h in HORIZONS}
    )

    # Forward net returns across horizons (in bps, after bid-ask crossing and 11 bps fees)
    net_returns_bps: Dict[str, Optional[float]] = field(
        default_factory=lambda: {h: None for h in HORIZONS}
    )
    # Fade / mean-reversion net returns (entering opposite direction)
    fade_net_returns_bps: Dict[str, Optional[float]] = field(
        default_factory=lambda: {h: None for h in HORIZONS}
    )
    # Gross mid returns (diagnostic baseline)
    gross_mid_returns_bps: Dict[str, Optional[float]] = field(
        default_factory=lambda: {h: None for h in HORIZONS}
    )
    exit_prices: Dict[str, Optional[float]] = field(
        default_factory=lambda: {h: None for h in HORIZONS}
    )

    has_ws_gap: bool = False
    resolved: bool = False


class SignalEvaluator:
    """Manages episode-debounced signal research with realistic execution and crash persistence."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        primary_horizon: str = PRIMARY_HORIZON,
        fee_bps: float = ROUNDTRIP_TAKER_FEE_BPS,
        episode_gap_seconds: float = EPISODE_GAP_SECONDS,
    ) -> None:
        self.output_dir = output_dir or Path(__file__).resolve().parent.parent / "logs" / "signal_research"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "signal_observations.jsonl"
        self.pending_path = self.output_dir / "pending_queue.json"
        self.summary_path = self.output_dir / "signal_research_summary.json"

        self.primary_horizon = primary_horizon
        self.fee_bps = fee_bps
        self.maker_fee_bps = 4.0  # 2.0 bps entry, 2.0 bps exit (VIP0 maker fee)
        self.episode_gap_seconds = episode_gap_seconds

        self.last_episode_time: Dict[str, float] = {}
        self.pending: List[SignalObservation] = []
        self.observations: Dict[str, SignalObservation] = {}
        self._last_summary_write = 0.0

        self._load_pending_queue()
        self._load_existing_observations()

    def _coerce_observation(self, r: dict) -> Optional[SignalObservation]:
        """Convert a raw dictionary to SignalObservation handling legacy field names."""
        try:
            valid_keys = {
                "signal_id", "timestamp", "time_utc", "symbol", "direction",
                "z_score", "turnover_surge", "velocity_pct", "detected_mid",
                "detected_bid", "detected_ask", "detected_spread_bps",
                "is_primary_episode", "is_control", "matched_signal_id", "z_bucket",
                "latency_seconds", "entry_fill_price", "entry_fill_ts",
                "net_returns_bps", "fade_net_returns_bps", "gross_mid_returns_bps",
                "exit_prices", "has_ws_gap", "resolved"
            }
            # Map legacy names if present
            if "entry_mid" in r and "detected_mid" not in r:
                r["detected_mid"] = r["entry_mid"]
            if "entry_bid" in r and "detected_bid" not in r:
                r["detected_bid"] = r["entry_bid"]
            if "entry_ask" in r and "detected_ask" not in r:
                r["detected_ask"] = r["entry_ask"]
            if "entry_spread_bps" in r and "detected_spread_bps" not in r:
                r["detected_spread_bps"] = r["entry_spread_bps"]
            if "returns_bps" in r and "net_returns_bps" not in r:
                r["net_returns_bps"] = r["returns_bps"]

            filtered = {k: v for k, v in r.items() if k in valid_keys}
            return SignalObservation(**filtered)
        except Exception as e:
            logger.warning("Could not coerce observation: %s", e)
            return None

    def _load_pending_queue(self) -> None:
        """Reload in-flight pending observations from disk on startup."""
        if self.pending_path.exists():
            try:
                data = json.loads(self.pending_path.read_text(encoding="utf-8"))
                for item in data:
                    obs = self._coerce_observation(item)
                    if obs:
                        self.pending.append(obs)
                        self.observations[obs.signal_id] = obs
                        if obs.is_primary_episode:
                            self.last_episode_time[obs.symbol] = max(
                                self.last_episode_time.get(obs.symbol, 0.0), obs.timestamp
                            )
                logger.info("Restored %d in-flight pending observations from %s", len(self.pending), self.pending_path)
            except Exception as e:
                logger.warning("Could not restore pending queue: %s", e)

    def _load_existing_observations(self) -> None:
        """Load resolved observations from JSONL."""
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            obs = self._coerce_observation(r)
                            if obs:
                                self.observations[obs.signal_id] = obs
                                if obs.is_primary_episode:
                                    self.last_episode_time[obs.symbol] = max(
                                        self.last_episode_time.get(obs.symbol, 0.0), obs.timestamp
                                    )
                logger.info("Loaded %d resolved observations from %s", len(self.observations), self.jsonl_path)
            except Exception as e:
                logger.warning("Error reading %s: %s", self.jsonl_path, e)

    def _save_pending_queue(self) -> None:
        """Atomically persist pending queue to disk."""
        try:
            temp_path = self.pending_path.with_suffix(".tmp")
            payload = [asdict(obs) for obs in self.pending]
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.pending_path)
        except Exception as e:
            logger.error("Failed saving pending queue: %s", e)

    def _get_z_bucket(self, z: float) -> str:
        abs_z = abs(z)
        if abs_z < 1.8:
            return "1.2-1.8"
        elif abs_z < 2.2:
            return "1.8-2.2"
        elif abs_z < 2.8:
            return "2.2-2.8"
        elif abs_z < 3.5:
            return "2.8-3.5"
        return "3.5+"

    def record_signal(
        self,
        symbol: str,
        direction: int,
        z_score: float,
        turnover_surge: float,
        velocity_pct: float,
        detected_bid: float,
        detected_ask: float,
        detected_mid: float,
        detected_spread_bps: float,
        latency_seconds: float = 0.75,
        prior_20s_move: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> SignalObservation:
        """Record an emitted signal with episode debouncing and matched naive-momentum control."""
        now = timestamp or time.time()
        now_utc = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        sig_id = str(uuid.uuid4())

        # Episode Debouncing: First signal on coin after >= 5m quiet is primary
        last_t = self.last_episode_time.get(symbol, 0.0)
        is_primary = (now - last_t) >= self.episode_gap_seconds
        if is_primary:
            self.last_episode_time[symbol] = now

        z_bucket = self._get_z_bucket(z_score)

        # 1. Actual signal observation
        sig_obs = SignalObservation(
            signal_id=sig_id,
            timestamp=now,
            time_utc=now_utc,
            symbol=symbol,
            direction=direction,
            z_score=z_score,
            turnover_surge=turnover_surge,
            velocity_pct=velocity_pct,
            detected_mid=detected_mid,
            detected_bid=detected_bid,
            detected_ask=detected_ask,
            detected_spread_bps=detected_spread_bps,
            is_primary_episode=is_primary,
            is_control=False,
            z_bucket=z_bucket,
            latency_seconds=latency_seconds,
        )
        self.pending.append(sig_obs)
        self.observations[sig_id] = sig_obs

        # 2. Matched Naive-Momentum Controls (Multiple controls to reduce control sampling error)
        # Tests whether Z/surge filters beat naive drift on the same coin
        ctrl_lookbacks = [10.0, 15.0, 20.0, 30.0, 45.0]
        for idx, lb in enumerate(ctrl_lookbacks):
            drift_dir = 1 if (prior_20s_move > 0 or (prior_20s_move == 0 and (idx % 2 == 0))) else -1
            ctrl_id = str(uuid.uuid4())
            ctrl_obs = SignalObservation(
                signal_id=ctrl_id,
                timestamp=now,
                time_utc=now_utc,
                symbol=symbol,
                direction=drift_dir,
                z_score=0.0,
                turnover_surge=1.0,
                velocity_pct=prior_20s_move,
                detected_mid=detected_mid,
                detected_bid=detected_bid,
                detected_ask=detected_ask,
                detected_spread_bps=detected_spread_bps,
                is_primary_episode=is_primary,
                is_control=True,
                matched_signal_id=sig_id,
                z_bucket="control",
                latency_seconds=latency_seconds,
            )
            self.pending.append(ctrl_obs)
            self.observations[ctrl_id] = ctrl_obs

        self._save_pending_queue()
        return sig_obs

    def evaluate_pending(
        self,
        feed,  # OmniWebSocketFeed instance
        now: Optional[float] = None,
    ) -> List[SignalObservation]:
        """Update realistic execution fills and forward returns across all pending observations."""
        ts = now or time.time()
        just_resolved: List[SignalObservation] = []
        still_pending: List[SignalObservation] = []

        for obs in self.pending:
            bid, ask, mid, spread, ex_ts = feed.get_top_of_book(obs.symbol)
            if mid <= 0:
                still_pending.append(obs)
                continue

            # Step 1: Simulate entry fill at t + latency (Ask for Buy, Bid for Sell)
            if obs.entry_fill_price is None:
                if ts >= (obs.timestamp + obs.latency_seconds):
                    fill_quote = feed.get_quote_at_timestamp(obs.symbol, obs.timestamp + obs.latency_seconds)
                    if fill_quote:
                        if obs.direction > 0:
                            obs.entry_fill_price = fill_quote.ask
                            obs.fade_entry_price = fill_quote.bid
                        else:
                            obs.entry_fill_price = fill_quote.bid
                            obs.fade_entry_price = fill_quote.ask
                        obs.entry_fill_ts = fill_quote.exchange_ts
                    else:
                        if obs.direction > 0:
                            obs.entry_fill_price = ask
                            obs.fade_entry_price = bid
                        else:
                            obs.entry_fill_price = bid
                            obs.fade_entry_price = ask
                        obs.entry_fill_ts = ts

            # Step 2: Forward horizon evaluations (Exit at Bid for Long, Ask for Short)
            if obs.entry_fill_price and obs.entry_fill_price > 0:
                for h_name, h_secs in HORIZONS.items():
                    if obs.net_returns_bps[h_name] is None and ts >= (obs.timestamp + h_secs):
                        target_ts = obs.timestamp + h_secs
                        exit_quote = feed.get_quote_at_timestamp(obs.symbol, target_ts)
                        exit_bid = exit_quote.bid if exit_quote else bid
                        exit_ask = exit_quote.ask if exit_quote else ask
                        exit_mid = exit_quote.mid if exit_quote else mid

                        # Realistic Fills:
                        # Momentum Long: Buy Ask, Sell Bid. Fade Short: Sell Bid, Buy Ask.
                        if obs.direction > 0:
                            exit_px = exit_bid
                            gross_trade_ret = (exit_px - obs.entry_fill_price) / obs.entry_fill_price
                            fade_entry = obs.fade_entry_price or obs.detected_bid
                            fade_gross_ret = (fade_entry - exit_ask) / fade_entry
                        else:
                            exit_px = exit_ask
                            gross_trade_ret = (obs.entry_fill_price - exit_px) / obs.entry_fill_price
                            fade_entry = obs.fade_entry_price or obs.detected_ask
                            fade_gross_ret = (exit_bid - fade_entry) / fade_entry

                        # True Net Return = Gross Trade Return - Roundtrip Taker Fees
                        net_bps = (gross_trade_ret * 10000.0) - self.fee_bps
                        fade_net_bps = (fade_gross_ret * 10000.0) - self.fee_bps
                        gross_mid_bps = obs.direction * ((exit_mid - obs.detected_mid) / obs.detected_mid) * 10000.0

                        obs.net_returns_bps[h_name] = round(net_bps, 2)
                        obs.fade_net_returns_bps[h_name] = round(fade_net_bps, 2)
                        obs.gross_mid_returns_bps[h_name] = round(gross_mid_bps, 2)
                        obs.exit_prices[h_name] = exit_px

                        # Simulated Maker Fill-Conditioned Return (4 bps fee)
                        if not obs.maker_filled and hasattr(feed, "check_maker_fill"):
                            side_str = "Buy" if obs.direction > 0 else "Sell"
                            limit_px = obs.detected_bid if obs.direction > 0 else obs.detected_ask
                            if limit_px > 0:
                                filled, fill_px, fill_t = feed.check_maker_fill(
                                    obs.symbol,
                                    side=side_str,
                                    limit_price=limit_px,
                                    start_ts=obs.timestamp + obs.latency_seconds,
                                    timeout_seconds=30.0,
                                )
                                if filled:
                                    obs.maker_filled = True
                                    obs.maker_fill_price = fill_px
                                    obs.maker_fill_ts = fill_t

                        if obs.maker_filled and obs.maker_fill_price:
                            if obs.direction > 0:
                                m_gross = (exit_bid - obs.maker_fill_price) / obs.maker_fill_price
                            else:
                                m_gross = (obs.maker_fill_price - exit_ask) / obs.maker_fill_price
                            obs.maker_net_returns_bps[h_name] = round((m_gross * 10000.0) - self.maker_fee_bps, 2)

                        # Flag if window spans a WebSocket disconnection gap
                        if feed.spans_gap(obs.timestamp, target_ts):
                            obs.has_ws_gap = True

            # Step 3: Check full resolution (15m elapsed)
            if obs.net_returns_bps["15m"] is not None or (ts >= obs.timestamp + 920.0):
                obs.resolved = True
                just_resolved.append(obs)
            else:
                still_pending.append(obs)

        self.pending = still_pending

        # Save state periodically (every 10s)
        if ts - self._last_summary_write >= 10.0:
            self._save_pending_queue()
            self._flush_observations()
            self.update_summary()
            self._last_summary_write = ts

        return just_resolved

    def _flush_observations(self) -> None:
        """Write all recorded observations to JSONL atomically."""
        try:
            temp_path = self.jsonl_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                for obs in self.observations.values():
                    f.write(json.dumps(asdict(obs)) + "\n")
            temp_path.replace(self.jsonl_path)
        except Exception as exc:
            logger.error("Failed flushing observations: %s", exc)

    def compute_metrics(self) -> dict:
        """Compute institutional metrics: Episode-level, Trimmed Means, Outlier Share, Dose-Response."""
        records = [asdict(obs) for obs in self.observations.values()]
        if not records:
            return {"status": "NO_DATA"}

        # Filter out WS gap corruptions
        clean_records = [r for r in records if not r.get("has_ws_gap", False)]
        
        # Primary analysis is strictly EPISODE-LEVEL (is_primary_episode == True)
        primary_signals = [r for r in clean_records if r.get("is_primary_episode") and not r.get("is_control")]
        secondary_signals = [r for r in clean_records if not r.get("is_primary_episode") and not r.get("is_control")]
        controls = [r for r in clean_records if r.get("is_primary_episode") and r.get("is_control")]

        horizon_stats: Dict[str, dict] = {}
        for h in HORIZONS:
            sig_net = [r["net_returns_bps"][h] for r in primary_signals if r.get("net_returns_bps", {}).get(h) is not None]
            sig_fade = [r["fade_net_returns_bps"][h] for r in primary_signals if r.get("fade_net_returns_bps", {}).get(h) is not None]
            ctrl_net = [r["net_returns_bps"][h] for r in controls if r.get("net_returns_bps", {}).get(h) is not None]

            if sig_net:
                arr = np.array(sig_net)
                arr_sorted = np.sort(arr)
                mean_net = float(np.mean(arr))
                med_net = float(np.median(arr))
                std_net = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

                # 5% Trimmed Mean (removes extreme 5% tails)
                trim_cut = int(len(arr) * 0.05)
                trimmed_arr = arr_sorted[trim_cut : len(arr) - trim_cut] if trim_cut > 0 and len(arr) > 20 else arr
                trim_mean_net = float(np.mean(trimmed_arr))

                # Top 1% Outlier Share (how much of positive sum comes from top 1% trades)
                pos_returns = arr[arr > 0]
                if len(pos_returns) > 10:
                    top_1pct_cut = max(1, int(len(pos_returns) * 0.01))
                    top_1pct_sum = float(np.sum(np.sort(pos_returns)[-top_1pct_cut:]))
                    outlier_share = (top_1pct_sum / float(np.sum(pos_returns))) * 100.0
                else:
                    outlier_share = 0.0

                fade_mean = float(np.mean(sig_fade)) if sig_fade else 0.0
                ctrl_mean = float(np.mean(ctrl_net)) if ctrl_net else 0.0
                excess_over_naive = mean_net - ctrl_mean
                win_rate = float(np.mean(arr > 0.0)) * 100.0

                horizon_stats[h] = {
                    "count": len(arr),
                    "mean_net_bps": round(mean_net, 2),
                    "median_net_bps": round(med_net, 2),
                    "trimmed_mean_net_bps": round(trim_mean_net, 2),
                    "std_net_bps": round(std_net, 2),
                    "win_rate_pct": round(win_rate, 2),
                    "fade_mean_net_bps": round(fade_mean, 2),
                    "naive_control_mean_net_bps": round(ctrl_mean, 2),
                    "excess_over_naive_bps": round(excess_over_naive, 2),
                    "top_1pct_outlier_share_pct": round(outlier_share, 2),
                }
            else:
                horizon_stats[h] = {"count": 0, "status": "PENDING"}

        # Dose-Response breakdown on Primary Horizon (5m)
        dose_response: Dict[str, dict] = {}
        for b_name in ("1.2-1.8", "1.8-2.2", "2.2-2.8", "2.8-3.5", "3.5+"):
            b_rets = [
                r["net_returns_bps"][self.primary_horizon]
                for r in primary_signals
                if r.get("z_bucket") == b_name and r.get("net_returns_bps", {}).get(self.primary_horizon) is not None
            ]
            if b_rets:
                dose_response[b_name] = {
                    "count": len(b_rets),
                    "mean_net_bps": round(float(np.mean(b_rets)), 2),
                    "median_net_bps": round(float(np.median(b_rets)), 2),
                    "win_rate_pct": round(float(np.mean(np.array(b_rets) > 0)) * 100.0, 1),
                }
            else:
                dose_response[b_name] = {"count": 0}

        # Maker Fill & Adverse Selection Metrics on Primary Horizon (5m)
        maker_5m_resolved = [r for r in primary_signals if r.get("net_returns_bps", {}).get(self.primary_horizon) is not None]
        maker_filled_5m = [r for r in maker_5m_resolved if r.get("maker_filled") and r.get("maker_net_returns_bps", {}).get(self.primary_horizon) is not None]
        maker_fill_rate = (len(maker_filled_5m) / len(maker_5m_resolved) * 100.0) if maker_5m_resolved else 0.0
        maker_mean_net = float(np.mean([r["maker_net_returns_bps"][self.primary_horizon] for r in maker_filled_5m])) if maker_filled_5m else 0.0
        maker_med_net = float(np.median([r["maker_net_returns_bps"][self.primary_horizon] for r in maker_filled_5m])) if maker_filled_5m else 0.0

        uncond_gross = float(np.mean([r["gross_mid_returns_bps"][self.primary_horizon] for r in maker_5m_resolved])) if maker_5m_resolved else 0.0
        filled_gross = float(np.mean([r["gross_mid_returns_bps"][self.primary_horizon] for r in maker_filled_5m])) if maker_filled_5m else 0.0
        adverse_selection_drag = uncond_gross - filled_gross

        maker_stats = {
            "evaluated_episodes": len(maker_5m_resolved),
            "filled_count": len(maker_filled_5m),
            "fill_rate_pct": round(maker_fill_rate, 1),
            "maker_fee_roundtrip_bps": self.maker_fee_bps,
            "maker_fill_conditioned_mean_net_bps": round(maker_mean_net, 2),
            "maker_fill_conditioned_median_net_bps": round(maker_med_net, 2),
            "adverse_selection_drag_bps": round(adverse_selection_drag, 2),
        }

        return {
            "total_primary_episodes": len(primary_signals),
            "total_secondary_ticks": len(secondary_signals),
            "total_naive_controls": len(controls),
            "pending_queue": len(self.pending),
            "primary_horizon": self.primary_horizon,
            "taker_fee_deducted_bps": self.fee_bps,
            "horizons": horizon_stats,
            "dose_response_5m": dose_response,
            "maker_execution_5m": maker_stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def update_summary(self) -> dict:
        summary = self.compute_metrics()
        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as exc:
            logger.warning("Failed writing summary to %s: %s", self.summary_path, exc)
        return summary

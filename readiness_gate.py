"""
Fail-closed readiness gate for live/paper trading bots.

Added per VALIDATION_REMEDIATION_PLAN.md, Section 0 ("Do this today — not a
phase, a live issue"): as of this commit, neither V1 nor V4 had any gate at
all blocking github_bot.py / github_bot_v4.py from trading on every scheduled
run, regardless of whether the strategy behind them had been honestly
validated (see STRATEGY_VERSION_AUDIT.md).

This module does NOT decide whether a strategy is good. It only decides
whether a *gate file* exists, is fresh, and explicitly authorizes trading.
Building a legitimate readiness file per strategy — computed from real
walk-forward / paper results, not from the same in-sample run being judged —
is Phase 5 of the remediation plan and is intentionally NOT done here.

Fail-closed semantics: missing file, unreadable/malformed JSON, missing the
`ready_for_live` key, missing/unparseable `generated_at`, a stale file, or
`ready_for_live` not being exactly `True` all mean NOT READY. There is no
default-to-running path.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MAX_GATE_AGE_HOURS = 24


def check_gate(strategy_name: str):
    """
    Check whether `strategy_name` (e.g. "v1", "v4") is cleared to trade.

    Looks for readiness_<strategy_name>.json in the current working directory,
    expected shape:
        {
          "ready_for_live": true,
          "generated_at": "2026-07-20T12:00:00+00:00",
          ... (whatever supporting metrics the generator wants to record)
        }

    Returns (ready: bool, reason: str) — reason is always a human-readable
    explanation, used for logging regardless of outcome.
    """
    gate_path = Path(f"readiness_{strategy_name}.json")

    if not gate_path.exists():
        return False, f"no readiness_{strategy_name}.json present"

    try:
        with open(gate_path, "r") as f:
            gate = json.load(f)
    except Exception as e:
        return False, f"readiness_{strategy_name}.json unreadable/malformed: {e}"

    if not isinstance(gate, dict) or "ready_for_live" not in gate:
        return False, f"readiness_{strategy_name}.json missing 'ready_for_live' key"

    generated_at = gate.get("generated_at")
    if not generated_at:
        return False, f"readiness_{strategy_name}.json missing 'generated_at' timestamp"

    try:
        ts = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception as e:
        return False, f"readiness_{strategy_name}.json has unparseable 'generated_at': {e}"

    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if age_hours > MAX_GATE_AGE_HOURS:
        return False, (
            f"readiness_{strategy_name}.json is stale "
            f"({age_hours:.1f}h old, max {MAX_GATE_AGE_HOURS}h)"
        )

    if gate["ready_for_live"] is not True:
        return False, f"readiness_{strategy_name}.json sets ready_for_live={gate['ready_for_live']!r}"

    return True, "gate passed"

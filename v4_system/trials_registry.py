"""Append-only trial registry for the V4 preregistered experiment budget.

Implements the budget rules registered in ``v4_system/PREREG.md`` Part II.4:

- **40 total trials** across all families.
- **12 trials per strategy family**.
- **200 configurations per trial**.
- **3 repeats** of the same (family, configuration) combination.
- Append-only: rows are never modified or deleted; a result is recorded as
  a NEW row referencing the same ``trial_id``.
- ``assert_registered`` lets runners reject unregistered execution.

The registry writes to ``v4_system/trials_log.csv`` (header already
committed). It performs NO strategy execution and consumes no budget by
itself — registration is a bookkeeping act that must happen BEFORE any
results exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

from v4_system.config import TRIALS_LOG

# Budget limits — PREREG II.4
MAX_TOTAL_TRIALS = 40
MAX_TRIALS_PER_FAMILY = 12
MAX_CONFIGS_PER_TRIAL = 200
MAX_REPEATS = 3

HEADER = ["phase", "trial_id", "timestamp", "description", "params",
          "result", "net_expectancy", "oos_trades", "max_dd",
          "passed_thresholds", "notes"]


class TrialBudgetError(Exception):
    """Raised when a registration would exceed the preregistered budget."""


class UnregisteredTrialError(Exception):
    """Raised when execution is attempted without a prior registration."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rows(path: str = TRIALS_LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _config_hash(params: dict) -> str:
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _family_of(row: dict) -> str:
    # family is encoded in the phase column as "phase2/<family>"
    phase = row.get("phase", "")
    return phase.split("/", 1)[1] if "/" in phase else phase


def register_trial(*, family: str, description: str, params: dict,
                   n_configs: int, phase: str = "phase2",
                   path: str = TRIALS_LOG) -> str:
    """Register a trial BEFORE any results exist. Returns the trial_id.

    Enforces the preregistered budget. ``params`` must include the full
    configuration matrix description; ``n_configs`` is the number of
    distinct configurations in the trial (<= 200).
    """
    if n_configs < 1 or n_configs > MAX_CONFIGS_PER_TRIAL:
        raise TrialBudgetError(
            f"n_configs={n_configs} outside 1..{MAX_CONFIGS_PER_TRIAL}")

    rows = _read_rows(path)
    registrations = [r for r in rows if r.get("result") == "REGISTERED"]
    total = len(registrations)
    family_count = sum(1 for r in registrations
                       if _family_of(r) == family)
    if total >= MAX_TOTAL_TRIALS:
        raise TrialBudgetError(
            f"trial budget exhausted: {total}/{MAX_TOTAL_TRIALS} registered")
    if family_count >= MAX_TRIALS_PER_FAMILY:
        raise TrialBudgetError(
            f"family '{family}' budget exhausted: "
            f"{family_count}/{MAX_TRIALS_PER_FAMILY}")

    # Repeat limit: same family + identical config hash <= MAX_REPEATS.
    # The hash is stored inside params at registration time; re-hashing the
    # stored params would include n_configs/config_hash and never match.
    chash = _config_hash(params)
    repeats = 0
    for r in registrations:
        if _family_of(r) != family:
            continue
        try:
            stored = json.loads(r.get("params") or "{}")
        except json.JSONDecodeError:
            continue
        if stored.get("config_hash") == chash:
            repeats += 1
    if repeats >= MAX_REPEATS:
        raise TrialBudgetError(
            f"repeat limit: family '{family}' config {chash} already "
            f"registered {repeats} times (max {MAX_REPEATS})")

    trial_id = f"{family}-{len(registrations) + 1:03d}-{chash}"
    row = {
        "phase": f"{phase}/{family}",
        "trial_id": trial_id,
        "timestamp": _now(),
        "description": description,
        "params": json.dumps({**params, "n_configs": n_configs,
                              "config_hash": chash}, sort_keys=True,
                             default=str),
        "result": "REGISTERED",
        "net_expectancy": "",
        "oos_trades": "",
        "max_dd": "",
        "passed_thresholds": "",
        "notes": "",
    }
    _append_row(row, path)
    return trial_id


def record_result(*, trial_id: str, result: str,
                  net_expectancy: str = "", oos_trades: str = "",
                  max_dd: str = "", passed_thresholds: str = "",
                  notes: str = "", path: str = TRIALS_LOG) -> None:
    """Append a result row for a previously registered trial.

    Append-only: the original REGISTERED row is never modified; the result
    is a second row with the same trial_id. Failed experiments MUST be
    recorded here too (result may be 'FAILED'/'NO_EDGE').
    """
    rows = _read_rows(path)
    if not any(r.get("trial_id") == trial_id and
               r.get("result") == "REGISTERED" for r in rows):
        raise UnregisteredTrialError(
            f"cannot record result for unregistered trial {trial_id!r}")
    reg = next(r for r in rows
               if r.get("trial_id") == trial_id and
               r.get("result") == "REGISTERED")
    row = {
        "phase": reg["phase"],
        "trial_id": trial_id,
        "timestamp": _now(),
        "description": reg["description"],
        "params": reg["params"],
        "result": result,
        "net_expectancy": net_expectancy,
        "oos_trades": oos_trades,
        "max_dd": max_dd,
        "passed_thresholds": passed_thresholds,
        "notes": notes,
    }
    _append_row(row, path)


def assert_registered(trial_id: str, path: str = TRIALS_LOG) -> None:
    """Reject execution of a trial that was never registered."""
    rows = _read_rows(path)
    if not any(r.get("trial_id") == trial_id and
               r.get("result") == "REGISTERED" for r in rows):
        raise UnregisteredTrialError(
            f"trial {trial_id!r} is not registered in {path}")


def _append_row(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if not exists or os.path.getsize(path) == 0:
            w.writeheader()
        w.writerow(row)


def budget_status(path: str = TRIALS_LOG) -> dict:
    """Read-only summary of budget consumption."""
    rows = _read_rows(path)
    regs = [r for r in rows if r.get("result") == "REGISTERED"]
    families: dict[str, int] = {}
    for r in regs:
        fam = _family_of(r)
        families[fam] = families.get(fam, 0) + 1
    return {
        "total_registered": len(regs),
        "max_total": MAX_TOTAL_TRIALS,
        "per_family": families,
        "max_per_family": MAX_TRIALS_PER_FAMILY,
        "max_configs_per_trial": MAX_CONFIGS_PER_TRIAL,
        "max_repeats": MAX_REPEATS,
    }

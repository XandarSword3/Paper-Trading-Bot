"""Bridge utilities to run liquidation-hunter from the BTC Strategy project."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LiquidationHunterBridgeError(RuntimeError):
    """Raised when the bridge cannot execute or sync liquidation-hunter runs."""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_liquidation_dir(explicit_dir: str | None) -> Path:
    if explicit_dir:
        candidate = Path(explicit_dir).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise LiquidationHunterBridgeError(f"Liquidation-hunter directory not found: {candidate}")

    env_dir = os.getenv("LIQUIDATION_HUNTER_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parent
    candidates = [
        (repo_root.parent / "Trading Bot" / "liquidation-hunter").resolve(),
        (repo_root / "liquidation-hunter").resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise LiquidationHunterBridgeError(
        "Could not locate liquidation-hunter project. Set LIQUIDATION_HUNTER_DIR or pass --lh-dir. "
        f"Searched:\n{searched}"
    )


def _build_command(
    mode: str,
    iterations: int | None,
    poll_seconds: int | None,
    candles: int | None,
    oos_ratio: float | None,
    folds: int | None,
    mc_iterations: int | None,
    compare_timeframes: bool,
) -> list[str]:
    cmd = [sys.executable, "main.py", "--mode", mode]

    if iterations is not None:
        cmd.extend(["--iterations", str(iterations)])
    if poll_seconds is not None:
        cmd.extend(["--poll-seconds", str(poll_seconds)])
    if candles is not None:
        cmd.extend(["--candles", str(candles)])
    if oos_ratio is not None:
        cmd.extend(["--oos-ratio", str(oos_ratio)])
    if folds is not None:
        cmd.extend(["--folds", str(folds)])
    if mc_iterations is not None:
        cmd.extend(["--mc-iterations", str(mc_iterations)])
    if compare_timeframes:
        cmd.append("--compare-timeframes")

    return cmd


def _extract_summary_from_reports(log_dir: Path) -> dict[str, Any]:
    robustness = _read_json_if_exists(log_dir / "robustness_report.json") or {}
    paper_readiness = _read_json_if_exists(log_dir / "paper_readiness.json") or {}
    static_analysis = _read_json_if_exists(log_dir / "static_analysis_report.json") or {}

    baseline = robustness.get("baseline", {}) if isinstance(robustness, dict) else {}
    walk_forward = {}
    monte_carlo = {}
    if isinstance(robustness, dict):
        walk_forward = (robustness.get("walk_forward", {}) or {}).get("summary", {}) or {}
        monte_carlo = robustness.get("monte_carlo", {}) or {}

    return {
        "robustness": {
            "baseline_total_return": _to_float(baseline.get("total_return")),
            "baseline_max_drawdown": _to_float(baseline.get("max_drawdown")),
            "baseline_sharpe": _to_float(baseline.get("sharpe")),
            "walk_forward_oos_avg_sharpe": _to_float(walk_forward.get("oos_avg_sharpe")),
            "mc_5pct_equity": _to_float(monte_carlo.get("p5_equity")),
            "mc_ruin_probability": _to_float(monte_carlo.get("probability_ruin")),
        },
        "paper_readiness": paper_readiness,
        "static_analysis": static_analysis,
    }


def _sync_artifacts(liquidation_dir: Path, destination_dir: Path) -> dict[str, Any]:
    log_dir = liquidation_dir / "logs"
    destination_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = [
        "robustness_report.json",
        "paper_readiness.json",
        "static_analysis_report.json",
    ]

    copied: list[str] = []
    for name in artifact_names:
        src = log_dir / name
        if not src.exists():
            continue
        dst = destination_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))

    summary = {
        "synced_at": _now_utc_iso(),
        "source_log_dir": str(log_dir),
        "copied_artifacts": copied,
    }
    summary.update(_extract_summary_from_reports(log_dir))

    summary_path = destination_dir / "last_sync_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    history_path = destination_dir / "integration_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, separators=(",", ":"), default=str) + "\n")

    return summary


def run_liquidation_hunter(
    mode: str,
    liquidation_dir: str | None = None,
    iterations: int | None = None,
    poll_seconds: int | None = None,
    candles: int | None = None,
    oos_ratio: float | None = None,
    folds: int | None = None,
    mc_iterations: int | None = None,
    compare_timeframes: bool = False,
    sync_artifacts: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_dir = _resolve_liquidation_dir(liquidation_dir)
    main_file = project_dir / "main.py"
    if not main_file.exists():
        raise LiquidationHunterBridgeError(f"main.py not found under liquidation-hunter directory: {project_dir}")

    command = _build_command(
        mode=mode,
        iterations=iterations,
        poll_seconds=poll_seconds,
        candles=candles,
        oos_ratio=oos_ratio,
        folds=folds,
        mc_iterations=mc_iterations,
        compare_timeframes=compare_timeframes,
    )

    result: dict[str, Any] = {
        "executed_at": _now_utc_iso(),
        "project_dir": str(project_dir),
        "command": command,
        "mode": mode,
        "dry_run": bool(dry_run),
    }

    if dry_run:
        return result

    completed = subprocess.run(command, cwd=str(project_dir), check=False)
    result["return_code"] = int(completed.returncode)
    if completed.returncode != 0:
        raise LiquidationHunterBridgeError(
            "Liquidation-hunter run failed with exit code "
            f"{completed.returncode}. Command: {' '.join(command)}"
        )

    if sync_artifacts:
        destination = Path(__file__).resolve().parent / "results" / "liquidation_hunter"
        result["artifact_sync"] = _sync_artifacts(project_dir, destination)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run liquidation-hunter through BTC Strategy bridge")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "research", "observe"], required=True)
    parser.add_argument("--lh-dir", default=None, help="Path to liquidation-hunter directory")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--candles", type=int, default=None)
    parser.add_argument("--oos-ratio", type=float, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--mc-iterations", type=int, default=None)
    parser.add_argument("--compare-timeframes", action="store_true")
    parser.add_argument("--no-sync-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_liquidation_hunter(
        mode=args.mode,
        liquidation_dir=args.lh_dir,
        iterations=args.iterations,
        poll_seconds=args.poll_seconds,
        candles=args.candles,
        oos_ratio=args.oos_ratio,
        folds=args.folds,
        mc_iterations=args.mc_iterations,
        compare_timeframes=args.compare_timeframes,
        sync_artifacts=not args.no_sync_artifacts,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

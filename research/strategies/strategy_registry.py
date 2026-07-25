"""
Registry mapping a strategy name to the timeframe it trades on, which
strategy FAMILY it belongs to, and the parameter ranges to search for it.

This is the ONE file to edit to validate a new strategy through
run_single_fold.py / aggregate_folds.py and the
.github/workflows/walk_forward_validation.yml GitHub Actions workflow — add
an entry below, then trigger the workflow with that strategy name as input.
Neither the scripts nor the workflow need to change.

`family` dispatches which strategy engine + fold-runner run_single_fold.py
and aggregate_folds.py use:
  - "turtle": TurtleDonchianStrategy / walk_forward.run_fold / RobustnessRanges
    (the original Donchian-breakout family: v1, v4).
  - "mean_reversion": BollingerRSIMeanReversion / mr_walk_forward.run_fold_mr /
    MRRobustnessRanges (research/strategies/mean_reversion_strategy.py) — a
    separate, independent strategy, not a variant of the turtle family.
Defaults to "turtle" if omitted, so this stays backward compatible with any
other reader of this registry that predates the `family` field.

Naming note: "v1" and "v4" below match the existing repo-wide convention
(walk_forward_test.py maps 4h->"v1", 1h->"v4"; build_readiness_gates.py and
edge_validation_test.py read walk_forward_results_v1.json /
walk_forward_results_v4.json by those exact names). If you add a strategy
that shares a timeframe with an existing one (e.g. testing V4's parameter
neighborhood on 4h, or this mean-reversion strategy on 4h), give it its own
key here so its output doesn't overwrite v1's/v4's files — aggregate_folds.py
names its output file after the registry key itself (args.strategy), not
after the timeframe, specifically so a same-timeframe addition like "mr_4h"
can never collide with "v1"'s walk_forward_results_v1.json.
"""
from config import DEFAULT_ROBUSTNESS

try:
    # Only needed for the mean_reversion family; keep the turtle-only path
    # (v1/v4) working even if mr_robustness.py isn't present for some reason.
    from mr_robustness import DEFAULT_MR_ROBUSTNESS
except ImportError:
    DEFAULT_MR_ROBUSTNESS = None

STRATEGY_REGISTRY = {
    "v1": {"timeframe": "4h", "family": "turtle", "ranges": DEFAULT_ROBUSTNESS},
    "v4": {"timeframe": "1h", "family": "turtle", "ranges": DEFAULT_ROBUSTNESS},
    "mr_4h": {"timeframe": "4h", "family": "mean_reversion", "ranges": DEFAULT_MR_ROBUSTNESS},
    "mr_1h": {"timeframe": "1h", "family": "mean_reversion", "ranges": DEFAULT_MR_ROBUSTNESS},
}


def get_strategy_config(name: str) -> dict:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Known strategies: {sorted(STRATEGY_REGISTRY)}. "
            "Add a new entry to strategy_registry.STRATEGY_REGISTRY to register one."
        )
    cfg = STRATEGY_REGISTRY[name]
    cfg.setdefault("family", "turtle")
    return cfg

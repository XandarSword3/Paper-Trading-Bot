"""
Registry mapping a strategy name to the timeframe it trades on and the
RobustnessRanges to search for it.

This is the ONE file to edit to validate a new strategy through
run_single_fold.py / aggregate_folds.py and the
.github/workflows/walk_forward_validation.yml GitHub Actions workflow — add
an entry below, then trigger the workflow with that strategy name as input.
Neither the scripts nor the workflow need to change.

Naming note: "v1" and "v4" below match the existing repo-wide convention
(walk_forward_test.py maps 4h->"v1", 1h->"v4"; build_readiness_gates.py and
edge_validation_test.py read walk_forward_results_v1.json /
walk_forward_results_v4.json by those exact names). If you add a strategy
that shares a timeframe with an existing one (e.g. testing V4's parameter
neighborhood on 4h), give it its own key here so its output doesn't
overwrite v1's/v4's files — but note that if it also shares the same
`ranges`, the search is identical to whichever existing strategy already
covers that timeframe, so a separate run of it is redundant rather than
wrong.
"""
from config import DEFAULT_ROBUSTNESS

STRATEGY_REGISTRY = {
    "v1": {"timeframe": "4h", "ranges": DEFAULT_ROBUSTNESS},
    "v4": {"timeframe": "1h", "ranges": DEFAULT_ROBUSTNESS},
}


def get_strategy_config(name: str) -> dict:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Known strategies: {sorted(STRATEGY_REGISTRY)}. "
            "Add a new entry to strategy_registry.STRATEGY_REGISTRY to register one."
        )
    return STRATEGY_REGISTRY[name]

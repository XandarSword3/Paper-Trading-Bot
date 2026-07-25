"""
Makes research/strategies, research/data, and research/validation importable
as flat modules (`from config import ...`, `from data_fetcher import ...`,
etc.) regardless of what directory a script is invoked from.

Every script in this directory that uses the repo's flat-import convention
(list_folds.py, run_single_fold.py, aggregate_folds.py, walk_forward.py,
mr_walk_forward.py, robustness_test.py, mr_robustness.py, deflated_sharpe.py,
strategy.py, mean_reversion_strategy.py, strategy_registry.py, config.py,
data_fetcher.py, data_splits.py, metrics_utils.py, ...) assumes these three
directories are already on sys.path. Nothing in this repo actually sets
PYTHONPATH to provide that — including .github/workflows/walk_forward_validation.yml,
which invokes these scripts as `python3 research/validation/<script>.py` from
the repo root with no PYTHONPATH set. That combination (flat imports +
no PYTHONPATH) meant the workflow's `list_folds.py` call failed at the very
first import, and because it was invoked inside `MATRIX=$(... | tail -n 1)`,
the failure was silently swallowed: MATRIX came out empty, the fold matrix
never got built, and the run reported "setup: success" with zero fold jobs.

The other import convention in this codebase (research.validation.build_readiness_gates,
run via `python -m research.validation.build_readiness_gates` in
refresh_gates.yml) takes the opposite approach: dotted imports off the repo
root, which it adds to sys.path itself. Both conventions coexist in this
repo; this file provides the sys.path side of the flat-import one, at the
single entry-point layer, so none of the many modules that already assume
it need to change.

Usage: `import _path_bootstrap` as the FIRST import, before any flat import,
at the top of any script in research/validation/ meant to be run directly.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../research/validation
_RESEARCH_DIR = os.path.dirname(_HERE)                       # .../research

for _p in (
    os.path.join(_RESEARCH_DIR, "strategies"),
    os.path.join(_RESEARCH_DIR, "data"),
    _HERE,  # research/validation itself, for cross-imports like `from walk_forward import ...`
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

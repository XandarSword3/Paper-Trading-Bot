# Strategy Validation Pipeline: How We Actually Test Whether a Strategy Is Valid

## Read this first: there are two pipelines in this repo, not one

**`main.py`** (root) runs a 7-phase pipeline (`phase_1_setup` → `phase_6_7_forward_test`).
It produces the "855% / 1572% return" headline numbers you'll see quoted around
this repo (and in older docs). It does not touch the frozen holdout window, but
it **optimizes and scores on the same data** — a single grid search over the
development window, with the best combination's own in-sample performance
reported as the result. That is exactly the methodology that produces
overfit-looking numbers, and it is why it's being superseded, not because the
code is buggy.

**`research/validation/`** is the corrected pipeline, built to fix that. Every
script in it exists to answer one question the old pipeline couldn't: *does
this edge survive on data the optimizer never saw?* This document describes
that pipeline — what each phase does, what script runs it, and, as of this
writing, what it actually found when run against V1 and V4.

If you only remember one thing: **an in-sample backtest number is not
evidence a strategy works.** Everything below exists to replace "it backtested
well" with "it held up on data it never touched."

---

## The frozen data split

One split, defined once in `research/strategies/config.py` (`DataSplitConfig`),
enforced by `research/data/data_splits.py`. Nothing else should hand-roll its
own date filter — an acceptance check greps the repo for the literal date
`2025-01-01` to catch leaks.

| Window | Range | Use |
|---|---|---|
| In-sample | 2017-01-01 → 2023-12-31 | Free to use for development, rule design, robustness/plateau checks |
| Validation | 2024-01-01 → 2024-12-31 | Walk-forward OOS folds, Monte Carlo, regime/survivability stress tests — "seen" during development but never used to pick final params |
| **Holdout** | **2025-01-01 → latest** | Touched by exactly one script, exactly once, after everything else is done |

"Development" = in-sample + validation combined (`get_development()`). The
holdout is a one-shot resource: spend it once, on parameters that are already
frozen, or it stops being a holdout.

---

## The phases

### Phase 0 — Fail-closed readiness gate
**Script:** `research/validation/readiness_gate.py`

Doesn't judge whether a strategy is good. Only checks whether a
`data/readiness_<strategy>.json` file exists, is younger than 24 hours, and
says `ready_for_live: true`. Missing file, stale file, malformed JSON, or
anything other than exactly `true` → **not ready, no exceptions.** This is
what `research/bots/bot_runner.py` calls before every scheduled run — nothing
trades live by default.

### Phase 1 — Data split enforcement
**Script:** `research/data/data_splits.py`

Provides `get_in_sample()`, `get_validation()`, `get_development()` so every
downstream script slices the same way. Fixes the original problem: every tool
used to fetch and use the *entire* dataset independently, silently erasing the
holdout.

### Phase 2 — Rolling walk-forward optimization
**Scripts:** `research/validation/walk_forward.py`, `walk_forward_test.py`

Each fold optimizes on a rolling window and is scored **only** on the
unseen window immediately after it (2-year train / 6-month test / 6-month
step, sliding). The in-sample number from each fold's grid search is used
purely to pick that fold's parameters and is never blended into the reported
result — only the stitched-together out-of-sample curve counts.

### Phase 3 — Honest risk/skill testing
**Scripts:** `robustness_test.py` (in-sample plateau check only — explicitly
labeled as not a performance claim), `monte_carlo.py`
(`OOSBlockBootstrap` — block-bootstraps the walk-forward OOS *returns*),
`deflated_sharpe.py` (Probabilistic/Deflated Sharpe Ratio, corrects the
best-of-2,800-combinations Sharpe for how many combinations were tried),
`edge_validation_test.py` (runs both side by side).

This is where "reshuffling the same in-sample trades" (old Monte Carlo) gets
replaced by "resample returns that were never touched by parameter selection."

### Phase 4 — Cross-market validation
**Script:** `research/validation/cross_market_validation.py`

Runs the frozen V1 rule set, unchanged, on ETH/USDT, BTC/USDT 1h, and Kraken
BTC/USD, restricted to the development window. No re-fitting per market.
Reports numbers side by side with no automated pass/fail — checks whether the
edge is structural or a BTC-2017–2025-shaped coincidence. Anything under ~20
trades in a leg's window is flagged as too thin to conclude from either way.

### Phase 5 — Readiness gate generator
**Script:** `research/validation/build_readiness_gates.py`

Computes real metrics from the strategy's **own recorded paper-trading
trades** (`data/trades.json` for V1, `data/trades_v4.json` for V4) — never
from a backtest. Thresholds (`GateThresholds` in `config.py`):

| Threshold | Value |
|---|---|
| Minimum completed paper trades | 30 |
| Minimum annualized paper Sharpe | 0.5 |
| Minimum total paper return | 0.0% (floor, not a target) |
| Maximum paper drawdown | 25% |

Also flags (without blocking) if paper Sharpe has degraded more than 50%
versus the walk-forward OOS Sharpe from Phase 2. This is the script that
writes the file Phase 0 reads.

### Final holdout
**Script:** `research/validation/final_holdout_validation.py`

The one script allowed to touch 2025-01-01 → latest, and only after Phases
2–4 are done and parameters are frozen. Requires the explicit flag
`--i-understand-this-burns-the-holdout`. **As of this writing, this has not
been run for either strategy** — no `data/final_holdout_*.json` exists.
Running it now, before paper-trading gates clear, would just spend the
holdout on a second in-sample number wearing a different date range.

### Phase 6–7 — Paper trading
3–6 months live paper trading against real fills, tracked in
`data/trades.json` / `data/trades_v4.json`, per the thresholds in
[`PAPER_TRADING_CHECKLIST.md`](PAPER_TRADING_CHECKLIST.md), before any real
capital moves.

---

## Where V1 and V4 actually stand right now

Pulled directly from the current `data/*.json` gate files — not summarized
from memory, not re-derived. Numbers will drift as more paper trades and
walk-forward runs accumulate; re-check the JSON files directly for the
latest state rather than trusting this table indefinitely.

### V1 (4H Turtle-Donchian)

| Check | Result |
|---|---|
| Walk-forward OOS (`walk_forward_results_v1.json`, 9 folds, 2020-05→2024-11) | Sharpe **0.69**, total return **+120.5%**, max drawdown **-61.7%**, 321 OOS trades |
| Compare to in-sample headline | 855% return / 1.98 Sharpe / -39.3% drawdown (main.py's flat grid search) |
| Paper trading gate (`readiness_v1.json`) | **NOT READY** — 0 completed paper trades recorded (need ≥30) |
| `exit_len` parameter stability across folds | Unstable (4 distinct values across 9 folds) — everything else stable |

The OOS Sharpe (0.69) and drawdown (-61.7%) are both materially worse than
the in-sample 855%/1.98/-39.3% headline. That gap is the entire point of
walk-forward testing: it's telling you the in-sample number overstated the
edge. V1 hasn't accumulated any real paper trades yet, so the live-readiness
gate has nothing to evaluate.

### V4 (1H fast variant)

| Check | Result |
|---|---|
| Walk-forward OOS (`walk_forward_results_v4.json`, 9 folds, 2020-05→2024-11) | Sharpe **-0.39** (negative), total return +489.4%, max drawdown **-121.3%** |
| Compare to in-sample headline | 1572% return / 1.20 Sharpe / -65% drawdown |
| Paper trading gate (`readiness_v4.json`) | **NOT READY** — confirmed loss: -28.7% return, Sharpe -1.36, 39.8% max drawdown, 20.2% win rate over 104 real paper trades (2025-12-28 → 2026-07-20) |

A max drawdown past -100% on the stitched OOS equity curve means the curve
went negative in at least one fold — a strategy-breaking result, not a rounding
artifact. Combined with a negative walk-forward Sharpe and a **confirmed
losing real paper-trading track record**, V4's headline 1572% number should
not be treated as representative of anything. See
`docs/archive/backups/README.md` for additional context: V4's live bot code
doesn't even import the same `strategy.py`/`config.py` that the validation
pipeline tests — that inconsistency is an open item, not yet resolved.

**Bottom line: neither strategy is currently authorized to trade live.**
Both `readiness_v1.json` and `readiness_v4.json` say `ready_for_live: false`,
and Phase 0's gate will block `bot_runner.py` from trading either one until
that changes.

---

## Running it yourself

```bash
# Phase 2 — walk-forward (full grid per fold takes hours; --fast for a sanity check)
cd research/validation
python walk_forward_test.py --fast
python walk_forward_test.py --jobs 8                    # full grid, parallelized

# Phase 3 — edge validation (needs Phase 2's output first)
python edge_validation_test.py

# Phase 4 — cross-market
python cross_market_validation.py

# Phase 5 — rebuild both readiness gates from current paper trade logs
python build_readiness_gates.py --all

# Check whether a strategy is currently cleared to trade
python -c "from readiness_gate import check_gate; print(check_gate('v1'))"

# Final holdout — DO NOT run until Phases 2-4 are done and params are frozen
python final_holdout_validation.py --i-understand-this-burns-the-holdout
```

## What NOT to trust as a validity claim

- **`python main.py`** — runs the legacy 7-phase in-sample pipeline. Useful
  for quick sanity checks on strategy code, not evidence of a working edge.
- **`python validate_all.py`** (research/validation/validate_all.py) — checks
  that a backtest reproduces expected *in-sample* numbers (e.g. "Total Return
  > 800%"). It's a regression test for the code, not a validation test for
  the strategy. A strategy that's badly overfit would still pass every check
  in this file.
- **Any number quoted without saying in-sample or out-of-sample** — assume
  in-sample (optimistic) unless it explicitly comes from
  `walk_forward_results_*.json`, `readiness_*.json`, or the final holdout.

---

*Last updated: July 24, 2026, from the current contents of `data/readiness_v1.json`,
`data/readiness_v4.json`, `data/walk_forward_results_v1.json`, and
`data/walk_forward_results_v4.json`. Re-derive rather than trust this table once
those files have moved on.*

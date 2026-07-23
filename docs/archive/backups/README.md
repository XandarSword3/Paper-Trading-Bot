# Archived / non-authoritative files

Phase 6 of VALIDATION_REMEDIATION_PLAN.md: "move anything non-authoritative
into backups/ or archive/ with a one-line reason." Everything below was
confirmed, by grepping every `.py`/`.yml` file in the repo, to be imported by
nothing outside this same group — not by either live bot (`github_bot.py`,
`github_bot_v4.py`), not by the dashboard, not by any test/validation script.

| File | Reason archived |
|---|---|
| `strategy_v1_winning.py`, `config_v1_winning.py`, `sp500_reinvest_v1.py` | Pre-existing archive (superseded V1 iteration), predates this pass |
| `strategy_v2.py`, `config_v2.py` | Superseded experimental version; not imported anywhere live |
| `strategy_v3.py`, `strategy_v3_fast.py`, `v3_strategy.py` | Superseded experimental versions; not imported anywhere live |
| `strategy_v4.py`, `strategy_v4_fast.py`, `config_v4_optimal.py` | The live V4 bot (`github_bot_v4.py`) hardcodes its own params inline and never imports these — the "1572% backtest return" this config's docstring cites was in-sample only (same flaw the whole remediation plan exists to fix), and V4's real paper track record is now a confirmed loss (see `readiness_v4.json`) |
| `enhanced_strategy.py` | Experimental variant; not imported anywhere live |

**Canonical strategy code going forward:** `strategy.py` + `config.py`'s
`DEFAULT_PARAMS` — this is what `walk_forward.py`, `monte_carlo.py`,
`cross_market_validation.py`, `robustness_test.py`, and `final_holdout_validation.py`
all actually exercise, and its parameters match what `github_bot.py` (V1)
hardcodes. V4 has no equivalent tested module — see the open Phase 6 decision
in `VALIDATION_REMEDIATION_PLAN.md` (fix it, revert to V1, or retire it).

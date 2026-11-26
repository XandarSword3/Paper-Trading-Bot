# New Edge Lab

This directory is deliberately separate from the existing production strategies.
Nothing here touches RGMT, funding carry, OmniPulse, or any live path.

## Mandate

Find a **new**, uncorrelated edge by testing pre-registered hypotheses against
the research data already cached in this workspace. The loop is allowed to fail;
it is not allowed to data-mine until something looks good.

## Data

| Dataset | Coverage | Use |
| :--- | :--- | :--- |
| Binance USDT-perp 1m | 2025-11-01 .. 2026-10-31, 20 symbols | intraday pilots |
| Binance USDT-perp 1h | 2020-01-01 .. 2026-10-31, 36 symbols | multi-day / daily pilots |
| Binance funding | 2020-01-01 .. 2026-10-31, 36 symbols | crowding / carry signals |
| Order-flow bars | 2026-10-23 .. 2026-11-22, 20 symbols | microstructure pilots only |

Discovery is always the earlier slice. The holdout is loaded only after a
candidate passes discovery.

## Loop

1. **Pre-register** one hypothesis: mechanism, universe, signal, entry, exit,
   sizing, costs, and the exact primary parameters. One primary config per
   hypothesis.
2. **Implement** the smallest backtest that answers the question.
3. **Run discovery** only. Report gross and net separately.
4. **Screen** with hard gates:
   - net Sharpe >= 1.0
   - max drawdown <= 20%
   - >= 100 trades or >= 250 trading days
   - positive in at least 2 of 3 sub-periods
   - top-5-day share of positive PnL <= 50%
   - placebo p-value < 0.05
5. **Holdout once** for survivors: net Sharpe >= 0.5, positive return,
   max drawdown <= 25%, DSR >= 0.95 with all trials counted.
6. **Robustness** for survivors: 2x costs, +/-20% parameters, leave-one-symbol
   out, sub-period stability, correlation to the existing book.
7. **Verdict**: KILL, PARK, or PROMOTE_TO_PAPER. A promoted idea gets its own
   paper runner before any live capital.
8. **Log** every trial, including failures, in `registry.json`.

## Rules

- No re-tuning on the holdout.
- Every parameter combination counts as a trial.
- If a hypothesis fails its primary config, it is killed. A variant is a new
  hypothesis, not a rescue.
- Report the failures. A lab that only reports winners is a marketing
  department.

## Results

See `FINDINGS.md` for the write-up and `registry.json` for the trial log.
Result payloads live in `results/`.

**Status: 9 trials, 0 deployable edges.** Every hypothesis was killed by
costs, by the holdout, or by a fill/latency artifact. The lab is deliberately
not promoting anything to paper trading.

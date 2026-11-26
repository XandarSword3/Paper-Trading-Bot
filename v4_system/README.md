# V4 — Multi-Horizon AI Crypto Trading System

Aggressive, automated, multi-timeframe crypto trading research platform.
Target: ≥6%/month net on $1,000 (target, not assumption — see PREREG).

**Status: Phase 1 (data engine) in progress.** See `PREREG.md` for
hypotheses, trial budgets and pass/fail thresholds. Every trial is logged
in `trials_log.csv`.

## Module map

| Module | Status | Purpose |
|---|---|---|
| `data_engine/` | ✅ implemented | Binance public data: perp+spot klines (1m–1d), funding, OI, mark/index, aggTrades, contract specs; parquet store + manifest + gap detection |
| `news_engine/` | ⬜ Phase 5 | Event ingestion, classification, PIT-safe features |
| `macro_engine/` | ⬜ Phase 6 | Cross-asset regimes, directional bias |
| `technical_engine/` | ⬜ Phase 2 | Indicator library (incremental-value tested) |
| `microstructure_engine/` | ⬜ Phase 3 | OI/funding/CVD/book/liquidation features |
| `ml_engine/` | ⬜ Phase 4 | LightGBM/XGBoost baselines, calibrated, purged WF |
| `strategy_engine/` | ⬜ Phase 6 | 6 independent strategies + central decision engine |
| `risk_engine/` | ⬜ Phase 6 | Sizing, leverage ladder, liquidation modeling |
| `execution_engine/` | ⬜ Phase 7 | Order routing, reconciliation (paper only) |
| `backtesting/` | ⬜ Phase 2+ | Event-driven sim: fees, slippage, funding, liquidation |
| `paper_trading/` | ⬜ Phase 7 | Live paper deployment, OI/liquidation recorders |
| `reports/` | ⬜ ongoing | Equity curves, trade logs, robustness reports |

## Quick start

```powershell
# Availability probe (all endpoints, ~1 min)
python -m v4_system.data_engine.fetch_phase1 --probe

# Smoke test: BTCUSDT 1h only
python -m v4_system.data_engine.fetch_phase1 --quick

# Full Phase 1 fetch (~15–25 min, resumable — skips existing datasets)
python -m v4_system.data_engine.fetch_phase1
```

Data lands in `v4_system/data/*.parquet` with provenance in
`v4_system/data/MANIFEST.json`.

## Hard data limitations (see reports/DATA_AVAILABILITY.md)

- No historical liquidations or order-book depth from Binance public API
  (websocket-only, live). Self-recording starts in paper-trading phase.
- Open-interest history capped at ~30 days.
- aggTrades full-history is available but heavy; bounded windows only for now.

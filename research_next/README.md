# research_next — Automated Crypto Income ($1,000 total capital)

Separate research project; all S1–S6 work in `../slow_study/` is preserved
untouched. Three new strategy directions tested under a strict $1,000 TOTAL
capital constraint with a full margin/cost/liquidation model.

## Structure

```
research_next/
  data/            # fetched market data (see below)
    spot/          # OKX USDT spot 1D, 12 majors (copied from slow_study)
    funding_*_binance.csv   # Binance USD-M perp funding, full history
    quarterly/      # Binance USD-M quarterly futures daily klines
    fetch_binance_funding.py
    fetch_quarterly_futures.py
  strategies/      # frozen rule definitions
    sA_funding_arb.py    # A: multi-asset funding arb
    sC_adaptive.py       # C: adaptive funding carry
  backtest/
    engine.py      # multi-asset margin engine (S5 v2.1 rules generalized)
    cash_carry.py  # B: cash-and-carry futures basis
  execution/
    venue_notes.md # OKX/Bybit specs, fees, Lebanon access findings
  reports/         # equity_*.csv, trades_*.csv, summary.csv,
                   # sensitivity_B.csv, REPORT.md
  run_all.py       # reproduces everything
  make_report.py   # regenerates REPORT.md from summary.csv
```

## Reproduce

```powershell
cd research_next
python run_all.py        # ~15-20 min, writes reports/*.csv
python make_report.py    # writes reports/REPORT.md
```

Data fetchers (only needed if re-fetching):
`python data/fetch_binance_funding.py`, `python data/fetch_quarterly_futures.py`.

## Frozen rules (fixed before runs, no post-hoc tuning)

- **A — multi-asset funding arb**: top-3 trailing-30d funding, weekly rebalance,
  negative-funding avoidance, alloc 0.90/K, L∈{1,2}.
- **B — cash-and-carry**: long spot + short nearest quarterly future with
  >14d to expiry, enter only if annualized basis >3%/yr, IMR=20%, MMR_F=0.5%,
  roll at expiry, BTC & ETH independently at $1,000.
- **C — adaptive funding carry**: BTC carry, weekly exposure multiplier from
  trailing funding + 30d vol (thresholds in `strategies/sC_adaptive.py`),
  L∈{1,2}.

Margin/liquidation/cost rules are inherited from the audited S5 v2.1 model
(isolated margin, MMR 0.40%, 0.15%/side costs, Rule A defense, Rule B
rebalance, daily-high liquidation checks, no re-entry after liquidation).

## V2 validation (2026-11-26) — READ THIS FIRST

`reports_v2/REPORT_V2.md` **supersedes the Strategy B results below.** A
data-alignment bug (OKX 16:00 UTC candle-open labels vs Binance 00:00 UTC
daily futures candles) inflated all v1 B numbers. Corrected: ETH/BTC
cash-and-carry earns ~0.25–0.35%/month with zero liquidations only under
Binance Multi-Assets cross-margin, and **underperforms S5 1x**. Isolated
margin loses money or liquidates at every tested IMR. Final answer: **no**,
B is not deployable.

V2 reproduction:
```powershell
python data/fetch_quarterly_4h.py   # 4H futures (alignment fix)
python run_v2.py                    # audit + margin sweep + common period
```

## Headline results (FULL, cost 1.0x) — SUPERSEDED for Strategy B, see V2

| strategy | final | CAGR | mean/mo | maxDD | liq | verdict |
|---|---|---|---|---|---|---|
| A funding arb L1 | $1,037 | 0.5% | 0.01% | -3.0% | 0 | rejected (costs kill it) |
| B cash-carry ETH | $1,652 | 13.6% | 1.15% | -13.0% | 9 | best return, fragile |
| B cash-carry BTC | $1,212 | 5.0% | 0.46% | -16.7% | 8 | marginal |
| C adaptive L1 | $1,134 | 3.3% | 0.26% | -0.6% | 0 | safe, below S5 1x |
| C adaptive L2 | $1,187 | 4.5% | 0.36% | -0.9% | 0 | below S5 1x |
| S5 1x (benchmark) | $1,434 | 5.4% | 0.44% | -0.9% | 0 | still best risk-adjusted |
| BTC buy&hold | $3,615 | 38.6% | 4.3% | -53.3% | 0 | not income |

See `reports/REPORT.md` for the full comparison, splits (2023+/2025+), cost
stress, sensitivity, and the final recommendation.

## Known limitations

- Funding is Binance, spot is OKX — cross-venue basis ignored.
- Perp marked at spot price (no alt perp OHLC history).
- Daily-HIGH liquidation approximation; no intraday path.
- Survivorship-biased current top-12 universe.
- Lebanon: not on OKX/Bybit excluded lists, but BDL Article 10 blocks bank
  funding rails and account approval is discretionary — see
  `execution/venue_notes.md`.

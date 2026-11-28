# Slow-Strategy Study — Results (net of costs)

Costs: 0.10% taker + 0.05% slippage per side (0.15%/side). Start equity $1,000. Signals on daily close, executed next open.

Bootstrap: circular block bootstrap over monthly returns, block=6 months, 10,000 reps. 99% CI = Bonferroni correction for 5 strategies.

## Period splits

| strategy | split | n_months | mean/mo | median/mo | %mo>=6% | worst/mo | CAGR | maxDD | underwater_d | CI99_lo | CI99_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_ma_trend|FULL | FULL | 94 | 2.66% | 0.00% | 17% | -14.1% | 28.0% | -33.1% | 962 | -0.56% | 7.56% |
| S1_ma_trend|2019_2022 | 2019_2022 | 47 | 3.53% | 0.00% | 17% | -14.1% | 37.7% | -33.1% | 627 | -1.49% | 13.18% |
| S1_ma_trend|2023_plus | 2023_plus | 47 | 1.80% | 0.00% | 17% | -10.3% | 18.8% | -30.9% | 708 | -0.91% | 5.55% |
| S3_donchian55|FULL | FULL | 94 | 3.80% | 0.00% | 26% | -24.7% | 39.4% | -48.7% | 825 | -0.44% | 9.45% |
| S3_donchian55|2019_2022 | 2019_2022 | 47 | 5.27% | 0.00% | 26% | -24.7% | 57.8% | -48.7% | 508 | -1.68% | 15.17% |
| S3_donchian55|2023_plus | 2023_plus | 47 | 2.32% | 0.00% | 26% | -17.7% | 22.8% | -39.6% | 415 | -1.48% | 6.81% |
| S6_buyhold|FULL | FULL | 94 | 4.74% | 2.69% | 44% | -37.1% | 41.8% | -76.7% | 846 | -0.46% | 10.87% |
| S6_buyhold|2019_2022 | 2019_2022 | 47 | 5.67% | 2.67% | 45% | -37.1% | 44.2% | -76.7% | 483 | -3.83% | 16.44% |
| S6_buyhold|2023_plus | 2023_plus | 47 | 3.82% | 2.71% | 43% | -22.0% | 39.2% | -53.1% | 415 | -1.25% | 8.97% |
| S2_tsmom_voltarget|FULL | FULL | 94 | 3.54% | 0.00% | 28% | -47.2% | 30.1% | -69.3% | 846 | -1.25% | 9.71% |
| S2_tsmom_voltarget|2019_2022 | 2019_2022 | 47 | 4.58% | 0.00% | 28% | -47.2% | 37.2% | -69.3% | 550 | -4.17% | 15.71% |
| S2_tsmom_voltarget|2023_plus | 2023_plus | 47 | 2.50% | 0.00% | 28% | -18.6% | 23.2% | -46.0% | 415 | -1.28% | 7.35% |
| S4_xsmom_weekly|FULL | FULL | 94 | -0.93% | 0.00% | 3% | -24.7% | -13.1% | -70.0% | 395 | -3.21% | 0.49% |
| S4_xsmom_weekly|2019_2022 | 2019_2022 | 47 | 0.00% | 0.00% | 0% | 0.0% | 0.0% | 0.0% | 0 | 0.00% | 0.00% |
| S4_xsmom_weekly|2023_plus | 2023_plus | 47 | -1.86% | 0.00% | 6% | -24.7% | -24.7% | -70.0% | 395 | -6.03% | 0.94% |
| S5_funding_carry|FULL | FULL | 82 | 2.31% | 0.87% | 9% | -0.3% | 30.2% | -1.1% | 136 | 0.86% | 5.14% |
| S5_funding_carry|2019_2022 | 2019_2022 | 35 | 3.88% | 0.89% | 17% | -0.2% | 53.9% | -1.1% | 60 | 0.84% | 9.73% |
| S5_funding_carry|2023_plus | 2023_plus | 47 | 1.14% | 0.85% | 2% | -0.3% | 14.5% | -0.7% | 136 | 0.56% | 1.86% |

## Verdicts (2023+ out-of-sample)

| strategy | mean/mo | CI99 | maxDD | VIABLE | TARGET(>=6%/mo) |
|---|---|---|---|---|---|
| S1_ma_trend | 1.80% | [-0.91%, 5.55%] | -30.9% | False | False |
| S3_donchian55 | 2.32% | [-1.48%, 6.81%] | -39.6% | False | False |
| S6_buyhold | 3.82% | [-1.25%, 8.97%] | -53.1% | False | False |
| S2_tsmom_voltarget | 2.50% | [-1.28%, 7.35%] | -46.0% | False | False |
| S4_xsmom_weekly | -1.86% | [-6.03%, 0.94%] | -70.0% | False | False |
| S5_funding_carry | 1.14% | [0.56%, 1.86%] | -0.7% | True | False |

## FINAL VERDICT

**VIABLE but below target** — a strategy has a positive OOS CI and DD<=40%, but mean < 6%/month.
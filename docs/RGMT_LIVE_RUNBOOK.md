# RGMT Live Runbook

Regime-Gated Multi-Horizon Trend (RGMT) is the only strategy in this repo with a
validated out-of-sample edge: Sharpe 1.22, 27.2% CAGR, -19.0% max drawdown
(2020-10-31 to 2026-10-31), holdout Sharpe 0.71, DSR 0.999.

The gate is BTC above its 200-day EMA. **When BTC is below the EMA the correct
position is cash.** Do not bypass this to "make something happen"; the flat
periods are part of the edge.

## Current state (2026-11-24)

| Item | Value |
| :--- | :--- |
| BTC close | $61,241 |
| BTC 200-day EMA | $75,128 |
| Distance to gate | -18.5% |
| Regime | OFF |
| Paper equity | $10,000 (flat, no positions) |

The paper runner is scheduled daily at 00:05 UTC by
`.github/workflows/rgmt_daily.yml`. It runs the accounting/safety tests, executes
the paper rebalance, verifies state health, and commits the snapshot.

## Credentials

The executor reads `BINANCE_API_KEY`, `BINANCE_API_SECRET`, and
`BINANCE_TESTNET` from the environment or from the gitignored `.env` file.

### Testnet (do this first)

1. Register at <https://testnet.binancefuture.com>.
2. Create an API key there with Futures enabled.
3. Put it in `.env`:
   ```dotenv
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   BINANCE_TESTNET=true
   ```
4. Verify:
   ```powershell
   python scripts/run_rgmt_live.py --check
   ```

A spot-testnet key does **not** work on futures testnet. The current `.env` key
returns `401 -2015` on `/fapi/v2/account`, so it is not a futures key.

### Mainnet (only after testnet works)

1. Create a Binance Futures API key with **Enable Futures**.
2. Leave withdrawals disabled.
3. Optionally whitelist the machine's IP.
4. Set `BINANCE_TESTNET=false` in `.env`.

## Commands

Print the plan without sending anything (default):

```powershell
python scripts/run_rgmt_live.py
```

Execute on testnet:

```powershell
$env:ALLOW_LIVE = "true"
python scripts/run_rgmt_live.py --execute
```

Execute on mainnet (both opt-ins required):

```powershell
$env:BINANCE_TESTNET = "false"
$env:ALLOW_LIVE = "true"
$env:ALLOW_MAINNET_LIVE = "true"
python scripts/run_rgmt_live.py --execute
```

## Safety behavior

- Default is a dry run. `--execute` is required to send orders.
- `ALLOW_LIVE=true` is required for any execution.
- Mainnet additionally requires `ALLOW_MAINNET_LIVE=true`.
- Frozen limits: 3x gross, 15% per asset, 40 orders per run.
- The account must be in one-way mode and dedicated to this strategy. Positions
  outside the target set are flattened.
- A 30% drawdown from peak flattens the book and latches a halt.
- Create `data/RGMT_LIVE_HALT` to flatten on the next run.
- Clear a latched halt only after reviewing the cause:
  ```powershell
  python scripts/run_rgmt_live.py --resume
  ```
- Reduce-only orders are never blocked by notional caps, so the book can always
  be de-risked.

## What "ready" means

The code path is ready. Actual capital deployment is blocked on two things:

1. BTC reclaiming its 200-day EMA (currently ~$75.1k).
2. A valid Binance **Futures** API key. I cannot create or authorize one.

Until both are true, the honest expected return from RGMT is 0% (cash), not a
forced trade.

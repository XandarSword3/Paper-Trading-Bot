# Trading Bot Guide

> This replaces an older version of this doc that described a standalone
> `paper_bot.py` script polling Binance testnet in a loop. That script no
> longer exists — it was consolidated into a single gated runner. If you have
> notes or muscle memory from the old workflow (`python paper_bot.py`,
> `KILLSWITCH.txt`), they no longer apply.

## What actually runs today

**Script:** `research/bots/bot_runner.py` — one unified runner for both V1
and V4, invoked as a single cycle per run (not a long-lived loop).

```bash
python -m research.bots.bot_runner --strategy v1     # single V1 cycle
python -m research.bots.bot_runner --strategy v4     # single V4 cycle
python -m research.bots.bot_runner --strategy all    # both, V4 then V1
```

**How it's actually scheduled:** GitHub Actions (`.github/workflows/bot.yml`),
not a local always-on process:
- V4 runs every hour (`5 * * * *`)
- V1 runs every 4 hours (at 0, 4, 8, 12, 16, 20 UTC)
- State (`data/bot_state.json`, `data/bot_state_v4.json`) and trade logs
  (`data/trades.json`, `data/trades_v4.json`) are committed back to the repo
  after each run — that's the persistence layer.

**Data source:** Kraken's public API (`XBTUSD`), not Binance testnet. No API
keys are required to fetch candles for either bot.

## The gate check happens on every single run

Before any entry/exit logic executes, `bot_runner.py` calls
`research.validation.readiness_gate.check_gate(strategy)`:

- **Gate passes** → normal entry/pyramiding/exit logic runs against
  live candles.
- **Gate blocked** → the run still fetches candles, logs the current
  price/ATR/entry-high/exit-low, and saves state — but takes **no trading
  action**. This is "log-only mode," visible in the run logs as
  `GATE BLOCKED — log-only mode: <reason>`.

Right now, both `data/readiness_v1.json` and `data/readiness_v4.json` set
`ready_for_live: false`, so every scheduled run of either bot is currently in
log-only mode. See
[TESTING_AND_OPTIMIZATION_GUIDE.md](TESTING_AND_OPTIMIZATION_GUIDE.md) for
why, and what has to happen (30+ real paper trades meeting the Phase 5
thresholds) before that flips.

The gate file itself expires after 24 hours (`MAX_GATE_AGE_HOURS` in
`readiness_gate.py`), so it has to be refreshed regularly — see
`.github/workflows/refresh_gates.yml`, which reruns
`build_readiness_gates.py --all` on a schedule.

## Strategy parameters actually used by the runner

These are hardcoded in `research/bots/bot_runner.py`'s `STRATEGY_SPECS`, not
read from `research/strategies/config.py` — worth knowing if you're comparing
live behavior to the validation pipeline's `DEFAULT_PARAMS`.

| | V1 | V4 |
|---|---|---|
| Interval | 240 min (4H) | 60 min (1H) |
| Entry length | 40 | 8 |
| Exit length | 16 | 16 |
| ATR length | 20 | 14 |
| Trail multiplier | 4.0 | 3.5 |
| Risk % | 1% | 1% |
| Max pyramid units | 4 | 4 |

V1's parameters match `config.py`'s `DEFAULT_PARAMS`. V4's live parameters
are a separately-maintained copy — see `docs/archive/backups/README.md` for
the open question about V4 not sharing a canonical, tested strategy module
with the validation pipeline.

## Monitoring

- **Logs:** GitHub Actions run logs for `bot.yml` (each scheduled run is a
  separate job log), not a local `paper_bot.log` file.
- **State:** `data/bot_state.json` (V1) / `data/bot_state_v4.json` (V4) —
  current equity, open position, trade count.
- **Trades:** `data/trades.json` (V1) / `data/trades_v4.json` (V4) — full
  ENTRY/EXIT history, the same files `build_readiness_gates.py` reads to
  compute the paper-trading readiness metrics.
- **Telegram:** set `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` as repo secrets to
  get trade notifications (see `research/bots/telegram_bot.py`).

## Troubleshooting

**Bot shows "GATE BLOCKED" every run** — expected right now for both
strategies. It's not a bug; it means Phase 5's paper-trading thresholds
haven't been met yet. Check `data/readiness_<strategy>.json` for the current
`reasons` list.

**No candle data / "Aborting strategy run"** — Kraken's public API had an
issue, or the workflow's network access failed. Check the Actions run log.

**Want to check readiness manually:**
```bash
python -c "from research.validation.readiness_gate import check_gate; print(check_gate('v1'))"
python -c "from research.validation.readiness_gate import check_gate; print(check_gate('v4'))"
```

## Before going live

Live capital is a separate, larger decision than clearing the paper-trading
gate — see [PAPER_TRADING_CHECKLIST.md](PAPER_TRADING_CHECKLIST.md) and
[CRITICAL_WARNINGS.md](CRITICAL_WARNINGS.md). Clearing Phase 5's gate makes
`ready_for_live: true`, which only means `bot_runner.py` will stop
suppressing trades — it is not itself a recommendation to fund the bot with
real money.

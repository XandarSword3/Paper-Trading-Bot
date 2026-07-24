# Quant Alpha Terminal — Complete Build Plan

*Grounded in `XandarSword3/Paper-Trading-Bot` as of July 23, 2026 — not the two mockup images. Where the mockups and the repo disagree, the repo wins and it's called out explicitly.*

---

## 0. TL;DR

- **Only 1 of your 8 sidebar destinations is real.** `switchNavTab()` in `app.js` just toggles a CSS class — Strategies, Backtests, Paper Trading, Risk, CI/CD, Alerts, Settings all render nothing. Command is the only live page.
- **The backend is not the bottleneck.** FastAPI + SQLAlchemy, 11 routes, a genuinely institutional-grade analytics module (bootstrap Sharpe CI, Sortino, Calmar, Monte Carlo ruin-survival, VaR/CVaR, rolling metrics, PnL histograms, duration scatter). This is a frontend and instrumentation project riding on a serious quant backend, not a backend rebuild.
- **The current frontend is 100% hand-rolled** — vanilla HTML/CSS/JS, zero libraries, `<script src="/static/app.js"></script>` is the *only* script tag. The radar chart, donut chart, particle globe, and brain icon are all drawn by hand on `<canvas>`. That's the ceiling you're hitting, not a skill issue — you can't get TradingView-grade candles or a real 3D mesh out of hand-rolled Canvas 2D without reinventing several libraries.
- **Image 1's numbers are the old fake stub values, not a target.** `StrategyHealthSnapshot` in `models.py` carries a comment explicitly saying `core_stability_pct=98.0, confidence_level_pct=81.0, market_regime="TRENDING"` used to be hardcoded and were deliberately ripped out. The mockup was almost certainly generated from a screenshot taken while those defaults were still live — it inherited the fake numbers along with the visual style. Chase the *visual fidelity*, not `98%` / `81%` / `TRENDING`.
- **I found three things that look real on screen but aren't**, in increasing order of severity — full detail in §7:
  1. Eight "system health" badges around the Command globe (`SENTIMENT: POSITIVE`, `NEWS FEED: NOMINAL`, `MOMENTUM: ACTIVE`, `RISK ENGINE: OPTIMAL`, etc.) are **static text baked into `index.html`**. No JS function ever touches them. They'll say `OPTIMAL` and `POSITIVE` forever, including right now, while the account is down 28.67% with a 0.71 profit factor.
  2. The Execution Timeline's 9 stage timestamps look like real per-stage telemetry but are **fabricated** — `trigger_bot_run()` takes the total wall-clock duration of one bot cycle and splits it at fixed proportions (0%, 5%, 10%, 15%, 40%, 50%, 60%, 70%, 90%). Nothing inside the actual strategy execution emits these.
  3. `/api/v1/terrain/{id}` — the endpoint the "3D Volatility & Price Mesh" / "Risk Terrain" concept needs — **exists and returns 200, but `terrain_matrix` is never populated anywhere in `analytics.py`.** It's a stub that always returns `[]`.
- This plan gives you all 11 pages, reconciled from your real nav + both mockups' navs, each with a signature visual, a widget-by-widget data/library spec, and an honest real/stub/needs-new-work tag on every single widget.

---

## 1. Ground Truth Table

Before any page design, here's what's actually true right now. Everything below this table is designed against these facts, not against the mockups.

| Backend route | Wired to any UI? | Status |
|---|---|---|
| `GET /api/v1/command_center/{id}` | Yes — the only live page | Real, well-computed |
| `GET /api/v1/strategies` | No | Real, ready to use |
| `GET /api/v1/strategies/{id}/trades` | No | Real, ready to use |
| `POST /api/v1/trades` | Indirectly (bot writes to it) | Real, has write-path dedup guard |
| `GET /api/v1/readiness` | No | Real, ready to use |
| `POST /api/v1/bot/run` | Yes — "RUN BOT EXECUTION" button | Real, but see fabricated timestamps below |
| `GET /api/v1/analytics/{id}` | No | Real — 20+ metrics incl. `pnl_histogram`, `duration_scatter`, `rolling_metrics`, `sharpe_ci` that **neither mockup even shows** |
| `GET /api/v1/candles` | No | Real — live Kraken OHLCV + server-computed dynamic Donchian channel + ATR per candle |
| `GET /api/v1/equity_curve/{id}` | No | Real |
| `GET /api/v1/terrain/{id}` | No | **Stub — always returns `terrain_matrix: []`** |
| `GET /api/v1/events` | No | Real endpoint, correct category filtering, but table is essentially empty (see §7) |

| Mockup element | Reality |
|---|---|
| Core Stability 98% / Confidence 81% / "TRENDING" (Image 1) | Old hardcoded defaults, explicitly deprecated in `models.py`. Real formula now exists (win rate + Sharpe + drawdown + profit factor for stability; 500-iteration Monte Carlo ruin-survival for confidence). |
| Capital Allocation: BTC/ETH/SOL/USDT/OTHERS (Image 1) | Bot is single-asset (BTC only). `README.md`'s own roadmap lists "Portfolio mode (BTC, ETH, SOL)" as **unbuilt**. Real payload only ever returns a BTC/USDT split today. |
| 8 satellite node badges around the globe (both images) | **Static HTML, never updated.** See §7.1. |
| Execution Timeline 9-stage trace (Image 1) | Real row gets written per run, but internal stage timestamps are a fabricated proportional split, not genuine telemetry. See §7.2. |
| "3D Volatility & Price Mesh" (Image 3) | No backend support exists for a *volatility* mesh at all. A close cousin — a *drawdown terrain* mesh — has a named, intended endpoint (`/api/v1/terrain`) that is currently a stub. |
| Live Signal Scanner, multi-symbol (ETH/SOL/AVAX/LINK, Image 3) | Bot trades BTC only. This widget as drawn is fully aspirational — see §6 for the honest scope-down. |
| Order Book / Market Depth (Image 3) | Nothing in the backend touches order book data at all. Needs new infrastructure — see §6. |
| System Metrics: CPU/memory/latency (Image 3) | Doesn't exist. Cheap to add — see §6. |

---

## 2. Global Shell (persistent on every page)

Both mockups and the real app agree on this chrome, so it's built once as an app shell, not per-page.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ [Logo] QUANT ALPHA TERMINAL   STATUS  MISSION TIME  STRATEGY▾  BTC  EQUITY  [RUN BOT]│
├────────┬──────────────────────────────────────────────────────────────────┤
│ LIVE   │                                                                    │
│ ○Command│                                                                   │
│ ○Live Term.                                                                 │
│ ○Paper Tr.                        <page content>                           │
│ ○Alerts│                                                                    │
│ ANALYSIS                                                                    │
│ ○Strategies                                                                 │
│ ○Backtests                                                                  │
│ ○Risk  │                                                                    │
│ ○Risk Terrain                                                               │
│ ○Reports                                                                    │
│ SYSTEM │                                                                    │
│ ○Readiness & CI                                                             │
│ ○Settings                                                                   │
│ [QA OPERATOR • ONLINE]│                                                     │
└────────┴──────────────────────────────────────────────────────────────────┘
```

**Changes from the current shell:**
- **Nav gets 3 grouped clusters with small caption labels** (`LIVE` / `ANALYSIS` / `SYSTEM`) instead of one flat list. This isn't decoration — it encodes a real distinction (operational/real-time vs. research/analytical vs. configuration), and it's the only way 11 destinations don't read as a wall of icons.
- **Strategy selector becomes global state**, not Command-page-local. Every analytical page needs to know the active `strategy_id`. Put it in a URL query param (`?strategy=v4`) so pages are linkable/shareable, mirrored into a small global store.
- **`switchNavTab()` gets replaced by real client-side routing.** With React this is React Router; each nav item becomes a `<Link>`, each page a route, and the "active" state falls out of the router instead of manual class toggling.

---

## 3. Information Architecture — Reconciled

Your real nav and the mockups' nav don't agree with each other (Strategies/Backtests/CI-CD exist in your real sidebar but not in the Image 3 mockup; Live Terminal/3D Analytics/Risk Terrain/Reports exist in the mockup but not your real sidebar). I merged them into one IA rather than picking one arbitrarily:

| # | Page | Cluster | In real nav? | In mockup nav? | Primary endpoint(s) |
|---|---|---|---|---|---|
| 1 | Command | Live | ✅ | ✅ | `/command_center/{id}` |
| 2 | Live Terminal | Live | — | ✅ | `/candles`, new order-book source |
| 3 | Paper Trading | Live | ✅ | ✅ | `/strategies/{id}/trades`, `/overview` |
| 4 | Alerts | Live | ✅ | ✅ | `/events` |
| 5 | Strategies | Analysis | ✅ | — | `/strategies` |
| 6 | Backtests | Analysis | ✅ | — | `/analytics/{id}` (rolling_metrics, duration_scatter) |
| 7 | Risk | Analysis | ✅ | — | `/analytics/{id}` (var_95, cvar_95, pnl_histogram) |
| 8 | Risk Terrain | Analysis | — | ✅ (merged with "3D Analytics") | `/terrain/{id}` *(needs building)* |
| 9 | Reports | Analysis | — | ✅ | `/equity_curve/{id}`, `/analytics/{id}` |
| 10 | Readiness & CI | System | ✅ (as "CI/CD") | — | `/readiness`, GitHub Actions API |
| 11 | Settings | System | ✅ | ✅ | new |

**Two deliberate consolidations, with reasoning:**
- Image 3's "3D Analytics" and "Risk Terrain" are folded into **one** page. I couldn't find enough distinct real content to justify two separate 3D pages — both mockup concepts point at the same underlying idea (spatial exploration of risk/volatility over time), and `/api/v1/terrain` is the one endpoint that already gestures at it.
- **"CI/CD" is reinterpreted as "Readiness & CI."** I checked — you have six real GitHub Actions workflows (`bot.yml`, `refresh_gates.yml`, `fetch-data.yml`, `fetch-funding-data.yml`, `walk_forward_validation.yml`, `funding_carry_validation.yml`). Pairing their run status with the `ReadinessGate` checklist (the "is this strategy allowed to trade live" business logic) is the most literal, defensible reading of what "CI/CD" means for this specific app. Flag it if you meant something else.

---

## 4. Design System — Extend, Don't Replace

`style.css` already has a real, non-generic token system. It doesn't need to be reinvented — it needs a real rendering pipeline behind it.

**Existing tokens (keep all of these):**

| Token | Value | Role |
|---|---|---|
| `--bg-deep` | `#050811` | App background |
| `--bg-panel` / `--bg-card` | `rgba(8,14,26,.85)` / `rgba(12,20,36,.9)` | Panel/card surfaces |
| `--accent-cyan` | `#00f5d4` | Primary accent, borders, glow |
| `--accent-blue` | `#00d2ff` | Secondary data series |
| `--accent-emerald` | `#00f5a0` | Positive/bullish |
| `--accent-crimson` | `#ff3366` | Negative/risk |
| `--accent-amber` | `#ffb800` | Warning |
| `--accent-purple` | `#a855f7` | Tertiary series |
| `--font-title` | Space Grotesk | Headers, big numbers |
| `--font-sans` | Inter | Body/labels |
| `--font-mono` | JetBrains Mono | All numeric data — prices, %, timestamps |

**Tokens to add:**
- `--glow-strong` / `--glow-soft` — two-tier glow intensity so not every panel competes for attention at once
- A 3-stop gradient scale for the terrain mesh (deep purple → cyan → white-hot) reusing existing hues rather than inventing new ones
- `--motion-fast` (120ms), `--motion-base` (240ms), `--motion-slow` (600ms) — named durations so animation timing is consistent instead of ad hoc

**Library decisions (opinionated, not a menu):**

| Need | Pick | Why |
|---|---|---|
| App framework | **React + TypeScript + Vite** | Not Next.js. This is a single-tenant, client-heavy, real-time dashboard talking to your own FastAPI JSON API — Next's SSR/routing machinery is overhead you don't need here, and your `README.md` roadmap already says "Web-based dashboard (React)" in the planned column. Vite gives the fastest dev loop for this shape of app. |
| Candlesticks + volume | **lightweight-charts** (TradingView, Apache-2.0) | Purpose-built for exactly this. Don't hand-roll it. |
| Bespoke gauges (core-stability ring, confidence capsule, risk radar) | **D3 (scales/shape generators) driving React-rendered SVG** | These shapes are custom enough that a chart-kit fights you. D3 gives full control without hand-rolled Canvas math. |
| Standard charts (equity curve, PnL histogram, duration scatter) | **Recharts** or the same D3 approach | Faster to ship than custom D3 for line/area/bar; use it where the shape isn't bespoke. |
| 3D drawdown terrain | **three.js + react-three-fiber + drei** | This is real, data-driven geometry — it earns actual WebGL. |
| Quant Alpha Core globe | **2D canvas/SVG particle system, not WebGL** | It's atmospheric, not data-encoding (once the 8 badges are wired to real fields — see §7.1). Three.js would be overkill for a decorative element. |
| Motion | **Framer Motion** | Page transitions, card entrance, pipeline flow, live-value tick flashes |
| Data fetching/caching | **TanStack Query** | Polling, revalidation, cache — don't hand-roll `fetch` + `useEffect` across 11 pages |
| Real-time | **New FastAPI WebSocket endpoint** (e.g. `/ws/live`) | Replaces the current pattern of the frontend hitting a synchronous Kraken REST proxy on every poll. FastAPI supports WS natively — server subscribes once, fans out to all connected clients. |

---

## 5. Page Specs

Cross-cutting rule, stated once so it isn't repeated 40 times below: **every widget below inherits the principle already written into `compute_command_center_telemetry()`'s own docstring** — *"When data is missing, values are 0 / None / 'NO DATA' — never faked."* Every widget needs an explicit empty/loading state, not a plausible-looking placeholder.

### 5.1 Command
**Purpose:** one glance — is the bot healthy, what's it thinking, what's it holding, what just happened.
**Signature element:** the Quant Alpha Core globe, once its 8 satellite badges actually reflect reality instead of static text.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Header HUD | `command_center` | — | Already real, just needs the shell rebuild from §2 |
| Core Stability ring | `command_center.strategy_health` | D3 arc, animated fill | Real formula (win rate + Sharpe + drawdown + PF) |
| Confidence capsule | `command_center.strategy_health` | D3 vertical fill | Real Monte Carlo (500-iter ruin-survival) — worth a tooltip explaining what it actually means, since "confidence" undersells "probability of not hitting -50% drawdown across 500 resampled paths" |
| Market Regime card | `command_center.market_regime` | Text + 2 sub-stats | Real |
| Quant Alpha Core globe | 8 fields — **audit which are real before rebuilding** | Canvas/SVG particle sphere | See §7.1. I traced volatility/liquidity into the risk radar with confidence; I did **not** find where sentiment/news-feed/trend/momentum/volume are computed (or if they are at all) — confirm this before wiring the rebuilt globe |
| AI Copilot feed | `command_center.ai_copilot_insights` | Scrolling list, color-coded by severity | Genuinely real — templated off real analytics numbers. Give it a proper illustrated brain graphic and a staggered entrance animation (Framer Motion), it's currently the most under-designed real thing on the page |
| Risk radar | `command_center.risk_radar` | D3 radar/spider | Real (`var_95`, `max_dd`, `exposure`, `leverage`) |
| Capital Allocation donut | `command_center.assets[]` | D3 donut | Real, but backend returns an array — **build the widget to map over N assets**, don't hardcode a 2- or 5-slice shape, so it "just works" when portfolio mode ships |
| Execution Timeline | `ExecutionPipelineStep` | Horizontal stepped tracker, particle flow between nodes | Build the *component* now; gate turning it on for real bot runs behind the P0 fix in §7.2 |
| Live Event Feed | `/api/v1/events` | Filterable timestamped list | Same — component is ready, data isn't (§7.3) |
| Footer ticker | `command_center` header fields | Sparklines (D3) | Real |

### 5.2 Live Terminal
**Purpose:** the trading-desk view — price action, book, and system telemetry in one glance.
**Signature element:** candlestick + order book + depth chart sharing a single hovered-price crosshair across all three.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Candlestick chart | `/api/v1/candles` | lightweight-charts | Real — and it's richer than either mockup shows: server already computes dynamic Donchian entry/exit channel + ATR per candle. Overlay those as real strategy context, don't just plot bare candles |
| Volume + indicator strip (ATR/RSI/EMA) | `/api/v1/candles` (ATR real; RSI/EMA need adding) | Mini sparkline panels | ATR is real. RSI(14)/EMA(50) shown in the mockup aren't in the payload yet — cheap server-side add (pandas) |
| Order Book | **New — nothing exists today** | Two-sided depth list | Needs Kraken's public order-book WS channel or REST snapshot polling. Flag clearly: this is 100% new infrastructure, not a wiring job |
| Market Depth chart | Same new source | D3 area/mountain chart | Depends on Order Book above |
| Order Flow | Same new source | Small bar oscillator | Depends on Order Book above |
| Live Signal Scanner | **Scope decision needed** | Table, confidence progress bars | Mockup shows BTC/ETH/SOL/AVAX/LINK. Bot is BTC-only. Either (a) honestly rescope to multi-*setup*, single-symbol (e.g. different timeframes/strategies on BTC), or (b) treat multi-symbol as a real roadmap item requiring the strategy engine to evaluate other symbols. Don't fake rows for symbols the bot doesn't trade |
| Strategy Telemetry | `/api/v1/analytics/{id}` | D3 radial win-rate gauge + stat list | Real — win rate, profit factor, expectancy, avg win/loss, streaks, Sharpe all already computed |
| 3D Volatility & Price Mesh | **New computation** | react-three-fiber | Not the same thing as the drawdown terrain (§5.8) — this would need a new rolling-volatility-by-lookback matrix. Lower priority than Risk Terrain since that one already has a named endpoint |
| Live Execution Pipeline | Same as Command's Execution Timeline | Same component, reused | — |
| System Metrics (CPU/mem/latency/WS) | **New — trivial to add** | Sparkline tiles | `psutil` on the backend + a timing wrapper; genuinely cheap, unlike Order Book |
| Data Stream status | Derived from WS connection state | Status dots | Real once the WS relay from §4 exists |

### 5.3 Paper Trading
**Purpose:** what's open, what closed, and why.
**Signature element:** a single position "vitals" card with live-ticking unrealized PnL — styled like a patient-monitor readout, since it's genuinely one position (single-asset bot).

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Open position card | `/api/v1/overview` | Big numeric readout + live PnL | Real |
| Trade ledger | `/api/v1/strategies/{id}/trades` | Sortable/filterable table | Real, and already deduplicated at the query layer |
| Trade detail drawer | Same, per-row | Slide-in panel | Entry/exit/reason/PnL per trade |
| Equity mini-chart | `/api/v1/equity_curve/{id}` | Small area chart | Real |

### 5.4 Alerts
**Purpose:** one unified, prioritized notification stream.
**Signature element:** severity-colored stream merging three genuinely different sources into one feed.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Unified feed | `/api/v1/events` + readiness-gate flips + drawdown-threshold breaches | Timeline list, filter chips | The events half is real (once instrumented, §7.3); gate-flip and threshold alerts are new logic, cheap to add since the underlying numbers already exist |
| Alert rules / thresholds | New settings table | Simple form list | e.g. "notify if drawdown > 20%" |

### 5.5 Strategies
**Purpose:** compare and manage every strategy the bot runs — not just V1 and V4.
**Signature element:** parallel scorecards laid out for instant head-to-head comparison, not a table.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Strategy scorecards | `/api/v1/strategies` | Card grid, one per strategy | **Build data-driven off this list, not hardcoded to v1/v4.** `Strategy` rows are generic (id/name/timeframe/config) — when Liquidation Hunter or the funding-carry strategy get onboarded into this same schema, they should appear automatically. This mirrors the "unified table over hardcoded module naming" principle from V2-Ecosystem's engine architecture |
| Config viewer | `Strategy.config_json` | Formatted JSON/key-value view | Real |
| Readiness badge per card | `/api/v1/readiness` | Pass/fail chip | Real |
| Per-strategy quick stats | `/api/v1/analytics/{id}` | Sparkline + 3 numbers | Real |

### 5.6 Backtests
**Purpose:** the research record — walk-forward validation, robustness, historical performance.
**Signature element:** a walk-forward fold timeline, each out-of-sample fold as a colored segment. This is one of the few places numbered/sequential markers are earned rather than decorative — walk-forward folds genuinely *are* an ordered sequence.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Fold timeline | `research/validation/walk_forward.py` output — needs an API route | Horizontal segmented bar, pass/fail color | You have `walk_forward_results_v1.json` / `_v4.json` and a CI workflow (`walk_forward_validation.yml`) already producing this — needs a thin FastAPI route to expose it, not new computation |
| PnL distribution | `/api/v1/analytics/{id}.pnl_histogram` | D3/Recharts histogram | Real, unused by either mockup |
| Duration scatter | `/api/v1/analytics/{id}.duration_scatter` | Scatter plot | Real, unused by either mockup |
| Rolling metrics | `/api/v1/analytics/{id}.rolling_metrics` | Dual-line chart (rolling win rate + Sharpe) | Real, unused by either mockup |
| Yearly breakdown | Legacy `SYSTEM_SUMMARY.md` data — needs a real route if you want this live rather than static | Bar chart by year | Currently only exists as a markdown table from the old project |

### 5.7 Risk
**Purpose:** the numbers that matter if this goes live — made auditable, not just displayed.
**Signature element:** the return-distribution histogram with VaR/CVaR threshold lines drawn directly on it, so an abstract percentage becomes something you can visually verify.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Return histogram with VaR/CVaR markers | `/api/v1/analytics/{id}` | D3 histogram + threshold lines | This directly addresses the VaR/CVaR trust problem in §7.4 — instead of a bare "2.14%," you see the actual tail |
| Risk radar (expanded) | Same as Command's, larger | D3 radar | — |
| Underwater equity / drawdown table | `equity_curve[].drawdown_pct` | Area chart + sortable table of worst drawdowns | Real, already computed per-point |
| Exposure & leverage | `command_center.risk_radar` | Gauge pair | Real |
| Sharpe with bootstrap CI | `analytics.sharpe_ci` | Point estimate + CI band | Already real and already has a confidence interval — **VaR/CVaR don't, yet** (§7.4). Show them side by side so the gap is visible while you close it |

### 5.8 Risk Terrain
**Purpose:** spatial exploration of drawdown over time — literally your trade history's own terrain.
**Signature element:** the mesh itself. Once `/api/v1/terrain` is actually built, this is genuinely one-of-a-kind geometry (it's *your* trades), not decorative eye candy — which is what makes it worth the WebGL budget.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| 3D drawdown terrain | `/api/v1/terrain/{id}` | react-three-fiber mesh, orbit controls | **Blocked on §7.5** — build the frontend component against a mocked matrix shape now, swap to real data once the backend piece lands |
| View/timeframe/mesh controls | Local UI state | Dropdowns + auto-rotate toggle | As in Image 3 |
| Color legend | Derived from data range | Vertical gradient bar | Reuse the gradient token from §4 |

### 5.9 Reports
**Purpose:** a clean, exportable summary — the one page that should look calm on purpose.
**Signature element:** deliberately quiet. Per the design principle "spend your boldness in one place" — this page shouldn't compete with Risk Terrain for visual noise. It should read like an institutional tearsheet.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Summary stat grid | `/api/v1/analytics/{id}` | Plain numeric grid, mono font | No glow, no gradient — restraint is the point here |
| Equity curve | `/api/v1/equity_curve/{id}` | Clean line chart | — |
| Trade ledger extract | `/api/v1/strategies/{id}/trades` | Table | — |
| Export | New | PDF/CSV buttons | `reportlab` or `weasyprint` server-side for PDF |

### 5.10 Readiness & CI
**Purpose:** two related but distinct questions — "is this strategy allowed to trade live?" and "are the scheduled jobs healthy?"
**Signature element:** the readiness checklist rendered as a literal pass/fail circuit gating a path to a "LIVE" indicator — making the gate metaphor visual instead of just a list.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Readiness gate checklist | `/api/v1/readiness` | Circuit-style gated checklist | Real (`ready_for_live`, `sharpe_ratio`, `win_rate`, `checks_json`) |
| GitHub Actions status | **New** — GitHub Actions API | 6 workflow run tiles (`bot`, `refresh_gates`, `fetch-data`, `fetch-funding-data`, `walk_forward_validation`, `funding_carry_validation`) | These workflows already exist in `.github/workflows/`. Needs a backend route that calls `GET /repos/XandarSword3/Paper-Trading-Bot/actions/runs` and caches it |

### 5.11 Settings
**Purpose:** configuration. Deliberately the most restrained page in the app.
**Signature element:** none, on purpose. Plain forms, no glow — per the design skill's "quiet everything around the signature," this page is where the eye rests.

| Widget | Data source | Visualization | Notes |
|---|---|---|---|
| Exchange/API key management | New | Plain form | — |
| Strategy parameter editor | `Strategy.config_json` | Form bound to schema | — |
| Alert thresholds | New | Form | Feeds §5.4 |
| Appearance | Local | Toggle (motion reduce, density) | Respect `prefers-reduced-motion` regardless |

---

## 6. New Backend Surface Required

These don't exist in any form today — distinct from §7's "exists but broken" list.

| What | Effort | Notes |
|---|---|---|
| Order book / market depth | High | Needs Kraken WS order-book channel subscription server-side, fanned out over your own WS |
| System metrics (CPU/mem/latency) | Low | `psutil` + request timing middleware |
| Multi-symbol signal scanner | Medium–High (scope decision) | Bot is BTC-only today; decide honest scope per §5.2 before building |
| Volatility surface mesh (distinct from drawdown terrain) | Medium | New rolling-volatility-by-lookback matrix computation |
| GitHub Actions status route | Low | Thin wrapper around the GitHub REST API, cached |
| Walk-forward fold API route | Low | Data already exists in `walk_forward_results_*.json`; just needs exposing |
| PDF/CSV export | Low–Medium | `weasyprint`/`reportlab` |

---

## 7. Data Integrity Punch List

Ranked by how much they undermine trust in whatever you build on top of them. Not the main deliverable of this doc, but skipping this section means building a beautiful UI on top of numbers that lie.

**P0 — fix before the corresponding visual ships, not after**

1. **The 8 globe satellite badges are static HTML** (`node-val-exec`, `node-val-sent`, `node-val-risk`, `node-val-trend`, `node-val-mom`, `node-val-liq`, `node-val-vol`, `node-val-news` in `index.html`) — no function in `app.js` ever writes to them. Right now, mid-losing-streak, the UI still says `RISK ENGINE: OPTIMAL` and `SENTIMENT: POSITIVE`. This is worse than an empty state — it's confidently wrong. Either wire each to a real computed field or replace with an honest "not yet computed" treatment, consistent with the rest of the app's own stated principle.
2. **Execution Timeline sub-stage timestamps are fabricated.** `trigger_bot_run()` in `main.py` distributes 9 timestamps at fixed proportions (0/5/10/15/40/50/60/70/90%) of total run duration — it doesn't measure when risk-check actually passed or when the order actually filled. Needs real event emission from inside the strategy execution path itself.
3. **`terrain_matrix` is never populated.** `/api/v1/terrain/{id}` always returns `[]`. The route, docstring ("Duration x Depth x Time"), and even the frontend concept all exist — the actual matrix construction from `equity_curve`'s drawdown series doesn't.

**P1 — fix while building the relevant page**

4. **VaR/CVaR denominator.** `trade_pct_returns = pnl / initial_capital` uses a **fixed** $1,000 base rather than point-in-time equity. With current equity at ~$713, this *understates* real percentage risk. Either switch the denominator to equity-at-trade-time or relabel clearly if the fixed-base version is intentional.
5. **VaR/CVaR sample size.** With ~104 trades, the 5th-percentile tail is estimated from roughly 5 data points — noisy. Sharpe already gets a proper bootstrap 95% CI in the same file; VaR/CVaR should get the same treatment, and the UI should show sample size, not just a bare percentage (this is exactly what §5.7's histogram-with-markers widget is designed to make visible).
6. **Trade dedup is check-then-insert, not DB-enforced.** `record_trade_endpoint` queries for an existing match before inserting — solid, but not race-proof under concurrent writers. A DB-level `UNIQUE(strategy_id, price, quantity, timestamp)` constraint would make the guarantee real instead of best-effort. (Good news: there are already three independent layers of defense here — the write-path guard, `compute_strategy_analytics`'s in-memory dedup, and a standalone `dedupe_trades.py` cleanup script — so this is a hardening item, not an open wound.)

**P2 — lower urgency, worth knowing about**

7. `StrategyHealthSnapshot` and `CapitalAllocation` DB tables are vestigial — `compute_command_center_telemetry()` computes those values inline per-request rather than reading from either table. Either wire a periodic snapshot job (which would enable a genuinely nice "core stability over the last 30 days" trend line on Command) or drop them to avoid schema drift confusion.
8. `/api/v1/candles` hits Kraken's public REST API synchronously on every request with no caching. Fine at your current traffic; will not scale once multiple pages poll it. The WebSocket relay from §4 solves this as a side effect.

---

## 8. Phased Roadmap

| Phase | Scope | Depends on |
|---|---|---|
| 0 | Vite + React + TS scaffold; port existing CSS tokens; pick and install charting libs; global shell (§2) with real routing | — |
| 1 | Rebuild Command with real libraries, widget by widget, replacing hand-rolled Canvas pieces | Phase 0 |
| 1.5 | **P0 punch-list items** (§7.1–7.3) — do this alongside Phase 1, since Command is where all three live | Phase 0 |
| 2 | Paper Trading + Strategies — endpoints already exist, mostly wiring + design | Phase 0 |
| 3 | Risk + Backtests + Reports — biggest unlock, since `pnl_histogram`/`duration_scatter`/`rolling_metrics`/`sharpe_ci` are already computed and currently shown nowhere | Phase 0 |
| 4 | Risk Terrain | Backend terrain_matrix build (§7.3) |
| 5 | Live Terminal — candles/ATR immediately; order book, signal scanner, system metrics per the scope decisions in §5.2/§6 | Phase 0; new infra for order book |
| 6 | Readiness & CI, Alerts, Settings | Phases 2–3 for shared components |
| 7 | Real-time infra pass (WS relay replacing REST polling) + motion/animation polish pass across all pages | All above |

---

## 9. Closing Notes

The headline finding is worth repeating because it changes how you should spend your time: **the hard quant work is already done.** Bootstrap confidence intervals, Monte Carlo ruin survival, dynamic Donchian/ATR overlays, rolling metrics — that's not "junior dev vibe coded in an hour" work, and it's not the thing holding this UI back. The gap is entirely in (a) rendering libraries, (b) page count, and (c) a handful of specific instrumentation gaps that are now precisely identified in §7 rather than vaguely "suspected."

This doc is written to double as the kind of spec you'd hand to a parallel Claude session cold — every widget has its real data source named, every fake-looking thing is labeled as fake, and every genuinely new piece of work is separated from the wiring work. Worth dropping into the repo as `docs/QUANT_ALPHA_TERMINAL_BUILD_PLAN.md` if you want it to survive across sessions the way `CONTEXT.md` does for V2-Ecosystem.

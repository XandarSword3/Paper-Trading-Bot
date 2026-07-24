// ============================================================================
// QUANT ALPHA TERMINAL — API CLIENT
// Thin typed wrapper around the FastAPI backend. No mock data, no fallback
// numbers — if a fetch fails, callers see an error/empty state, not a
// plausible-looking placeholder. This mirrors the backend's own stated
// principle in compute_command_center_telemetry(): "never faked."
// ============================================================================

const BASE = '' // same-origin; Vite dev proxy forwards /api -> :8000

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      /* no-op */
    }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json()
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const b = await res.json()
      detail = b.detail || detail
    } catch {
      /* no-op */
    }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json()
}

// ---------------------------------------------------------------------------
// Types (mirroring backend/analytics.py + backend/main.py response shapes)
// ---------------------------------------------------------------------------

export interface StrategyOverview {
  id: string
  name: string
  timeframe: string
  current_equity: number
  position_size: number
  trade_count: number
  ready_for_live: boolean
}

export interface Trade {
  id: number
  strategy_id: string
  trade_type: string
  price: number
  quantity: number
  pnl: number | null
  reason: string | null
  timestamp: string
}

export interface ReadinessGate {
  strategy_id: string
  ready_for_live: boolean
  sharpe_ratio: number | null
  win_rate: number | null
  checks_json: string | null
  timestamp: string
}

export interface EquityCurvePoint {
  timestamp: string
  equity: number
  peak: number
  drawdown_pct: number
}

export interface RollingMetricPoint {
  trade_index: number
  win_rate: number
  sharpe: number
}

export interface PnlHistogramBin {
  bin_label: string
  count: number
  is_win: boolean
}

export interface DurationScatterPoint {
  duration_hrs: number
  pnl: number
  trade_type: string
}

export interface StrategyAnalytics {
  strategy_id: string
  strategy_name: string
  timeframe: string
  current_equity: number
  initial_capital: number
  total_return_pct: number
  cagr: number
  net_realized_pnl: number
  sharpe_ratio: number
  sharpe_ci: [number, number]
  sortino_ratio: number
  calmar_ratio: number
  profit_factor: number
  win_rate: number
  loss_rate: number
  total_trades: number
  win_count: number
  loss_count: number
  avg_win: number
  avg_loss: number
  payoff_ratio: number
  expectancy: number
  max_consecutive_wins: number
  max_consecutive_losses: number
  max_drawdown_pct: number
  max_drawdown_amount: number
  max_drawdown_duration_hrs: number
  var_95: number
  var_99: number
  cvar_95: number
  position_size: number
  ready_for_live: boolean
  equity_curve: EquityCurvePoint[]
  rolling_metrics: RollingMetricPoint[]
  pnl_histogram: PnlHistogramBin[]
  duration_scatter: DurationScatterPoint[]
}

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  entry_high: number
  exit_low: number
  atr: number
}

export interface PipelineStep {
  key: string
  label: string
  status: 'DONE' | 'PENDING'
  time: string
}

export interface AiInsight {
  time: string
  text: string
  highlight: boolean
}

export interface RiskRadar {
  max_drawdown_pct: string
  var_95_pct: string
  exposure_pct: string
  leverage: string
  radar_scores: {
    drawdown: number
    volatility: number
    liquidity: number
    leverage: number
    exposure: number
  }
}

export interface Asset {
  name: string
  pct: number
  color: string
  value: number
}

export interface EventLogItem {
  time: string
  category: string
  message: string
  level: string
  id?: number
}

export interface GlobeNodes {
  execution: string
  sentiment: string
  risk_engine: string
  trend: string
  momentum: string
  liquidity: string
  volume: string
  news_feed: string
}

export interface CommandCenterData {
  strategy_id: string
  btc_price: number | null
  btc_price_delta_pct: number | null
  eth_price: number | null
  eth_price_delta_pct: number | null
  strategy_health: {
    core_stability_pct: number
    confidence_level_pct: number
    status: string
  }
  market_regime: {
    regime: string
    volatility: string
    liquidity: string
  }
  analytics_summary: {
    current_equity: number
    initial_capital: number
    equity_delta_pct: number
    net_realized_pnl: number
    win_rate: number
    total_trades: number
    sharpe_ratio: number
    profit_factor: number
    max_drawdown_pct: number
    position_size: number
  }
  pipeline_steps: PipelineStep[]
  ai_copilot_insights: AiInsight[]
  risk_radar: RiskRadar
  capital_allocation: { total_equity: number; assets: Asset[] }
  event_feed: EventLogItem[]
  globe_nodes: GlobeNodes
  sparkline_data: {
    btc_prices: number[]
    eth_prices: number[]
    equity_history: number[]
    pnl_cumulative: number[]
    win_rate_rolling: number[]
  }
}

export interface TerrainVertex {
  step: number
  timestamp: string
  time_norm: number
  depth_pct: number
  duration_hrs: number
  equity: number
  peak: number
}

export interface TerrainData {
  strategy_id: string
  terrain_matrix: TerrainVertex[]
  max_depth_pct: number
  max_duration_hrs: number
  sample_count: number
}

export interface WalkForwardFold {
  fold_index: number
  params: Record<string, number>
}

export interface WalkForwardResult {
  strategy_id: string
  timeframe: string
  generated_at: string
  acceptance_check_passed: boolean
  num_folds: number
  total_oos_trades: number
  oos_win_rate_pct: number
  oos_sharpe: number
  oos_total_return_pct: number
  oos_cagr_pct: number
  oos_max_drawdown_pct: number
  oos_calmar_ratio: number
  oos_coverage_start: string
  oos_coverage_end: string
  folds: WalkForwardFold[]
  parameter_stability: Record<string, { is_stable: boolean; distinct_values: number; sequence: number[] }>
}

export interface CiWorkflowRun {
  workflow_name: string
  status: string
  conclusion: string | null
  run_number: number
  html_url: string
  created_at: string
  updated_at: string
  head_branch: string
}

export interface CiStatus {
  repo: string
  workflows: CiWorkflowRun[]
  error?: string
}

export interface PortfolioOverview {
  total_equity: number
  total_trades: number
  strategies: StrategyOverview[]
}

// ---------------------------------------------------------------------------
// Endpoint functions
// ---------------------------------------------------------------------------

export const api = {
  overview: () => getJSON<PortfolioOverview>('/api/v1/overview'),
  strategies: () => getJSON<StrategyOverview[]>('/api/v1/strategies'),
  strategyTrades: (id: string, limit = 100) =>
    getJSON<Trade[]>(`/api/v1/strategies/${id}/trades?limit=${limit}`),
  readiness: () => getJSON<ReadinessGate[]>('/api/v1/readiness'),
  runBot: (strategyId: string) =>
    postJSON<{ status: string; executed_strategy: string; duration_seconds: number }>(
      `/api/v1/bot/run?strategy_id=${strategyId}`
    ),
  analytics: (id: string) => getJSON<StrategyAnalytics>(`/api/v1/analytics/${id}`),
  candles: (strategyId: string, limit = 150) =>
    getJSON<Candle[]>(`/api/v1/candles?strategy_id=${strategyId}&limit=${limit}`),
  equityCurve: (id: string) =>
    getJSON<{ strategy_id: string; equity_curve: EquityCurvePoint[] }>(`/api/v1/equity_curve/${id}`),
  terrain: (id: string) => getJSON<TerrainData>(`/api/v1/terrain/${id}`),
  commandCenter: (id: string) => getJSON<CommandCenterData>(`/api/v1/command_center/${id}`),
  events: (category = 'ALL', limit = 30) =>
    getJSON<EventLogItem[]>(`/api/v1/events?category=${category}&limit=${limit}`),
  walkForward: (id: string) => getJSON<WalkForwardResult>(`/api/v1/walkforward/${id}`),
  ciStatus: () => getJSON<CiStatus>('/api/v1/ci/status'),
}

// Known strategy ids currently onboarded onto this schema (bot is v1 + v4
// today per Strategies table; extend this list, or better, derive it from
// api.strategies() wherever the caller can await it).
export const KNOWN_STRATEGY_IDS = ['v4', 'v1']

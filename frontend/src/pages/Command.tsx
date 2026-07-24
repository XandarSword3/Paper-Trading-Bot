import { useState } from 'react'
import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, Badge, toneForWord, EmptyState, LoadingState, Grid, Span } from '../components/ui/Primitives'
import { GaugeRing, VerticalCapsule, RadarChart, DonutChart, Sparkline } from '../components/charts/index'
import { QuantAlphaGlobe } from '../components/QuantAlphaGlobe'
import {
  Brain, Radio, ShieldCheck, Calculator, Send, CheckCircle2, FolderOpen,
  ShieldAlert, Target, Compass, CircleDot,
} from 'lucide-react'

const EVENT_CATEGORIES = ['ALL', 'EXECUTIONS', 'RISK', 'SYSTEM', 'MARKET'] as const

const EVENT_DOT_COLOR: Record<string, string> = {
  EXECUTION: 'var(--accent-blue)',
  EXECUTIONS: 'var(--accent-blue)',
  RISK: 'var(--accent-amber)',
  SYSTEM: 'var(--accent-emerald)',
  MARKET: 'var(--accent-purple)',
}

// Pipeline steps are backend-driven text labels, not a fixed enum — match by
// keyword so any wording the API sends still gets a sensible icon.
function iconForPipelineStep(label: string) {
  const l = label.toUpperCase()
  if (l.includes('SIGNAL')) return Radio
  if (l.includes('RISK')) return ShieldCheck
  if (l.includes('SIZE')) return Calculator
  if (l.includes('SUBMIT')) return Send
  if (l.includes('FILL')) return CheckCircle2
  if (l.includes('POSITION')) return FolderOpen
  if (l.includes('SL') || l.includes('STOP')) return ShieldAlert
  if (l.includes('TP') || l.includes('PROFIT')) return Target
  if (l.includes('TRAIL')) return Compass
  return CircleDot
}

export default function Command() {
  const { strategyId } = useStrategy()
  const [eventFilter, setEventFilter] = useState<(typeof EVENT_CATEGORIES)[number]>('ALL')

  const { data: commandData, isLoading, error } = useQuery({
    queryKey: ['commandCenter', strategyId],
    queryFn: () => api.commandCenter(strategyId),
    enabled: !!strategyId,
    refetchInterval: 5000,
  })

  if (isLoading) return <LoadingState label="Loading Command Center…" />
  if (error) return <div className="qat-error">Failed to load command center data</div>
  if (!commandData) return <EmptyState label="NO DATA" />

  console.log('Command Center Data:', commandData)

  const { strategy_health, market_regime, analytics_summary, risk_radar, capital_allocation, ai_copilot_insights, globe_nodes, pipeline_steps, event_feed, sparkline_data } = commandData

  return (
    <div className="qat-command">
      <Grid cols={12} gap={16}>
        {/* Core Health Row */}
        <Span n={3}>
          <Card title="Strategy Health" glow>
            <div className="qat-command__gauge-wrapper">
              <GaugeRing
                value={strategy_health.core_stability_pct}
                size={280}
                label="CORE STABILITY"
                status={strategy_health.status}
              />
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Confidence Level" subtitle="Monte Carlo Ruin Survival">
            <div className="qat-command__gauge-wrapper">
              <VerticalCapsule
                value={strategy_health.confidence_level_pct}
                width={90}
                height={220}
                color="var(--accent-emerald)"
                label="MONTE CARLO"
              />
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Market Regime">
            <div className="qat-command__regime-card">
              <div className="qat-command__regime-main">
                <span className="qat-command__regime-value">{market_regime.regime}</span>
                <Badge tone={toneForWord(market_regime.regime)}>{strategy_health.status}</Badge>
              </div>
              <div className="qat-command__regime-stats">
                <div className="qat-command__regime-stat">
                  <span className="qat-command__regime-stat-label">VOLATILITY</span>
                  <span className="qat-command__regime-stat-value">{market_regime.volatility}</span>
                </div>
                <div className="qat-command__regime-stat">
                  <span className="qat-command__regime-stat-label">LIQUIDITY</span>
                  <span className="qat-command__regime-stat-value">{market_regime.liquidity}</span>
                </div>
              </div>
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Capital Allocation">
            <div className="qat-command__donut-row">
              <div className="qat-command__donut-wrapper">
                <DonutChart
                  data={capital_allocation.assets.map(a => ({ name: a.name, pct: a.pct, color: a.color }))}
                  size={120}
                />
                <div className="qat-command__donut-center">
                  <div className="qat-command__donut-label">TOTAL</div>
                  <div className="qat-command__donut-value">${capital_allocation.total_equity.toLocaleString()}</div>
                </div>
              </div>
              <div className="qat-command__donut-legend">
                {capital_allocation.assets.map((a) => (
                  <div key={a.name} className="qat-command__donut-legend-row">
                    <span className="qat-command__donut-legend-dot" style={{ background: a.color }} />
                    <span className="qat-command__donut-legend-name">{a.name}</span>
                    <span className="qat-command__donut-legend-pct">{a.pct.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </Span>

        {/* Quant Alpha Core Globe */}
        <Span n={6}>
          <Card title="Quant Alpha Core" subtitle="Real-time strategy intelligence" glow>
            {globe_nodes ? (
              <QuantAlphaGlobe nodes={globe_nodes} />
            ) : (
              <EmptyState label="NO GLOBE DATA" />
            )}
          </Card>
        </Span>

        {/* Risk Radar */}
        <Span n={3}>
          <Card title="Risk Radar">
            <RadarChart
              axes={[
                { label: 'DRAWDOWN', value: risk_radar.radar_scores.drawdown },
                { label: 'VOLATILITY', value: risk_radar.radar_scores.volatility },
                { label: 'LIQUIDITY', value: risk_radar.radar_scores.liquidity },
                { label: 'LEVERAGE', value: risk_radar.radar_scores.leverage },
                { label: 'EXPOSURE', value: risk_radar.radar_scores.exposure },
              ]}
              size={200}
              color="var(--accent-crimson)"
            />
          </Card>
        </Span>

        {/* AI Copilot */}
        <Span n={3}>
          <Card title="AI Copilot" subtitle="Real-time strategy insights">
            <div className="qat-command__ai-feed">
              {ai_copilot_insights.length === 0 ? (
                <EmptyState label="NO INSIGHTS" />
              ) : (
                ai_copilot_insights.slice(0, 5).map((insight, i) => (
                  <div key={i} className={`qat-command__ai-item ${insight.highlight ? 'is-highlight' : ''}`}>
                    <Brain size={14} className="qat-command__ai-icon" />
                    <div className="qat-command__ai-content">
                      <span className="qat-command__ai-time">{insight.time}</span>
                      <span className="qat-command__ai-text">{insight.text}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>
        </Span>

        {/* Execution Timeline */}
        <Span n={8}>
          <Card title="Execution Pipeline">
            {pipeline_steps.length === 0 ? (
              <EmptyState label="NO PIPELINE DATA" />
            ) : (
              <div className="qat-pipeline">
                {pipeline_steps.map((step) => {
                  const Icon = iconForPipelineStep(step.label)
                  return (
                    <div key={step.key} className={`qat-pipeline__node ${step.status.toLowerCase()}`}>
                      <div className="qat-pipeline__connector" />
                      <div className="qat-pipeline__node-icon">
                        <Icon size={17} />
                      </div>
                      <span className="qat-pipeline__node-label">{step.label}</span>
                      <span className="qat-pipeline__node-time">{step.time}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </Card>
        </Span>

        {/* Live Event Feed */}
        <Span n={4}>
          <Card title="Event Feed">
            <div className="qat-command__event-filters">
              {EVENT_CATEGORIES.map((cat) => (
                <button
                  key={cat}
                  className={`qat-command__event-filter ${eventFilter === cat ? 'is-active' : ''}`}
                  onClick={() => setEventFilter(cat)}
                >
                  {cat}
                </button>
              ))}
            </div>
            <div className="qat-command__events">
              {(() => {
                const filtered = eventFilter === 'ALL'
                  ? event_feed
                  : event_feed.filter((e) => e.category?.toUpperCase().startsWith(eventFilter.slice(0, -1)) || e.category?.toUpperCase() === eventFilter)
                return filtered.length === 0 ? (
                  <EmptyState label="NO EVENTS" />
                ) : (
                  filtered.slice(0, 8).map((event, i) => {
                    const catKey = event.category?.toUpperCase() ?? ''
                    return (
                      <div key={event.id || i} className="qat-command__event-item">
                        <span className="qat-command__event-time">{event.time}</span>
                        <span
                          className="qat-command__event-dot"
                          style={{ background: EVENT_DOT_COLOR[catKey] ?? 'var(--text-muted)' }}
                        />
                        <span
                          className="qat-command__event-category"
                          style={{ color: EVENT_DOT_COLOR[catKey] ?? 'var(--text-dim)' }}
                        >
                          {event.category}
                        </span>
                        <span className="qat-command__event-message">{event.message}</span>
                      </div>
                    )
                  })
                )
              })()}
            </div>
          </Card>
        </Span>

        {/* Footer Ticker */}
        <Span n={12}>
          <Card className="qat-command__ticker-card">
            <div className="qat-command__ticker">
              <div className="qat-command__ticker-item">
                <span className="qat-command__ticker-label">BTC</span>
                <Sparkline values={sparkline_data.btc_prices} width={120} height={30} color="var(--accent-cyan)" />
              </div>
              <div className="qat-command__ticker-item">
                <span className="qat-command__ticker-label">ETH</span>
                <Sparkline values={sparkline_data.eth_prices} width={120} height={30} color="var(--accent-blue)" />
              </div>
              <div className="qat-command__ticker-item">
                <span className="qat-command__ticker-label">EQUITY</span>
                <Sparkline values={sparkline_data.equity_history} width={120} height={30} color="var(--accent-emerald)" />
              </div>
              <div className="qat-command__ticker-item">
                <span className="qat-command__ticker-label">PnL</span>
                <Sparkline values={sparkline_data.pnl_cumulative} width={120} height={30} color="var(--accent-crimson)" />
              </div>
              <div className="qat-command__ticker-item">
                <span className="qat-command__ticker-label">WIN RATE</span>
                <Sparkline values={sparkline_data.win_rate_rolling} width={120} height={30} color="var(--accent-amber)" />
              </div>
            </div>
          </Card>
        </Span>
      </Grid>
    </div>
  )
}

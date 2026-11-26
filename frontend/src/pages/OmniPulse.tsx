import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../lib/api'
import { Card, Badge, LoadingState, Grid, Span } from '../components/ui/Primitives'
import {
  Zap,
  Activity,
  TrendingUp,
  ShieldCheck,
  DollarSign,
  Clock,
  RefreshCw,
  Cpu,
  Radio,
  ArrowUpRight,
  ArrowDownRight,
  Terminal,
  FlaskConical,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Flame,
  Layers,
} from 'lucide-react'

function formatTelemetryTime(timeStr: string): string {
  if (!timeStr) return ''
  if (timeStr.includes('T') || timeStr.endsWith('Z')) {
    const d = new Date(timeStr)
    return isNaN(d.getTime()) ? timeStr : d.toLocaleTimeString([], { hour12: false })
  }
  const match = timeStr.match(/^(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$/)
  if (match) {
    const now = new Date()
    const utcDate = new Date(Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      parseInt(match[1]),
      parseInt(match[2]),
      parseInt(match[3]),
      parseInt(match[4] || '0')
    ))
    return isNaN(utcDate.getTime()) ? timeStr : utcDate.toLocaleTimeString([], { hour12: false })
  }
  return timeStr
}

export default function OmniPulse() {
  const [pulseTick, setPulseTick] = useState(0)
  const [activeTab, setActiveTab] = useState<'research' | 'execution'>('research')

  // 1. Live Paper Trader State Query
  const { data: state, isLoading: stateLoading, refetch: refetchState } = useQuery({
    queryKey: ['omnipulseState'],
    queryFn: api.omnipulseState,
    refetchInterval: 1500,
  })

  // 2. Institutional Signal Research Query
  const { data: research, isLoading: researchLoading, refetch: refetchResearch } = useQuery({
    queryKey: ['omnipulseResearch'],
    queryFn: api.omnipulseResearch,
    refetchInterval: 2000,
  })

  // 3. Pre-Registered Research Variants Query
  const { data: variantsReport, refetch: refetchVariants } = useQuery({
    queryKey: ['omnipulseVariants'],
    queryFn: api.omnipulseVariants,
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (state?.last_updated || research?.last_updated) {
      setPulseTick((prev) => prev + 1)
    }
  }, [state?.last_updated, research?.last_updated])

  if (stateLoading && researchLoading) {
    return <LoadingState label="Connecting to OmniPulse Real-Time Telemetry & Research Stream…" />
  }

  const equity = state?.equity ?? 10000.0
  const cash = state?.cash ?? 10000.0
  const realizedPnl = state?.realized_pnl ?? 0.0
  const feesPaid = state?.total_fees_paid ?? 0.0
  const winRate = state?.win_rate_pct ?? 0.0
  const universeSize = state?.universe_size ?? 122
  const openPositions = state?.open_positions_detail ?? []
  const recentTrades = state?.recent_trades ?? []
  const topMovers = state?.top_movers ?? []
  const telemetryEvents = state?.telemetry_events ?? []

  // Research metrics
  const totalPrimaryEpisodes = research?.total_primary_episodes ?? 0
  const totalSecondaryTicks = research?.total_secondary_ticks ?? 0
  const totalNaiveControls = research?.total_naive_controls ?? 0
  const pendingQueue = research?.pending_queue ?? 0
  const horizons = research?.horizons ?? {}
  const bootstrap5m = research?.bootstrap_5m ?? {}
  const killGate = research?.kill_gate ?? {}
  const doseResponse = research?.dose_response_5m ?? {}
  const dsr = research?.dsr ?? {}
  const trialsRegistered = research?.trials_registered ?? 10

  const killStatus = killGate?.decision ?? 'COLLECTING_SAMPLE'
  const killTone = killStatus === 'PROCEED_TO_MAKER_EXECUTION_ENGINEERING' ? 'good' : killStatus === 'KILL_STRATEGY' ? 'bad' : 'warn'

  return (
    <div className="qat-omnipulse" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Top Header & View Switcher */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <motion.div
                animate={{ rotate: [0, 15, -15, 0], scale: [1, 1.2, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Zap size={26} style={{ color: 'var(--accent-cyan, #00d4ff)' }} />
              </motion.div>
              <h2 style={{ margin: 0, fontSize: '1.45rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
                OmniPulse Quantitative Terminal
              </h2>
              <Badge tone="good">BYBIT V5 WEBSOCKET LIVE</Badge>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', background: 'rgba(0, 255, 136, 0.1)', border: '1px solid rgba(0, 255, 136, 0.3)', borderRadius: 16 }}>
                <span style={{ position: 'relative', display: 'flex', height: 8, width: 8 }}>
                  <motion.span
                    animate={{ scale: [1, 2, 1], opacity: [0.8, 0, 0.8] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
                    style={{ position: 'absolute', display: 'inline-flex', height: '100%', width: '100%', borderRadius: '50%', background: '#00ff88' }}
                  />
                  <span style={{ position: 'relative', display: 'inline-flex', borderRadius: '50%', height: 8, width: 8, background: '#00ff88' }} />
                </span>
                <span style={{ fontSize: '0.74rem', fontWeight: 700, color: '#00ff88', letterSpacing: '0.04em' }}>
                  ACTIVE (Tick #{pulseTick})
                </span>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 12, marginLeft: 4 }}>
                  {[8, 12, 6, 14, 10, 16, 7].map((barH, bi) => (
                    <motion.div
                      key={bi}
                      animate={{ height: [barH * 0.3, barH, barH * 0.2] }}
                      transition={{ duration: 0.6 + (bi * 0.12), repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }}
                      style={{ width: 2.5, background: '#00d4ff', borderRadius: 1 }}
                    />
                  ))}
                </div>
              </div>
            </div>
            <p style={{ margin: '6px 0 0', color: 'var(--text-muted, #8b949e)', fontSize: '0.88rem' }}>
              Streaming top-of-book orderbook.1 & execution tape across {universeSize} linear perpetuals with 5m episode debouncing
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* View Switcher Tabs */}
            <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', borderRadius: 8, padding: 3, border: '1px solid rgba(255,255,255,0.1)' }}>
              <button
                onClick={() => setActiveTab('research')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '7px 14px',
                  borderRadius: 6,
                  border: 'none',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  background: activeTab === 'research' ? 'rgba(0, 212, 255, 0.2)' : 'transparent',
                  color: activeTab === 'research' ? '#00d4ff' : '#8b949e',
                }}
              >
                <FlaskConical size={14} /> 🔬 Signal Research & Kill Gate
              </button>
              <button
                onClick={() => setActiveTab('execution')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '7px 14px',
                  borderRadius: 6,
                  border: 'none',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  background: activeTab === 'execution' ? 'rgba(0, 255, 136, 0.2)' : 'transparent',
                  color: activeTab === 'execution' ? '#00ff88' : '#8b949e',
                }}
              >
                <Cpu size={14} /> ⚡ Paper Trader & Execution
              </button>
            </div>

            <button
              onClick={() => {
                refetchState()
                refetchResearch()
                refetchVariants()
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.15)',
                color: '#fff',
                padding: '8px 14px',
                borderRadius: 6,
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>
      </Card>

      {/* ========================================================================= */}
      {/* TAB 1: INSTITUTIONAL SIGNAL RESEARCH & KILL GATE                           */}
      {/* ========================================================================= */}
      {activeTab === 'research' && (
        <>
          {/* Pre-Registered Stopping Rule Banner */}
          <Card>
            <div style={{ borderLeft: '4px solid #00d4ff', background: 'rgba(0, 212, 255, 0.04)', padding: 14, borderRadius: 8 }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div style={{ padding: 6, borderRadius: 8, background: 'rgba(0, 212, 255, 0.1)', color: '#00d4ff' }}>
                  <ShieldCheck size={22} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                    <span style={{ fontWeight: 700, fontSize: '0.94rem', color: '#fff' }}>
                      Institutional Pre-Registered Stopping Rule: Target N ≥ 300 Primary Episodes at 5m
                    </span>
                    <Badge tone="info">SAMPLE PROGRESS: {totalPrimaryEpisodes} / 300 EPISODES</Badge>
                  </div>
                  <p style={{ margin: '6px 0 0', fontSize: '0.82rem', color: '#8b949e', lineHeight: 1.5 }}>
                    Per institutional peer review: interim peeking swamps true means with noise. The final 100 episodes (last 33%) are strictly reserved as an untouched confirmation holdout set.
                    Alpha clearance requires: (1) 95% Bootstrap CI Lower Bound &gt; 0.0 bps Net at 5m, (2) Positive Median, (3) Sign Agreement across 1m, 5m, and 15m, and (4) Monotonic Dose-Response across Z-score buckets.
                  </p>
                  <div style={{ marginTop: 10, height: 6, width: '100%', background: 'rgba(255,255,255,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, (totalPrimaryEpisodes / 300) * 100)}%` }}
                      style={{ height: '100%', background: 'linear-gradient(90deg, #00d4ff, #00ff88)', borderRadius: 3 }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Institutional Kill Criterion Hero Banner */}
          <Card>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 20px',
                borderRadius: 8,
                background:
                  killTone === 'good'
                    ? 'rgba(0, 255, 136, 0.08)'
                    : killTone === 'bad'
                    ? 'rgba(255, 68, 68, 0.08)'
                    : 'rgba(255, 170, 0, 0.08)',
                border: `1px solid ${
                  killTone === 'good'
                    ? 'rgba(0, 255, 136, 0.3)'
                    : killTone === 'bad'
                    ? 'rgba(255, 68, 68, 0.3)'
                    : 'rgba(255, 170, 0, 0.3)'
                }`,
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <ShieldCheck size={22} style={{ color: killTone === 'good' ? '#00ff88' : killTone === 'bad' ? '#ff4444' : '#ffaa00' }} />
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>
                    Institutional Kill Criterion Gate (Pre-Registered 5m Horizon)
                  </span>
                  <Badge tone={killTone}>{killStatus.replace(/_/g, ' ')}</Badge>
                </div>
                <p style={{ margin: '6px 0 0', color: 'var(--text-muted, #8b949e)', fontSize: '0.86rem' }}>
                  Rule: The <strong>Lower Bound of the 95% Moving Block Bootstrap CI on 5m</strong> must clear <strong>&gt; 0.0 bps Net</strong> after crossing Ask/Bid and deducting 11 bps taker fee.
                </p>
                <div style={{ marginTop: 8, fontSize: '0.84rem', color: '#e6edf3', fontWeight: 600 }}>
                  {killGate?.statement ?? 'Collecting primary episodes for temporal block bootstrap verification…'}
                </div>
              </div>

              <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>5m Primary Net Drift</span>
                <span
                  style={{
                    fontSize: '1.6rem',
                    fontWeight: 800,
                    color: (bootstrap5m?.net_mean_bps ?? 0) >= 0 ? '#00ff88' : '#ff4444',
                  }}
                >
                  {bootstrap5m?.net_mean_bps !== undefined ? `${bootstrap5m.net_mean_bps > 0 ? '+' : ''}${bootstrap5m.net_mean_bps.toFixed(2)} bps` : 'Pending'}
                </span>
                <span style={{ fontSize: '0.78rem', color: '#8b949e' }}>
                  95% CI:{' '}
                  {bootstrap5m?.net_95_ci_bps
                    ? `[${bootstrap5m.net_95_ci_bps[0] > 0 ? '+' : ''}${bootstrap5m.net_95_ci_bps[0].toFixed(2)}, ${bootstrap5m.net_95_ci_bps[1] > 0 ? '+' : ''}${bootstrap5m.net_95_ci_bps[1].toFixed(2)}] bps`
                    : 'Awaiting Blocks (< 50)'}
                </span>
              </div>
            </div>

            {/* Institutional 5-Criterion Scorecard */}
            {killGate?.criteria && (
              <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 14 }}>
                <div style={{ fontSize: '0.84rem', fontWeight: 700, color: '#fff', marginBottom: 10 }}>
                  Institutional Alpha Verification Criteria Scorecard
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                  <thead>
                    <tr style={{ color: '#8b949e', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px' }}>Criterion</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px' }}>Pre-Registered Threshold</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px' }}>Empirical Value</th>
                      <th style={{ textAlign: 'right', padding: '6px 8px' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(killGate.criteria).map(([k, c]: [string, any]) => (
                      <tr key={k} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '8px', fontWeight: 600, color: '#e6edf3' }}>{c.name}</td>
                        <td style={{ padding: '8px', color: '#8b949e' }}>{c.threshold}</td>
                        <td style={{ padding: '8px', fontFamily: 'monospace', color: c.passed ? '#00ff88' : '#ff7b72' }}>{c.value}</td>
                        <td style={{ padding: '8px', textAlign: 'right' }}>
                          <Badge tone={c.passed ? 'good' : 'bad'}>{c.passed ? 'PASS' : 'FAILING'}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Research Architecture Metrics Grid */}
          <Grid cols={12}>
            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Primary Episodes</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#00d4ff', marginTop: 4 }}>
                  {totalPrimaryEpisodes}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  1st signal per coin after ≥ 5m quiet
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Debounced Intra-Burst Ticks</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#a371f7', marginTop: 4 }}>
                  {totalSecondaryTicks}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  Filtered out of primary statistical tests
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Matched Naive Controls</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#ffaa00', marginTop: 4 }}>
                  {totalNaiveControls}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  Same-coin trend without Z/surge filter
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>In-Flight Evaluation Queue</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#00ff88', marginTop: 4 }}>
                  {pendingQueue}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  Persisted to disk across restarts
                </div>
              </Card>
            </Span>
          </Grid>

          {/* Realistic Multi-Horizon Forward Net Returns Table */}
          <Card title="Primary Episodes: Realistic Net Returns (Entry Ask/Bid + Latency + 11.0 bps VIP0 Taker Fee)">
            <p style={{ margin: '0 0 16px', color: '#8b949e', fontSize: '0.84rem' }}>
              All returns measured from entry fill at <strong>t + latency</strong> at the <strong>Ask</strong> (Long) or <strong>Bid</strong> (Short), exiting at opposite quote and deducting round-trip fees.
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.12)', color: '#8b949e' }}>
                    <th style={{ padding: '8px' }}>Horizon</th>
                    <th style={{ padding: '8px' }}>Episodes (N)</th>
                    <th style={{ padding: '8px' }}>Net Mean (bps)</th>
                    <th style={{ padding: '8px' }}>5% Trimmed Mean</th>
                    <th style={{ padding: '8px' }}>Median Net</th>
                    <th style={{ padding: '8px' }}>95% Bootstrap CI</th>
                    <th style={{ padding: '8px' }}>Win Rate (%)</th>
                    <th style={{ padding: '8px' }}>Fade / Reversal Mean</th>
                    <th style={{ padding: '8px' }}>Naive Control Mean</th>
                    <th style={{ padding: '8px' }}>Excess over Naive</th>
                  </tr>
                </thead>
                <tbody>
                  {['10s', '30s', '1m', '5m', '15m'].map((h) => {
                    const stat = horizons[h]
                    const hasData = stat && typeof stat === 'object' && stat.count > 0
                    const isPrim = h === '5m'
                    const netMean = stat?.mean_net_bps ?? 0
                    const excess = stat?.excess_over_naive_bps ?? 0

                    return (
                      <tr
                        key={h}
                        style={{
                          borderBottom: '1px solid rgba(255,255,255,0.06)',
                          background: isPrim ? 'rgba(0, 212, 255, 0.05)' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '10px 8px', fontWeight: 700, color: isPrim ? '#00d4ff' : '#fff' }}>
                          {h} {isPrim && <Badge tone="info">PRIMARY</Badge>}
                        </td>
                        <td style={{ padding: '10px 8px' }}>{hasData ? stat.count : '0 (in queue)'}</td>
                        <td
                          style={{
                            padding: '10px 8px',
                            fontWeight: 700,
                            color: hasData ? (netMean >= 0 ? '#00ff88' : '#ff4444') : '#8b949e',
                          }}
                        >
                          {hasData ? `${netMean > 0 ? '+' : ''}${netMean.toFixed(2)} bps` : 'Pending'}
                        </td>
                        <td style={{ padding: '10px 8px', color: '#e6edf3' }}>
                          {hasData ? `${stat.trimmed_mean_net_bps > 0 ? '+' : ''}${stat.trimmed_mean_net_bps.toFixed(2)} bps` : '—'}
                        </td>
                        <td style={{ padding: '10px 8px', color: '#e6edf3' }}>
                          {hasData ? `${stat.median_net_bps > 0 ? '+' : ''}${stat.median_net_bps.toFixed(2)} bps` : '—'}
                        </td>
                        <td style={{ padding: '10px 8px', fontFamily: 'monospace', fontSize: '0.8rem', color: '#00d4ff' }}>
                          {hasData && stat.bootstrap_ci_95 ? `[${stat.bootstrap_ci_95[0]} to ${stat.bootstrap_ci_95[1]}] bps` : 'Pending (Blocks<50)'}
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          {hasData ? (
                            <Badge tone={stat.win_rate_pct >= 50 ? 'good' : 'warn'}>{stat.win_rate_pct.toFixed(1)}%</Badge>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td style={{ padding: '10px 8px', color: (stat?.fade_mean_net_bps ?? 0) >= 0 ? '#00ff88' : '#8b949e' }}>
                          {hasData ? `${stat.fade_mean_net_bps > 0 ? '+' : ''}${stat.fade_mean_net_bps.toFixed(2)} bps` : '—'}
                        </td>
                        <td style={{ padding: '10px 8px', color: '#8b949e' }}>
                          {hasData ? `${stat.naive_control_mean_net_bps > 0 ? '+' : ''}${stat.naive_control_mean_net_bps.toFixed(2)} bps` : '—'}
                        </td>
                        <td
                          style={{
                            padding: '10px 8px',
                            fontWeight: 700,
                            color: hasData ? (excess >= 0 ? '#00ff88' : '#ffaa00') : '#8b949e',
                          }}
                        >
                          {hasData ? `${excess > 0 ? '+' : ''}${excess.toFixed(2)} bps` : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Dose-Response Monotonicity & Calibrated DSR Grid */}
          <Grid cols={12}>
            {/* Dose-Response Curve */}
            <Span n={6}>
              <Card title="Dose-Response Monotonicity Curve (5m Horizon)">
                <p style={{ margin: '0 0 14px', color: '#8b949e', fontSize: '0.82rem' }}>
                  Tests whether net forward return scales monotonically with signal conviction ($Z$-score floor).
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {['1.2-1.8', '1.8-2.2', '2.2-2.8', '2.8-3.5', '3.5+'].map((bucket) => {
                    const bData = doseResponse[bucket]
                    const bCount = bData?.count ?? 0
                    const bMean = bData?.mean_net_bps ?? 0
                    const bWr = bData?.win_rate_pct ?? 0

                    return (
                      <div
                        key={bucket}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '8px 12px',
                          background: 'rgba(255,255,255,0.03)',
                          borderRadius: 6,
                          border: '1px solid rgba(255,255,255,0.06)',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ fontWeight: 700, color: '#00d4ff', minWidth: 65 }}>Z {bucket}</span>
                          <span style={{ fontSize: '0.8rem', color: '#8b949e' }}>N = {bCount} episodes</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                          {bCount > 0 ? (
                            <>
                              <Badge tone={bWr >= 50 ? 'good' : 'warn'}>{bWr.toFixed(1)}% WR</Badge>
                              <span style={{ fontWeight: 700, color: bMean >= 0 ? '#00ff88' : '#ff4444' }}>
                                {bMean > 0 ? '+' : ''}{bMean.toFixed(2)} bps
                              </span>
                            </>
                          ) : (
                            <span style={{ fontSize: '0.78rem', color: '#6e7681' }}>Collecting…</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Card>
            </Span>

            {/* Calibrated DSR Audit */}
            <Span n={6}>
              <Card title="Calibrated Deflated Sharpe Ratio (DSR) Multi-Trial Audit">
                <p style={{ margin: '0 0 14px', color: '#8b949e', fontSize: '0.82rem' }}>
                  Accounts for all historical strategy iterations, parameter grids, and 5 horizons tested.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Lifetime Tested Configurations (N_trials):</span>
                    <span style={{ fontWeight: 700, color: '#fff' }}>{trialsRegistered}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Effective Independent Events (T):</span>
                    <span style={{ fontWeight: 700, color: '#00d4ff' }}>{totalPrimaryEpisodes} Episodes</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Per-Period Net Sharpe (unannualized):</span>
                    <span style={{ fontWeight: 700, color: '#fff' }}>
                      {dsr?.observed_sharpe !== undefined ? dsr.observed_sharpe : 'Awaiting N>=15'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Multi-Trial Deflation Hurdle (SR*):</span>
                    <span style={{ fontWeight: 700, color: '#ffaa00' }}>
                      {dsr?.sr_star !== undefined ? dsr.sr_star : '—'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0' }}>
                    <span style={{ color: '#8b949e' }}>DSR Statistical Confidence:</span>
                    <span>
                      {dsr?.dsr_prob_pct !== undefined ? (
                        <Badge tone={dsr.passed ? 'good' : 'warn'}>{dsr.dsr_prob_pct.toFixed(1)}%</Badge>
                      ) : (
                        <span style={{ color: '#8b949e' }}>Collecting baseline</span>
                      )}
                    </span>
                  </div>
                </div>
              </Card>
            </Span>
          </Grid>

          {/* ========================================================================= */}
          {/* PRE-REGISTERED RESEARCH VARIANTS & 70/30 TIME-SPLIT SCORECARD             */}
          {/* ========================================================================= */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Layers size={18} color="#00d4ff" /> Pre-Registered Research Variants & Offline Replay Scorecard
                </div>
                <div style={{ fontSize: '0.82rem', color: '#8b949e', marginTop: 3 }}>
                  Strict non-overlapping time split: 70% Discovery Partition ({variantsReport?.discovery_episodes ?? 0} Episodes) vs. 30% Locked Holdout Partition ({variantsReport?.holdout_episodes ?? 0} Episodes). Penalized under Deflated Sharpe Ratio (trial budget N=5).
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge tone="info">TOTAL: {variantsReport?.total_episodes_evaluated ?? 0} EPISODES</Badge>
                <Badge tone="warn">TRIAL BUDGET: 5 VARIANTS</Badge>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.12)', color: '#8b949e', textAlign: 'left' }}>
                    <th style={{ padding: '8px' }}>Variant ID & Name</th>
                    <th style={{ padding: '8px' }}>Core Hypothesis & Filter Rule</th>
                    <th style={{ padding: '8px' }}>Discovery Net (Med)</th>
                    <th style={{ padding: '8px' }}>Holdout Net (Med)</th>
                    <th style={{ padding: '8px' }}>Holdout Win%</th>
                    <th style={{ padding: '8px' }}>LOCO (Ex-Top Coin)</th>
                    <th style={{ padding: '8px' }}>DSR Confidence</th>
                    <th style={{ padding: '8px' }}>Holdout Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {(variantsReport?.variants ?? []).map((v: any) => {
                    const isV5 = v.variant_id === 'V5_FROZEN_HOLDOUT'
                    const holdMean = v.holdout_mean_net_bps ?? 0
                    const holdMed = v.holdout_median_net_bps ?? 0
                    const discMean = v.discovery_mean_net_bps ?? 0
                    const discMed = v.discovery_median_net_bps ?? 0
                    const loco = v.loco_ex_top_coin_mean_net_bps ?? 0
                    const dsrConf = v.dsr_confidence_pct ?? 0

                    return (
                      <tr
                        key={v.variant_id}
                        style={{
                          borderBottom: '1px solid rgba(255,255,255,0.06)',
                          background: isV5 ? 'rgba(255, 68, 68, 0.08)' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '10px 8px', fontWeight: 700, color: isV5 ? '#ff4444' : '#00d4ff' }}>
                          <div>{v.variant_id}</div>
                          <div style={{ fontSize: '0.74rem', color: '#8b949e', fontWeight: 400 }}>{v.name}</div>
                        </td>
                        <td style={{ padding: '10px 8px', color: '#8b949e', maxWidth: 280 }}>
                          {v.hypothesis}
                        </td>
                        <td style={{ padding: '10px 8px', color: '#e6edf3' }}>
                          <span style={{ fontWeight: 600, color: discMean >= 0 ? '#00ff88' : '#ff4444' }}>
                            {discMean > 0 ? '+' : ''}{discMean.toFixed(1)} bps
                          </span>
                          <span style={{ fontSize: '0.75rem', color: '#8b949e', marginLeft: 4 }}>
                            ({discMed > 0 ? '+' : ''}{discMed.toFixed(1)})
                          </span>
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          <span style={{ fontWeight: 700, color: holdMean >= 0 ? '#00ff88' : '#ff4444' }}>
                            {holdMean > 0 ? '+' : ''}{holdMean.toFixed(1)} bps
                          </span>
                          <span style={{ fontSize: '0.75rem', color: '#8b949e', marginLeft: 4 }}>
                            ({holdMed > 0 ? '+' : ''}{holdMed.toFixed(1)})
                          </span>
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          <Badge tone={v.holdout_win_rate_pct >= 50 ? 'good' : 'warn'}>
                            {v.holdout_win_rate_pct?.toFixed(1)}%
                          </Badge>
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          <span style={{ fontWeight: 700, color: loco >= 0 ? '#00ff88' : '#ff4444' }}>
                            {loco > 0 ? '+' : ''}{loco.toFixed(1)} bps
                          </span>
                          <div style={{ fontSize: '0.7rem', color: '#8b949e' }}>ex-{v.top_coin}</div>
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          <Badge tone={dsrConf >= 95 ? 'good' : 'warn'}>{dsrConf.toFixed(1)}%</Badge>
                        </td>
                        <td style={{ padding: '10px 8px' }}>
                          <Badge tone={isV5 ? 'bad' : 'warn'}>
                            {isV5 ? 'FALSIFIED' : v.institutional_gate_decision}
                          </Badge>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Maker Adverse Selection & Holdout Verdict Audit Grid */}
          <Grid cols={12}>
            {/* Maker Fill & Adverse Selection (Path B) */}
            <Span n={6}>
              <Card title="Variant 1: Maker Execution & Adverse Selection Audit (Path B)">
                <p style={{ margin: '0 0 12px', color: '#8b949e', fontSize: '0.82rem' }}>
                  Tests whether post-only limit execution (4.0 bps fee) captures gross drift or if adverse selection eliminates the edge.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Passive Limit Order Fill Rate:</span>
                    <span style={{ fontWeight: 700, color: '#ffaa00' }}>10.5% (89.5% unfill/cancelled)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Unconditional Gross Drift:</span>
                    <span style={{ fontWeight: 700, color: '#00d4ff' }}>+59.85 bps</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Fill-Conditioned Gross Drift:</span>
                    <span style={{ fontWeight: 700, color: '#00ff88' }}>+12.19 bps</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Empirical Adverse Selection Drag:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>-47.66 bps</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0' }}>
                    <span style={{ color: '#8b949e' }}>Net Advantage vs Taker:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>-40.66 bps (Drag exceeds 7 bps fee savings)</span>
                  </div>
                </div>
                <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,68,68,0.08)', borderRadius: 6, border: '1px solid rgba(255,68,68,0.2)', fontSize: '0.8rem', color: '#e6edf3', lineHeight: 1.5 }}>
                  <strong>Definitive Finding:</strong> Passive limit orders fill selectively when the market is reversing through the bid. The 47.7 bps adverse selection drag is roughly 7x larger than the 7.0 bps fee discount gained by trading maker.
                </div>
              </Card>
            </Span>

            {/* Holdout Verdict & Scientific Conclusion */}
            <Span n={6}>
              <Card title="Variant 5: Untouched 30% Holdout Evaluation">
                <p style={{ margin: '0 0 12px', color: '#8b949e', fontSize: '0.82rem' }}>
                  Pre-registered composite hypothesis (Clean Universe + Cost Gating + 5m Hold) tested once on virgin data.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Holdout Sample Size:</span>
                    <span style={{ fontWeight: 700, color: '#fff' }}>64 primary episodes</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Holdout Mean Net Return:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>-51.15 bps</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Holdout Median Net Return:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>-36.69 bps</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#8b949e' }}>Holdout Win Rate:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>12.5%</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0' }}>
                    <span style={{ color: '#8b949e' }}>DSR Multi-Trial Confidence:</span>
                    <span style={{ fontWeight: 700, color: '#ff4444' }}>26.7% (Required: &gt;= 95%)</span>
                  </div>
                </div>
                <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,170,0,0.08)', borderRadius: 6, border: '1px solid rgba(255,170,0,0.2)', fontSize: '0.8rem', color: '#e6edf3', lineHeight: 1.5 }}>
                  <strong>Scientific Conclusion:</strong> The strategy fails all 5 pre-registered institutional criteria on virgin holdout data. Pure momentum breakouts cannot beat Bybit VIP0 taker fees or overcome adverse selection under passive execution.
                </div>
              </Card>
            </Span>
          </Grid>
        </>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: LIVE PAPER TRADER & MICROSTRUCTURE EXECUTION                        */}
      {/* ========================================================================= */}
      {activeTab === 'execution' && (
        <>
          {/* Dynamic Data Flow Pipeline Visualizer */}
          <Card>
            <div
              style={{
                padding: '16px 20px',
                background: 'rgba(0, 0, 0, 0.35)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
              }}
            >
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted, #8b949e)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                REAL-TIME DATA PROCESSING PIPELINE
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}>
                {/* Stage 1: Ingestion */}
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(0, 212, 255, 0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#00d4ff', fontSize: '0.82rem', fontWeight: 600 }}>
                    <Radio size={14} /> 1. Ingestion Stream
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                    {universeSize} Coins / Tick
                  </div>
                </div>

                {/* Animated Flow Arrow 1 */}
                <div style={{ padding: '0 12px' }}>
                  <motion.div animate={{ x: [0, 8, 0], opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.2, repeat: Infinity }}>
                    <ArrowUpRight size={18} style={{ color: '#00d4ff', transform: 'rotate(45deg)' }} />
                  </motion.div>
                </div>

                {/* Stage 2: Scanning & Z-Score Filtering */}
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(163, 113, 247, 0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#a371f7', fontSize: '0.82rem', fontWeight: 600 }}>
                    <Activity size={14} /> 2. Statistical Anomaly Scan
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                    Z &gt; 1.2σ &amp; Vol &gt; 1.5x
                  </div>
                </div>

                {/* Animated Flow Arrow 2 */}
                <div style={{ padding: '0 12px' }}>
                  <motion.div animate={{ x: [0, 8, 0], opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.2, repeat: Infinity, delay: 0.3 }}>
                    <ArrowUpRight size={18} style={{ color: '#a371f7', transform: 'rotate(45deg)' }} />
                  </motion.div>
                </div>

                {/* Stage 3: Microstructure Filter */}
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(255, 170, 0, 0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#ffaa00', fontSize: '0.82rem', fontWeight: 600 }}>
                    <ShieldCheck size={14} /> 3. Microstructure Gate
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                    Spread &le; 12 bps + BTC Guard
                  </div>
                </div>

                {/* Animated Flow Arrow 3 */}
                <div style={{ padding: '0 12px' }}>
                  <motion.div animate={{ x: [0, 8, 0], opacity: [0.4, 1, 0.4] }} transition={{ duration: 1.2, repeat: Infinity, delay: 0.6 }}>
                    <ArrowUpRight size={18} style={{ color: '#ffaa00', transform: 'rotate(45deg)' }} />
                  </motion.div>
                </div>

                {/* Stage 4: Order Sizing & Ratchet */}
                <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(0, 255, 136, 0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#00ff88', fontSize: '0.82rem', fontWeight: 600 }}>
                    <Zap size={14} /> 4. Trailing Ratchet Execution
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                    +1.0% Ratchet / -1.5% Stop
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Financial Overview Metrics */}
          <Grid cols={12}>
            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Portfolio Equity</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#fff', marginTop: 4 }}>
                  ${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  Initial: $10,000.00
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Realized PnL</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: realizedPnl >= 0 ? '#00ff88' : '#ff4444', marginTop: 4 }}>
                  {realizedPnl >= 0 ? '+' : ''}${realizedPnl.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  Win Rate: {winRate.toFixed(1)}% ({recentTrades.length} closed)
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Total Fees Paid</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#ffaa00', marginTop: 4 }}>
                  ${feesPaid.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  11.0 bps VIP0 taker + real measured spread
                </div>
              </Card>
            </Span>

            <Span n={3}>
              <Card>
                <div style={{ fontSize: '0.78rem', color: '#8b949e', textTransform: 'uppercase' }}>Open Positions</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#00d4ff', marginTop: 4 }}>
                  {openPositions.length} / 5
                </div>
                <div style={{ fontSize: '0.76rem', color: '#8b949e', marginTop: 4 }}>
                  ${(openPositions.length * 2000).toLocaleString()} Active Notional
                </div>
              </Card>
            </Span>
          </Grid>

          {/* Active Positions */}
          <Card title={`Active Open Positions (${openPositions.length})`}>
            {openPositions.length === 0 ? (
              <div style={{ padding: '40px 0', textAlign: 'center', color: '#8b949e' }}>
                No active positions open. Scanner searching for high-velocity surge breakouts…
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#8b949e' }}>
                      <th style={{ padding: '8px' }}>Symbol</th>
                      <th style={{ padding: '8px' }}>Side</th>
                      <th style={{ padding: '8px' }}>Entry Price</th>
                      <th style={{ padding: '8px' }}>Current Price</th>
                      <th style={{ padding: '8px' }}>Unrealized PnL</th>
                      <th style={{ padding: '8px' }}>Stop Loss</th>
                      <th style={{ padding: '8px' }}>Trailing Ratchet</th>
                    </tr>
                  </thead>
                  <tbody>
                    {openPositions.map((pos: any) => {
                      const isProfit = (pos.unrealized_pnl ?? 0) >= 0
                      return (
                        <tr key={pos.symbol} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                          <td style={{ padding: '10px 8px', fontWeight: 700 }}>{pos.symbol}</td>
                          <td style={{ padding: '10px 8px' }}>
                            <Badge tone={pos.side === 'Buy' ? 'good' : 'bad'}>{pos.side}</Badge>
                          </td>
                          <td style={{ padding: '10px 8px' }}>${pos.entry_price?.toFixed(4)}</td>
                          <td style={{ padding: '10px 8px' }}>${pos.current_price?.toFixed(4)}</td>
                          <td style={{ padding: '10px 8px', fontWeight: 700, color: isProfit ? '#00ff88' : '#ff4444' }}>
                            {isProfit ? '+' : ''}${pos.unrealized_pnl?.toFixed(2)}
                          </td>
                          <td style={{ padding: '10px 8px', color: '#ff4444' }}>${pos.stop_loss?.toFixed(4)}</td>
                          <td style={{ padding: '10px 8px' }}>
                            {pos.trailing_active ? (
                              <Badge tone="good">TRAILING LOCKED (+0.5%)</Badge>
                            ) : (
                              <span style={{ color: '#8b949e' }}>Pending +1.0% Profit</span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Live Market Velocity Radar (Top 20 Key Movers Analyzed) */}
          <Card title={`Live Market Velocity Radar (${topMovers.length > 0 ? topMovers.length : 20} Key Movers Analyzed)`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div style={{ fontSize: '0.84rem', color: '#8b949e' }}>
                Real-time rolling 60-second momentum velocities and volume bursts refreshed every sub-second cycle:
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.76rem', color: '#00ff88', fontWeight: 600 }}>
                <span style={{ position: 'relative', display: 'flex', height: 8, width: 8 }}>
                  <motion.span
                    animate={{ scale: [1, 2, 1], opacity: [0.8, 0, 0.8] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
                    style={{ position: 'absolute', display: 'inline-flex', height: '100%', width: '100%', borderRadius: '50%', background: '#00ff88' }}
                  />
                  <span style={{ position: 'relative', display: 'inline-flex', borderRadius: '50%', height: 8, width: 8, background: '#00ff88' }} />
                </span>
                <span>BYBIT V5 LIVE RADAR SCANNER</span>
              </div>
            </div>

            {topMovers.length === 0 ? (
              <div style={{ padding: '30px 0', textAlign: 'center', color: '#8b949e' }}>
                Scanning live Bybit linear contract feed for top velocity movers…
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
                {topMovers.slice(0, 20).map((m: any, idx: number) => {
                  const vel = m.velocity_pct ?? 0.0
                  const isPump = vel >= 0.15
                  const isDump = vel <= -0.15
                  const borderCol = isPump ? 'rgba(0, 255, 136, 0.4)' : isDump ? 'rgba(255, 68, 68, 0.4)' : 'rgba(255, 255, 255, 0.08)'
                  const bgGlow = isPump ? 'rgba(0, 255, 136, 0.06)' : isDump ? 'rgba(255, 68, 68, 0.06)' : 'rgba(255, 255, 255, 0.02)'

                  return (
                    <motion.div
                      key={m.symbol || idx}
                      initial={{ opacity: 0, scale: 0.96 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.2 }}
                      style={{
                        padding: '10px 12px',
                        borderRadius: 8,
                        background: bgGlow,
                        border: `1px solid ${borderCol}`,
                        transition: 'all 0.3s ease',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fff' }}>{m.symbol}</span>
                        <span
                          style={{
                            fontSize: '0.78rem',
                            fontWeight: 700,
                            color: isPump ? '#00ff88' : isDump ? '#ff4444' : '#8b949e',
                          }}
                        >
                          {vel >= 0 ? '+' : ''}{vel.toFixed(2)}%
                        </span>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: '#8b949e', marginTop: 6 }}>
                        <span>${m.last_price?.toFixed(4)}</span>
                        <span>Vol: {m.turnover_surge?.toFixed(1)}x</span>
                        <span>{m.spread_bps?.toFixed(1)} bps</span>
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            )}
          </Card>

          {/* Telemetry Log & Closed Trades */}
          <Grid cols={12}>
            <Span n={6}>
              <Card title="Live Microstructure Event Tape">
                <div
                  style={{
                    maxHeight: 230,
                    overflowY: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.78rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    padding: '8px 12px',
                    background: '#0d1117',
                    borderRadius: 6,
                  }}
                >
                  {telemetryEvents.length === 0 ? (
                    <div style={{ color: '#8b949e' }}>Streaming live ticker events…</div>
                  ) : (
                    telemetryEvents.map((ev: any, idx: number) => (
                      <div key={idx} style={{ display: 'flex', gap: 8 }}>
                        <span style={{ color: '#8b949e' }}>[{formatTelemetryTime(ev.time)}]</span>
                        <span style={{ color: ev.type === 'ENTRY' ? '#00d4ff' : '#00ff88', fontWeight: 600 }}>{ev.type}:</span>
                        <span style={{ color: '#e6edf3' }}>{ev.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </Card>
            </Span>

            <Span n={6}>
              <Card title={`Recent Closed Trades (${recentTrades.length})`}>
                {recentTrades.length === 0 ? (
                  <div style={{ padding: '40px 0', textAlign: 'center', color: '#8b949e' }}>
                    No completed trades yet in this session.
                  </div>
                ) : (
                  <div style={{ maxHeight: 230, overflowY: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.84rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#8b949e' }}>
                          <th style={{ padding: '6px' }}>Symbol</th>
                          <th style={{ padding: '6px' }}>Side</th>
                          <th style={{ padding: '6px' }}>Net PnL</th>
                          <th style={{ padding: '6px' }}>Exit Reason</th>
                          <th style={{ padding: '6px' }}>Duration</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentTrades.slice(-5).reverse().map((t: any, idx: number) => {
                          const isProfit = (t.net_pnl ?? 0) >= 0
                          const durMin = ((t.duration_seconds ?? 0) / 60).toFixed(1)
                          return (
                            <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                              <td style={{ padding: '8px 6px', fontWeight: 700 }}>{t.symbol}</td>
                              <td style={{ padding: '8px 6px' }}>
                                <Badge tone={t.side === 'Buy' ? 'good' : 'bad'}>{t.side}</Badge>
                              </td>
                              <td
                                style={{
                                  padding: '8px 6px',
                                  fontWeight: 700,
                                  color: isProfit ? '#00ff88' : '#ff4444',
                                }}
                              >
                                {isProfit ? '+' : ''}${t.net_pnl?.toFixed(2)}
                              </td>
                              <td style={{ padding: '8px 6px' }}>
                                <span style={{ color: isProfit ? '#7ee787' : '#ffa657' }}>{t.exit_reason}</span>
                              </td>
                              <td style={{ padding: '8px 6px', color: '#8b949e' }}>{durMin}m</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            </Span>
          </Grid>
        </>
      )}
    </div>
  )
}

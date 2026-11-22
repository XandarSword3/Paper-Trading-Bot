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
} from 'lucide-react'

export default function OmniPulse() {
  const [pulseTick, setPulseTick] = useState(0)

  const { data: state, isLoading, refetch } = useQuery({
    queryKey: ['omnipulseState'],
    queryFn: api.omnipulseState,
    refetchInterval: 1500, // 1.5s refresh for fluid live feel
  })

  // Trigger pulse animation whenever data updates
  useEffect(() => {
    if (state?.last_updated) {
      setPulseTick((prev) => prev + 1)
    }
  }, [state?.last_updated])

  if (isLoading) return <LoadingState label="Connecting to OmniPulse Real-Time Telemetry…" />

  const equity = state?.equity ?? 10000.0
  const cash = state?.cash ?? 10000.0
  const realizedPnl = state?.realized_pnl ?? 0.0
  const feesPaid = state?.total_fees_paid ?? 0.0
  const universeSize = state?.universe_size ?? 124
  const openPositions = state?.open_positions_detail ?? []
  const recentTrades = state?.recent_trades ?? []
  const topMovers = state?.top_movers ?? []
  const telemetryEvents = state?.telemetry_events ?? []
  const winRate = state?.win_rate_pct ?? 0.0
  const signalsCount = state?.signals_detected ?? 0

  return (
    <div className="qat-omnipulse" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* 1. Header Banner & Live Pipeline Pulse */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <motion.div
                animate={{ rotate: [0, 15, -15, 0], scale: [1, 1.2, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Zap size={26} style={{ color: 'var(--accent-cyan, #00d4ff)' }} />
              </motion.div>
              <h2 style={{ margin: 0, fontSize: '1.45rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
                OmniPulse Real-Time Multi-Asset Scanner & Trader
              </h2>
              <Badge tone="good">LIVE TELEMETRY STREAM</Badge>
            </div>
            <p style={{ margin: '6px 0 0', color: 'var(--text-muted, #8b949e)', fontSize: '0.88rem' }}>
              Streaming live order flow & cross-sectional velocity anomalies across {universeSize} Bybit linear perpetuals
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ textAlign: 'right', fontSize: '0.82rem', color: 'var(--text-muted, #8b949e)' }}>
              <div>Stream Status: <span style={{ color: '#00ff88', fontWeight: 600 }}>Active (2.0s pulse)</span></div>
              <div>Buffer Depth: <span style={{ color: '#fff', fontWeight: 600 }}>600s rolling</span></div>
            </div>
            <button
              onClick={() => refetch()}
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

        {/* Dynamic Data Flow Pipeline Visualizer */}
        <div
          style={{
            marginTop: 20,
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
              <div style={{ fontSize: '0.74rem', color: '#8b949e' }}>Bybit V5 Linear Batch</div>
            </div>

            {/* Animated Connector 1 */}
            <div style={{ width: 40, position: 'relative', display: 'flex', justifyContent: 'center' }}>
              <motion.div
                key={`p1-${pulseTick}`}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 20, opacity: 1 }}
                transition={{ duration: 0.8, ease: 'linear' }}
                style={{ width: 8, height: 8, borderRadius: '50%', background: '#00d4ff', boxShadow: '0 0 8px #00d4ff' }}
              />
            </div>

            {/* Stage 2: Cross-Sectional Velocity */}
            <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(168, 85, 247, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#c084fc', fontSize: '0.82rem', fontWeight: 600 }}>
                <Activity size={14} /> 2. Statistical Z-Score
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                &gt; 2.2σ Threshold
              </div>
              <div style={{ fontSize: '0.74rem', color: '#8b949e' }}>Robust IQR Normalization</div>
            </div>

            {/* Animated Connector 2 */}
            <div style={{ width: 40, position: 'relative', display: 'flex', justifyContent: 'center' }}>
              <motion.div
                key={`p2-${pulseTick}`}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 20, opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.2, ease: 'linear' }}
                style={{ width: 8, height: 8, borderRadius: '50%', background: '#c084fc', boxShadow: '0 0 8px #c084fc' }}
              />
            </div>

            {/* Stage 3: Volume & Spread Gate */}
            <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(234, 179, 8, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#facc15', fontSize: '0.82rem', fontWeight: 600 }}>
                <ShieldCheck size={14} /> 3. Spread &amp; Volume Guard
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                &gt; 2.0x Vol &amp; &lt; 12 bps
              </div>
              <div style={{ fontSize: '0.74rem', color: '#8b949e' }}>BTC Macro Direction Check</div>
            </div>

            {/* Animated Connector 3 */}
            <div style={{ width: 40, position: 'relative', display: 'flex', justifyContent: 'center' }}>
              <motion.div
                key={`p3-${pulseTick}`}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 20, opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.4, ease: 'linear' }}
                style={{ width: 8, height: 8, borderRadius: '50%', background: '#00ff88', boxShadow: '0 0 8px #00ff88' }}
              />
            </div>

            {/* Stage 4: Trailing Ratchet Execution */}
            <div style={{ flex: 1, textAlign: 'center', padding: '10px 14px', background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(0, 255, 136, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: '#00ff88', fontSize: '0.82rem', fontWeight: 600 }}>
                <Zap size={14} /> 4. Trailing Ratchet
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
                {openPositions.length} Active / 5 Max
              </div>
              <div style={{ fontSize: '0.74rem', color: '#8b949e' }}>+1.0% Trailing Activation</div>
            </div>
          </div>
        </div>
      </Card>

      {/* 2. Top Metric Tiles */}
      <Grid cols={12} gap={16}>
        <Span n={3}>
          <Card title="Portfolio Equity" glow>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff', marginTop: 8 }}>
              ${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted, #8b949e)', marginTop: 4 }}>
              Available Cash: ${cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Realized Net PnL">
            <div
              style={{
                fontSize: '1.8rem',
                fontWeight: 800,
                color: realizedPnl >= 0 ? 'var(--accent-emerald, #00ff88)' : 'var(--accent-crimson, #ff4444)',
                marginTop: 8,
              }}
            >
              {realizedPnl >= 0 ? '+' : ''}${realizedPnl.toFixed(2)}
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted, #8b949e)', marginTop: 4 }}>
              Fees &amp; Slippage: ${feesPaid.toFixed(2)}
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Active Positions">
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-cyan, #00d4ff)', marginTop: 8 }}>
              {openPositions.length} <span style={{ fontSize: '1.1rem', color: '#8b949e' }}>/ 5 max</span>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted, #8b949e)', marginTop: 4 }}>
              Signals Triggered: {signalsCount}
            </div>
          </Card>
        </Span>

        <Span n={3}>
          <Card title="Win Rate &amp; Closed">
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff', marginTop: 8 }}>
              {winRate.toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted, #8b949e)', marginTop: 4 }}>
              Total Closed: {recentTrades.length} trades
            </div>
          </Card>
        </Span>
      </Grid>

      {/* 3. Live Market Pulse Heatmap / Matrix Radar */}
      <Card title={`Live Market Velocity Radar (${topMovers.length} Key Movers Analyzed)`}>
        <div style={{ marginBottom: 12, fontSize: '0.84rem', color: 'var(--text-muted, #8b949e)' }}>
          Real-time rolling 60-second momentum velocities and volume bursts refreshed every sub-second cycle:
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
          {topMovers.map((m: any, idx: number) => {
            const vel = m.velocity_pct ?? 0.0
            const isPump = vel >= 0.15
            const isDump = vel <= -0.15
            const borderCol = isPump ? 'rgba(0, 255, 136, 0.4)' : isDump ? 'rgba(255, 68, 68, 0.4)' : 'rgba(255, 255, 255, 0.08)'
            const bgGlow = isPump ? 'rgba(0, 255, 136, 0.06)' : isDump ? 'rgba(255, 68, 68, 0.06)' : 'rgba(255, 255, 255, 0.02)'

            return (
              <motion.div
                key={m.symbol}
                initial={{ opacity: 0.8 }}
                animate={{ scale: [1, 1.02, 1], opacity: 1 }}
                transition={{ duration: 0.3 }}
                style={{
                  padding: '10px 12px',
                  background: bgGlow,
                  border: `1px solid ${borderCol}`,
                  borderRadius: 8,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#fff' }}>{m.symbol}</span>
                  <span
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      fontWeight: 700,
                      fontSize: '0.84rem',
                      color: vel >= 0 ? 'var(--accent-emerald, #00ff88)' : 'var(--accent-crimson, #ff4444)',
                    }}
                  >
                    {vel >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
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
      </Card>

      {/* 4. Active Positions Table */}
      <Card title={`Active Open Positions (${openPositions.length})`}>
        {openPositions.length === 0 ? (
          <div style={{ padding: '30px 0', textAlign: 'center', color: 'var(--text-muted, #8b949e)' }}>
            <Activity size={32} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
            <div>No open positions currently. OmniPulse is actively scanning {universeSize} coins for breakout anomalies.</div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted, #8b949e)' }}>
                  <th style={{ padding: '10px 8px' }}>Symbol</th>
                  <th style={{ padding: '10px 8px' }}>Side</th>
                  <th style={{ padding: '10px 8px' }}>Entry Price</th>
                  <th style={{ padding: '10px 8px' }}>Mark Price</th>
                  <th style={{ padding: '10px 8px' }}>Notional</th>
                  <th style={{ padding: '10px 8px' }}>Unrealized PnL</th>
                  <th style={{ padding: '10px 8px' }}>Stop Loss</th>
                  <th style={{ padding: '10px 8px' }}>Take Profit</th>
                  <th style={{ padding: '10px 8px' }}>Trailing Status</th>
                  <th style={{ padding: '10px 8px' }}>Duration</th>
                </tr>
              </thead>
              <tbody>
                {openPositions.map((pos: any, idx: number) => {
                  const pnl = pos.unrealized_pnl ?? 0.0
                  const pnlPct = pos.notional > 0 ? (pnl / pos.notional) * 100 : 0
                  const isProfit = pnl >= 0
                  const holdMin = ((pos.hold_seconds ?? 0) / 60).toFixed(1)

                  return (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.06)',
                        background: idx % 2 === 0 ? 'rgba(255,255,255,0.015)' : 'transparent',
                      }}
                    >
                      <td style={{ padding: '12px 8px', fontWeight: 700 }}>{pos.symbol}</td>
                      <td style={{ padding: '12px 8px' }}>
                        <Badge tone={pos.side === 'Buy' ? 'good' : 'bad'}>{pos.side}</Badge>
                      </td>
                      <td style={{ padding: '12px 8px' }}>${pos.entry_price?.toFixed(4)}</td>
                      <td style={{ padding: '12px 8px' }}>${pos.current_price?.toFixed(4)}</td>
                      <td style={{ padding: '12px 8px' }}>${pos.notional?.toFixed(1)}</td>
                      <td
                        style={{
                          padding: '12px 8px',
                          fontWeight: 700,
                          color: isProfit ? 'var(--accent-emerald, #00ff88)' : 'var(--accent-crimson, #ff4444)',
                        }}
                      >
                        {isProfit ? '+' : ''}${pnl.toFixed(2)} ({isProfit ? '+' : ''}
                        {pnlPct.toFixed(2)}%)
                      </td>
                      <td style={{ padding: '12px 8px', color: '#ff7b72' }}>${pos.stop_loss?.toFixed(4)}</td>
                      <td style={{ padding: '12px 8px', color: '#7ee787' }}>${pos.take_profit?.toFixed(4)}</td>
                      <td style={{ padding: '12px 8px' }}>
                        {pos.trailing_active ? (
                          <Badge tone="good">ACTIVE (Trailing Peak)</Badge>
                        ) : (
                          <span style={{ color: 'var(--text-muted, #8b949e)', fontSize: '0.8rem' }}>Waiting +1.0%</span>
                        )}
                      </td>
                      <td style={{ padding: '12px 8px', color: 'var(--text-muted, #8b949e)' }}>{holdMin}m</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* 5. Live Telemetry Console & Recent Closed Trades */}
      <Grid cols={12} gap={16}>
        <Span n={6}>
          <Card title="Live Stream Telemetry Console">
            <div
              style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: 8,
                padding: '12px 14px',
                fontFamily: 'monospace',
                fontSize: '0.80rem',
                height: 230,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              {telemetryEvents.length === 0 ? (
                <div style={{ color: '#8b949e' }}>Streaming live order flow &amp; risk triggers...</div>
              ) : (
                telemetryEvents.map((ev: any, idx: number) => (
                  <div key={idx} style={{ display: 'flex', gap: 8 }}>
                    <span style={{ color: '#8b949e' }}>[{ev.time}]</span>
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
    </div>
  )
}

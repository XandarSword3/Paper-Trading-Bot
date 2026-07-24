import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom'
import { useStrategy } from '../lib/StrategyContext'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Badge, toneForWord } from './ui/Primitives'
import { Activity, Command, Play, Settings, BarChart3, TrendingUp, Shield, Globe, FileText, GitBranch, Bell } from 'lucide-react'

interface NavItem {
  path: string
  label: string
  icon: React.ReactNode
  cluster: 'LIVE' | 'ANALYSIS' | 'SYSTEM'
}

const NAV_ITEMS: NavItem[] = [
  { path: '/command', label: 'Command', icon: <Command size={18} />, cluster: 'LIVE' },
  { path: '/live-terminal', label: 'Live Terminal', icon: <Activity size={18} />, cluster: 'LIVE' },
  { path: '/paper-trading', label: 'Paper Trading', icon: <TrendingUp size={18} />, cluster: 'LIVE' },
  { path: '/alerts', label: 'Alerts', icon: <Bell size={18} />, cluster: 'LIVE' },
  { path: '/strategies', label: 'Strategies', icon: <BarChart3 size={18} />, cluster: 'ANALYSIS' },
  { path: '/backtests', label: 'Backtests', icon: <GitBranch size={18} />, cluster: 'ANALYSIS' },
  { path: '/risk', label: 'Risk', icon: <Shield size={18} />, cluster: 'ANALYSIS' },
  { path: '/risk-terrain', label: 'Risk Terrain', icon: <Globe size={18} />, cluster: 'ANALYSIS' },
  { path: '/reports', label: 'Reports', icon: <FileText size={18} />, cluster: 'ANALYSIS' },
  { path: '/readiness', label: 'Readiness & CI', icon: <GitBranch size={18} />, cluster: 'SYSTEM' },
  { path: '/settings', label: 'Settings', icon: <Settings size={18} />, cluster: 'SYSTEM' },
]

export default function Shell() {
  const location = useLocation()
  const navigate = useNavigate()
  const { strategyId, setStrategyId } = useStrategy()

  // Fetch strategies for the selector
  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: api.strategies,
  })

  // Fetch command center data for header stats
  const { data: commandData } = useQuery({
    queryKey: ['commandCenter', strategyId],
    queryFn: () => api.commandCenter(strategyId),
    enabled: !!strategyId,
  })

  const runBotMutation = useMutation({
    mutationFn: () => api.runBot(strategyId),
  })

  const handleRunBot = () => {
    runBotMutation.mutate()
  }

  const missionTime = new Date().toLocaleTimeString('en-US', { hour12: false })
  const btcPrice = commandData?.btc_price ?? null
  const btcDelta = commandData?.btc_price_delta_pct ?? null
  const equity = commandData?.analytics_summary?.current_equity ?? null

  return (
    <div className="qat-shell">
      {/* Header */}
      <header className="qat-shell__header">
        <div className="qat-shell__header-left">
          <div className="qat-shell__logo">
            <Command size={24} className="qat-shell__logo-icon" />
            <div className="qat-shell__logo-text">
              <span>QUANT ALPHA TERMINAL</span>
              <span className="qat-shell__logo-subtitle">STRATEGY COMMAND CENTER</span>
            </div>
          </div>
          <div className="qat-shell__header-stats">
            <div className="qat-shell__stat-group">
              <div className="qat-shell__stat">
                <span className="qat-shell__stat-label">STATUS</span>
                <Badge tone="good">ONLINE</Badge>
              </div>
              <div className="qat-shell__stat">
                <span className="qat-shell__stat-label">MISSION TIME</span>
                <span className="qat-shell__stat-value">{missionTime}</span>
              </div>
              <div className="qat-shell__stat">
                <span className="qat-shell__stat-label">STRATEGY</span>
                <select
                  className="qat-shell__strategy-select"
                  value={strategyId}
                  onChange={(e) => setStrategyId(e.target.value)}
                >
                  {strategies?.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name} ({s.id})
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="qat-shell__stat-group">
              <div className="qat-shell__stat">
                <span className="qat-shell__stat-label">BTC</span>
                <span className="qat-shell__stat-value">
                  {btcPrice ? `$${btcPrice.toLocaleString()}` : '--'}
                </span>
                {btcDelta !== null && (
                  <span className={`qat-shell__stat-delta ${btcDelta >= 0 ? 'is-good' : 'is-bad'}`}>
                    {btcDelta >= 0 ? '+' : ''}{btcDelta.toFixed(2)}%
                  </span>
                )}
              </div>
              <div className="qat-shell__stat">
                <span className="qat-shell__stat-label">EQUITY</span>
                <span className="qat-shell__stat-value">
                  {equity ? `$${equity.toLocaleString()}` : '--'}
                </span>
              </div>
            </div>
          </div>
        </div>
        <button
          className="qat-shell__run-btn"
          onClick={handleRunBot}
          disabled={runBotMutation.isPending}
        >
          <Play size={16} />
          {runBotMutation.isPending ? 'RUNNING...' : 'RUN BOT'}
        </button>
      </header>

      <div className="qat-shell__body">
        {/* Sidebar */}
        <aside className="qat-shell__sidebar">
          {(['LIVE', 'ANALYSIS', 'SYSTEM'] as const).map((cluster) => (
            <div key={cluster} className="qat-shell__nav-cluster">
              <div className="qat-shell__nav-cluster-label">{cluster}</div>
              {NAV_ITEMS.filter((item) => item.cluster === cluster).map((item) => {
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`qat-shell__nav-item ${isActive ? 'is-active' : ''}`}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </div>
          ))}
          <div className="qat-shell__sidebar-footer">
            <div className="qat-shell__operator">
              <span className="qat-shell__operator-dot"></span>
              <span>QA OPERATOR • ONLINE</span>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="qat-shell__main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

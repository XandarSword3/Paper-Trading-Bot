import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, StatTile, EmptyState, LoadingState, Grid, Span } from '../components/ui/Primitives'
import { RadarChart } from '../components/charts/index'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area } from 'recharts'

export default function Risk() {
  const { strategyId } = useStrategy()

  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics', strategyId],
    queryFn: () => api.analytics(strategyId),
    enabled: !!strategyId,
  })

  const { data: equityCurve } = useQuery({
    queryKey: ['equityCurve', strategyId],
    queryFn: () => api.equityCurve(strategyId),
    enabled: !!strategyId,
  })

  if (analyticsLoading) return <LoadingState label="Loading Risk Analysis…" />

  // Prepare data for charts
  const pnlHistogramData = analytics?.pnl_histogram.map(h => ({
    bin: h.bin_label,
    count: h.count,
    isWin: h.is_win,
    color: h.is_win ? '#00f5a0' : '#ff3366',
  })) || []

  const underwaterEquityData = equityCurve?.equity_curve
    .filter(p => p.drawdown_pct > 0)
    .map(p => ({
      timestamp: new Date(p.timestamp).toLocaleDateString(),
      drawdown: p.drawdown_pct,
      equity: p.equity,
    })) || []

  const var95 = analytics?.var_95 ?? 0
  const cvar95 = analytics?.cvar_95 ?? 0

  return (
    <div className="qat-risk">
      <Grid cols={12} gap={16}>
        {/* VaR/CVaR Stats */}
        <Span n={12}>
          <Card title="Value at Risk & Conditional VaR" subtitle="95% confidence level">
            <div className="qat-risk__var-stats">
              <StatTile label="VaR (95%)" value={`${var95.toFixed(2)}%`} />
              <StatTile label="CVaR (95%)" value={`${cvar95.toFixed(2)}%`} />
              <StatTile label="Max Drawdown" value={`${analytics?.max_drawdown_pct.toFixed(2)}%`} />
              <StatTile label="Max DD Duration" value={`${analytics?.max_drawdown_duration_hrs.toFixed(1)}h`} />
              <StatTile label="Sharpe Ratio" value={analytics?.sharpe_ratio?.toFixed(2) ?? 'N/A'} />
              <StatTile label="Sharpe CI" value={`[${analytics?.sharpe_ci?.[0]?.toFixed(2) ?? 'N/A'}, ${analytics?.sharpe_ci?.[1]?.toFixed(2) ?? 'N/A'}]`} />
            </div>
          </Card>
        </Span>

        {/* Return Distribution with VaR/CVaR Markers */}
        <Span n={8}>
          <Card title="Return Distribution" subtitle="PnL histogram with VaR/CVaR thresholds">
            {pnlHistogramData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={pnlHistogramData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="bin" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(8, 14, 26, 0.95)', border: '1px solid rgba(0, 245, 212, 0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <ReferenceLine x={var95} stroke="#ff3366" strokeWidth={2} strokeDasharray="5 5" label="VaR 95%" />
                  <ReferenceLine x={cvar95} stroke="#ffb800" strokeWidth={2} strokeDasharray="5 5" label="CVaR 95%" />
                  <Bar dataKey="count" fill="color" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState label="NO PnL DATA" />
            )}
          </Card>
        </Span>

        {/* Risk Radar */}
        <Span n={4}>
          <Card title="Risk Radar">
            {analytics && (
              <RadarChart
                axes={[
                  { label: 'DRAWDOWN', value: Math.min(100, (analytics.max_drawdown_pct / 50) * 100) },
                  { label: 'VOLATILITY', value: Math.min(100, (analytics.var_95 / 10) * 100) },
                  { label: 'LIQUIDITY', value: 75 },
                  { label: 'LEVERAGE', value: Math.min(100, (analytics.position_size / analytics.current_equity) * 1000) },
                  { label: 'EXPOSURE', value: Math.min(100, (analytics.position_size / analytics.current_equity) * 500) },
                ]}
                size={220}
                color="var(--accent-crimson)"
              />
            )}
          </Card>
        </Span>

        {/* Underwater Equity */}
        <Span n={12}>
          <Card title="Underwater Equity" subtitle="Drawdown periods over time">
            {underwaterEquityData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={underwaterEquityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="timestamp" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(8, 14, 26, 0.95)', border: '1px solid rgba(0, 245, 212, 0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Area type="monotone" dataKey="drawdown" stroke="#ff3366" fill="#ff3366" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState label="NO DRAWDOWN DATA" />
            )}
          </Card>
        </Span>

        {/* Exposure & Leverage */}
        <Span n={6}>
          <Card title="Exposure">
            <div className="qat-risk__exposure">
              <StatTile label="Position Size" value={`${analytics?.position_size.toFixed(6)} BTC`} />
              <StatTile label="Exposure %" value={`${((analytics?.position_size || 0) / (analytics?.current_equity || 1) * 100).toFixed(2)}%`} />
            </div>
          </Card>
        </Span>

        <Span n={6}>
          <Card title="Leverage">
            <div className="qat-risk__leverage">
              <StatTile label="Effective Leverage" value={`${((analytics?.position_size || 0) * (analytics?.current_equity || 1) / (analytics?.current_equity || 1)).toFixed(2)}x`} />
              <StatTile label="Max Safe Leverage" value="2.0x" />
            </div>
          </Card>
        </Span>
      </Grid>
    </div>
  )
}

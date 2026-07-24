import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, StatTile, EmptyState, LoadingState, Grid, Span, Badge, toneForWord } from '../components/ui/Primitives'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ZAxis } from 'recharts'

export default function Backtests() {
  const { strategyId } = useStrategy()

  const { data: walkForward, isLoading: wfLoading } = useQuery({
    queryKey: ['walkForward', strategyId],
    queryFn: () => api.walkForward(strategyId),
    enabled: !!strategyId,
  })

  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics', strategyId],
    queryFn: () => api.analytics(strategyId),
    enabled: !!strategyId,
  })

  if (wfLoading || analyticsLoading) return <LoadingState label="Loading Backtests…" />

  // Prepare data for charts
  const rollingMetricsData = analytics?.rolling_metrics.map(m => ({
    trade: m.trade_index,
    winRate: m.win_rate,
    sharpe: m.sharpe,
  })) || []

  const pnlHistogramData = analytics?.pnl_histogram.map(h => ({
    bin: h.bin_label,
    count: h.count,
    isWin: h.is_win,
    color: h.is_win ? '#00f5a0' : '#ff3366',
  })) || []

  const durationScatterData = analytics?.duration_scatter.map(d => ({
    duration: d.duration_hrs,
    pnl: d.pnl,
    type: d.trade_type,
    color: d.pnl >= 0 ? '#00f5a0' : '#ff3366',
  })) || []

  return (
    <div className="qat-backtests">
      <Grid cols={12} gap={16}>
        {/* Walk Forward Results */}
        <Span n={12}>
          <Card title="Walk-Forward Validation" subtitle={walkForward ? `${walkForward.num_folds} folds • OOS: ${walkForward.oos_coverage_start} to ${walkForward.oos_coverage_end}` : ''}>
            {walkForward ? (
              <div className="qat-backtests__wf-results">
                <div className="qat-backtests__wf-stats">
                  <StatTile label="OOS Win Rate" value={`${walkForward.oos_win_rate_pct.toFixed(1)}%`} />
                  <StatTile label="OOS Sharpe" value={walkForward.oos_sharpe.toFixed(2)} />
                  <StatTile label="OOS Return" value={`${walkForward.oos_total_return_pct.toFixed(1)}%`} />
                  <StatTile label="OOS CAGR" value={`${walkForward.oos_cagr_pct.toFixed(1)}%`} />
                  <StatTile label="OOS Max DD" value={`${walkForward.oos_max_drawdown_pct.toFixed(1)}%`} />
                  <StatTile label="OOS Calmar" value={walkForward.oos_calmar_ratio.toFixed(2)} />
                </div>
                <div className="qat-backtests__wf-status">
                  <Badge tone={walkForward.acceptance_check_passed ? 'good' : 'bad'}>
                    {walkForward.acceptance_check_passed ? 'ACCEPTANCE CHECK PASSED' : 'ACCEPTANCE CHECK FAILED'}
                  </Badge>
                </div>
              </div>
            ) : (
              <EmptyState label="NO WALK-FORWARD DATA" />
            )}
          </Card>
        </Span>

        {/* Rolling Metrics */}
        <Span n={6}>
          <Card title="Rolling Metrics" subtitle="Win rate and Sharpe over time">
            {rollingMetricsData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={rollingMetricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="trade" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(8, 14, 26, 0.95)', border: '1px solid rgba(0, 245, 212, 0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Line type="monotone" dataKey="winRate" stroke="#00f5a0" strokeWidth={2} name="Win Rate %" dot={false} />
                  <Line type="monotone" dataKey="sharpe" stroke="#00d2ff" strokeWidth={2} name="Sharpe" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState label="NO ROLLING METRICS" />
            )}
          </Card>
        </Span>

        {/* PnL Histogram */}
        <Span n={6}>
          <Card title="PnL Distribution" subtitle="Trade return histogram">
            {pnlHistogramData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={pnlHistogramData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="bin" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(8, 14, 26, 0.95)', border: '1px solid rgba(0, 245, 212, 0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Bar dataKey="count" fill="color" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState label="NO PnL DATA" />
            )}
          </Card>
        </Span>

        {/* Duration Scatter */}
        <Span n={12}>
          <Card title="Duration vs PnL" subtitle="Trade duration scatter plot">
            {durationScatterData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="duration" name="Duration (hrs)" stroke="#94a3b8" fontSize={11} />
                  <YAxis dataKey="pnl" name="PnL ($)" stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ background: 'rgba(8, 14, 26, 0.95)', border: '1px solid rgba(0, 245, 212, 0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Scatter data={durationScatterData} fill="color" />
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState label="NO DURATION DATA" />
            )}
          </Card>
        </Span>
      </Grid>
    </div>
  )
}

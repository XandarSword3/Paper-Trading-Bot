import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, StatTile, EmptyState, LoadingState, Grid, Span, Badge, toneForWord } from '../components/ui/Primitives'
import { Sparkline } from '../components/charts/index'
import { TrendingUp, TrendingDown } from 'lucide-react'

export default function PaperTrading() {
  const { strategyId } = useStrategy()

  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
  })

  const { data: trades, isLoading: tradesLoading } = useQuery({
    queryKey: ['trades', strategyId],
    queryFn: () => api.strategyTrades(strategyId, 100),
    enabled: !!strategyId,
  })

  const { data: equityCurve } = useQuery({
    queryKey: ['equityCurve', strategyId],
    queryFn: () => api.equityCurve(strategyId),
    enabled: !!strategyId,
  })

  if (overviewLoading || tradesLoading) return <LoadingState label="Loading Paper Trading…" />

  // Find the current strategy's data from overview
  const currentStrategy = overview?.strategies.find(s => s.id === strategyId)

  return (
    <div className="qat-paper-trading">
      <Grid cols={12} gap={16}>
        {/* Open Position Card */}
        <Span n={4}>
          <Card title="Open Position" glow>
            {currentStrategy ? (
              <div className="qat-paper-trading__position">
                <div className="qat-paper-trading__position-header">
                  <span className="qat-paper-trading__position-symbol">BTC/USDT</span>
                  <Badge tone={currentStrategy.ready_for_live ? 'good' : 'warn'}>
                    {currentStrategy.ready_for_live ? 'READY' : 'NOT READY'}
                  </Badge>
                </div>
                <div className="qat-paper-trading__position-main">
                  <div className="qat-paper-trading__position-size">
                    <span className="qat-paper-trading__position-label">SIZE</span>
                    <span className="qat-paper-trading__position-value">
                      {currentStrategy.position_size.toFixed(6)} BTC
                    </span>
                  </div>
                  <div className="qat-paper-trading__position-equity">
                    <span className="qat-paper-trading__position-label">EQUITY</span>
                    <span className="qat-paper-trading__position-value">
                      ${currentStrategy.current_equity.toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="qat-paper-trading__position-stats">
                  <StatTile label="Total Trades" value={currentStrategy.trade_count.toString()} />
                </div>
              </div>
            ) : (
              <EmptyState label="NO POSITION DATA" />
            )}
          </Card>
        </Span>

        {/* Equity Mini Chart */}
        <Span n={8}>
          <Card title="Equity Curve">
            {equityCurve && equityCurve.equity_curve.length > 0 ? (
              <div className="qat-paper-trading__equity-chart">
                <Sparkline
                  values={equityCurve.equity_curve.map(p => p.equity)}
                  width={600}
                  height={120}
                  color="var(--accent-emerald)"
                  fill
                />
                <div className="qat-paper-trading__equity-stats">
                  <div className="qat-paper-trading__equity-stat">
                    <span className="qat-paper-trading__equity-stat-label">PEAK</span>
                    <span className="qat-paper-trading__equity-stat-value">
                      ${Math.max(...equityCurve.equity_curve.map(p => p.peak)).toLocaleString()}
                    </span>
                  </div>
                  <div className="qat-paper-trading__equity-stat">
                    <span className="qat-paper-trading__equity-stat-label">MAX DD</span>
                    <span className="qat-paper-trading__equity-stat-value is-bad">
                      {Math.max(...equityCurve.equity_curve.map(p => p.drawdown_pct)).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState label="NO EQUITY DATA" />
            )}
          </Card>
        </Span>

        {/* Trade Ledger */}
        <Span n={12}>
          <Card title="Trade Ledger" subtitle={`Last ${trades?.length || 0} trades`}>
            {trades && trades.length > 0 ? (
              <div className="qat-paper-trading__ledger">
                <div className="qat-paper-trading__ledger-header">
                  <span>TIME</span>
                  <span>TYPE</span>
                  <span>PRICE</span>
                  <span>QTY</span>
                  <span>PnL</span>
                  <span>REASON</span>
                </div>
                <div className="qat-paper-trading__ledger-body">
                  {trades.map((trade) => (
                    <div key={trade.id} className="qat-paper-trading__ledger-row">
                      <span className="qat-paper-trading__ledger-time">{trade.timestamp}</span>
                      <span className={`qat-paper-trading__ledger-type ${trade.trade_type.toLowerCase()}`}>
                        {trade.trade_type}
                      </span>
                      <span className="qat-paper-trading__ledger-price">${trade.price.toLocaleString()}</span>
                      <span className="qat-paper-trading__ledger-qty">{trade.quantity.toFixed(6)}</span>
                      <span className={`qat-paper-trading__ledger-pnl ${trade.pnl && trade.pnl >= 0 ? 'is-good' : 'is-bad'}`}>
                        {trade.pnl !== null ? `$${trade.pnl.toFixed(2)}` : '--'}
                      </span>
                      <span className="qat-paper-trading__ledger-reason">{trade.reason || '--'}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <EmptyState label="NO TRADES" />
            )}
          </Card>
        </Span>
      </Grid>
    </div>
  )
}

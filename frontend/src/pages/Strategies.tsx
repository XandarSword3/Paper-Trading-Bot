import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, StatTile, EmptyState, LoadingState, Grid, Span, Badge, toneForWord } from '../components/ui/Primitives'
import { Sparkline } from '../components/charts/index'
import { Link } from 'react-router-dom'

export default function Strategies() {
  const { setStrategyId } = useStrategy()

  const { data: strategies, isLoading: strategiesLoading } = useQuery({
    queryKey: ['strategies'],
    queryFn: api.strategies,
  })

  const { data: readiness } = useQuery({
    queryKey: ['readiness'],
    queryFn: api.readiness,
  })

  if (strategiesLoading) return <LoadingState label="Loading Strategies…" />

  return (
    <div className="qat-strategies">
      <Grid cols={12} gap={16}>
        {strategies && strategies.length > 0 ? (
          strategies.map((strategy) => {
            const readinessData = readiness?.find(r => r.strategy_id === strategy.id)
            const isReady = readinessData?.ready_for_live ?? false

            return (
              <Span n={4} key={strategy.id}>
                <Card
                  title={strategy.name}
                  subtitle={strategy.id}
                  glow={strategy.id === 'v4'}
                  actions={
                    <button
                      className="qat-strategies__select-btn"
                      onClick={() => setStrategyId(strategy.id)}
                    >
                      SELECT
                    </button>
                  }
                >
                  <div className="qat-strategies__card">
                    <div className="qat-strategies__card-header">
                      <Badge tone={isReady ? 'good' : 'warn'}>
                        {isReady ? 'READY FOR LIVE' : 'NOT READY'}
                      </Badge>
                      <span className="qat-strategies__card-timeframe">{strategy.timeframe}</span>
                    </div>

                    <div className="qat-strategies__card-stats">
                      <StatTile label="Equity" value={`$${strategy.current_equity.toLocaleString()}`} />
                      <StatTile label="Position" value={`${strategy.position_size.toFixed(6)} BTC`} />
                      <StatTile label="Trades" value={strategy.trade_count.toString()} />
                    </div>

                    <div className="qat-strategies__card-footer">
                      <Link to={`/backtests?strategy=${strategy.id}`} className="qat-strategies__card-link">
                        View Backtests →
                      </Link>
                      <Link to={`/risk?strategy=${strategy.id}`} className="qat-strategies__card-link">
                        View Risk →
                      </Link>
                    </div>
                  </div>
                </Card>
              </Span>
            )
          })
        ) : (
          <Span n={12}>
            <EmptyState label="NO STRATEGIES FOUND" />
          </Span>
        )}
      </Grid>
    </div>
  )
}

import { useRef, useEffect, useState } from 'react'
import { useStrategy } from '../lib/StrategyContext'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { Card, StatTile, EmptyState, LoadingState, Grid, Span, Badge, toneForWord } from '../components/ui/Primitives'
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, Time } from 'lightweight-charts'
import { TrendingUp, TrendingDown, Activity, Zap, Target, BarChart3 } from 'lucide-react'

const TIME_RANGES = ['1H', '4H', '1D', '1W'] as const

export default function LiveTerminal() {
  const { strategyId } = useStrategy()
  const [timeRange, setTimeRange] = useState<(typeof TIME_RANGES)[number]>('4H')
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  const { data: candles, isLoading: candlesLoading, error: candlesError } = useQuery({
    queryKey: ['candles', strategyId, timeRange],
    queryFn: () => api.candles(strategyId, 150),
    enabled: !!strategyId,
    refetchInterval: 5000,
  })

  const { data: analytics } = useQuery({
    queryKey: ['analytics', strategyId],
    queryFn: () => api.analytics(strategyId),
    enabled: !!strategyId,
    refetchInterval: 10000,
  })

  const { data: commandCenter } = useQuery({
    queryKey: ['commandCenter', strategyId],
    queryFn: () => api.commandCenter(strategyId),
    enabled: !!strategyId,
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: 'transparent' },
        textColor: '#64748b',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.05)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.05)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(0, 245, 212, 0.4)', width: 1, style: 2 },
        horzLine: { color: 'rgba(0, 245, 212, 0.4)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.1)',
      },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00f5a0',
      downColor: '#ff3366',
      borderUpColor: '#00f5a0',
      borderDownColor: '#ff3366',
      wickUpColor: '#00f5a0',
      wickDownColor: '#ff3366',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    })

    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    })

    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries
    volumeSeriesRef.current = volumeSeries

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current || !candles) return

    console.log('Candles received:', candles.length, candles[0])

    const candlestickData: CandlestickData[] = candles.map((candle) => ({
      time: (candle.time / 1000) as Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }))

    const volumeData: HistogramData[] = candles.map((candle) => ({
      time: (candle.time / 1000) as Time,
      value: candle.volume,
      color: candle.close >= candle.open ? 'rgba(0, 245, 160, 0.3)' : 'rgba(255, 51, 102, 0.3)',
    }))

    console.log('Setting chart data:', candlestickData.length, volumeData.length)
    candlestickSeriesRef.current.setData(candlestickData)
    volumeSeriesRef.current.setData(volumeData)
  }, [candles])

  if (candlesLoading) return <LoadingState label="Loading Live Terminal…" />
  if (candlesError) return <div className="qat-error">Failed to load market data</div>

  const latestCandle = candles?.[candles.length - 1]
  const prevCandle = candles?.[candles.length - 2]
  const priceChange = latestCandle && prevCandle ? ((latestCandle.close - prevCandle.close) / prevCandle.close * 100) : 0
  const isUp = priceChange >= 0

  return (
    <div className="qat-live-terminal">
      {/* Price Header */}
      <div className="qat-live-terminal__price-header">
        <div className="qat-live-terminal__price-main">
          <span className="qat-live-terminal__price-symbol">BTC/USDT</span>
          <span className="qat-live-terminal__price-value">
            ${latestCandle?.close.toFixed(2) ?? '--'}
          </span>
          <span className={`qat-live-terminal__price-change ${isUp ? 'is-up' : 'is-down'}`}>
            {isUp ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
            {Math.abs(priceChange).toFixed(2)}%
          </span>
        </div>
        <div className="qat-live-terminal__time-range">
          {TIME_RANGES.map((range) => (
            <button
              key={range}
              className={`qat-live-terminal__time-btn ${timeRange === range ? 'is-active' : ''}`}
              onClick={() => setTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      <Grid cols={12} gap={16}>
        {/* Candlestick Chart */}
        <Span n={9}>
          <Card title="" glow>
            <div ref={chartContainerRef} className="qat-live-terminal__chart" />
            {!candles || candles.length === 0 && <EmptyState label="NO CANDLE DATA" />}
          </Card>
        </Span>

        {/* Strategy Telemetry */}
        <Span n={3}>
          <Card title="Strategy Telemetry" subtitle="Real-time performance metrics">
            {analytics ? (
              <div className="qat-live-terminal__telemetry-grid">
                <div className="qat-live-terminal__tele-item">
                  <Activity size={16} className="qat-live-terminal__tele-icon" />
                  <div className="qat-live-terminal__tele-content">
                    <span className="qat-live-terminal__tele-label">Win Rate</span>
                    <span className={`qat-live-terminal__tele-value ${analytics.win_rate >= 50 ? 'is-good' : 'is-bad'}`}>
                      {analytics.win_rate.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="qat-live-terminal__tele-item">
                  <Zap size={16} className="qat-live-terminal__tele-icon" />
                  <div className="qat-live-terminal__tele-content">
                    <span className="qat-live-terminal__tele-label">Profit Factor</span>
                    <span className={`qat-live-terminal__tele-value ${analytics.profit_factor >= 1.5 ? 'is-good' : analytics.profit_factor >= 1 ? 'is-neutral' : 'is-bad'}`}>
                      {analytics.profit_factor.toFixed(2)}x
                    </span>
                  </div>
                </div>
                <div className="qat-live-terminal__tele-item">
                  <Target size={16} className="qat-live-terminal__tele-icon" />
                  <div className="qat-live-terminal__tele-content">
                    <span className="qat-live-terminal__tele-label">Expectancy</span>
                    <span className={`qat-live-terminal__tele-value ${analytics.expectancy >= 0 ? 'is-good' : 'is-bad'}`}>
                      ${analytics.expectancy.toFixed(2)}
                    </span>
                  </div>
                </div>
                <div className="qat-live-terminal__tele-item">
                  <BarChart3 size={16} className="qat-live-terminal__tele-icon" />
                  <div className="qat-live-terminal__tele-content">
                    <span className="qat-live-terminal__tele-label">Sharpe Ratio</span>
                    <span className={`qat-live-terminal__tele-value ${analytics.sharpe_ratio >= 1 ? 'is-good' : 'is-neutral'}`}>
                      {analytics.sharpe_ratio.toFixed(2)}
                    </span>
                  </div>
                </div>
                <div className="qat-live-terminal__tele-divider" />
                <div className="qat-live-terminal__tele-row">
                  <span className="qat-live-terminal__tele-row-label">Avg Win</span>
                  <span className="qat-live-terminal__tele-row-value is-good">${analytics.avg_win.toFixed(2)}</span>
                </div>
                <div className="qat-live-terminal__tele-row">
                  <span className="qat-live-terminal__tele-row-label">Avg Loss</span>
                  <span className="qat-live-terminal__tele-row-value is-bad">${analytics.avg_loss.toFixed(2)}</span>
                </div>
                <div className="qat-live-terminal__tele-row">
                  <span className="qat-live-terminal__tele-row-label">Max Win Streak</span>
                  <span className="qat-live-terminal__tele-row-value">{analytics.max_consecutive_wins}</span>
                </div>
                <div className="qat-live-terminal__tele-row">
                  <span className="qat-live-terminal__tele-row-label">Max Loss Streak</span>
                  <span className="qat-live-terminal__tele-row-value">{analytics.max_consecutive_losses}</span>
                </div>
              </div>
            ) : (
              <EmptyState label="NO TELEMETRY" />
            )}
          </Card>
        </Span>

        {/* Technical Indicators */}
        <Span n={12}>
          <Card title="Technical Indicators" subtitle="Donchian Channels & ATR">
            {candles && candles.length > 0 && latestCandle ? (
              <div className="qat-live-terminal__indicators">
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">ATR (14)</span>
                  <span className="qat-live-terminal__ind-value">${latestCandle.atr.toFixed(2)}</span>
                </div>
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">Volume (24h)</span>
                  <span className="qat-live-terminal__ind-value">{latestCandle.volume.toLocaleString()}</span>
                </div>
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">Donchian Upper</span>
                  <span className="qat-live-terminal__ind-value is-good">${latestCandle.entry_high.toFixed(2)}</span>
                </div>
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">Donchian Lower</span>
                  <span className="qat-live-terminal__ind-value is-bad">${latestCandle.exit_low.toFixed(2)}</span>
                </div>
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">Position Size</span>
                  <span className="qat-live-terminal__ind-value">{analytics?.position_size.toFixed(6)} BTC</span>
                </div>
                <div className="qat-live-terminal__ind-group">
                  <span className="qat-live-terminal__ind-label">Exposure</span>
                  <span className="qat-live-terminal__ind-value">{(commandCenter?.analytics_summary?.position_size ?? 0) > 0 ? 'ACTIVE' : 'FLAT'}</span>
                </div>
              </div>
            ) : (
              <EmptyState label="NO INDICATOR DATA" />
            )}
          </Card>
        </Span>

        {/* Market Regime */}
        <Span n={6}>
          <Card title="Market Regime">
            {commandCenter?.market_regime ? (
              <div className="qat-live-terminal__regime">
                <Badge tone={toneForWord(commandCenter.market_regime.regime)}>
                  {commandCenter.market_regime.regime}
                </Badge>
                <div className="qat-live-terminal__regime-details">
                  <div className="qat-live-terminal__regime-item">
                    <span className="qat-live-terminal__regime-label">Volatility</span>
                    <Badge tone={toneForWord(commandCenter.market_regime.volatility)}>
                      {commandCenter.market_regime.volatility}
                    </Badge>
                  </div>
                  <div className="qat-live-terminal__regime-item">
                    <span className="qat-live-terminal__regime-label">Liquidity</span>
                    <Badge tone={toneForWord(commandCenter.market_regime.liquidity)}>
                      {commandCenter.market_regime.liquidity}
                    </Badge>
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState label="NO REGIME DATA" />
            )}
          </Card>
        </Span>

        {/* Signal Status */}
        <Span n={6}>
          <Card title="Signal Status">
            {analytics ? (
              <div className="qat-live-terminal__signal">
                <div className="qat-live-terminal__signal-main">
                  <span className="qat-live-terminal__signal-label">Current Signal</span>
                  <Badge tone={analytics.position_size > 0 ? 'good' : 'neutral'}>
                    {analytics.position_size > 0 ? 'LONG ACTIVE' : 'NO POSITION'}
                  </Badge>
                </div>
                <div className="qat-live-terminal__signal-details">
                  <div className="qat-live-terminal__signal-row">
                    <span>Entry Price</span>
                    <span className="qat-live-terminal__signal-value">${latestCandle?.close.toFixed(2) ?? '--'}</span>
                  </div>
                  <div className="qat-live-terminal__signal-row">
                    <span>Stop Loss</span>
                    <span className="qat-live-terminal__signal-value">${latestCandle?.exit_low.toFixed(2) ?? '--'}</span>
                  </div>
                  <div className="qat-live-terminal__signal-row">
                    <span>Take Profit</span>
                    <span className="qat-live-terminal__signal-value">${latestCandle?.entry_high.toFixed(2) ?? '--'}</span>
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState label="NO SIGNAL DATA" />
            )}
          </Card>
        </Span>
      </Grid>
    </div>
  )
}

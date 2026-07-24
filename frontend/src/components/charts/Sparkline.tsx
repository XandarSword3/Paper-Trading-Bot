import { scaleLinear } from 'd3-scale'
import { line as d3line, curveMonotoneX } from 'd3-shape'

export function Sparkline({
  values,
  width = 120,
  height = 32,
  color = 'var(--accent-cyan)',
  fill = false,
}: {
  values: number[]
  width?: number
  height?: number
  color?: string
  fill?: boolean
}) {
  if (!values || values.length < 2) {
    return (
      <svg width={width} height={height}>
        <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
      </svg>
    )
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const x = scaleLinear().domain([0, values.length - 1]).range([2, width - 2])
  const y = scaleLinear().domain([min === max ? min - 1 : min, min === max ? max + 1 : max]).range([height - 4, 4])

  const gen = d3line<number>()
    .x((_d, i) => x(i))
    .y((d) => y(d))
    .curve(curveMonotoneX)

  const d = gen(values) ?? ''
  const areaD = fill ? `${d} L${x(values.length - 1)},${height} L${x(0)},${height} Z` : undefined
  const last = values[values.length - 1]
  const first = values[0]
  const trendColor = last >= first ? 'var(--accent-emerald)' : 'var(--accent-crimson)'

  return (
    <svg width={width} height={height}>
      {fill && <path d={areaD} fill={trendColor} opacity={0.12} />}
      <path d={d} fill="none" stroke={color === 'auto' ? trendColor : color} strokeWidth={1.75} />
      <circle cx={x(values.length - 1)} cy={y(last)} r={2.5} fill={color === 'auto' ? trendColor : color} />
    </svg>
  )
}

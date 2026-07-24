import { scaleLinear } from 'd3-scale'
import { lineRadial, curveLinearClosed } from 'd3-shape'

export interface RadarAxis {
  label: string
  value: number // 0-100
}

export function RadarChart({
  axes,
  size = 260,
  color = 'var(--accent-cyan)',
  labelPad = 44,
}: {
  axes: RadarAxis[]
  size?: number
  color?: string
  labelPad?: number
}) {
  const radius = size / 2 - 36
  const vb = size + labelPad * 2
  const angleScale = scaleLinear().domain([0, axes.length]).range([0, Math.PI * 2])
  const rScale = scaleLinear().domain([0, 100]).range([0, radius])

  const points: [number, number][] = axes.map((a, i) => {
    const angle = angleScale(i) - Math.PI / 2
    const r = rScale(Math.max(0, Math.min(100, a.value)))
    return [r * Math.cos(angle), r * Math.sin(angle)]
  })

  const lineGen = lineRadial()
    .radius((d) => rScale(Math.max(0, Math.min(100, d[1] as unknown as number))))
    .angle((_d, i) => angleScale(i))
    .curve(curveLinearClosed)

  const shapeD = lineGen(axes.map((a, i) => [i, a.value]) as any)

  const rings = [25, 50, 75, 100]

  return (
    <svg width={size} height={size} viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`}>
      {rings.map((ringVal) => (
        <polygon
          key={ringVal}
          points={axes
            .map((_a, i) => {
              const angle = angleScale(i) - Math.PI / 2
              const r = rScale(ringVal)
              return `${r * Math.cos(angle)},${r * Math.sin(angle)}`
            })
            .join(' ')}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={1}
        />
      ))}
      {axes.map((_a, i) => {
        const angle = angleScale(i) - Math.PI / 2
        return (
          <line
            key={i}
            x1={0}
            y1={0}
            x2={radius * Math.cos(angle)}
            y2={radius * Math.sin(angle)}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={1}
          />
        )
      })}
      <path
        d={shapeD ?? undefined}
        fill={color}
        fillOpacity={0.18}
        stroke={color}
        strokeWidth={2}
        style={{ filter: `drop-shadow(0 0 6px ${color})` }}
      />
      {points.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={3} fill={color} />
      ))}
      {axes.map((a, i) => {
        const angle = angleScale(i) - Math.PI / 2
        const lx = (radius + 22) * Math.cos(angle)
        const ly = (radius + 22) * Math.sin(angle)
        return (
          <text
            key={a.label}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="middle"
            fontFamily="var(--font-sans)"
            fontSize={11}
            fill="var(--text-dim)"
          >
            {a.label}
          </text>
        )
      })}
    </svg>
  )
}

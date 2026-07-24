import { pie as d3pie, arc as d3arc } from 'd3-shape'

export interface DonutSlice {
  name: string
  pct: number
  color: string
}

// Builds itself off however many slices the backend returns — no hardcoded
// 2- or 5-slice shape, so it keeps working once portfolio mode ships more assets.
export function DonutChart({
  data,
  size = 200,
  thickness = 26,
  centerLabel,
  centerValue,
}: {
  data: DonutSlice[]
  size?: number
  thickness?: number
  centerLabel?: string
  centerValue?: string
}) {
  const r = size / 2
  const pieGen = d3pie<DonutSlice>()
    .value((d) => d.pct)
    .sort(null)
  const arcs = pieGen(data)
  const arcGen = d3arc<any>().innerRadius(r - thickness).outerRadius(r)

  if (!data.length || data.every((d) => d.pct === 0)) {
    return (
      <svg width={size} height={size} viewBox={`${-r} ${-r} ${size} ${size}`}>
        <circle r={r - thickness / 2} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={thickness} />
        <text textAnchor="middle" dy={4} fontFamily="var(--font-mono)" fontSize={12} fill="var(--text-muted)">
          NO DATA
        </text>
      </svg>
    )
  }

  return (
    <svg width={size} height={size} viewBox={`${-r} ${-r} ${size} ${size}`}>
      {arcs.map((a, i) => (
        <path key={i} d={arcGen(a) ?? undefined} fill={data[i].color} stroke="var(--bg-card)" strokeWidth={2} />
      ))}
      {centerValue && (
        <text
          textAnchor="middle"
          dy={centerLabel ? -2 : 5}
          fontFamily="var(--font-mono)"
          fontWeight={700}
          fontSize={size * 0.1}
          fill="var(--text-main)"
        >
          {centerValue}
        </text>
      )}
      {centerLabel && (
        <text
          textAnchor="middle"
          dy={16}
          fontFamily="var(--font-sans)"
          fontSize={size * 0.05}
          letterSpacing="0.06em"
          fill="var(--text-dim)"
        >
          {centerLabel}
        </text>
      )}
    </svg>
  )
}

import { arc as d3arc } from 'd3-shape'
import { useId } from 'react'

// Tactical, targeting-reticle style gauge with cyber/success palette
export function GaugeRing({
  value, // 0-100
  size = 280,
  thickness = 12,
  color = 'var(--accent-cyan)',
  trackColor = 'rgba(255,255,255,0.06)',
  label,
  sublabel,
  status,
}: {
  value: number
  size?: number
  thickness?: number
  color?: string
  trackColor?: string
  label?: string
  sublabel?: string
  status?: string
}) {
  const clamped = Math.max(0, Math.min(100, value))
  const centerX = 0
  const centerY = 0
  
  // Color zones based on value
  const getColorForValue = (v: number) => {
    if (v >= 80) return '#00f5a0'
    if (v >= 65) return '#00E5FF'
    if (v >= 50) return '#ffb800'
    if (v >= 35) return '#ff6b35'
    return '#ff3366'
  }

  const primaryColor = getColorForValue(clamped)
  const secondaryColor = clamped >= 65 ? '#00B8D4' : (clamped >= 50 ? '#ff9500' : '#ff4500')
  const glowColor = primaryColor
  
  // D3 arc parameters for gradient track
  const startAngle = -Math.PI * 0.75
  const endAngle = Math.PI * 0.75
  const valueAngle = startAngle + (endAngle - startAngle) * (clamped / 100)
  
  // Ring dimensions - more nested circles
  const outerRadius = size / 2 - 10
  const ring1Radius = size / 2 - 20
  const ring2Radius = size / 2 - 30
  const ring3Radius = size / 2 - 40
  const ring4Radius = size / 2 - 50
  const innerRadius = size / 2 - 60
  
  // D3 arcs for gradient track
  const trackArc = d3arc()({
    innerRadius: ring2Radius,
    outerRadius: ring2Radius + thickness,
    startAngle,
    endAngle,
  } as any)

  const valueArc = d3arc()({
    innerRadius: ring2Radius,
    outerRadius: ring2Radius + thickness,
    startAngle,
    endAngle: valueAngle,
  } as any)
  
  // Calculate segmented outer ring (tactical blocks)
  const segments = 48
  const activeSegments = Math.floor((clamped / 100) * segments)
  
  // Calculate circumference for stroke-dasharray
  const outerCircumference = 2 * Math.PI * outerRadius
  const ring1Circumference = 2 * Math.PI * ring1Radius
  const ring3Circumference = 2 * Math.PI * ring3Radius
  const ring4Circumference = 2 * Math.PI * ring4Radius
  const innerCircumference = 2 * Math.PI * innerRadius
  
  // Progress offsets for continuous rings
  const ring1Offset = ring1Circumference - (clamped / 100) * ring1Circumference
  const ring3Offset = ring3Circumference - (clamped / 100) * ring3Circumference

  return (
    <svg width={size} height={size + 60} viewBox={`${-size / 2} ${-size / 2 - 30} ${size} ${size + 60}`}>
      <defs>
        {/* Intense neon glow filters */}
        <filter id="neon-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feGaussianBlur stdDeviation="6" result="coloredBlur2" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="coloredBlur2" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <filter id="intense-glow" x="-150%" y="-150%" width="400%" height="400%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feGaussianBlur stdDeviation="8" result="blur2" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="blur2" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <filter id="text-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Radial gradient for background glow */}
        <radialGradient id="bg-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={primaryColor} stopOpacity={0.15} />
          <stop offset="50%" stopColor={primaryColor} stopOpacity={0.05} />
          <stop offset="100%" stopColor={primaryColor} stopOpacity={0} />
        </radialGradient>

        {/* Linear gradient for segments */}
        <linearGradient id="segment-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={primaryColor} />
          <stop offset="100%" stopColor={secondaryColor} />
        </linearGradient>

        {/* Gradient track */}
        <linearGradient id="track-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
          <stop offset="50%" stopColor="rgba(255,255,255,0.08)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0.03)" />
        </linearGradient>

        {/* Value arc gradient */}
        <linearGradient id="value-arc-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={primaryColor} stopOpacity={0.6} />
          <stop offset="50%" stopColor={primaryColor} stopOpacity={1} />
          <stop offset="100%" stopColor={secondaryColor} stopOpacity={0.8} />
        </linearGradient>
      </defs>

      {/* Background radial glow */}
      <circle cx={centerX} cy={centerY} r={outerRadius + 20} fill="url(#bg-glow)" />

      {/* Outer ring - segmented tactical blocks */}
      <circle
        cx={centerX}
        cy={centerY}
        r={outerRadius}
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeWidth={4}
        strokeDasharray={`${(outerCircumference / segments) - 2} 2`}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`0 ${centerX} ${centerY}`}
          to={`360 ${centerX} ${centerY}`}
          dur="60s"
          repeatCount="indefinite"
        />
      </circle>
      
      {/* Active segments on outer ring */}
      <circle
        cx={centerX}
        cy={centerY}
        r={outerRadius}
        fill="none"
        stroke="url(#segment-gradient)"
        strokeWidth={4}
        strokeDasharray={`${(outerCircumference / segments) - 2} 2`}
        strokeDashoffset={outerCircumference - (activeSegments * (outerCircumference / segments))}
        filter="url(#neon-glow)"
        transform={`rotate(-90 ${centerX} ${centerY})`}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`-90 ${centerX} ${centerY}`}
          to={`270 ${centerX} ${centerY}`}
          dur="60s"
          repeatCount="indefinite"
          additive="sum"
        />
      </circle>

      {/* Ring 1 - continuous track */}
      <circle
        cx={centerX}
        cy={centerY}
        r={ring1Radius}
        fill="none"
        stroke="rgba(255,255,255,0.05)"
        strokeWidth={2}
      />
      
      {/* Active progress on ring 1 */}
      <circle
        cx={centerX}
        cy={centerY}
        r={ring1Radius}
        fill="none"
        stroke={primaryColor}
        strokeWidth={2}
        strokeDasharray={ring1Circumference}
        strokeDashoffset={ring1Offset}
        strokeLinecap="round"
        opacity={0.6}
        filter="url(#neon-glow)"
        transform={`rotate(-90 ${centerX} ${centerY})`}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`-90 ${centerX} ${centerY}`}
          to={`-450 ${centerX} ${centerY}`}
          dur="45s"
          repeatCount="indefinite"
          additive="sum"
        />
      </circle>

      {/* Ring 2 - gradient track (D3 arc) */}
      <path d={trackArc ?? undefined} fill="url(#track-gradient)" />
      
      {/* Value arc on ring 2 */}
      <path 
        d={valueArc ?? undefined} 
        fill="url(#value-arc-gradient)" 
        filter="url(#intense-glow)"
      />

      {/* Ring 3 - thin decorative */}
      <circle
        cx={centerX}
        cy={centerY}
        r={ring3Radius}
        fill="none"
        stroke="rgba(255,255,255,0.03)"
        strokeWidth={1}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`0 ${centerX} ${centerY}`}
          to={`360 ${centerX} ${centerY}`}
          dur="30s"
          repeatCount="indefinite"
        />
      </circle>
      
      {/* Active progress on ring 3 */}
      <circle
        cx={centerX}
        cy={centerY}
        r={ring3Radius}
        fill="none"
        stroke={primaryColor}
        strokeWidth={1}
        strokeDasharray={ring3Circumference}
        strokeDashoffset={ring3Offset}
        strokeLinecap="round"
        opacity={0.4}
        transform={`rotate(-90 ${centerX} ${centerY})`}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`-90 ${centerX} ${centerY}`}
          to={`270 ${centerX} ${centerY}`}
          dur="30s"
          repeatCount="indefinite"
          additive="sum"
        />
      </circle>

      {/* Ring 4 - thin decorative */}
      <circle
        cx={centerX}
        cy={centerY}
        r={ring4Radius}
        fill="none"
        stroke={primaryColor}
        strokeWidth={0.5}
        opacity={0.1}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`360 ${centerX} ${centerY}`}
          to={`0 ${centerX} ${centerY}`}
          dur="20s"
          repeatCount="indefinite"
        />
      </circle>

      {/* Inner ring - thin decorative */}
      <circle
        cx={centerX}
        cy={centerY}
        r={innerRadius}
        fill="none"
        stroke={primaryColor}
        strokeWidth={0.5}
        opacity={0.08}
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from={`0 ${centerX} ${centerY}`}
          to={`360 ${centerX} ${centerY}`}
          dur="25s"
          repeatCount="indefinite"
        />
      </circle>

      {/* Tactical accents - cardinal direction crosshairs */}
      {[0, 90, 180, 270].map((angle) => {
        const rad = (angle * Math.PI) / 180
        const x1 = Math.cos(rad) * (innerRadius - 10)
        const y1 = Math.sin(rad) * (innerRadius - 10)
        const x2 = Math.cos(rad) * (innerRadius + 10)
        const y2 = Math.sin(rad) * (innerRadius + 10)
        return (
          <line
            key={angle}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={primaryColor}
            strokeWidth={1.5}
            opacity={0.4}
            filter="url(#neon-glow)"
          />
        )
      })}

      {/* Cardinal direction dots */}
      {[0, 90, 180, 270].map((angle) => {
        const rad = (angle * Math.PI) / 180
        const x = Math.cos(rad) * (outerRadius + 15)
        const y = Math.sin(rad) * (outerRadius + 15)
        return (
          <circle
            key={`dot-${angle}`}
            cx={x}
            cy={y}
            r={2.5}
            fill={primaryColor}
            filter="url(#neon-glow)"
          />
        )
      })}

      {/* Tactical ticks at 45-degree intervals */}
      {[45, 135, 225, 315].map((angle) => {
        const rad = (angle * Math.PI) / 180
        const x1 = Math.cos(rad) * (outerRadius - 5)
        const y1 = Math.sin(rad) * (outerRadius - 5)
        const x2 = Math.cos(rad) * (outerRadius + 5)
        const y2 = Math.sin(rad) * (outerRadius + 5)
        return (
          <line
            key={`tick-${angle}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={primaryColor}
            strokeWidth={1}
            opacity={0.3}
          />
        )
      })}

      {/* Additional tactical ticks at 22.5-degree intervals */}
      {[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5].map((angle) => {
        const rad = (angle * Math.PI) / 180
        const x1 = Math.cos(rad) * (ring1Radius - 3)
        const y1 = Math.sin(rad) * (ring1Radius - 3)
        const x2 = Math.cos(rad) * (ring1Radius + 3)
        const y2 = Math.sin(rad) * (ring1Radius + 3)
        return (
          <line
            key={`minor-tick-${angle}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={primaryColor}
            strokeWidth={0.5}
            opacity={0.2}
          />
        )
      })}

      {/* Center display */}
      <g>
        {/* Status text - prominent with glow */}
        {status && (
          <text
            x={centerX}
            y={170}
            textAnchor="middle"
            fontFamily="var(--font-mono)"
            fontWeight={900}
            fontSize={size * 0.07}
            letterSpacing="0.2em"
            fill={primaryColor}
            style={{ 
              textShadow: `0 0 10px ${glowColor}, 0 0 20px ${glowColor}, 0 0 30px ${glowColor}`,
              filter: 'url(#text-glow)'
            }}
          >
            {status}
          </text>
        )}

        {/* Main percentage - large and bold */}
        <text
          x={centerX}
          y={status ? 15 : 5}
          textAnchor="middle"
          fontFamily="var(--font-mono)"
          fontWeight={900}
          fontSize={size * 0.28}
          fill="#FFFFFF"
          style={{ 
            textShadow: `0 0 15px ${glowColor}, 0 0 30px ${glowColor}`,
            filter: 'url(#intense-glow)'
          }}
        >
          {clamped.toFixed(0)}%
        </text>

        {/* Label - smaller, inside the rings */}
        {label && (
          <text
            x={centerX}
            y={45}
            textAnchor="middle"
            fontFamily="var(--font-mono)"
            fontWeight={700}
            fontSize={size * 0.055}
            letterSpacing="0.15em"
            fill={primaryColor}
            style={{ 
              textShadow: `0 0 8px ${glowColor}`,
              filter: 'url(#text-glow)'
            }}
          >
            {label}
          </text>
        )}

        {/* Sublabel */}
        {sublabel && (
          <text
            x={centerX}
            y={65}
            textAnchor="middle"
            fontFamily="var(--font-mono)"
            fontWeight={500}
            fontSize={size * 0.04}
            letterSpacing="0.1em"
            fill="rgba(255,255,255,0.5)"
          >
            {sublabel}
          </text>
        )}
      </g>

      {/* Center targeting reticle */}
      <circle
        cx={centerX}
        cy={centerY}
        r={5}
        fill="none"
        stroke={primaryColor}
        strokeWidth={1}
        opacity={0.5}
      />
      <circle
        cx={centerX}
        cy={centerY}
        r={2}
        fill={primaryColor}
        filter="url(#neon-glow)"
      />
      
      {/* Additional nested circles for depth */}
      <circle
        cx={centerX}
        cy={centerY}
        r={15}
        fill="none"
        stroke={primaryColor}
        strokeWidth={0.5}
        opacity={0.15}
      />
      <circle
        cx={centerX}
        cy={centerY}
        r={25}
        fill="none"
        stroke={primaryColor}
        strokeWidth={0.5}
        opacity={0.1}
      />
    </svg>
  )
}

// 3D Glass Canister variant (used for the Confidence / Monte Carlo Ruin
// Survival gauge).
//
// This component previously had three separate bugs that combined into a
// near-empty-looking tube with text bleeding into neighboring cards:
//
// 1. COORDINATE FRAME MISMATCH: the liquid fill's y-position was computed
//    against the outer (pre-translate) coordinate space, then drawn inside
//    a <g> already translated into a different local frame — so the
//    "42% full" math landed near the bottom of the tube no matter what the
//    value was. Fixed by computing fillY/fillH in the SAME local frame the
//    <g> actually draws in (0..height), using the same inset the 25/50/75%
//    tick marks use so the liquid surface lines up with them.
// 2. DANGLING GRADIENT/FILTER IDS: several fills referenced ids that were
//    never defined in this component (`liquid-cylindrical`, `bg-glow`,
//    `neon-glow`, `super-intense-glow`) — some only "worked" by accident
//    because a sibling GaugeRing on the same page happened to define an
//    id with the same name. Every gradient/filter this component uses is
//    now defined locally and namespaced with `uid` so it never depends on
//    what else is mounted on the page.
// 3. TEXT LIVING INSIDE THE SVG VIEWPORT: the value/tier/label text used
//    to be drawn inside the SVG, offset to the right of a viewBox sized
//    only for the tube — with `overflow: visible` that text rendered
//    outside the SVG's own box and bled into whatever card sat next to it.
//    The readout now lives in plain HTML below the tube, styled with CSS
//    text-shadow for the glow, so it can never escape its container.
export function VerticalCapsule({
  value,
  width = 90,
  height = 220,
  color = 'var(--accent-emerald)',
  label,
}: {
  value: number
  width?: number
  height?: number
  color?: string
  label?: string
}) {
  const uid = useId().replace(/[:]/g, '')
  const clamped = Math.max(0, Math.min(100, value))

  // Color zones based on value (cyberpunk neon)
  const getColorForValue = (v: number) => {
    if (v >= 80) return '#00f5d4'
    if (v >= 65) return '#00d4ff'
    if (v >= 50) return '#ffd60a'
    if (v >= 35) return '#ff9e00'
    return '#ff0054'
  }

  const primaryColor = getColorForValue(clamped)
  const secondaryColor = clamped >= 65 ? '#00b4d8' : (clamped >= 50 ? '#ffb703' : '#fb5607')
  const glowColor = primaryColor

  // Get qualitative tier
  const getTier = (v: number) => {
    if (v >= 80) return 'HIGH CONFIDENCE'
    if (v >= 65) return 'GOOD CONFIDENCE'
    if (v >= 50) return 'MODERATE'
    if (v >= 35) return 'LOW CONFIDENCE'
    return 'CRITICAL'
  }

  const tier = getTier(clamped)

  // ---- tube geometry — the SVG holds ONLY the tube; text lives outside it ----
  const pad = 30 // fixed glow-bloom margin; safe regardless of container width since svg is responsive
  const tubeW = Math.max(40, width - 20)
  const capH = tubeW * 0.32 // cap-zone height scales with tube width so the dome always looks proportional

  const tubeX = pad
  const tubeTopY = pad + capH

  const svgW = tubeW + pad * 2
  const svgH = height + capH * 2 + pad * 2

  // Liquid fill level, in the SAME local frame as everything drawn inside
  // the <g> below (0 = top of glass, height = bottom of glass). FILL_INSET
  // matches the inset the 25/50/75% tick marks use, so the liquid surface
  // lines up with those marks at the matching value.
  const FILL_INSET = 15
  const usableH = height - FILL_INSET * 2
  const fillY = FILL_INSET + usableH * (1 - clamped / 100)
  const fillH = height - FILL_INSET - fillY

  const midX = tubeW / 2
  const meniscusPath = `M 3 ${fillY + 8} Q ${midX} ${fillY - 7} ${tubeW - 3} ${fillY + 8} L ${tubeW - 3} ${height - 4} Q ${midX} ${height + 6} 3 ${height - 4} Z`
  const surfacePath = `M ${tubeW * 0.08} ${fillY + 7} Q ${midX} ${fillY - 8} ${tubeW * 0.92} ${fillY + 7}`

  const particles = Array.from({ length: 10 }, (_, i) => ({
    x: 10 + ((i * 37 + 13) % (tubeW - 20)),
    y: fillY + 14 + ((i * 53 + 7) % Math.max(fillH - 28, 1)),
    size: 1.4 + ((i * 7) % 5) * 0.5,
    speed: 1.4 + ((i * 3) % 5) * 0.4,
    wobble: 2 + ((i * 11) % 6),
  }))

  return (
    <div className="qat-capsule">
      <svg
        className="qat-capsule__svg"
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ width: svgW, maxWidth: '100%', height: 'auto', overflow: 'visible', display: 'block' }}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <filter id={`${uid}-liquid-glow`} x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <radialGradient id={`${uid}-bg-glow`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={primaryColor} stopOpacity={0.18} />
            <stop offset="55%" stopColor={primaryColor} stopOpacity={0.05} />
            <stop offset="100%" stopColor={primaryColor} stopOpacity={0} />
          </radialGradient>

          <linearGradient id={`${uid}-liquid`} x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor={primaryColor} stopOpacity={0.85} />
            <stop offset="45%" stopColor={secondaryColor} stopOpacity={1} />
            <stop offset="100%" stopColor={primaryColor} stopOpacity={0.85} />
          </linearGradient>

          {/* Horizontal cylindrical lighting gradient - more pronounced */}
          <linearGradient id={`${uid}-cylindrical-light`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(0,0,0,0.8)" />
            <stop offset="10%" stopColor="rgba(0,0,0,0.4)" />
            <stop offset="20%" stopColor="rgba(0,0,0,0.15)" />
            <stop offset="30%" stopColor="rgba(255,255,255,0.3)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0.5)" />
            <stop offset="70%" stopColor="rgba(255,255,255,0.3)" />
            <stop offset="80%" stopColor="rgba(0,0,0,0.15)" />
            <stop offset="90%" stopColor="rgba(0,0,0,0.4)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.8)" />
          </linearGradient>

          {/* Glass tube with enhanced cylindrical refraction */}
          <linearGradient id={`${uid}-glass-cylindrical`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="6%" stopColor="rgba(255,255,255,0.1)" />
            <stop offset="15%" stopColor="rgba(255,255,255,0.22)" />
            <stop offset="25%" stopColor="rgba(255,255,255,0.35)" />
            <stop offset="35%" stopColor="rgba(255,255,255,0.45)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0.55)" />
            <stop offset="65%" stopColor="rgba(255,255,255,0.45)" />
            <stop offset="75%" stopColor="rgba(255,255,255,0.35)" />
            <stop offset="85%" stopColor="rgba(255,255,255,0.22)" />
            <stop offset="94%" stopColor="rgba(255,255,255,0.1)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.03)" />
          </linearGradient>

          {/* Specular highlight for curved surface */}
          <linearGradient id={`${uid}-curved-specular`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.95)" />
            <stop offset="30%" stopColor="rgba(255,255,255,0.7)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.2)" />
          </linearGradient>

          {/* Metallic cap gradient - enhanced 3D effect */}
          <linearGradient id={`${uid}-metal-cap`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1a1a1a" />
            <stop offset="5%" stopColor="#333333" />
            <stop offset="15%" stopColor="#666666" />
            <stop offset="30%" stopColor="#999999" />
            <stop offset="50%" stopColor="#cccccc" />
            <stop offset="53%" stopColor="#e6e6e6" />
            <stop offset="56%" stopColor="#cccccc" />
            <stop offset="70%" stopColor="#999999" />
            <stop offset="85%" stopColor="#666666" />
            <stop offset="95%" stopColor="#333333" />
            <stop offset="100%" stopColor="#1a1a1a" />
          </linearGradient>

          {/* Cap vertical gradient for 3D depth */}
          <linearGradient id={`${uid}-metal-cap-vertical`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.5)" />
            <stop offset="45%" stopColor="rgba(255,255,255,0.1)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.4)" />
          </linearGradient>

          {/* Cap rim gradient */}
          <linearGradient id={`${uid}-metal-cap-rim`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#000000" />
            <stop offset="50%" stopColor="#444444" />
            <stop offset="100%" stopColor="#000000" />
          </linearGradient>

          {/* Glassmorphic ellipse gradient */}
          <linearGradient id={`${uid}-glassmorphic-ellipse`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={primaryColor} stopOpacity={0.6} />
            <stop offset="50%" stopColor={secondaryColor} stopOpacity={0.85} />
            <stop offset="100%" stopColor={primaryColor} stopOpacity={0.6} />
          </linearGradient>

          {/* Particle gradient for sparkle effect */}
          <radialGradient id={`${uid}-particle-3d`}>
            <stop offset="0%" stopColor="rgba(255,255,255,1)" />
            <stop offset="40%" stopColor="rgba(255,255,255,0.7)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.2)" />
          </radialGradient>

          {/* Drop shadow */}
          <filter id={`${uid}-drop-shadow`} x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feOffset in="blur" dx="6" dy="8" result="offsetBlur" />
            <feFlood floodColor="rgba(0,0,0,0.7)" result="shadowColor" />
            <feComposite in="shadowColor" in2="offsetBlur" operator="in" result="shadow" />
            <feMerge>
              <feMergeNode in="shadow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background radial glow */}
        <ellipse cx={svgW / 2} cy={svgH / 2} rx={svgW} ry={svgH * 0.6} fill={`url(#${uid}-bg-glow)`} />

        {/* Canister container — local frame: (0,0) top of glass, (0,height) bottom of glass */}
        <g transform={`translate(${tubeX}, ${tubeTopY})`}>
          {/* Background shadow */}
          <ellipse cx={tubeW / 2 + 8} cy={height / 2 + 10} rx={tubeW / 2} ry={height / 2} fill="rgba(0,0,0,0.7)" filter={`url(#${uid}-drop-shadow)`} />

          {/* Glass tube body */}
          <rect x={0} y={0} width={tubeW} height={height} rx={tubeW / 2} fill="rgba(5, 5, 15, 0.98)" stroke="rgba(255,255,255,0.15)" strokeWidth={3} />

          {/* Glass cylindrical reflection */}
          <rect x={0} y={0} width={tubeW} height={height} rx={tubeW / 2} fill={`url(#${uid}-glass-cylindrical)`} opacity={0.85} />

          {/* Curved specular highlight */}
          <rect x={tubeW * 0.2} y={0} width={tubeW * 0.2} height={height} rx={tubeW * 0.1} fill={`url(#${uid}-curved-specular)`} opacity={0.75} />

          {/* Secondary curved highlight */}
          <rect x={tubeW * 0.6} y={0} width={tubeW * 0.15} height={height} rx={tubeW * 0.075} fill="rgba(255,255,255,0.25)" opacity={0.65} />

          {/* Liquid fill with cylindrical 3D effect */}
          <g>
            {/* Deep shadow for volume */}
            <path d={meniscusPath} fill="rgba(0,0,0,0.7)" transform="translate(8, 8)" />

            {/* Left dark edge - creates strong cylindrical depth */}
            <path
              d={`M 4 ${fillY + 15} L 4 ${fillY + fillH} Q ${midX} ${fillY + fillH + 10} ${tubeW - 4} ${fillY + fillH} L ${tubeW - 4} ${fillY + fillH} L 4 ${fillY + fillH} Z`}
              fill="rgba(0,0,0,0.7)"
            />

            {/* Right dark edge - creates strong cylindrical depth */}
            <path
              d={`M ${tubeW - 4} ${fillY + 15} L ${tubeW - 4} ${fillY + fillH} L ${tubeW - 4} ${fillY + fillH} Q ${midX} ${fillY + fillH + 10} 4 ${fillY + fillH} L 4 ${fillY + fillH} Z`}
              fill="rgba(0,0,0,0.6)"
            />

            {/* Liquid base with curved meniscus for 3D effect */}
            <path
              d={meniscusPath}
              fill={`url(#${uid}-liquid)`}
              style={{ transition: 'all 800ms ease' }}
              filter={`url(#${uid}-liquid-glow)`}
            />

            {/* Left highlight strip - shows cylindrical curvature */}
            <path
              d={`M 4 ${fillY + 12} L 4 ${fillY + fillH - 10} Q ${midX} ${fillY + fillH} ${tubeW * 0.25} ${fillY + fillH - 10} L ${tubeW * 0.25} ${fillY + 12} Q ${midX} ${fillY + 5} 4 ${fillY + 12} Z`}
              fill="rgba(255,255,255,0.15)"
            />

            {/* Center highlight strip - creates cylindrical volume */}
            <path
              d={`M ${tubeW * 0.3} ${fillY + 10} L ${tubeW * 0.3} ${fillY + fillH - 8} Q ${midX} ${fillY + fillH + 6} ${tubeW * 0.7} ${fillY + fillH - 8} L ${tubeW * 0.7} ${fillY + 10} Q ${midX} ${fillY} ${tubeW * 0.3} ${fillY + 10} Z`}
              fill="rgba(255,255,255,0.35)"
            />

            {/* Right highlight strip - shows cylindrical curvature */}
            <path
              d={`M ${tubeW * 0.75} ${fillY + 12} L ${tubeW * 0.75} ${fillY + fillH - 10} Q ${midX} ${fillY + fillH} ${tubeW - 4} ${fillY + fillH - 10} L ${tubeW - 4} ${fillY + 12} Q ${midX} ${fillY + 5} ${tubeW * 0.75} ${fillY + 12} Z`}
              fill="rgba(255,255,255,0.12)"
            />

            {/* Cylindrical lighting overlay */}
            <path d={meniscusPath} fill={`url(#${uid}-cylindrical-light)`} opacity={0.85} />

            {/* Top highlight curve - creates surface tension effect */}
            <path d={surfacePath} fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth={5} />

            {/* Secondary surface highlight */}
            <path
              d={`M ${tubeW * 0.18} ${fillY + 4} Q ${midX} ${fillY - 5} ${tubeW * 0.82} ${fillY + 4}`}
              fill="none"
              stroke="rgba(255,255,255,0.55)"
              strokeWidth={4}
            />

            {/* Glassmorphic meniscus ellipse - matches water color */}
            <ellipse
              cx={midX}
              cy={fillY + 10}
              rx={midX - 5}
              ry={6}
              fill={`url(#${uid}-glassmorphic-ellipse)`}
              opacity={0.9}
              style={{ transition: 'all 800ms ease' }}
              filter={`url(#${uid}-liquid-glow)`}
            >
              <animate attributeName="opacity" values="0.8;1;0.8" dur="2s" repeatCount="indefinite" />
            </ellipse>

            {/* Meniscus shadow for depth */}
            <ellipse cx={midX} cy={fillY + 12} rx={midX - 7} ry={5} fill="rgba(0,0,0,0.6)" />
          </g>

          {/* Sparkles/particles in liquid */}
          {fillH > 40 && particles.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={p.size} fill={`url(#${uid}-particle-3d)`} opacity={0.85}>
              <animate attributeName="cy" values={`${p.y};${p.y - 20};${p.y}`} dur={`${p.speed}s`} repeatCount="indefinite" begin={`${i * 0.15}s`} />
              <animate attributeName="cx" values={`${p.x};${p.x + p.wobble};${p.x - p.wobble};${p.x}`} dur={`${p.speed}s`} repeatCount="indefinite" begin={`${i * 0.15}s`} />
              <animate attributeName="opacity" values="0.7;1;0.7" dur={`${p.speed}s`} repeatCount="indefinite" begin={`${i * 0.15}s`} />
            </circle>
          ))}

          {/* Curved surface meniscus */}
          {fillH > 20 && (
            <g>
              {/* Meniscus shadow */}
              <ellipse cx={midX} cy={fillY + 9} rx={midX - 7} ry={5.5} fill="rgba(0,0,0,0.6)" />
              {/* Meniscus base */}
              <ellipse cx={midX} cy={fillY + 7} rx={midX - 7} ry={5} fill={primaryColor} opacity={0.5} />
              {/* Meniscus highlight curve */}
              <ellipse cx={midX} cy={fillY + 5} rx={midX - 8} ry={3} fill="rgba(255,255,255,0.75)" />
              {/* Specular on meniscus */}
              <ellipse cx={midX - tubeW * 0.25} cy={fillY + 4} rx={tubeW * 0.15} ry={1.8} fill="rgba(255,255,255,1)" />
            </g>
          )}

          {/* Top metallic cap - enhanced 3D */}
          <g>
            {/* Cap shadow */}
            <rect x={-7} y={-10} width={tubeW + 14} height={22} rx={6} fill="rgba(0,0,0,0.7)" />

            {/* Main cap body */}
            <rect x={-4} y={-12} width={tubeW + 8} height={22} rx={6} fill={`url(#${uid}-metal-cap)`} stroke="rgba(255,255,255,0.25)" strokeWidth={3} />

            {/* Vertical lighting overlay */}
            <rect x={-4} y={-12} width={tubeW + 8} height={22} rx={6} fill={`url(#${uid}-metal-cap-vertical)`} opacity={0.75} />

            {/* Cap rim highlight */}
            <rect x={-4} y={-12} width={tubeW + 8} height={22} rx={6} fill="none" stroke={`url(#${uid}-metal-cap-rim)`} strokeWidth={4} opacity={0.95} />

            {/* Top highlight ridge */}
            <rect x={-3} y={-11} width={tubeW + 6} height={5} rx={2.5} fill="rgba(255,255,255,0.6)" />

            {/* Center highlight line */}
            <rect x={tubeW * 0.35} y={-10} width={tubeW * 0.3} height={14} rx={2} fill="rgba(255,255,255,0.25)" />

            {/* Screw detail */}
            <circle cx={midX} cy={-1} r={5} fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
            <line x1={midX - 3} y1={-1} x2={midX + 3} y2={-1} stroke="rgba(255,255,255,0.35)" strokeWidth={1} />
          </g>

          {/* Bottom metallic cap - enhanced 3D */}
          <g>
            {/* Cap shadow */}
            <rect x={-7} y={height - 9} width={tubeW + 14} height={22} rx={6} fill="rgba(0,0,0,0.7)" />

            {/* Main cap body */}
            <rect x={-4} y={height - 10} width={tubeW + 8} height={22} rx={6} fill={`url(#${uid}-metal-cap)`} stroke="rgba(255,255,255,0.25)" strokeWidth={3} />

            {/* Vertical lighting overlay */}
            <rect x={-4} y={height - 10} width={tubeW + 8} height={22} rx={6} fill={`url(#${uid}-metal-cap-vertical)`} opacity={0.75} />

            {/* Cap rim highlight */}
            <rect x={-4} y={height - 10} width={tubeW + 8} height={22} rx={6} fill="none" stroke={`url(#${uid}-metal-cap-rim)`} strokeWidth={4} opacity={0.95} />

            {/* Bottom highlight ridge */}
            <rect x={-3} y={height - 5} width={tubeW + 6} height={5} rx={2.5} fill="rgba(255,255,255,0.55)" />

            {/* Center highlight line */}
            <rect x={tubeW * 0.35} y={height - 9} width={tubeW * 0.3} height={14} rx={2} fill="rgba(255,255,255,0.22)" />

            {/* Screw detail */}
            <circle cx={midX} cy={height + 1} r={5} fill="rgba(0,0,0,0.5)" stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
            <line x1={midX - 3} y1={height + 1} x2={midX + 3} y2={height + 1} stroke="rgba(255,255,255,0.35)" strokeWidth={1} />
          </g>

          {/* Measurement marks */}
          {[25, 50, 75].map((mark) => {
            const y = FILL_INSET + usableH * (1 - mark / 100)
            return (
              <g key={mark}>
                <line x1={tubeW - 10} y1={y} x2={tubeW - 4} y2={y} stroke="rgba(255,255,255,0.45)" strokeWidth={1.5} />
                <text x={tubeW - 15} y={y + 5} fontFamily="var(--font-mono)" fontSize={11} fill="rgba(255,255,255,0.65)" textAnchor="end" fontWeight="bold">
                  {mark}%
                </text>
              </g>
            )
          })}

          {/* Decorative rings */}
          {[0.12, 0.3, 0.55, 0.8].map((pos, i) => (
            <ellipse key={i} cx={midX} cy={height * pos} rx={midX - 2} ry={5} fill="none" stroke={primaryColor} strokeWidth={2} opacity={0.35}>
              <animate attributeName="opacity" values="0.25;0.5;0.25" dur={`${1.8 + i * 0.4}s`} repeatCount="indefinite" />
            </ellipse>
          ))}
        </g>
      </svg>

      {/* Readout — plain HTML/CSS, not SVG, so long tier labels wrap and glow
          with a CSS text-shadow instead of risking SVG-viewport clipping */}
      <div className="qat-capsule__readout">
        <div className="qat-capsule__value" style={{ textShadow: `0 0 14px ${glowColor}, 0 0 28px ${glowColor}` }}>
          {clamped.toFixed(0)}%
        </div>
        <div className="qat-capsule__tier" style={{ color: primaryColor, textShadow: `0 0 10px ${primaryColor}, 0 0 20px ${primaryColor}` }}>
          {tier}
        </div>
        {label && <div className="qat-capsule__label">{label}</div>}
      </div>
    </div>
  )
}

import { useRef, useMemo, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Target, Shield, TrendingUp, Activity, Droplet, BarChart2, Newspaper, Brain, LucideIcon } from 'lucide-react'
import { Badge, toneForWord } from './ui/Primitives'

// Fibonacci-sphere point distribution — even coverage without pole clustering.
function useSpherePositions(count: number, radius: number) {
  return useMemo(() => {
    const positions = new Float32Array(count * 3)
    const offset = 2 / count
    const increment = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < count; i++) {
      const y = i * offset - 1 + offset / 2
      const r = Math.sqrt(Math.max(0, 1 - y * y))
      const phi = i * increment
      positions[i * 3] = Math.cos(phi) * r * radius
      positions[i * 3 + 1] = y * radius
      positions[i * 3 + 2] = Math.sin(phi) * r * radius
    }
    return positions
  }, [count, radius])
}

function ParticleSphere() {
  const pointsRef = useRef<THREE.Points>(null)
  // Denser field (2200 vs the old 900) so the sphere reads as a solid,
  // data-dense "core" rather than a sparse dot cloud.
  const positions = useSpherePositions(2200, 1.55)

  useFrame((state, delta) => {
    if (!pointsRef.current) return
    pointsRef.current.rotation.y += delta * 0.12
    pointsRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.25) * 0.15
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.022}
        color="#00f5d4"
        transparent
        opacity={0.95}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Sparse long-line lat/long-style wireframe sitting just inside the particle
// field — gives the "network globe" mesh read the mockup has, instead of a
// pure point cloud with no connective structure.
function WireframeMesh() {
  const meshRef = useRef<THREE.LineSegments>(null)
  const geometry = useMemo(() => new THREE.IcosahedronGeometry(1.42, 2), [])
  const wireframe = useMemo(() => new THREE.WireframeGeometry(geometry), [geometry])

  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y -= delta * 0.05
  })

  return (
    <lineSegments ref={meshRef} geometry={wireframe}>
      <lineBasicMaterial color="#00d2ff" transparent opacity={0.18} depthWrite={false} blending={THREE.AdditiveBlending} />
    </lineSegments>
  )
}

// Soft additive-glow core behind the particles — the bright "hot center"
// visible in the mockup, absent from the flat point cloud.
function GlowCore() {
  return (
    <mesh scale={1.35}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshBasicMaterial color="#00f5d4" transparent opacity={0.08} depthWrite={false} blending={THREE.AdditiveBlending} />
    </mesh>
  )
}

function OrbitRing({ radius, tilt, color, speed }: { radius: number; tilt: number; color: string; speed: number }) {
  const ringRef = useRef<THREE.LineSegments>(null)
  const geometry = useMemo(() => {
    const pts: THREE.Vector3[] = []
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2
      pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius))
    }
    return new THREE.BufferGeometry().setFromPoints(pts)
  }, [radius])

  useFrame((_, delta) => {
    if (ringRef.current) ringRef.current.rotation.y += delta * speed
  })

  return (
    <lineSegments ref={ringRef} geometry={geometry} rotation={[tilt, 0, 0]}>
      <lineBasicMaterial color={color} transparent opacity={0.35} blending={THREE.AdditiveBlending} />
    </lineSegments>
  )
}

function GlobeScene() {
  return (
    <>
      <ambientLight intensity={0.6} />
      <GlowCore />
      <WireframeMesh />
      <ParticleSphere />
      <OrbitRing radius={2.05} tilt={0.4} color="#00d2ff" speed={0.05} />
      <OrbitRing radius={2.35} tilt={-0.55} color="#a855f7" speed={-0.035} />
    </>
  )
}

export interface GlobeNodeData {
  execution: string
  sentiment: string
  risk_engine: string
  trend: string
  momentum: string
  liquidity: string
  volume: string
  news_feed: string
}

// Angles run clockwise from the top (-90deg) at 45deg steps, one per node.
// Position is computed from the measured container box (not fixed % corners),
// so satellites never collide regardless of card width — see index.css .qat-globe.
const SATELLITES: { key: keyof GlobeNodeData; label: string; icon: LucideIcon; angle: number }[] = [
  { key: 'execution', label: 'EXECUTION', icon: Target, angle: -90 },
  { key: 'risk_engine', label: 'RISK ENGINE', icon: Shield, angle: -45 },
  { key: 'trend', label: 'TREND', icon: TrendingUp, angle: 0 },
  { key: 'momentum', label: 'MOMENTUM', icon: Activity, angle: 45 },
  { key: 'liquidity', label: 'LIQUIDITY', icon: Droplet, angle: 90 },
  { key: 'volume', label: 'VOLUME', icon: BarChart2, angle: 135 },
  { key: 'news_feed', label: 'NEWS FEED', icon: Newspaper, angle: 180 },
  { key: 'sentiment', label: 'SENTIMENT', icon: Brain, angle: 225 },
]

export function QuantAlphaGlobe({ nodes }: { nodes: GlobeNodeData | null }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setBox({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Half-node footprint reserved so the badge itself never clips the card edge.
  const radiusX = Math.max(0, box.w / 2 - 62)
  const radiusY = Math.max(0, box.h / 2 - 30)

  return (
    <div ref={containerRef} className="qat-globe">
      <div className="qat-globe__canvas">
        <Canvas camera={{ position: [0, 0, 5.2], fov: 42 }} dpr={[1, 1.5]}>
          <GlobeScene />
        </Canvas>
      </div>
      {box.w > 0 &&
        SATELLITES.map((sat) => {
          const rad = (sat.angle * Math.PI) / 180
          const x = Math.cos(rad) * radiusX
          const y = Math.sin(rad) * radiusY
          const Icon = sat.icon
          const value = nodes ? nodes[sat.key] : '—'
          return (
            <div
              key={sat.key}
              className="qat-globe__node"
              style={{ transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))` }}
            >
              <Icon size={13} className="qat-globe__node-icon" />
              <span className="qat-globe__node-label">{sat.label}</span>
              <Badge tone={nodes ? toneForWord(value) : 'neutral'}>{value}</Badge>
            </div>
          )
        })}
    </div>
  )
}

import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import './primitives.css'

// ---------------------------------------------------------------------------
// Card — the base panel surface used everywhere
// ---------------------------------------------------------------------------
export function Card({
  title,
  subtitle,
  children,
  className = '',
  glow = false,
  actions,
}: {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
  glow?: boolean
  actions?: ReactNode
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24 }}
      className={`qat-card ${glow ? 'qat-card--glow' : ''} ${className}`}
    >
      {(title || actions) && (
        <div className="qat-card__head">
          <div>
            {title && <h3 className="qat-card__title">{title}</h3>}
            {subtitle && <p className="qat-card__subtitle">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className="qat-card__body">{children}</div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// StatTile — a single mono-font numeric readout with label + optional delta
// ---------------------------------------------------------------------------
export function StatTile({
  label,
  value,
  delta,
  deltaGood,
  hint,
  size = 'md',
}: {
  label: string
  value: string
  delta?: string
  deltaGood?: boolean
  hint?: string
  size?: 'sm' | 'md' | 'lg'
}) {
  return (
    <div className={`qat-stat qat-stat--${size}`} title={hint}>
      <div className="qat-stat__label">{label}</div>
      <div className="qat-stat__value">{value}</div>
      {delta && (
        <div className={`qat-stat__delta ${deltaGood ? 'is-good' : 'is-bad'}`}>{delta}</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Badge — small status pill
// ---------------------------------------------------------------------------
type BadgeTone = 'neutral' | 'good' | 'bad' | 'warn' | 'info'
export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: BadgeTone }) {
  return <span className={`qat-badge qat-badge--${tone}`}>{children}</span>
}

export function toneForWord(word: string): BadgeTone {
  const w = word.toUpperCase()
  if (['OPTIMAL', 'ONLINE', 'BULLISH', 'ACTIVE', 'NOMINAL', 'STRONG', 'GOOD', 'EXCELLENT', 'PASS', 'SUCCESS', 'DONE', 'ELEVATED_GOOD'].includes(w))
    return 'good'
  if (['BREACH', 'PAUSED', 'BEARISH', 'CRITICAL', 'ALERT', 'FAIL', 'FAILURE'].includes(w)) return 'bad'
  if (['ELEVATED', 'CAUTION', 'WARNING', 'PENDING', 'QUEUED', 'IN_PROGRESS'].includes(w)) return 'warn'
  return 'neutral'
}

// ---------------------------------------------------------------------------
// Empty / Loading / Error states — used instead of ever faking a number
// ---------------------------------------------------------------------------
export function EmptyState({ label = 'NO DATA' }: { label?: string }) {
  return <div className="qat-empty">{label}</div>
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return <div className="qat-loading">{label}</div>
}

export function ErrorState({ message }: { message: string }) {
  return <div className="qat-error">⚠ {message}</div>
}

// ---------------------------------------------------------------------------
// PageHeader — consistent per-page title row
// ---------------------------------------------------------------------------
export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="qat-pageheader">
      <div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {actions}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Grid — simple responsive grid wrapper
// ---------------------------------------------------------------------------
export function Grid({ cols = 12, gap = 16, children }: { cols?: number; gap?: number; children: ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap }}>{children}</div>
  )
}

export function Span({ n, children }: { n: number; children: ReactNode }) {
  // display:flex so the Card inside can stretch (height:100%) to match the
  // tallest card in the same grid row, instead of every card sizing to its
  // own content and leaving ragged bottom edges across a row.
  return <div style={{ gridColumn: `span ${n}`, display: 'flex' }}>{children}</div>
}

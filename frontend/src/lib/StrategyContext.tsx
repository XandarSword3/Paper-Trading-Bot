import { createContext, useContext, ReactNode, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

// Global strategy selection, mirrored into the URL (?strategy=v4) so every
// analytical page is linkable/shareable and stays in sync with the header
// selector, per the build plan's "strategy selector becomes global state"
// requirement.
interface StrategyCtx {
  strategyId: string
  setStrategyId: (id: string) => void
}

const Ctx = createContext<StrategyCtx | null>(null)

export function StrategyProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const strategyId = params.get('strategy') || 'v4'

  const value = useMemo<StrategyCtx>(
    () => ({
      strategyId,
      setStrategyId: (id: string) => {
        const next = new URLSearchParams(params)
        next.set('strategy', id)
        setParams(next, { replace: true })
      },
    }),
    [strategyId, params, setParams]
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useStrategy() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useStrategy must be used within StrategyProvider')
  return ctx
}

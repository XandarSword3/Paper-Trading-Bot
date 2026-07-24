import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrategyProvider } from './lib/StrategyContext'
import Shell from './components/Shell'
import Command from './pages/Command'
import LiveTerminal from './pages/LiveTerminal'
import PaperTrading from './pages/PaperTrading'
import Alerts from './pages/Alerts'
import Strategies from './pages/Strategies'
import Backtests from './pages/Backtests'
import Risk from './pages/Risk'
import RiskTerrain from './pages/RiskTerrain'
import Reports from './pages/Reports'
import Readiness from './pages/Readiness'
import Settings from './pages/Settings'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 15_000,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <StrategyProvider>
          <Routes>
            <Route path="/" element={<Shell />}>
              <Route index element={<Navigate to="/command" replace />} />
              <Route path="command" element={<Command />} />
              <Route path="live-terminal" element={<LiveTerminal />} />
              <Route path="paper-trading" element={<PaperTrading />} />
              <Route path="alerts" element={<Alerts />} />
              <Route path="strategies" element={<Strategies />} />
              <Route path="backtests" element={<Backtests />} />
              <Route path="risk" element={<Risk />} />
              <Route path="risk-terrain" element={<RiskTerrain />} />
              <Route path="reports" element={<Reports />} />
              <Route path="readiness" element={<Readiness />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </StrategyProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App

import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { OverviewPage } from './pages/OverviewPage'
import { RiskPage } from './pages/RiskPage'
import { StockPage } from './pages/StockPage'
import { SentimentPage } from './pages/SentimentPage'
import { ResearchPage } from './pages/ResearchPage'

export default function App() {
  return <AppShell><Routes>
    <Route path="/" element={<OverviewPage />} />
    <Route path="/risk" element={<RiskPage />} />
    <Route path="/stock" element={<StockPage />} />
    <Route path="/sentiment" element={<SentimentPage />} />
    <Route path="/research" element={<ResearchPage />} />
    <Route path="*" element={<OverviewPage />} />
  </Routes></AppShell>
}

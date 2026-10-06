import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Activity, ArrowUpRight, BookOpen, Gauge, Menu, MessageSquareText, ShieldCheck, TrendingUp, Wifi, WifiOff, X } from 'lucide-react'
import { api, type Health } from '../api'

const navigation = [
  { to: '/', label: 'Overview', icon: Gauge },
  { to: '/risk', label: 'Risk intelligence', icon: ShieldCheck },
  { to: '/stock', label: 'Market prediction', icon: TrendingUp },
  { to: '/sentiment', label: 'Financial sentiment', icon: MessageSquareText },
  { to: '/research', label: 'AI research', icon: BookOpen },
]
const pageNames: Record<string, string> = {
  '/': 'Your intelligence workspace', '/risk': 'Risk intelligence', '/stock': 'Market prediction',
  '/sentiment': 'Financial sentiment', '/research': 'Financial research',
}

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  useEffect(() => {
    let active = true
    const check = () => api.health().then(value => { if (active) { setHealth(value); setHealthError(false) } }).catch(() => { if (active) { setHealth(null); setHealthError(true) } })
    check()
    const timer = window.setInterval(check, 30000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  useEffect(() => setMenuOpen(false), [location.pathname])
  const title = pageNames[location.pathname] ?? pageNames['/']
  return <div className="app-frame">
    <div className="ambient ambient-a"/><div className="ambient ambient-b"/>
    <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`}>
      <NavLink to="/" className="brand" aria-label="FinSight AI overview"><span className="brand-symbol"><Activity size={19}/></span><span className="brand-word">FinSight<span> AI</span><small>FINANCIAL INTELLIGENCE</small></span></NavLink>
      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="primary-nav" aria-label="Main navigation">{navigation.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><Icon size={17}/><span>{label}</span><ArrowUpRight className="nav-arrow" size={14}/></NavLink>)}</nav>
      <div className="sidebar-bottom"><div className="status-card"><span className={`status-dot ${health ? 'online' : ''}`}/><span><strong>{health ? 'Systems operational' : healthError ? 'API unavailable' : 'Connecting to API'}</strong><small>{health ? health.service : 'Backend connection'}</small></span></div><p>Models inform.<br/>People decide.</p></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><button className="mobile-menu" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={20}/> : <Menu size={20}/>}</button><div><div className="topbar-kicker">FINSIGHT AI <span>/</span> WORKSPACE</div><div className="topbar-title">{title}</div></div><div className={`connection ${health ? 'connected' : ''}`}><span className="connection-icon">{health ? <Wifi size={15}/> : <WifiOff size={15}/>}</span><span>{health ? 'API connected' : 'API status'}</span></div></header>
      <div className="page-content" key={location.pathname}>{children}</div>
      <footer className="app-footer"><span>FinSight AI</span><i/> Research and analysis tools <i/> For informational use only</footer>
    </main>
    {menuOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)}/>}
  </div>
}

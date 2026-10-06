import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Activity, ArrowRight, ArrowUpRight, BookOpen, BrainCircuit, ChartNoAxesCombined, ShieldCheck, Sparkles } from 'lucide-react'
import { api, type Health } from '../api'

const capabilities = [
  { to: '/risk', title: 'Risk intelligence', desc: 'Explore bankruptcy probability and the factors shaping the model result.', model: 'XGBoost · SHAP', icon: ShieldCheck, num: '01' },
  { to: '/stock', title: 'Market prediction', desc: 'Review an AAPL next-day return estimate from the trained sequence model.', model: 'AAPL · LSTM', icon: ChartNoAxesCombined, num: '02' },
  { to: '/sentiment', title: 'Financial sentiment', desc: 'Understand the tone expressed in company and market language.', model: 'FinBERT', icon: BrainCircuit, num: '03' },
  { to: '/research', title: 'AI research', desc: 'Ask questions grounded in passages from Apple’s annual report.', model: 'RAG · Gemini', icon: BookOpen, num: '04' },
]
export function OverviewPage() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => { let active = true; api.health().then(v => { if (active) { setHealth(v); setError(false) } }).catch(() => { if (active) setError(true) }); return () => { active = false } }, [])
  return <div className="overview-page">
    <section className="hero-card">
      <div className="hero-orb orb-one"/><div className="hero-orb orb-two"/>
      <div className="hero-copy"><div className="hero-kicker"><span className="kicker-mark"><Sparkles size={13}/></span> AI FINANCIAL INTELLIGENCE <i/></div>
        <h1>Clarity for the<br/><em>moments that matter.</em></h1>
        <p>Bring company risk, market signals, financial language, and reported results into one thoughtful research workspace.</p>
        <NavLink className="button-primary hero-cta" to="/research">Explore financial research<ArrowRight size={16}/></NavLink>
        <div className="hero-health"><span className={`status-dot ${health ? 'online' : ''}`}/><span>{health ? `${health.service} is ready` : error ? 'Backend could not be reached' : 'Connecting to backend'}</span></div>
      </div>
      <div className="hero-visual" aria-hidden="true"><div className="visual-halo halo-outer"/><div className="visual-halo halo-inner"/><div className="visual-core"><span><Activity size={29}/></span><small>FINANCIAL<br/>SIGNALS</small></div><span className="visual-node node-risk"><ShieldCheck size={17}/></span><span className="visual-node node-market"><ChartNoAxesCombined size={17}/></span><span className="visual-node node-research"><BookOpen size={17}/></span><div className="visual-caption">SIGNAL <b>→</b> CONTEXT</div></div>
      <div className="hero-index">01 <span>—</span> 04</div>
    </section>
    <section className="capabilities-section"><div className="section-heading"><div><div className="eyebrow">YOUR INTELLIGENCE TOOLKIT</div><h2>Four ways to see the full picture.</h2></div><span className="section-aside">BUILT AROUND THE DATA</span></div>
      <div className="capability-grid">{capabilities.map(({ to, title, desc, model, icon: Icon, num }) => <NavLink to={to} key={to} className="capability-card"><div className="cap-top"><span className="cap-icon"><Icon size={19}/></span><span className="cap-number">{num}</span></div><h3>{title}</h3><p>{desc}</p><div className="cap-bottom"><span>{model}</span><ArrowUpRight size={16}/></div></NavLink>)}</div>
    </section>
    <div className="disclosure"><span><Activity size={16}/></span><p>Model outputs are experimental research signals. They are not investment recommendations or financial advice.</p></div>
  </div>
}

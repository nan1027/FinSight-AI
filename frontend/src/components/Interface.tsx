import type { ReactNode } from 'react'
import { ArrowRight, CircleAlert, LoaderCircle, Sparkles, type LucideIcon } from 'lucide-react'

export function PageIntro({ label, title, text, icon: Icon }: { label: string; title: string; text: string; icon: LucideIcon }) {
  return <div className="page-intro"><span className="intro-icon"><Icon size={20}/></span><div><div className="eyebrow"><Sparkles size={12}/>{label}</div><h1>{title}</h1><p>{text}</p></div></div>
}
export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) { return <section className={`panel ${className}`}>{children}</section> }
export function SubmitButton({ loading, children }: { loading: boolean; children: string }) {
  return <button className="button-primary" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={16}/>Working…</> : <>{children}<ArrowRight size={16}/></>}</button>
}
export function Alert({ error }: { error: string }) { return <div className="alert-error" role="alert"><CircleAlert size={17}/><span>{error}</span></div> }
export function Metadata({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(([, value]) => value != null && value !== '')
  return <>{entries.length ? entries.map(([key, value]) => <span key={key} className="metadata-item"><b>{key.replace(/_/g, ' ')}</b>{String(value)}</span>) : <span className="metadata-item">Annual report</span>}</>
}

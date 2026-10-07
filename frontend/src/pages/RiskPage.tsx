import { useMemo, useState, type FormEvent } from 'react'
import { ShieldCheck, Sparkles } from 'lucide-react'
import metadata from '../riskModelMetadata.json'
import { api, type RiskResult } from '../api'
import { Alert, PageIntro, Panel, SubmitButton } from '../components/Interface'

const features = (metadata as { features: string[] }).features
const groups: [string, RegExp][] = [['Profitability & margins', /ROA|Margin|Profit Rate|Interest Rate|Net profit|Profit Growth|Net Profit|EPS|Income|Sales Gross|expense|DFL|Interest Coverage/i], ['Liquidity & working capital', /Current|Quick|Cash|Working Capital|Inventory|Receivable|Collection|Turnover/i], ['Debt & capital structure', /Debt|Liability|Equity|Borrow|fund suitability|Contingent|Net worth|Assets/i], ['Growth & operations', /Growth|Revenue|Operating|Asset Turnover|Person|Allocation|GNP|Reinvestment|Per Share|CFO|Funds to/i]]
export function RiskPage() {
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<RiskResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const grouped = useMemo(() => {
    const rows = groups.map(([label, pattern], index) => ({ label, items: features.filter(f => pattern.test(f) && !groups.slice(0, index).some(([, previous]) => previous.test(f))) }))
    const assigned = new Set(rows.flatMap(row => row.items))
    rows.push({ label: 'Other model features', items: features.filter(f => !assigned.has(f)) })
    return rows
  }, [])
  async function submit(event: FormEvent) {
    event.preventDefault(); setError(''); setResult(null)
    const missing = features.filter(feature => values[feature] === undefined || values[feature].trim() === '')
    if (missing.length) { setError(`Enter a numeric value for all ${features.length} model inputs. ${missing.length} remaining.`); return }
    const parsed: Record<string, number> = {}
    for (const feature of features) { const value = Number(values[feature]); if (!Number.isFinite(value)) { setError(`“${feature}” must be a finite number.`); return } parsed[feature] = value }
    setLoading(true); try { setResult(await api.risk(parsed)) } catch (cause) { setError((cause as Error).message) } finally { setLoading(false) }
  }
  const probability = result ? Math.max(0, Math.min(100, result.bankruptcy_probability * 100)) : 0
  return <><PageIntro label="XGBOOST CLASSIFIER · SHAP EXPLANATIONS" title="See the factors behind risk." text="Enter values for the model’s expected financial indicators to review its bankruptcy probability and feature attribution." icon={ShieldCheck}/>
    <div className="risk-layout"><form onSubmit={submit} className="risk-form"><Panel><div className="panel-heading"><div><div className="eyebrow">COMPANY INPUTS</div><h2>Financial indicators</h2></div><span className="count-pill">{features.length} FEATURES</span></div><p className="helper">Use the source dataset’s feature scale. Ratio fields accept decimals; all values are sent to the existing model API.</p>
      {grouped.map((group, index) => <details className="feature-group" key={group.label} open={index === 0}><summary><span>{group.label}</span><small>{group.items.length} inputs</small></summary><div className="feature-grid">{group.items.map(feature => <label className="field" key={feature}><span title={feature}>{feature.trim()}</span><input type="number" step="any" value={values[feature] ?? ''} onChange={e => setValues(current => ({ ...current, [feature]: e.target.value }))} placeholder="Enter value" required aria-label={feature}/></label>)}</div></details>)}
      {error && <Alert error={error}/>}<div className="form-actions"><button type="button" className="button-secondary" onClick={() => { setValues({}); setResult(null); setError('') }}>Clear inputs</button><SubmitButton loading={loading}>Analyze risk</SubmitButton></div>
    </Panel></form><div className="result-column"><Panel className="risk-result-card"><div className="result-overline"><span className="eyebrow">MODEL OUTPUT</span><span className="result-live"><i/>INFERENCE</span></div>{result ? <><div className="risk-dial" style={{ '--risk-value': `${probability}%` } as React.CSSProperties}><div className="risk-dial-inner"><strong>{probability.toFixed(1)}<small>%</small></strong><span>BANKRUPTCY<br/>PROBABILITY</span></div></div><span className={`risk-level ${result.risk_level.toLowerCase()}`}>{result.risk_level} risk</span><div className="risk-meter"><span style={{ width: `${probability}%` }}/></div><p className="helper">Probability and category returned by the trained classifier.</p></> : <div className="empty-result"><span className="empty-icon"><ShieldCheck size={23}/></span><h3>Your risk profile</h3><p>Complete the model inputs to see its probability estimate and explanation.</p><div className="risk-scale"><span className="low">LOW</span><span className="medium">MEDIUM</span><span className="high">HIGH</span></div></div>}</Panel>
      {result ? <Panel className="contribution-panel"><div className="eyebrow"><Sparkles size={12}/> WHY THIS PREDICTION?</div><h2 className="panel-title">Top SHAP contributors</h2><p className="helper">Contribution values are returned by the model. Positive or negative indicates direction in the model output.</p><div className="contributor-list">{result.top_contributors.map(item => <div className="contributor" key={item.feature}><span title={item.feature}>{item.feature.trim()}</span><strong className={item.shap_value >= 0 ? 'positive-shap' : 'negative-shap'}>{item.shap_value >= 0 ? '+' : ''}{item.shap_value.toFixed(4)}</strong></div>)}</div></Panel> : <div className="why-card"><span className="why-icon"><Sparkles size={16}/></span><div><strong>Why this prediction?</strong><p>After inference, the model’s SHAP values show which provided features contributed most to its result.</p></div></div>}
      <div className="small-note">Risk scores are statistical estimates and should be interpreted with appropriate context.</div></div></div>
  </>
}

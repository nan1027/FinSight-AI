import { useState, type FormEvent } from 'react'
import { Activity, ArrowDownRight, ArrowUpRight, ChartNoAxesCombined, TrendingUp } from 'lucide-react'
import { api, type StockResult } from '../api'
import { Alert, PageIntro, Panel, SubmitButton } from '../components/Interface'

function ModelStepChart({ result }: { result: StockResult }) {
  const values = [result.latest_close, result.predicted_next_close]
  const min = Math.min(...values); const max = Math.max(...values); const spread = max - min || Math.abs(max) * 0.02 || 1
  const y = (value: number) => 156 - ((value - (min - spread * .35)) / (spread * 1.7)) * 116
  const positive = result.predicted_next_day_return >= 0
  return <div className="stock-chart-wrap"><div className="chart-legend"><span><i className="legend-observed"/>Latest close</span><span><i className="legend-predicted"/>Model estimate</span></div><svg className="stock-chart" viewBox="0 0 680 220" role="img" aria-label={`Latest close ${result.latest_close.toFixed(2)} and predicted next close ${result.predicted_next_close.toFixed(2)}`}>
    <defs><linearGradient id="estimate-line" x1="0" x2="1"><stop offset="0%" stopColor="#a7a2ff"/><stop offset="100%" stopColor="#c5a5ff"/></linearGradient><linearGradient id="estimate-area" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#9277f8" stopOpacity=".19"/><stop offset="100%" stopColor="#9277f8" stopOpacity="0"/></linearGradient></defs>
    {[40, 80, 120, 160].map(line => <line key={line} x1="48" x2="632" y1={line} y2={line} className="chart-gridline"/>)}
    <path d={`M100 ${y(result.latest_close)} L580 ${y(result.predicted_next_close)} L580 178 L100 178 Z`} fill="url(#estimate-area)"/>
    <line x1="100" x2="580" y1={y(result.latest_close)} y2={y(result.predicted_next_close)} stroke="url(#estimate-line)" strokeWidth="3" strokeDasharray={positive ? '0' : '8 7'} strokeLinecap="round"/>
    <circle cx="100" cy={y(result.latest_close)} r="6" fill="#b6b4ff" stroke="#18182b" strokeWidth="4"/><circle cx="580" cy={y(result.predicted_next_close)} r="7" fill="#c5a5ff" stroke="#18182b" strokeWidth="4"/>
    <text x="100" y="202" textAnchor="middle" className="chart-label">LATEST CLOSE</text><text x="580" y="202" textAnchor="middle" className="chart-label">NEXT-DAY ESTIMATE</text>
    <text x="100" y={y(result.latest_close) - 16} textAnchor="middle" className="chart-value">${result.latest_close.toFixed(2)}</text><text x="580" y={y(result.predicted_next_close) - 16} textAnchor="middle" className="chart-value">${result.predicted_next_close.toFixed(2)}</text>
  </svg><div className="chart-footnote"><Activity size={13}/> Two observed/model values from the API response; historical daily points are not provided.</div></div>
}

export function StockPage() {
  const [ticker, setTicker] = useState('AAPL'); const [result, setResult] = useState<StockResult | null>(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(false)
  async function submit(event: FormEvent) { event.preventDefault(); setError(''); setResult(null); setLoading(true); try { setResult(await api.stock(ticker.trim().toUpperCase())) } catch (cause) { setError((cause as Error).message) } finally { setLoading(false) } }
  const positive = (result?.predicted_next_day_return ?? 0) >= 0
  return <><PageIntro label="MULTIVARIATE RETURN-TARGET LSTM" title="A clearer view of the next day." text="Explore the trained model’s estimate alongside its latest observed close. This view reflects the data available from the stock API." icon={TrendingUp}/>
    <div className="stock-workspace"><Panel className="ticker-panel"><div className="ticker-identity"><span className="ticker-monogram">{result?.ticker?.slice(0, 1) ?? ticker.slice(0, 1).toUpperCase()}</span><div><div className="eyebrow">STOCK INTELLIGENCE</div><h2>{result?.ticker ?? ticker.toUpperCase()}<span className="exchange-tag">MODEL VIEW</span></h2></div></div><form className="ticker-form" onSubmit={submit}><label className="field"><span>Company ticker</span><input value={ticker} onChange={e => setTicker(e.target.value)} placeholder="AAPL" required aria-label="Stock ticker"/></label><SubmitButton loading={loading}>Run prediction</SubmitButton></form>{error && <Alert error={error}/>}</Panel>
      {result ? <><Panel className="stock-analysis"><div className="stock-analysis-head"><div><div className="eyebrow">NEXT-DAY MODEL OUTPUT</div><h2>Market signal</h2></div><span className="model-pill"><span className="model-indicator"/>{result.model_type}</span></div><div className="stock-metrics"><div className="stock-metric"><span className="metric-label">Latest close</span><strong>${result.latest_close.toFixed(2)}</strong><small>Most recent value in model response</small></div><div className={`stock-metric return-metric ${positive ? 'up' : 'down'}`}><span className="metric-label">Predicted return</span><strong>{positive ? <ArrowUpRight size={20}/> : <ArrowDownRight size={20}/>} {(result.predicted_next_day_return * 100).toFixed(3)}%</strong><small>Next-day model estimate</small></div><div className="stock-metric estimate-metric"><span className="metric-label">Predicted next close</span><strong>${result.predicted_next_close.toFixed(2)}</strong><small>Implied by model output</small></div></div><div className="stock-chart-heading"><div><span className="eyebrow">OBSERVED TO ESTIMATED</span><h3>One-step model view</h3></div><span className="sequence-pill">{result.sequence_length} day input sequence</span></div><ModelStepChart result={result}/></Panel>
        <div className="model-context"><span className="model-context-icon"><ChartNoAxesCombined size={18}/></span><p><strong>How to read this view</strong><br/>The highlighted second point is the model’s predicted next close, not a live quote. The endpoint provides a latest close and a one-step estimate, not a historical series.</p><span className="context-model">{result.model_type}</span></div>
        <div className="disclosure warning"><span><Activity size={16}/></span><p>This is a next-day model prediction, not a market quote or investment recommendation. On the evaluated test period, the return-target LSTM did not outperform the persistence baseline; past evaluation does not predict future performance.</p></div>
      </> : <div className="empty-page stock-empty"><span className="empty-icon"><TrendingUp size={22}/></span><h3>Model view awaits a ticker</h3><p>The trained model currently supports AAPL. The API will return an error for unsupported tickers or missing model artifacts.</p></div>}
      <div className="small-note">For research and educational use only. Not investment advice.</div>
    </div>
  </>
}

export type Health = { status: string; service: string }
export type RiskResult = { bankruptcy_probability: number; risk_level: string; top_contributors: { feature: string; shap_value: number }[] }
export type StockResult = { ticker: string; latest_close: number; predicted_next_day_return: number; predicted_next_close: number; model_type: string; sequence_length: number }
export type SentimentResult = { sentiment: string; confidence: number; probabilities: { negative: number; neutral: number; positive: number } }
export type RetrievalResult = { chunk_id: string; score: number; text: string; metadata: Record<string, unknown> }
export type RetrieveResponse = { query: string; results: RetrievalResult[] }
export type AskResponse = { query: string; answer: string; sources: { chunk_id: string; metadata: Record<string, unknown> }[] }

const base = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')
async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${base}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch { throw new Error('Could not reach the FinSight API. Check that the backend is running and VITE_API_BASE_URL is correct.') }
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.toLowerCase().includes('json')) {
    if (!response.ok) throw new Error(`Request failed (${response.status})`)
    throw new Error('The API returned a non-JSON response. Check VITE_API_BASE_URL or the Vite API proxy.')
  }
  let data: Record<string, unknown>
  try {
    data = await response.json()
  } catch {
    if (!response.ok) throw new Error(`Request failed (${response.status})`)
    throw new Error('The API returned invalid JSON.')
  }
  if (!response.ok) {
    const detail = data.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).join(', ') : `Request failed (${response.status})`)
  }
  return data as T
}
export const api = {
  health: () => request<Health>('/health'),
  risk: (features: Record<string, number>) => request<RiskResult>('/risk/predict', { features }),
  stock: (ticker: string) => request<StockResult>('/stock/predict', { ticker }),
  sentiment: (text: string) => request<SentimentResult>('/sentiment/predict', { text }),
  retrieve: (query: string, top_k: number) => request<RetrieveResponse>('/rag/retrieve', { query, top_k }),
  ask: (query: string, top_k: number) => request<AskResponse>('/rag/ask', { query, top_k }),
}

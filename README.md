# FinSight AI

FinSight AI is a local financial analysis application that brings together bankruptcy-risk inference, AAPL next-day return prediction, financial-text sentiment classification, and grounded question answering over an Apple 2024 annual report. A React and TypeScript frontend communicates with a FastAPI backend that loads the project's local model and retrieval artifacts. FinSight AI is an educational/research project, not investment advice.

## Overview

FinSight AI currently provides four implemented workflows:

- **Bankruptcy Risk Analysis** — estimates bankruptcy probability from 94 financial features and returns a risk category with SHAP contributors.
- **Stock Return Prediction** — predicts a next-day return and projected closing price for **AAPL only**.
- **Financial Sentiment Analysis** — classifies a financial text snippet as positive, negative, or neutral.
- **Financial Research / RAG** — retrieves relevant passages from an indexed Apple Inc. 2024 Form 10-K and can use Gemini to generate an answer grounded in those passages.

This is a local financial-analysis application. Stock inference is limited to AAPL and uses local data; the research corpus is limited to the Apple 2024 annual report. It is not a live or general market-data platform, a general-purpose financial research platform, or investment advice.

## Key Features

- Bankruptcy probability, Low/Medium/High category, and five leading SHAP contributors.
- AAPL next-day return and projected close from a multivariate return-target LSTM.
- Financial sentence sentiment with confidence and per-class probabilities.
- Semantic search and Gemini answer generation over the local annual-report corpus, with source references.
- React/TypeScript views integrated with versioned FastAPI endpoints.

## System Architecture

```mermaid
flowchart LR
  subgraph RT["A. Runtime - solid arrows"]
    direction TB
    UIIN["Frontend"] --> API["FastAPI API /api/v1"]

    API -->|"risk/predict"| RS["Risk Service"] --> XGB["XGBoost"]
    XGB --> RPROB["Risk probability + level"] --> RRESULT["Risk result"]
    XGB --> SHAP["SHAP top five"] --> RRESULT
    RRESULT --> APIRESP["FastAPI response"]

    API -->|"stock/predict"| SS["Stock Service"] --> LSTM["AAPL return LSTM"] --> STOCKOUT["Next-day return + close; local, not live"] --> APIRESP
    API -->|"sentiment/predict"| SES["Sentiment Service"] --> FINBERT["Fine-tuned FinBERT"] --> SENTOUT["Label + probabilities"] --> APIRESP

    API -->|"/rag/retrieve"| RRETR["Retrieve route"] --> EMBED["SentenceTransformer"] --> FAISS["FAISS search"] --> CHUNKS["Retrieved chunks + metadata"]
    CHUNKS -->|"retrieve results; no Gemini"| RETRESP["Results + metadata"] --> APIRESP
    API -->|"/rag/ask"| RASK["Ask route"] --> EMBED
    CHUNKS -->|"context found"| ASKCTX["Retrieved context"] --> ANSWER["Grounded Answer Service"] --> GEMINI["Gemini API: answer only"] --> ASKRESP["Answer + sources"] --> APIRESP
    CHUNKS -->|"no context"| EMPTY["Insufficient-context answer"] --> APIRESP

    APIRESP --> UIOUT["Frontend"]
  end

  subgraph ART["B. Local Model/Data Artifacts"]
    direction TB
    RISKART["Risk model JSON + metadata"]
    STOCKART["AAPL LSTM + scalers + engineered data"]
    SENTART["FinBERT model + metadata"]
    RAGART["FAISS index + embeddings + chunks + metadata"]
  end

  subgraph OFF["C. Offline Preparation - dashed artifact generation"]
    direction TB
    UCI["UCI Bankruptcy"] -.-> RPREP["Preprocess"] -.-> RTRAIN["XGBoost training"] -.-> RISKART
    YF["Yahoo Finance AAPL"] -.-> SPREP["Feature engineering"] -.-> STRAIN["LSTM train/evaluate"] -.-> STOCKART
    FPB["Financial PhraseBank"] -.-> PPREP["Preprocess + splits"] -.-> PTRAIN["FinBERT fine-tune/evaluate"] -.-> SENTART
    SEC["Apple 2024 Form 10-K"] -.-> EXTRACT["Extract text"] -.-> CHUNK["Chunk"] -.-> EMBGEN["Embed all-MiniLM-L6-v2"] -.-> BUILD["Build FAISS index"] -.-> RAGART
  end

  RISKART --> XGB
  STOCKART --> SS
  SENTART --> FINBERT
  RAGART --> FAISS
```

The RAG retrieval endpoint returns retrieved chunks directly. The RAG ask endpoint retrieves context and passes it to the answer-generation provider; when there is no retrieved context, the answer service returns an insufficient-context response without calling the LLM.

## ML & RAG Components

### 1. Bankruptcy Risk Prediction

The risk workflow uses an **XGBoost `XGBClassifier`** with 94 named financial input features. It returns bankruptcy probability and a Low, Medium, or High category. **SHAP TreeExplainer** supplies the five feature contributions with the largest absolute values for each prediction.

### 2. Stock Return Prediction

The served workflow uses a **multivariate return-target LSTM** for **AAPL only**. It uses 13 engineered features over a 60-step input sequence to predict next-day return and derive a projected close from the latest close in local engineered data. This is not live pricing. The repository also contains univariate direct-close and multivariate direct-close LSTM experiments.

### 3. Financial Sentiment Analysis

The sentiment workflow uses a fine-tuned **`ProsusAI/finbert`** model trained on **Financial PhraseBank** data. It returns positive, negative, or neutral sentiment with confidence and probabilities for all three classes.

### 4. Financial Research / RAG

The RAG corpus contains 134 chunks from Apple's **2024 Form 10-K**. Retrieval uses **`sentence-transformers/all-MiniLM-L6-v2`** embeddings (384 dimensions) and a FAISS index. The answer-generation layer uses the configured Google Gemini model and grounds answers in retrieved context.

## Model Evaluation & Results

The figures below are transcribed from repository evaluation artifacts; no metrics were recomputed for this README.

| Workflow | Dataset and evaluation setup | Recorded results |
| --- | --- | --- |
| **Risk** | UCI Taiwanese Bankruptcy Prediction (repository ID 572); stratified 80/20 train/test split, `random_state=42`; XGBoost classifier with 94 retained features. Training handles class imbalance with `scale_pos_weight`. | No saved numeric evaluation summary is currently present. The training script calculates accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix on the test split, but only prints them when run. |
| **Stock** | AAPL Yahoo Finance daily data; served model is a 13-feature, 60-step LSTM targeting `Next_Day_Return`, using local engineered features. The repository also includes univariate and multivariate direct-close models. | No saved numeric MAE/RMSE/MAPE results or persistence-baseline comparison summary is currently present. |
| **Sentiment** | Financial PhraseBank: 2,264 examples split 80/10/10 (1,811 train / 227 validation / 226 test); fine-tuned `ProsusAI/finbert`. | **Held-out test set (226 examples):** accuracy **0.9779**, macro F1 **0.9729**, weighted F1 **0.9780**. |
| **RAG retrieval** | Apple 2024 Form 10-K; 134 chunks; `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional embeddings. FAISS `IndexFlatIP` uses L2-normalized vectors for cosine similarity. Evaluation set: 8 questions. | Recall@1 **0.750** · Recall@3 **1.000** · Recall@5 **1.000**. |

The recorded sentiment training configuration is 3 epochs, maximum sequence length 128, learning rate `2e-5`, batch size 8, weight decay `0.01`, and seed `42`. The validation metrics recorded during training are separate from the held-out test results shown above and are not presented as test performance.

### Evaluation limitations

- Numeric risk and stock evaluation summaries are not currently persisted in the repository; no performance claim against a baseline is made.
- RAG retrieval was evaluated on only eight project questions, so these results are a small project evaluation, not a broad benchmark.
- Sentiment scores are held-out test-set results and should not be presented as production performance.
- Stock prediction uses historical/local AAPL data rather than live-market inference.
## Frontend

The React and TypeScript application contains five views:

- **Overview** — service status and entry points to the analysis workflows.
- **Risk** — feature input and risk prediction results.
- **Stock** — AAPL next-day model output.
- **Sentiment** — financial text input and classification results.
- **Research** — document retrieval and grounded question answering with source references.

The frontend includes responsive styling and calls the existing backend API. During development, Vite proxies `/api` requests to the local FastAPI server.

## API Endpoints

All endpoints use the configured API prefix, which defaults to `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Return API health status. |
| `POST` | `/api/v1/risk/predict` | Predict bankruptcy probability and return risk category and SHAP contributors. |
| `POST` | `/api/v1/stock/predict` | Predict the next-day AAPL return and projected close. |
| `POST` | `/api/v1/sentiment/predict` | Classify a financial text snippet and return class probabilities. |
| `POST` | `/api/v1/rag/retrieve` | Retrieve matching chunks from the annual-report corpus. |
| `POST` | `/api/v1/rag/ask` | Retrieve context and generate a grounded answer with source references. |

## API Reference

The API prefix is configurable and defaults to `/api/v1`. Request bodies are JSON. Invalid request models return FastAPI's standard `422` validation response. The response examples below are schema illustrations; dynamic outputs use pseudo-JSON type placeholders and are not literal, copy-pasteable API responses.

### `GET /api/v1/health`

Returns API process health. It does **not** verify that ML models or RAG artifacts are ready.

- **Request:** No body.
- **Response:** `status` and `service` strings.

```json
{"status":"ok","service":"FinSight AI API"}
```

### `POST /api/v1/risk/predict`

Predicts bankruptcy probability and returns the top SHAP contributors.

- **Request:** Required `features` object mapping every exact trained feature name to a numeric value. The required feature set is recorded in [`ml/risk_prediction/model_metadata.json`](ml/risk_prediction/model_metadata.json); the model uses 94 features. To avoid suggesting incomplete or fabricated inputs, no partial feature payload is shown.
- **Validation:** Missing or extra feature names return **400**. Invalid body/value types return **422**.
- **Response:** `bankruptcy_probability`, `risk_level`, and `top_contributors` entries (`feature`, `shap_value`).
- **Other errors:** **500** if the risk model or metadata file is missing.

**Response schema illustration (pseudo-JSON; placeholders are types, not values):**

```text
{
  "bankruptcy_probability": <number>,
  "risk_level": <string>,
  "top_contributors": [{"feature": <string>, "shap_value": <number>}]
}
```

### `POST /api/v1/stock/predict`

Predicts next-day return and projected close using the local AAPL model and engineered data. This is **not** a live-market lookup. Only `AAPL` is supported.

- **Request:** Optional `ticker` string; defaults to `"AAPL"`.
- **Validation:** Any other ticker returns **400**.
- **Response:** `ticker`, `latest_close`, `predicted_next_day_return`, `predicted_next_close`, `model_type`, and `sequence_length`.
- **Other errors:** **500** if model or artifact files are missing.

**Request example:**

```json
{"ticker":"AAPL"}
```

**Response schema illustration (pseudo-JSON; placeholders are types, not values):**

```text
{
  "ticker": <string>,
  "latest_close": <number>,
  "predicted_next_day_return": <number>,
  "predicted_next_close": <number>,
  "model_type": <string>,
  "sequence_length": <integer>
}
```

### `POST /api/v1/sentiment/predict`

Classifies a financial text snippet with the saved FinBERT model.

- **Request:** Required `text` string.
- **Validation:** Empty or whitespace-only text returns **400**; a missing or non-string value returns **422**.
- **Response:** `sentiment`, `confidence`, and `probabilities` with `negative`, `neutral`, and `positive` values.
- **Other errors:** **500** if the sentiment model or metadata is missing.

**Request example:**

```json
{"text":"<financial text to classify>"}
```

**Response schema illustration (pseudo-JSON; placeholders are types, not values):**

```text
{
  "sentiment": <string>,
  "confidence": <number>,
  "probabilities": {"negative": <number>, "neutral": <number>, "positive": <number>}
}
```

### `POST /api/v1/rag/retrieve`

Searches the local FAISS index and returns matching annual-report chunks and metadata. This endpoint does **not** call Gemini.

- **Request:** Required `query` string; optional `top_k` integer, default `5`.
- **Validation:** `top_k` must be from `1` through `10` (inclusive). Empty query returns **422**; whitespace-only query returns **400**.
- **Response:** Original `query` and `results`; each result contains `chunk_id`, `score`, `text`, and `metadata`.
- **Other errors:** **500** if the retrieval index or document artifacts are unavailable.

**Request example:**

```json
{"query":"<question to search>","top_k":5}
```

**Response schema illustration (pseudo-JSON; placeholders are types, not values):**

```text
{
  "query": <string>,
  "results": [{"chunk_id": <string>, "score": <number>, "text": <string>, "metadata": <object>}]
}
```

### `POST /api/v1/rag/ask`

Retrieves report context first, then generates a grounded answer with the configured Gemini provider when context is available.

- **Request:** Required `query` string; optional `top_k` integer, default `5`.
- **Validation:** `query` must remain non-empty after trimming; `top_k` must be from `1` through `10` (inclusive). Invalid values return **422**.
- **Response:** `query`, generated `answer`, and `sources` containing retrieved chunk IDs and metadata.
- **Behavior:** If retrieval returns no chunks, the answer service returns its insufficient-context response without calling Gemini.
- **Other errors:** **500** for retrieval failure, unavailable/unconfigured Gemini provider, or answer-generation failure.

**Request example:**

```json
{"query":"<question about the report>","top_k":5}
```

**Response schema illustration (pseudo-JSON; placeholders are types, not values):**

```text
{
  "query": <string>,
  "answer": <string>,
  "sources": [{"chunk_id": <string>, "metadata": <object>}]
}
```

### OpenAPI documentation

FastAPI exposes interactive documentation at `/docs` and the OpenAPI schema at `/openapi.json`.
## Tech Stack

- **Frontend:** React, TypeScript, Vite, React Router.
- **Backend:** Python, FastAPI, Pydantic Settings, Uvicorn.
- **Machine learning:** XGBoost, SHAP, TensorFlow/Keras, scikit-learn.
- **NLP:** PyTorch, Hugging Face Transformers, FinBERT.
- **RAG:** Sentence Transformers, FAISS, PDF/text ingestion and chunking utilities.
- **LLM:** Google Gemini through the `google-genai` Python SDK.

## Project Structure

```text
.
├── backend/                 FastAPI app, API routes, schemas, service adapters
├── data/                    Local raw and processed data/artifacts
├── frontend/                React + TypeScript + Vite application
├── ml/
│   ├── risk_prediction/     XGBoost model metadata, training and evaluation code
│   ├── sentiment/           FinBERT workflow, metadata, evaluation and documentation
│   └── stock_prediction/    LSTM workflows, feature engineering and evaluation
├── notebooks/               Analysis notebooks
├── rag/
│   ├── documents/           Annual-report chunks and document metadata
│   ├── embeddings/          Embedding generation and metadata
│   ├── ingestion/           Document extraction, chunking and validation
│   └── retrieval/           FAISS index, retrieval code and evaluation
├── tests/                   Project-level tests
├── .env.example             Backend environment-variable template
└── requirements.txt         Python dependency declarations
```

## Local Development

The commands below show the repository's Windows development workflow. Replace the example repository path with the location where you cloned the project.

### Backend

```powershell
cd C:\path\to\FinSight-AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

### Frontend

In a separate terminal:

```powershell
cd C:\path\to\FinSight-AI\frontend
npm ci
npm run dev
```

Vite uses port `5173` by default and may select another port if `5173` is occupied. A frontend `frontend/.env` file is optional for the normal development setup. Without `VITE_API_BASE_URL`, the frontend uses the relative API base `/api/v1`, and Vite proxies `/api` requests to `http://127.0.0.1:8000`.

If you explicitly set `VITE_API_BASE_URL=http://localhost:8000/api/v1` (as shown in `frontend/.env.example`), requests go directly to FastAPI and use CORS rather than the Vite proxy. The example file is not loaded automatically; create `frontend/.env` to use that override.

For frontend checks and a production build, run from `frontend/`:

```powershell
npm run typecheck
npm run build
```

### Setup Caveats

- `requirements.txt` declares TensorFlow, SHAP, FAISS, and pytest.

- The current development environment was validated with Python 3.10.11, Node.js 24.13.0, and npm 11.21.0. These are tested development versions, not declared minimum requirements; the repository does not specify mandatory Python or Node.js/npm versions.
- The `.gitignore` excludes several generated or local ML/RAG artifacts, including stock `.keras` models and scaler artifacts, processed stock data, the FinBERT model directory, RAG embedding JSONL files, and the FAISS index. Git tracking of these artifacts has not been verified, so a fresh clone may require acquisition or regeneration of ignored model/data artifacts before all inference workflows can run.
- `/api/v1/health` reports API process health; it does not verify that the ML models or RAG artifacts are present or ready.
- `GEMINI_API_KEY` is required for grounded answer generation when retrieved context is available. `/api/v1/rag/retrieve` does not require Gemini; `/api/v1/rag/ask` uses Gemini when retrieval returns context.


## Environment Variables

Set backend variables in a local `.env` file based on `.env.example`. Do not commit secrets.

| Variable | Current default/example | Purpose |
| --- | --- | --- |
| `APP_NAME` | `FinSight AI` | FastAPI application title. |
| `APP_ENV` | `development` | Application environment label. |
| `API_V1_PREFIX` | `/api/v1` | Prefix for versioned API routes. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000` | Comma-separated browser origins allowed by the backend. |
| `GEMINI_API_KEY` | Empty in `.env.example` | Gemini credential used for answer generation. Supply it locally; do not publish it. |
| `FINSIGHT_LLM_MODEL` | `gemini-3.5-flash-lite` | Gemini model used by the answer-generation provider. |
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | Frontend API base URL, documented in `frontend/.env.example`. |

For local Vite development, `frontend/.env` is optional: without the override, requests use `/api/v1` through the Vite proxy to `http://127.0.0.1:8000`. Setting `VITE_API_BASE_URL` to the example's absolute URL sends requests directly to FastAPI and relies on the backend's CORS configuration.

## Limitations

- Stock inference is limited to AAPL and requires local model and data artifacts.
- Stock output is a model prediction; the application does not provide live market data.
- RAG answers are grounded in the local Apple 2024 Form 10-K corpus, not a general financial-document collection.
- This project is not investment advice or a general financial research platform.
- The reported sentiment and retrieval metrics describe repository evaluations only; they are not production performance guarantees.
- No deployment or hosted-service configuration is documented here.

## Future Improvements

Possible future work, not currently implemented, includes broader ticker and data support, live market-data integration, a richer financial-document corpus, expanded evaluation and monitoring, production deployment, and more complete dependency/artifact reproducibility.

## License

A project license has not yet been specified. Review applicable dataset, source-document, and model terms before redistribution or use.








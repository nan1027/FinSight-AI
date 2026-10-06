from fastapi import APIRouter, HTTPException, status

from backend.api.v1.rag import router as rag_router
from backend.schemas.common import HealthResponse
from backend.schemas.rag import RagRetrieveRequest, RagRetrieveResponse
from backend.schemas.risk import RiskPredictionRequest, RiskPredictionResponse
from backend.schemas.sentiment import SentimentPredictionRequest, SentimentPredictionResponse
from backend.schemas.stock import StockPredictionRequest, StockPredictionResponse
from backend.services.rag.retrieval_service import RagRetrievalService
from backend.services.risk.risk_service import RiskPredictionService
from backend.services.sentiment.sentiment_service import SentimentPredictionService
from backend.services.stock.stock_service import StockPredictionService

router = APIRouter()
router.include_router(rag_router)
risk_service: RiskPredictionService | None = None
sentiment_service: SentimentPredictionService | None = None
stock_service: StockPredictionService | None = None
rag_service: RagRetrievalService | None = None


def _get_risk_service() -> RiskPredictionService:
    global risk_service
    if risk_service is None:
        risk_service = RiskPredictionService()
    return risk_service


def _get_sentiment_service() -> SentimentPredictionService:
    global sentiment_service
    if sentiment_service is None:
        sentiment_service = SentimentPredictionService()
    return sentiment_service


def _get_stock_service() -> StockPredictionService:
    global stock_service
    if stock_service is None:
        stock_service = StockPredictionService()
    return stock_service


def _get_rag_service() -> RagRetrievalService:
    global rag_service
    if rag_service is None:
        rag_service = RagRetrievalService()
    return rag_service


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="FinSight AI API")


@router.post("/risk/predict", response_model=RiskPredictionResponse)
def predict_bankruptcy_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """Predict bankruptcy risk and return the top SHAP-driven feature contributors for the inference."""
    try:
        result = _get_risk_service().predict(request.features)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The risk model or metadata could not be loaded.",
        ) from exc

    return RiskPredictionResponse(**result)


@router.post("/sentiment/predict", response_model=SentimentPredictionResponse)
def predict_sentiment(request: SentimentPredictionRequest) -> SentimentPredictionResponse:
    """Predict the sentiment of a single financial text snippet using the calibrated FinBERT model."""
    try:
        result = _get_sentiment_service().predict(request.text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The sentiment model or metadata could not be loaded.",
        ) from exc

    return SentimentPredictionResponse(**result)


@router.post("/stock/predict", response_model=StockPredictionResponse)
def predict_stock_return(request: StockPredictionRequest) -> StockPredictionResponse:
    """Predict the next-day AAPL return and projected close using the trained return-target LSTM."""
    try:
        result = _get_stock_service().predict(request.ticker)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stock model or artifact files could not be loaded.",
        ) from exc

    return StockPredictionResponse(**result)


@router.post("/rag/retrieve", response_model=RagRetrieveResponse)
def retrieve_from_corpus(request: RagRetrieveRequest) -> RagRetrieveResponse:
    """Run semantic search against the annual report corpus and return the top matching chunks."""
    try:
        result = _get_rag_service().retrieve(query=request.query, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The retrieval index or document artifacts could not be loaded.",
        ) from exc
    return RagRetrieveResponse(**result)

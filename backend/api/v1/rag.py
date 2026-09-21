from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.rag import RagAskRequest, RagAskResponse
from backend.services.rag.answer_service import AnswerGenerationService
from rag.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=RagAskResponse)
def ask_question(request: RagAskRequest) -> RagAskResponse:
    """Retrieve relevant annual-report chunks and generate a grounded answer from them."""
    try:
        retrieval_result = RetrievalService().retrieve(query=request.query, top_k=request.top_k)
    except Exception as exc:  # pragma: no cover - boundary guard for retrieval failures
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval failed while searching the annual-report corpus.",
        ) from exc

    retrieved_chunks = retrieval_result.get("results", [])
    answer_service = AnswerGenerationService()

    if not retrieved_chunks:
        answer_result = answer_service.generate(request.query, [])
        return RagAskResponse(
            query=request.query,
            answer=answer_result["answer"],
            sources=answer_result.get("sources", []),
        )

    try:
        answer_result = answer_service.generate(request.query, retrieved_chunks)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The configured LLM provider is unavailable or not configured.",
        ) from exc
    except Exception as exc:  # pragma: no cover - provider execution guard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed while grounding the response.",
        ) from exc

    return RagAskResponse(
        query=request.query,
        answer=answer_result["answer"],
        sources=answer_result.get("sources", []),
    )

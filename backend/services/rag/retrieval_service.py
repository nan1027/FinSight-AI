from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from rag.retrieval.retrieval_service import RetrievalService as CoreRetrievalService


class RagRetrievalService:
    """Thin backend adapter around the reusable core retrieval service."""

    def __init__(self, core_service: CoreRetrievalService | None = None) -> None:
        self.core_service = core_service or CoreRetrievalService()

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        try:
            return self.core_service.retrieve(query=query, top_k=top_k)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The retrieval index or document artifacts could not be loaded.",
            ) from exc

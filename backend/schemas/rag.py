from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetrievalChunkMetadata(BaseModel):
    """Metadata associated with a retrieved chunk from the annual report."""

    page: int | None = Field(default=None, description="Page number from the original PDF when available.")
    section: str | None = Field(default=None, description="Section label or heading captured in the source document.")
    source_document: str | None = Field(default=None, description="Original document name or corpus identifier.")
    chunk_index: int | None = Field(default=None, description="Chunk position within the source document.")
    chunk_length: int | None = Field(default=None, description="Approximate character length of the chunk.")


class RetrievalResult(BaseModel):
    """A single document chunk returned by the RAG retriever."""

    chunk_id: str = Field(..., description="Stable chunk identifier for the retrieved document segment.")
    score: float = Field(..., description="Similarity score for the retrieved chunk.")
    text: str = Field(..., description="The chunk text returned to the caller.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Chunk metadata preserved from the ingestion pipeline.")


class RagRetrieveRequest(BaseModel):
    """Request body for semantic retrieval from the Apple annual report."""

    query: str = Field(..., min_length=1, description="Natural-language question to search against the annual report corpus.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of matching chunks to return.")


class RagRetrieveResponse(BaseModel):
    """Retrieval response containing the request query and matched chunks."""

    query: str = Field(..., description="Original search query.")
    results: list[RetrievalResult] = Field(default_factory=list, description="Top matching chunks in score order.")


class RagAskRequest(BaseModel):
    """Request body for grounded question answering over the annual report."""

    query: str = Field(..., min_length=1, description="Question to answer using the retrieved annual-report context.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of relevant annual-report chunks to retrieve before answering.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("query must be a string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must be a non-empty string.")
        return cleaned


class RagAskResponse(BaseModel):
    """Grounded answer and source references for the user's question."""

    query: str = Field(..., description="Original question asked by the user.")
    answer: str = Field(..., description="Grounded answer produced from the retrieved annual-report context.")
    sources: list[dict[str, object]] = Field(default_factory=list, description="Source references preserved from the retrieved chunks.")

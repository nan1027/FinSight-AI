from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = REPO_ROOT / "rag" / "embeddings" / "AAPL_annual_report_2024_embeddings.jsonl"
INDEX_PATH = REPO_ROOT / "rag" / "retrieval" / "AAPL_annual_report_2024.faiss"
INDEX_METADATA_PATH = REPO_ROOT / "rag" / "retrieval" / "index_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FAISSRetriever:
    def __init__(self, index_path: Path | str = INDEX_PATH, embedding_path: Path | str = EMBEDDINGS_PATH, model_name: str = MODEL_NAME):
        self.index_path = Path(index_path)
        self.embedding_path = Path(embedding_path)
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = self._load_metadata()
        self.chunk_id_mapping = self._load_chunk_id_mapping()
        self.embedding_records = self._load_embedding_records()
        self.embedding_dimension = int(self.metadata.get("embedding_dimension", 384))

    def _load_metadata(self) -> dict[str, Any]:
        if not INDEX_METADATA_PATH.exists():
            raise FileNotFoundError(f"Index metadata file not found: {INDEX_METADATA_PATH}")
        with INDEX_METADATA_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_chunk_id_mapping(self) -> dict[int, str]:
        mapping = self.metadata.get("chunk_id_mapping", {})
        return {int(key): str(value) for key, value in mapping.items()}

    def _load_embedding_records(self) -> list[dict[str, Any]]:
        with self.embedding_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return self._normalize(embedding)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.embed_query(query)
        if query_vector.shape[0] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {query_vector.shape[0]}.")

        scores, indices = self.index.search(np.asarray([query_vector], dtype=np.float32), top_k)
        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            record = self.embedding_records[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": str(record.get("chunk_id")),
                    "score": float(score),
                    "text": str(record.get("text", "")),
                    "company": str(record.get("company", "")),
                    "document_year": int(record.get("document_year", 0)),
                    "source_filename": str(record.get("source_filename", "")),
                    "page_start": int(record.get("page_start", 0)),
                    "page_end": int(record.get("page_end", 0)),
                }
            )
        return results


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    retriever = FAISSRetriever()
    return retriever.retrieve(query=query, top_k=top_k)


if __name__ == "__main__":
    sample = retrieve("What risks does Apple identify in its annual report?", top_k=3)
    for item in sample:
        print(item)

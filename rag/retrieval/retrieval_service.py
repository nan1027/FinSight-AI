from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "rag" / "retrieval" / "AAPL_annual_report_2024.faiss"
INDEX_METADATA_PATH = REPO_ROOT / "rag" / "retrieval" / "index_metadata.json"
EMBEDDING_METADATA_PATH = REPO_ROOT / "rag" / "embeddings" / "embedding_metadata.json"
EMBEDDINGS_PATH = REPO_ROOT / "rag" / "embeddings" / "AAPL_annual_report_2024_embeddings.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalService:
    def __init__(
        self,
        index_path: Path | str = INDEX_PATH,
        index_metadata_path: Path | str = INDEX_METADATA_PATH,
        embedding_metadata_path: Path | str = EMBEDDING_METADATA_PATH,
        embedding_path: Path | str = EMBEDDINGS_PATH,
        model_name: str | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.index_metadata_path = Path(index_metadata_path)
        self.embedding_metadata_path = Path(embedding_metadata_path)
        self.embedding_path = Path(embedding_path)

        self._validate_file_exists(self.index_path, "FAISS index")
        self._validate_file_exists(self.index_metadata_path, "index metadata")
        self._validate_file_exists(self.embedding_metadata_path, "embedding metadata")
        self._validate_file_exists(self.embedding_path, "embedding JSONL file")

        self.index_metadata = self._load_json(self.index_metadata_path)
        self.embedding_metadata = self._load_json(self.embedding_metadata_path)

        if not self.index_metadata.get("chunk_id_mapping"):
            raise ValueError("Missing chunk_id_mapping in index metadata.")

        self.model_name = model_name or str(self.embedding_metadata.get("embedding_model") or DEFAULT_MODEL_NAME)
        self.model = SentenceTransformer(self.model_name, device="cpu")

        self.index = faiss.read_index(str(self.index_path))
        self.embedding_records = self._load_embedding_records()
        self.chunk_id_mapping = self._load_chunk_id_mapping()
        self.embedding_dimension = self._resolve_embedding_dimension()

        if self.index.d != self.embedding_dimension:
            raise ValueError(
                f"FAISS index dimension mismatch: index dim={self.index.d}, expected {self.embedding_dimension}."
            )
        if self.index.ntotal != len(self.embedding_records):
            raise ValueError(
                f"FAISS index/document mismatch: index has {self.index.ntotal} vectors, embeddings file has {len(self.embedding_records)} records."
            )

        self._validate_chunk_lookup()

    def _validate_file_exists(self, path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"JSON metadata file is not a dictionary: {path}")
        return data

    def _load_embedding_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.embedding_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid embedding JSON on line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Embedding record on line {line_number} is not a JSON object.")
                records.append(record)
        if not records:
            raise ValueError("Embedding record file is empty.")
        return records

    def _load_chunk_id_mapping(self) -> dict[int, str]:
        mapping = self.index_metadata.get("chunk_id_mapping", {})
        if not mapping:
            raise ValueError("Missing chunk_id_mapping in the index metadata file.")

        normalized: dict[int, str] = {}
        for key, value in mapping.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid chunk ID mapping key: {key!r}") from exc
            normalized[idx] = str(value)

        if len(normalized) != self.index.ntotal:
            raise ValueError(
                f"chunk_id_mapping length mismatch: metadata has {len(normalized)} entries but index has {self.index.ntotal}."
            )
        return normalized

    def _resolve_embedding_dimension(self) -> int:
        explicit = self.embedding_metadata.get("embedding_dimension")
        if explicit is not None:
            return int(explicit)
        explicit_index = self.index_metadata.get("embedding_dimension")
        if explicit_index is not None:
            return int(explicit_index)
        return 384

    def _validate_chunk_lookup(self) -> None:
        known_ids = {str(record.get("chunk_id")) for record in self.embedding_records}
        for index in range(len(self.embedding_records)):
            expected_chunk_id = self.chunk_id_mapping.get(index)
            if expected_chunk_id is None:
                raise ValueError(f"Missing chunk mapping for FAISS index position {index}.")
            if expected_chunk_id not in known_ids:
                raise ValueError(f"Chunk mapping mismatch at index {index}: {expected_chunk_id} not found in embedding records.")

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.copy()
        return (vector / norm).astype(np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {query_vector.shape[0]}."
            )
        return self._normalize(query_vector)

    def _build_result(self, idx: int, score: float) -> dict[str, Any]:
        if idx < 0:
            raise ValueError("FAISS returned a negative index.")
        if idx >= len(self.embedding_records):
            raise ValueError(f"FAISS index {idx} exceeds loaded embedding record count {len(self.embedding_records)}.")

        record = self.embedding_records[int(idx)]
        chunk_id = str(record.get("chunk_id"))
        if not chunk_id:
            raise ValueError(f"Embedding record at position {idx} is missing chunk_id.")

        metadata = {key: value for key, value in record.items() if key not in {"chunk_id", "text", "embedding"}}
        return {
            "chunk_id": chunk_id,
            "score": float(score),
            "text": str(record.get("text", "")),
            "metadata": metadata,
        }

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        query_vector = self._embed_query(query)
        search_limit = min(int(top_k), int(self.index.ntotal))

        scores, indices = self.index.search(np.asarray([query_vector], dtype=np.float32), search_limit)

        if scores.shape[1] != indices.shape[1]:
            raise ValueError("FAISS returned mismatched score and index arrays.")

        seen_chunk_ids: set[str] = set()
        result_list: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = self._build_result(int(idx), float(score))
            if result["chunk_id"] in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id returned by FAISS search: {result['chunk_id']}")
            seen_chunk_ids.add(result["chunk_id"])
            result_list.append(result)

        if len(result_list) != len(seen_chunk_ids):
            raise ValueError("FAISS retrieval returned duplicate chunk_ids for the same query.")

        if not result_list:
            raise ValueError("FAISS retrieval returned no valid results.")

        return {
            "query": query,
            "results": result_list,
        }


def retrieve(query: str, top_k: int = 5) -> dict[str, Any]:
    service = RetrievalService()
    return service.retrieve(query=query, top_k=top_k)


__all__ = ["RetrievalService", "retrieve"]

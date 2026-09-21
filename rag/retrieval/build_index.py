from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = REPO_ROOT / "rag" / "embeddings" / "AAPL_annual_report_2024_embeddings.jsonl"
METADATA_PATH = REPO_ROOT / "rag" / "embeddings" / "embedding_metadata.json"
INDEX_PATH = REPO_ROOT / "rag" / "retrieval" / "AAPL_annual_report_2024.faiss"
INDEX_METADATA_PATH = REPO_ROOT / "rag" / "retrieval" / "index_metadata.json"
EXPECTED_DIMENSION = 384


def load_embedding_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Embedding record on line {line_number} is not a JSON object.")
            records.append(record)
    return records


def validate_embedding_records(records: list[dict[str, Any]]) -> np.ndarray:
    if not records:
        raise ValueError("No embedding records were loaded.")

    chunk_ids = [str(record.get("chunk_id")) for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Embedding chunk IDs are not unique.")

    matrix = []
    for record in records:
        vector = record.get("embedding")
        if not isinstance(vector, list):
            raise ValueError(f"Embedding record {record.get('chunk_id')} does not contain a list-based embedding.")
        if not vector:
            raise ValueError(f"Embedding record {record.get('chunk_id')} has an empty embedding.")
        array = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Embedding record {record.get('chunk_id')} contains NaN or infinite values.")
        matrix.append(array)

    matrix_array = np.vstack(matrix)
    if matrix_array.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(f"Expected embedding dimension {EXPECTED_DIMENSION}, found {matrix_array.shape[1]}.")
    return matrix_array


def build_faiss_index(vectors: np.ndarray) -> tuple[faiss.Index, np.ndarray]:
    normalized = vectors.astype(np.float32, copy=True)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-12

    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


def build_chunk_id_mapping(records: list[dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for idx, record in enumerate(records):
        chunk_id = str(record.get("chunk_id"))
        if not chunk_id:
            raise ValueError(f"Record at position {idx} is missing a chunk_id.")
        mapping[idx] = chunk_id
    return mapping


def write_index(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def write_metadata(index_path: Path, embedding_file: Path, index_metadata_path: Path, vector_count: int, embedding_dimension: int, embedding_model: str, normalized_for_indexing: bool, mapping: dict[int, str]) -> None:
    metadata = {
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "similarity_metric": "cosine_similarity",
        "normalized_for_indexing": normalized_for_indexing,
        "vector_count": vector_count,
        "source_embedding_file": str(embedding_file.relative_to(REPO_ROOT)),
        "index_file": str(index_path.relative_to(REPO_ROOT)),
        "chunk_id_mapping": mapping,
        "validation_results": {
            "vector_count_matches": vector_count == 134,
            "dimension_matches": embedding_dimension == 384,
            "index_built_successfully": True,
        },
    }
    with index_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index for the saved AAPL chunk embeddings.")
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH, help="JSONL embedding file.")
    parser.add_argument("--index", type=Path, default=INDEX_PATH, help="Output FAISS index path.")
    parser.add_argument("--metadata", type=Path, default=INDEX_METADATA_PATH, help="Output metadata path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embeddings_path = args.embeddings if args.embeddings.is_absolute() else REPO_ROOT / args.embeddings
    index_path = args.index if args.index.is_absolute() else REPO_ROOT / args.index
    metadata_path = args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata

    embedding_records = load_embedding_records(embeddings_path)
    matrix = validate_embedding_records(embedding_records)
    mapping = build_chunk_id_mapping(embedding_records)
    index, normalized_vectors = build_faiss_index(matrix)
    write_index(index, index_path)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    write_metadata(
        index_path=index_path,
        embedding_file=embeddings_path,
        index_metadata_path=metadata_path,
        vector_count=int(normalized_vectors.shape[0]),
        embedding_dimension=int(normalized_vectors.shape[1]),
        embedding_model=str(metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        normalized_for_indexing=True,
        mapping={int(k): str(v) for k, v in mapping.items()},
    )

    print(f"Index built: {index_path}")
    print(f"Vector count: {normalized_vectors.shape[0]}")
    print(f"Index dimension: {normalized_vectors.shape[1]}")
    print(f"Similarity metric: cosine_similarity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

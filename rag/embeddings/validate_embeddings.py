from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_PATH = REPO_ROOT / "rag" / "embeddings" / "AAPL_annual_report_2024_embeddings.jsonl"
CHUNKS_PATH = REPO_ROOT / "rag" / "documents" / "chunks" / "AAPL_annual_report_2024_chunks.jsonl"
METADATA_PATH = REPO_ROOT / "rag" / "embeddings" / "embedding_metadata.json"
EXPECTED_CHUNK_COUNT = 134


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_embedding_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_embedding_records(chunks: list[dict[str, object]], embeddings: list[dict[str, object]]) -> tuple[bool, dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(chunks) != EXPECTED_CHUNK_COUNT:
        errors.append(f"Expected {EXPECTED_CHUNK_COUNT} source chunks, found {len(chunks)}.")
    if len(embeddings) != EXPECTED_CHUNK_COUNT:
        errors.append(f"Expected {EXPECTED_CHUNK_COUNT} embeddings, found {len(embeddings)}.")
    if len(chunks) != len(embeddings):
        errors.append("Chunk and embedding counts do not match.")

    source_ids = [str(chunk.get("chunk_id")) for chunk in chunks]
    embedding_ids = [str(record.get("chunk_id")) for record in embeddings]
    if len(source_ids) != len(set(source_ids)):
        errors.append("Source chunk IDs are not unique.")
    if len(embedding_ids) != len(set(embedding_ids)):
        errors.append("Embedding chunk IDs are not unique.")
    if source_ids != embedding_ids:
        errors.append("Embedding order does not match chunk order by chunk_id.")

    dimensions: set[int] = set()
    for index, record in enumerate(embeddings):
        chunk_id = record.get("chunk_id")
        if chunk_id is None:
            errors.append(f"Embedding row {index} is missing chunk_id.")
            continue

        chunk = next((item for item in chunks if str(item.get("chunk_id")) == str(chunk_id)), None)
        if chunk is None:
            errors.append(f"Embedding chunk_id {chunk_id} does not exist in the source chunk file.")
            continue

        for field in ["document_id", "company", "document_year", "source_filename", "page_start", "page_end", "text"]:
            if field not in record:
                errors.append(f"Embedding record {chunk_id} is missing required field {field}.")
            elif field in chunk and record[field] != chunk[field]:
                errors.append(f"Embedding record {chunk_id} does not preserve metadata field {field}.")

        vector = record.get("embedding")
        if not isinstance(vector, list):
            errors.append(f"Embedding record {chunk_id} does not contain an embedding vector.")
            continue
        if not vector:
            errors.append(f"Embedding record {chunk_id} has an empty embedding vector.")
            continue

        array = np.asarray(vector, dtype=np.float64)
        dimensions.add(array.shape[0])
        if not np.isfinite(array).all():
            errors.append(f"Embedding record {chunk_id} contains NaN or infinite values.")

    if len(dimensions) > 1:
        errors.append(f"Embedding dimensions are inconsistent across the output: {sorted(dimensions)}.")

    if not errors:
        warnings.append("No warnings.")

    return (not errors), {"errors": errors, "warnings": warnings, "dimensions": sorted(dimensions)}


def main() -> int:
    if not EMBEDDING_PATH.exists():
        print(f"EMBEDDING VALIDATION FAILED: missing embedding file {EMBEDDING_PATH}")
        return 1
    if not CHUNKS_PATH.exists():
        print(f"EMBEDDING VALIDATION FAILED: missing chunk file {CHUNKS_PATH}")
        return 1

    chunks = load_chunk_records(CHUNKS_PATH)
    embeddings = load_embedding_records(EMBEDDING_PATH)
    valid, summary = validate_embedding_records(chunks, embeddings)

    metadata = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    if not valid:
        print("EMBEDDING VALIDATION FAILED")
        for error in summary["errors"]:
            print(f"- {error}")
        return 1

    print("EMBEDDING VALIDATION PASSED")
    print(f"Embedding model: {metadata.get('embedding_model', 'unknown')}")
    print(f"Embedding dimension: {metadata.get('embedding_dimension', 'unknown')}")
    print(f"Number of chunks embedded: {len(embeddings)}")
    print(f"Device used: {metadata.get('model_device', 'unknown')}")
    print(f"Output path: {EMBEDDING_PATH}")
    print("Validation results: all checks passed")
    print(f"Dependency/environment warnings: {summary['warnings'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

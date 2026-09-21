from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "rag" / "documents" / "chunks" / "AAPL_annual_report_2024_chunks.jsonl"
OUTPUT_PATH = REPO_ROOT / "rag" / "embeddings" / "AAPL_annual_report_2024_embeddings.jsonl"
METADATA_PATH = REPO_ROOT / "rag" / "embeddings" / "embedding_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Chunk record on line {line_number} is not a JSON object.")
            chunks.append(record)
    return chunks


def validate_chunk_records(chunks: list[dict[str, object]]) -> None:
    if not chunks:
        raise ValueError("No chunks were loaded from the chunk file.")

    chunk_ids = [str(chunk.get("chunk_id")) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs are not unique across the source chunk file.")

    for chunk in chunks:
        text = chunk.get("text")
        if text is None or str(text).strip() == "":
            raise ValueError(f"Chunk {chunk.get('chunk_id')} has empty text and cannot be embedded.")


def generate_embeddings_for_chunks(chunks: list[dict[str, object]], model: SentenceTransformer, batch_size: int) -> np.ndarray:
    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def write_embeddings_jsonl(chunks: list[dict[str, object]], embeddings: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk, embedding in zip(chunks, embeddings):
            payload = dict(chunk)
            payload["embedding"] = embedding.astype(float).tolist()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_embedding_metadata(
    *,
    chunk_count: int,
    model_name: str,
    device: str,
    input_path: Path,
    output_path: Path,
    dimension: int,
    validation_passed: bool,
    timestamp: str,
) -> dict[str, object]:
    return {
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "chunk_count": chunk_count,
        "source_chunk_file": str(input_path.relative_to(REPO_ROOT)),
        "output_file": str(output_path.relative_to(REPO_ROOT)),
        "normalization_status": "not_applied",
        "model_device": device,
        "generation_timestamp_utc": timestamp,
        "validation_results": {
            "chunk_count_matches": validation_passed,
            "all_embeddings_have_same_dimension": validation_passed,
            "all_embeddings_are_finite": validation_passed,
            "chunk_ids_unique": validation_passed,
            "order_matches_chunk_order": validation_passed,
        },
    }


def validate_generated_embeddings(chunks: list[dict[str, object]], embeddings: np.ndarray) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunk count mismatch: {len(chunks)} chunks loaded but {len(embeddings)} embeddings generated.")

    chunk_ids = [str(chunk.get("chunk_id")) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs are not unique after embedding generation.")

    if embeddings.size == 0:
        raise ValueError("No embedding data was generated.")

    dims = {tuple(row.shape) for row in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {dims}")

    if not np.isfinite(embeddings).all():
        raise ValueError("One or more embeddings contain NaN or infinite values.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"Embedding rows mismatch: expected {len(chunks)} rows, got {embeddings.shape[0]}.")

    for row_index, chunk in enumerate(chunks):
        if not isinstance(embeddings[row_index], np.ndarray):
            raise ValueError(f"Embedding row {row_index} is not an array.")
        if embeddings[row_index].size != embeddings.shape[1]:
            raise ValueError(f"Embedding row {row_index} has inconsistent shape.")
        if chunk.get("chunk_id") is None:
            raise ValueError(f"Chunk at index {row_index} does not include a chunk_id.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic embeddings for the chunked Apple annual report.")
    parser.add_argument("--input", type=Path, default=CHUNKS_PATH, help="Chunk JSONL file to embed.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="JSONL output file for chunk metadata plus embeddings.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Metadata JSON file describing the embedding run.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="SentenceTransformers embedding model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    metadata_path = args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata

    chunks = load_chunk_records(input_path)
    validate_chunk_records(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    embeddings = generate_embeddings_for_chunks(chunks, model, batch_size=args.batch_size)
    validate_generated_embeddings(chunks, embeddings)

    write_embeddings_jsonl(chunks, embeddings, output_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = build_embedding_metadata(
        chunk_count=len(chunks),
        model_name=args.model_name,
        device=device,
        input_path=input_path,
        output_path=output_path,
        dimension=int(embeddings.shape[1]),
        validation_passed=True,
        timestamp=timestamp,
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

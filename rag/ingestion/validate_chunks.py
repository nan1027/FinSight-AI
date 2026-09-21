from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "rag" / "documents" / "chunks" / "AAPL_annual_report_2024_chunks.jsonl"
METADATA_PATH = REPO_ROOT / "rag" / "documents" / "chunks" / "chunk_metadata.json"
TARGET_CHUNK_SIZE = 4400
OVERLAP_CHARS = 550


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def load_chunks(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        chunks.append(payload)
    return chunks


def has_overlap(previous_text: str, current_text: str) -> bool:
    if not previous_text or not current_text:
        return False
    max_overlap = min(len(previous_text), len(current_text), OVERLAP_CHARS * 2)
    for size in range(max_overlap, 0, -1):
        if previous_text.endswith(current_text[:size]):
            return True
    return False


def validate_chunks(chunks: list[dict[str, object]]) -> tuple[bool, dict[str, object]]:
    errors: list[str] = []

    if not chunks:
        errors.append("No chunks were detected in the chunk file.")

    seen_ids: set[str] = set()
    previous_normalized_text: str | None = None
    page_numbers: list[int] = []
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            errors.append(f"Chunk {index} is missing chunk_id.")
        elif chunk_id in seen_ids:
            errors.append(f"Duplicate chunk_id detected: {chunk_id}")
        else:
            seen_ids.add(str(chunk_id))

        if not chunk.get("document_id"):
            errors.append(f"Chunk {chunk_id} is missing document_id.")
        if not chunk.get("company"):
            errors.append(f"Chunk {chunk_id} is missing company.")
        if not chunk.get("document_year"):
            errors.append(f"Chunk {chunk_id} is missing document_year.")
        if not chunk.get("source_filename"):
            errors.append(f"Chunk {chunk_id} is missing source_filename.")

        text = str(chunk.get("text", "")).strip()
        if not text:
            errors.append(f"Chunk {chunk_id} has empty text.")

        normalized_text = " ".join(text.split())
        if previous_normalized_text is not None and normalized_text == previous_normalized_text:
            errors.append(f"Adjacent duplicate chunk text detected for chunk {chunk_id}.")
        previous_normalized_text = normalized_text

        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start is None or page_end is None:
            errors.append(f"Chunk {chunk_id} is missing page_start/page_end.")
        else:
            try:
                start = int(page_start)
                end = int(page_end)
            except (TypeError, ValueError):
                errors.append(f"Chunk {chunk_id} has non-integer page metadata.")
            else:
                page_numbers.append(start)
                page_numbers.append(end)
                if start > end:
                    errors.append(f"Chunk {chunk_id} has page_start greater than page_end.")

        chunk_text_length = len(text)
        if chunk_text_length > TARGET_CHUNK_SIZE * 2:
            errors.append(f"Chunk {chunk_id} is unreasonably large ({chunk_text_length} characters).")

    if len(chunks) > 1:
        for previous, current in zip(chunks, chunks[1:]):
            prev_text = str(previous.get("text", "")).strip()
            curr_text = str(current.get("text", "")).strip()
            if not has_overlap(prev_text, curr_text):
                errors.append(f"No meaningful overlap detected between {previous.get('chunk_id')} and {current.get('chunk_id')}.")

    chunk_order: list[int] = []
    for chunk in chunks:
        page_start = chunk.get("page_start")
        if page_start is not None:
            try:
                chunk_order.append(int(page_start))
            except (TypeError, ValueError):
                pass
    if len(chunk_order) != len(chunks):
        errors.append("Chunk order could not be fully validated because some page_start values were missing or invalid.")
    elif chunk_order != sorted(chunk_order):
        errors.append("Chunks are not in document order by page_start.")

    lengths = [len(str(chunk.get("text", "")).strip()) for chunk in chunks]
    estimated_tokens = [estimate_tokens(str(chunk.get("text", "")).strip()) for chunk in chunks]
    summary = {
        "total_chunks": len(chunks),
        "min_chunk_characters": min(lengths) if lengths else 0,
        "max_chunk_characters": max(lengths) if lengths else 0,
        "average_chunk_characters": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "median_chunk_characters": median(lengths) if lengths else 0,
        "average_estimated_tokens": round(sum(estimated_tokens) / len(estimated_tokens), 2) if estimated_tokens else 0,
        "page_coverage": {
            "min_page": min(page_numbers) if page_numbers else 0,
            "max_page": max(page_numbers) if page_numbers else 0,
        },
        "errors": errors,
    }
    return (not errors), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a chunk JSONL file and summarize chunk statistics.")
    parser.add_argument("--input", type=Path, default=CHUNKS_PATH, help="Chunk JSONL file to validate.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Chunk metadata JSON file to compare against the validation output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks_file = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    metadata_file = args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata

    try:
        chunks = load_chunks(chunks_file)
    except FileNotFoundError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1
    except ValueError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1

    valid, summary = validate_chunks(chunks)

    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Minimum chunk characters: {summary['min_chunk_characters']}")
    print(f"Maximum chunk characters: {summary['max_chunk_characters']}")
    print(f"Average chunk characters: {summary['average_chunk_characters']}")
    print(f"Median chunk characters: {summary['median_chunk_characters']}")
    print(f"Average estimated tokens: {summary['average_estimated_tokens']}")
    print(f"Page coverage: {summary['page_coverage']}")

    if not valid:
        print("Validation errors:")
        for error in summary["errors"]:
            print(f"- {error}")
        return 1

    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(f"Metadata file: {metadata_file}")
        print(f"Chunking strategy: {metadata.get('chunking_strategy')}")
        print(f"Overlap: {metadata.get('overlap')} characters")

    print("CHUNKING VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

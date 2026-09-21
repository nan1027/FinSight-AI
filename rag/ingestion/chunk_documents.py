from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = REPO_ROOT / "rag" / "documents" / "processed" / "AAPL_annual_report_2024.txt"
CHUNKS_DIR = REPO_ROOT / "rag" / "documents" / "chunks"
CHUNK_PATH = CHUNKS_DIR / "AAPL_annual_report_2024_chunks.jsonl"
METADATA_PATH = CHUNKS_DIR / "chunk_metadata.json"

DOCUMENT_ID = "AAPL_annual_report_2024"
COMPANY = "Apple Inc."
DOCUMENT_YEAR = 2024
SOURCE_FILENAME = "AAPL_annual_report_2024.pdf"
TARGET_CHUNK_SIZE = 4400
OVERLAP_CHARS = 550

PAGE_MARKER_RE = re.compile(r"^--- PAGE (\d+) ---\s*$", re.MULTILINE)


def load_processed_document(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Processed document not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Processed document is empty: {path}")
    if "--- PAGE " not in text:
        raise ValueError(f"Processed document does not contain page markers: {path}")
    return text


def extract_pages(text: str) -> list[tuple[int, str]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        raise ValueError("No page markers were found in the processed document.")

    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        pages.append((page_number, page_text))
    return pages


def split_page_into_segments(page_text: str) -> list[str]:
    working_blocks: list[str] = []
    current: list[str] = []

    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                block = " ".join(current).strip()
                if block:
                    working_blocks.append(block)
                current = []
            continue

        current.append(line)

    if current:
        block = " ".join(current).strip()
        if block:
            working_blocks.append(block)

    segments: list[str] = []
    for block in working_blocks:
        if len(block) > TARGET_CHUNK_SIZE:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            piece: list[str] = []
            piece_chars = 0
            for sentence in sentences:
                candidate = (sentence + " ").strip()
                if not candidate:
                    continue
                if piece and piece_chars + 1 + len(candidate) > TARGET_CHUNK_SIZE:
                    segments.append(" ".join(piece).strip())
                    piece = [candidate]
                    piece_chars = len(candidate)
                else:
                    piece.append(candidate)
                    piece_chars += len(candidate)
            if piece:
                segments.append(" ".join(piece).strip())
        else:
            segments.append(block)

    filtered = [segment for segment in segments if segment and segment.strip()]
    return filtered


def estimate_tokens(text: str) -> int:
    words = text.split()
    return max(1, len(words))


def create_chunks_for_document(text: str) -> list[dict[str, object]]:
    pages = extract_pages(text)
    segments: list[dict[str, object]] = []
    for page_number, page_text in pages:
        for segment in split_page_into_segments(page_text):
            segments.append({"page_number": page_number, "text": segment})

    if not segments:
        raise ValueError("No meaningful content was found to chunk.")

    chunks: list[dict[str, object]] = []
    current_pages: list[int] = []
    current_text = ""
    current_chars = 0
    chunk_index = 1

    def emit_chunk(chunk_text: str, page_numbers: list[int]) -> None:
        nonlocal chunk_index
        cleaned = " ".join(chunk_text.split())
        if not cleaned:
            return

        previous_text = " ".join(str(chunks[-1]["text"]).split()) if chunks else ""
        if previous_text and previous_text == cleaned:
            # Prevent the overlap tail from creating a no-progress duplicate chunk at the end of the document.
            return

        chunk = {
            "chunk_id": f"{DOCUMENT_ID}_chunk_{chunk_index:04d}",
            "document_id": DOCUMENT_ID,
            "company": COMPANY,
            "document_year": DOCUMENT_YEAR,
            "source_filename": SOURCE_FILENAME,
            "page_start": min(page_numbers),
            "page_end": max(page_numbers),
            "text": cleaned,
        }
        chunks.append(chunk)
        chunk_index += 1

    for item in segments:
        page_number = int(item["page_number"])
        segment_text = str(item["text"])

        if not current_text:
            current_text = segment_text
            current_pages = [page_number]
            current_chars = len(current_text)
            continue

        candidate = f"{current_text} {segment_text}".strip()
        if len(candidate) <= TARGET_CHUNK_SIZE:
            current_text = candidate
            if page_number not in current_pages:
                current_pages.append(page_number)
            current_chars = len(current_text)
            continue

        emit_chunk(current_text, current_pages)
        overlap_text = current_text[-OVERLAP_CHARS:]
        next_text = f"{overlap_text} {segment_text}".strip()

        if next_text == current_text:
            # Keep advancing when the overlap would otherwise recreate the same tail content as the chunk just emitted.
            current_text = segment_text
            current_pages = [page_number]
            current_chars = len(current_text)
            continue

        current_text = next_text
        current_pages = sorted(set(current_pages + [page_number]))
        current_chars = len(current_text)

    if current_text.strip():
        emit_chunk(current_text, current_pages)

    if not chunks:
        raise ValueError("Chunking produced no output.")

    return chunks


def write_jsonl(chunks: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def build_metadata(chunks: list[dict[str, object]]) -> dict[str, object]:
    lengths = [len(chunk["text"]) for chunk in chunks]
    estimated_tokens = [estimate_tokens(str(chunk["text"])) for chunk in chunks]

    metadata = {
        "document_id": DOCUMENT_ID,
        "company": COMPANY,
        "document_year": DOCUMENT_YEAR,
        "source_filename": SOURCE_FILENAME,
        "chunking_strategy": "recursive_structure_aware_character_approximation",
        "target_chunk_size": TARGET_CHUNK_SIZE,
        "overlap": OVERLAP_CHARS,
        "total_chunks": len(chunks),
        "size_statistics": {
            "min_chunk_characters": min(lengths),
            "max_chunk_characters": max(lengths),
            "average_chunk_characters": round(sum(lengths) / len(lengths), 2),
            "median_chunk_characters": median(lengths),
            "average_estimated_tokens": round(sum(estimated_tokens) / len(estimated_tokens), 2),
            "estimated_token_min": min(estimated_tokens),
            "estimated_token_max": max(estimated_tokens),
        },
        "pages_covered": {
            "page_start": min(int(chunk["page_start"]) for chunk in chunks),
            "page_end": max(int(chunk["page_end"]) for chunk in chunks),
            "chunks_per_page": {},
        },
    }

    page_counts: dict[int, int] = {}
    for chunk in chunks:
        for page_number in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1):
            page_counts[page_number] = page_counts.get(page_number, 0) + 1
    metadata["pages_covered"]["chunks_per_page"] = dict(sorted(page_counts.items()))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk the processed Apple annual report text into deterministic JSONL chunks with page-level metadata.")
    parser.add_argument("--input", type=Path, default=SOURCE_FILE, help="Processed text file to chunk.")
    parser.add_argument("--output", type=Path, default=CHUNK_PATH, help="JSONL output file for chunk records.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Metadata file output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input

    text = load_processed_document(document_path)
    chunks = create_chunks_for_document(text)

    write_jsonl(chunks, args.output if args.output.is_absolute() else REPO_ROOT / args.output)
    metadata = build_metadata(chunks)
    with (args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Document: {DOCUMENT_ID}")
    print(f"Source file: {SOURCE_FILENAME}")
    print(f"Chunks written: {len(chunks)}")
    print(f"Pages covered: {metadata['pages_covered']['page_start']} - {metadata['pages_covered']['page_end']}")
    print(f"Average chunk characters: {metadata['size_statistics']['average_chunk_characters']}")
    print(f"Average estimated tokens: {metadata['size_statistics']['average_estimated_tokens']}")
    print(f"Chunk JSONL: {args.output if args.output.is_absolute() else REPO_ROOT / args.output}")
    print(f"Chunk metadata: {args.metadata if args.metadata.is_absolute() else REPO_ROOT / args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

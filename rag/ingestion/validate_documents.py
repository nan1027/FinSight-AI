from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "rag" / "documents" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "AAPL_annual_report_2024.txt"


def validate_processed_document(file_path: Path) -> dict[str, object]:
    if not file_path.exists():
        raise FileNotFoundError(f"Processed document not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Processed document is empty: {file_path}")

    page_markers = re.findall(r"^--- PAGE \d+ ---$", text, flags=re.MULTILINE)
    if not page_markers:
        raise ValueError(f"Processed document has no page markers: {file_path}")

    lines = text.splitlines()
    page_count = len(page_markers)
    char_count = len(text)
    word_count = len(re.findall(r"\S+", text))

    empty_pages: list[int] = []
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^--- PAGE \d+ ---$", line):
            if current:
                sections.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sections.append(current)

    for index, section in enumerate(sections, start=1):
        if not any(segment.strip() for segment in section if segment.strip() != "[No text extracted on this page]"):
            empty_pages.append(index)

    first_lines = "\n".join(lines[:5])
    last_lines = "\n".join(lines[-5:])

    payload = {
        "filename": file_path.name,
        "page_count": page_count,
        "character_count": char_count,
        "word_count": word_count,
        "empty_pages": empty_pages,
        "first_lines": first_lines,
        "last_lines": last_lines,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a processed PDF text file and summarize page structure, content size, and emptiness.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Extracted text file to validate. Defaults to rag/documents/processed/AAPL_annual_report_2024.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input

    try:
        result = validate_processed_document(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        return 1

    print(f"Filename: {result['filename']}")
    print(f"Page count: {result['page_count']}")
    print(f"Character count: {result['character_count']}")
    print(f"Word count: {result['word_count']}")
    print(f"Empty pages: {result['empty_pages'] if result['empty_pages'] else 'none'}")
    print("First few lines:")
    print(result["first_lines"])
    print("Last few lines:")
    print(result["last_lines"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

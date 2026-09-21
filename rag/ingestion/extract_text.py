from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "rag" / "documents" / "raw"
PROCESSED_DIR = REPO_ROOT / "rag" / "documents" / "processed"
DEFAULT_PDF = RAW_DIR / "AAPL_annual_report_2024.pdf"
DEFAULT_OUTPUT = PROCESSED_DIR / "AAPL_annual_report_2024.txt"


def extract_pdf_text(pdf_path: Path) -> tuple[str, list[int]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - pypdf raises library-specific exceptions
        raise ValueError(f"PDF is not readable: {pdf_path}") from exc

    pages = reader.pages
    if not pages:
        raise ValueError(f"No pages were found in the PDF: {pdf_path}")

    page_fragments: list[str] = []
    empty_pages: list[int] = []

    for page_index, page in enumerate(pages, start=1):
        page_marker = f"--- PAGE {page_index} ---"
        page_text = page.extract_text() or ""
        page_fragments.append(page_marker)
        if page_text.strip():
            page_fragments.append(page_text.rstrip())
        else:
            page_fragments.append("[No text extracted on this page]")
            empty_pages.append(page_index)
        page_fragments.append("")

    text = "\n".join(page_fragments).rstrip() + "\n"
    if not text.strip():
        raise ValueError(f"No text could be extracted from the PDF: {pdf_path}")
    if "--- PAGE " not in text:
        raise ValueError("Page markers were not written to the extracted output.")

    return text, empty_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a raw financial PDF and save it under rag/documents/processed/ with explicit page markers.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF path under rag/documents/raw/ to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the extracted text. Defaults to rag/documents/processed/AAPL_annual_report_2024.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf if args.pdf.is_absolute() else REPO_ROOT / args.pdf
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    try:
        text, empty_pages = extract_pdf_text(pdf_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Extraction failed: {exc}")
        print(f"Expected PDF location: {DEFAULT_PDF}")
        print("Place the official Apple annual report PDF under rag/documents/raw/ before running extraction.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"Output file was not created correctly: {output_path}")

    char_count = len(text)
    word_count = len(text.split())
    page_count = text.count("--- PAGE ")
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_path}")
    print(f"Page count: {page_count}")
    print(f"Character count: {char_count}")
    print(f"Word count: {word_count}")
    print(f"Empty pages: {empty_pages if empty_pages else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

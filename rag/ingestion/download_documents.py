from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "rag" / "documents" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "AAPL_annual_report_2024.pdf"

OFFICIAL_SOURCE = {
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "document_type": "Form 10-K annual report",
    "source": "U.S. Securities and Exchange Commission (SEC) EDGAR",
    "year": 2024,
    "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000106/a10-k20240928.pdf",
}

MANUAL_INSTRUCTIONS = f"""
Official Apple annual report source:
- Company: {OFFICIAL_SOURCE['company']} ({OFFICIAL_SOURCE['ticker']})
- Document type: {OFFICIAL_SOURCE['document_type']}
- Source: {OFFICIAL_SOURCE['source']}
- Year: {OFFICIAL_SOURCE['year']}
- Official filing URL: {OFFICIAL_SOURCE['url']}

Because SEC direct PDF retrieval can be blocked in automated environments, this step intentionally does not invent a source.
Please download the official PDF manually from the SEC filing page and save it here:
{DEFAULT_OUTPUT}

If the PDF is already present, do not re-download it.
"""


def download_document(url: str, output_path: Path) -> Path:
    """Download an explicitly provided PDF to the raw documents directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, output_path.open("wb") as handle:
        handle.write(response.read())

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a specific financial PDF document into rag/documents/raw/ from an explicitly provided URL.",
    )
    parser.add_argument(
        "--url",
        help="Exact official document URL to download. If omitted, the script prints the required manual-download instructions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Target path for the downloaded PDF. Defaults to rag/documents/raw/AAPL_annual_report_2024.pdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if not args.url:
        print(MANUAL_INSTRUCTIONS)
        return 0

    try:
        downloaded = download_document(args.url.strip(), output_path)
    except Exception as exc:  # pragma: no cover - network behavior is environment dependent
        print(f"Automatic download failed for {args.url}: {exc}")
        print(MANUAL_INSTRUCTIONS)
        return 1

    print(f"Downloaded document to: {downloaded}")
    print(f"Source: {OFFICIAL_SOURCE['source']}")
    print(f"Document: {OFFICIAL_SOURCE['company']} ({OFFICIAL_SOURCE['ticker']}) {OFFICIAL_SOURCE['document_type']} {OFFICIAL_SOURCE['year']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

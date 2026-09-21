# RAG ingestion setup for financial annual reports

## Initial company and document

- Company: Apple Inc. (AAPL)
- Document type: annual report / financial filing
- Filing form: Form 10-K (annual report)
- Source: U.S. Securities and Exchange Commission (SEC) EDGAR
- Year: 2024
- Expected filename for project use: `AAPL_annual_report_2024.pdf`
- Expected processed text filename: `AAPL_annual_report_2024.txt`

## Source and manual-download instructions

The preferred official source is the SEC EDGAR filing for Apple’s annual report:

`https://www.sec.gov/Archives/edgar/data/320193/000032019324000106/a10-k20240928.pdf`

This is the official Apple annual report / 10-K filing source and is a valid document origin for the RAG proof-of-concept.

Because direct SEC PDF retrieval can be blocked in automated environments, the project intentionally does not invent a fallback URL. If the direct URL cannot be downloaded reliably from the current environment, the document should be downloaded manually from the SEC EDGAR filing page and saved here:

`rag/documents/raw/AAPL_annual_report_2024.pdf`

Do not place random or non-official PDFs in the raw folder.

## Licensing and source considerations

- Use the official filing supplied by the issuer or regulator.
- Respect the SEC’s document terms and any usage restrictions associated with the filing.
- This repository uses the document only as a local retrieval corpus for the project’s RAG development workflow.

## Intended downstream use

This document will be used later in the ingestion pipeline for:

1. PDF text extraction
2. document cleaning
3. page-aware text preservation
4. chunking for later embedding generation
5. retrieval and grounded Gemini responses

This step covers only source identification, extraction, and validation. It does not create embeddings, a vector database, or retrieval logic.

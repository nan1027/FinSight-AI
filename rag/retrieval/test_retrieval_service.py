from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.retrieval.retrieval_service import RetrievalService

VALID_QUERIES = [
    "What was Apple's total net sales in 2024?",
    "How much cash and cash equivalents did Apple report?",
    "What risks does Apple identify in its annual report?",
    "How much did Apple spend on research and development?",
]


def validate_valid_query(service: RetrievalService, query: str) -> None:
    results = service.retrieve(query, top_k=5)
    assert isinstance(results, dict), "Result payload must be a dictionary."
    assert list(results.keys()) == ["query", "results"], "Result payload keys are unexpected."
    assert results["query"] == query, "Query was not preserved in the response."
    assert len(results["results"]) == 5, "Each query must return exactly 5 results."

    seen_chunk_ids: set[str] = set()
    previous_score: float | None = None
    for item in results["results"]:
        assert set(item.keys()) == {"chunk_id", "score", "text", "metadata"}, "Each result must include chunk_id, score, text, and metadata."
        chunk_id = str(item["chunk_id"])
        assert chunk_id not in seen_chunk_ids, "Duplicate chunk_id returned within a single query."
        seen_chunk_ids.add(chunk_id)
        assert isinstance(item["text"], str) and item["text"].strip(), "Text must be a non-empty string."
        assert isinstance(item["metadata"], dict), "Metadata must be a dictionary."

        score = float(item["score"])
        assert math.isfinite(score), "Similarity score must be finite."
        if previous_score is not None:
            assert score <= previous_score + 1e-12, "Results are not sorted in descending score order."
        previous_score = score


def validate_invalid_inputs(service: RetrievalService) -> None:
    invalid_cases = [
        ("", "empty query"),
        ("   ", "whitespace query"),
        (0, "top_k=0"),
        (-1, "top_k=-1"),
    ]

    for query, _ in invalid_cases[:2]:
        try:
            service.retrieve(query, top_k=5)
            raise AssertionError(f"Expected ValueError for invalid query: {query!r}")
        except (TypeError, ValueError):
            pass

    for top_k, _ in invalid_cases[2:]:
        try:
            service.retrieve("test query", top_k=top_k)
            raise AssertionError(f"Expected ValueError for invalid top_k: {top_k!r}")
        except (TypeError, ValueError):
            pass


def main() -> None:
    service = RetrievalService()
    for query in VALID_QUERIES:
        validate_valid_query(service, query)
    validate_invalid_inputs(service)
    print("RETRIEVAL SERVICE VALIDATION PASSED")


if __name__ == "__main__":
    main()

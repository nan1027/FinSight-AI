from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.retrieval.retrieve import retrieve

INDEX_METADATA_PATH = REPO_ROOT / "rag" / "retrieval" / "index_metadata.json"

QUESTIONS = [
    "What was Apple's total net sales in 2024?",
    "What risks does Apple identify in its annual report?",
    "How much cash and cash equivalents did Apple report?",
    "What products and services does Apple provide?",
]


def print_result(label: str, results: list[dict[str, object]]) -> None:
    print(f"\n{label}")
    for item in results:
        text = str(item["text"])[:180].replace("\n", " ")
        print(
            f"  Rank {item['rank']}: score={item['score']:.6f} | pages={item['page_start']}-{item['page_end']} | "
            f"chunk={item['chunk_id']} | preview={text}"
        )


def validate_results(results: list[dict[str, object]], valid_chunk_ids: set[str]) -> None:
    seen_ids: set[str] = set()
    scores = []
    for item in results:
        chunk_id = str(item["chunk_id"])
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID in retrieval: {chunk_id}")
        seen_ids.add(chunk_id)
        if chunk_id not in valid_chunk_ids:
            raise ValueError(f"Chunk ID not found in embedding metadata: {chunk_id}")
        score = float(item["score"])
        if not __import__("math").isfinite(score):
            raise ValueError(f"Non-finite similarity score for {chunk_id}")
        scores.append(score)
    if scores != sorted(scores, reverse=True):
        raise ValueError("Results are not ordered by descending similarity score.")


def main() -> int:
    metadata = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
    mapping = {str(v) for v in metadata.get("chunk_id_mapping", {}).values()}

    for question in QUESTIONS:
        result = retrieve(question, top_k=5)
        validate_results(result, mapping)
        label = question
        print_result(label, result)

    print("RETRIEVAL VALIDATION PASSED")
    print(f"Index path: {REPO_ROOT / 'rag' / 'retrieval' / 'AAPL_annual_report_2024.faiss'}")
    print(f"Vector count: {metadata.get('vector_count')}")
    print(f"Dimension: {metadata.get('embedding_dimension')}")
    print(f"Similarity metric: {metadata.get('similarity_metric')}")
    print(f"Top-k used: 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

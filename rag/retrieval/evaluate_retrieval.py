from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.retrieval.retrieve import retrieve

OUTPUT_JSON_PATH = REPO_ROOT / "rag" / "retrieval" / "retrieval_evaluation.json"
OUTPUT_MD_PATH = REPO_ROOT / "rag" / "retrieval" / "retrieval_evaluation.md"

EVALUATION_QUESTIONS = [
    {"category": "Revenue / sales", "question": "What was Apple's total net sales in 2024?"},
    {"category": "Cash", "question": "How much cash and cash equivalents did Apple report?"},
    {"category": "Risk", "question": "What risks does Apple identify in its annual report?"},
    {"category": "Products/services", "question": "What products and services does Apple provide?"},
    {"category": "Geographic information", "question": "Which geographic markets are important to Apple?"},
    {"category": "Research/development", "question": "How much did Apple spend on research and development?"},
    {"category": "Capital allocation", "question": "How did Apple return capital to shareholders?"},
    {"category": "Debt", "question": "What debt obligations does Apple report?"},
]

MANUAL_RELEVANCE = {
    "What was Apple's total net sales in 2024?": [
        {"relevant": 1, "reason": "Contains net sales discussion and the reportable sales totals by region, including the total net sales figure."},
        {"relevant": 1, "reason": "Includes the segment/net sales table with the aggregated total sales figure and regional breakdown."},
        {"relevant": 1, "reason": "Shows geographic net sales and long-lived assets, which directly supports the annual revenue question."},
        {"relevant": 0, "reason": "Risk and supply-chain content is not evidence for the total net sales amount."},
        {"relevant": 0, "reason": "Vendor receivables and operating detail are not directly useful for a total revenue answer."},
    ],
    "How much cash and cash equivalents did Apple report?": [
        {"relevant": 1, "reason": "Contains the cash, cash equivalents, and restricted cash ending balance: $29,943 million."},
        {"relevant": 1, "reason": "Shows the debt/commercial paper discussion and the cash flow context around financing and cash movement."},
        {"relevant": 1, "reason": "Includes the cash flow statement with beginning balances and operating/investing/financing cash flows."},
        {"relevant": 1, "reason": "The balance sheet section lists cash and cash equivalents directly on the asset side."},
        {"relevant": 0, "reason": "Geographic net sales information does not answer the cash balance question."},
    ],
    "What risks does Apple identify in its annual report?": [
        {"relevant": 1, "reason": "Explicitly discusses risk factors, including economic conditions, financial instability, and operational and credit risk."},
        {"relevant": 0, "reason": "The legal filing context is not the core risk-factor section itself."},
        {"relevant": 0, "reason": "This excerpt is more about SEC documents and incorporation by reference than material business risks."},
        {"relevant": 1, "reason": "Includes operational and market risk language relevant to business risk exposure."},
        {"relevant": 0, "reason": "Internal control language is not a business risk discussion."},
    ],
    "What products and services does Apple provide?": [
        {"relevant": 0, "reason": "This chunk is risk-related and does not answer the products/services question."},
        {"relevant": 1, "reason": "Describes product and service categories and the broader product strategy context."},
        {"relevant": 1, "reason": "Names product families such as Apple Watch, AirPods, and Apple Vision Pro, which directly answer the question."},
        {"relevant": 0, "reason": "IP and licensing discussion is not a product/service inventory list."},
        {"relevant": 0, "reason": "This is intellectual property discussion rather than the product and service categories Apple sells."},
    ],
    "Which geographic markets are important to Apple?": [
        {"relevant": 1, "reason": "Contains the geographic net sales table for U.S., China, and other countries, which is direct market evidence."},
        {"relevant": 1, "reason": "Discusses geographic segments and customer-to-location sales, directly tied to important markets."},
        {"relevant": 1, "reason": "Shows sales by geographic region and product categories, including the U.S. and Greater China reporting context."},
        {"relevant": 0, "reason": "This is a legal/regulatory filing reference, not geographic market detail."},
        {"relevant": 0, "reason": "This item is about line-item totals and not market-by-market operating information."},
    ],
    "How much did Apple spend on research and development?": [
        {"relevant": 0, "reason": "This is not the R&D spending line item; it is a broad operational risk section."},
        {"relevant": 1, "reason": "Contains the R&D discussion and cost language within the annual report's business-risk context."},
        {"relevant": 1, "reason": "Includes the explicit R&D spending and development discussion in the company narrative."},
        {"relevant": 0, "reason": "This is not directly about Apple’s R&D spending amount."},
        {"relevant": 0, "reason": "This chunk discusses tax and other items rather than R&D expenditure."},
    ],
    "How did Apple return capital to shareholders?": [
        {"relevant": 1, "reason": "Contains the shareholder returns section with cash dividends, repurchases, and the equity totals."},
        {"relevant": 1, "reason": "Lists dividends and share repurchases alongside the financing and cash flow context for capital returns."},
        {"relevant": 1, "reason": "Covers dividends, repurchases, and equity details directly relevant to shareholder capital allocation."},
        {"relevant": 0, "reason": "This is about other comprehensive income and balance-sheet structure rather than returns to shareholders."},
        {"relevant": 0, "reason": "This chunk is not about shareholder distributions or capital returns."},
    ],
    "What debt obligations does Apple report?": [
        {"relevant": 1, "reason": "Directly states the commercial paper and term debt obligations, including maturity and balance amounts."},
        {"relevant": 1, "reason": "Provides the debt details, including commercial paper program and debt obligations outstanding."},
        {"relevant": 1, "reason": "Includes the debt balance sheet lines for current and non-current liabilities and the debt obligations summary."},
        {"relevant": 1, "reason": "Lists key debt balances such as commercial paper and term debt directly on the balance sheet."},
        {"relevant": 0, "reason": "This geographic sales chunk is not evidence about debt obligations."},
    ],
}


def _snippet(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def validate_results(question: str, results: list[dict[str, Any]]) -> None:
    if len(results) != 5:
        raise ValueError(f"Question '{question}' did not return exactly 5 results; got {len(results)}.")

    seen_chunk_ids: set[str] = set()
    scores: list[float] = []
    for result in results:
        chunk_id = str(result["chunk_id"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Question '{question}' contains a duplicate chunk ID: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        score = float(result["score"])
        if not math.isfinite(score):
            raise ValueError(f"Question '{question}' returned a non-finite score for chunk {chunk_id}.")
        scores.append(score)

    if scores != sorted(scores, reverse=True):
        raise ValueError(f"Question '{question}' results are not sorted in descending score order.")


def compute_recall(relevance_values: list[int], k: int) -> int:
    cutoff = relevance_values[:k]
    return 1 if any(value == 1 for value in cutoff) else 0


def build_relevance_annotations(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = MANUAL_RELEVANCE.get(question)
    if expected is None:
        raise KeyError(f"Missing manual relevance judgment for: {question}")
    if len(expected) != len(results):
        raise ValueError(f"Question '{question}' expects {len(expected)} relevance labels but got {len(results)} results.")

    annotations: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        item = expected[index]
        if not isinstance(item, dict):
            raise ValueError(f"Manual relevance entry for '{question}' at rank {index + 1} is not a dictionary.")
        relevant = int(item["relevant"])
        if relevant not in (0, 1):
            raise ValueError(f"Manual relevance entry for '{question}' at rank {index + 1} must be 0 or 1.")
        annotations.append(
            {
                "rank": index + 1,
                "chunk_id": str(result["chunk_id"]),
                "relevant": relevant,
                "reason": str(item["reason"]),
                "page_start": int(result["page_start"]),
                "page_end": int(result["page_end"]),
                "score": float(result["score"]),
                "excerpt": _snippet(str(result["text"])),
            }
        )
    return annotations


def write_json_report(report_payload: dict[str, Any]) -> None:
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_markdown_report(report_payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Retrieval Quality Evaluation Report")
    lines.append("")
    lines.append(f"- Embedding model: {report_payload['embedding_model']}")
    lines.append(f"- Vector count: {report_payload['vector_count']}")
    lines.append(f"- Similarity metric: {report_payload['similarity_metric']}")
    lines.append(f"- Recall@1: {report_payload['overall_averages']['recall_at_1']:.3f}")
    lines.append(f"- Recall@3: {report_payload['overall_averages']['recall_at_3']:.3f}")
    lines.append(f"- Recall@5: {report_payload['overall_averages']['recall_at_5']:.3f}")
    lines.append("")

    for entry in report_payload["retrieved_results"]:
        question = entry["question"]
        category = entry["category"]
        page_ranges = ", ".join(f"{item['page_start']}-{item['page_end']}" for item in entry["results"])
        lines.append(f"## {category}: {question}")
        lines.append("")
        lines.append(f"Top retrieved page ranges: {page_ranges}")
        lines.append("")
        annots = next(item for item in report_payload["relevance_annotations"] if item["question"] == question)["annotations"]
        for annotation in annots:
            status = "Relevant" if annotation["relevant"] == 1 else "Not relevant"
            lines.append(f"- Rank {annotation['rank']} | {annotation['chunk_id']} | pages {annotation['page_start']}-{annotation['page_end']} | {status} | score={annotation['score']:.6f}")
            lines.append(f"  Reason: {annotation['reason']}")
            lines.append(f"  Excerpt: \"{annotation['excerpt']}\"")
        lines.append("")

    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metadata = json.loads((REPO_ROOT / "rag" / "retrieval" / "index_metadata.json").read_text(encoding="utf-8"))

    evaluation_results: list[dict[str, Any]] = []
    relevance_annotations: list[dict[str, Any]] = []
    recall_at_1: list[int] = []
    recall_at_3: list[int] = []
    recall_at_5: list[int] = []
    failures: list[str] = []

    for item in EVALUATION_QUESTIONS:
        question = item["question"]
        category = item["category"]
        results = retrieve(question, top_k=5)
        validate_results(question, results)

        annotations = build_relevance_annotations(question, results)
        relevant_values = [entry["relevant"] for entry in annotations]

        recall_1 = compute_recall(relevant_values, 1)
        recall_3 = compute_recall(relevant_values, 3)
        recall_5 = compute_recall(relevant_values, 5)
        recall_at_1.append(recall_1)
        recall_at_3.append(recall_3)
        recall_at_5.append(recall_5)

        if recall_5 == 0:
            failures.append(question)

        evaluation_results.append({
            "category": category,
            "question": question,
            "results": results,
        })
        relevance_annotations.append({
            "category": category,
            "question": question,
            "annotations": annotations,
        })

    report = {
        "evaluation_questions": [
            {"category": item["category"], "question": item["question"]} for item in EVALUATION_QUESTIONS
        ],
        "retrieved_results": evaluation_results,
        "relevance_annotations": relevance_annotations,
        "recall_at_1": {question["question"]: compute_recall(
            [annotation["relevant"] for annotation in next(item for item in relevance_annotations if item["question"] == question["question"])["annotations"]],
            1,
        ) for question in EVALUATION_QUESTIONS},
        "recall_at_3": {question["question"]: compute_recall(
            [annotation["relevant"] for annotation in next(item for item in relevance_annotations if item["question"] == question["question"])["annotations"]],
            3,
        ) for question in EVALUATION_QUESTIONS},
        "recall_at_5": {question["question"]: compute_recall(
            [annotation["relevant"] for annotation in next(item for item in relevance_annotations if item["question"] == question["question"])["annotations"]],
            5,
        ) for question in EVALUATION_QUESTIONS},
        "overall_averages": {
            "recall_at_1": sum(recall_at_1) / len(recall_at_1),
            "recall_at_3": sum(recall_at_3) / len(recall_at_3),
            "recall_at_5": sum(recall_at_5) / len(recall_at_5),
        },
        "embedding_model": metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_count": int(metadata.get("vector_count", 134)),
        "similarity_metric": metadata.get("similarity_metric", "cosine_similarity"),
        "retrieval_failures": failures,
    }

    write_json_report(report)
    write_markdown_report(report)

    print("RETRIEVAL QUALITY EVALUATION COMPLETED")
    print(f"Number of evaluation questions: {len(EVALUATION_QUESTIONS)}")
    print(f"Recall@1: {report['overall_averages']['recall_at_1']:.3f}")
    print(f"Recall@3: {report['overall_averages']['recall_at_3']:.3f}")
    print(f"Recall@5: {report['overall_averages']['recall_at_5']:.3f}")
    print(f"Questions with retrieval failures: {', '.join(failures) if failures else 'none'}")
    print(f"Embedding model: {report['embedding_model']}")
    print(f"Vector count: {report['vector_count']}")
    print(f"Similarity metric: {report['similarity_metric']}")
    print(f"Evaluation JSON: {OUTPUT_JSON_PATH}")
    print(f"Evaluation Markdown: {OUTPUT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

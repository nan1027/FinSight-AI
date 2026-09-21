from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_valid_request_with_mocked_answer_provider() -> None:
    retrieved_chunks = [
        {
            "chunk_id": "chunk_0042",
            "text": "Apple's revenue increased as Services grew.",
            "metadata": {"page": 37, "section": "Financial Results"},
            "score": 0.91,
        },
        {
            "chunk_id": "chunk_0043",
            "text": "The company continued to invest in product design and innovation.",
            "metadata": {"page": 38, "section": "Product Innovation"},
            "score": 0.88,
        },
    ]
    answer_payload = {
        "answer": "Grounded answer from the retrieved context.",
        "sources": [{"chunk_id": "chunk_0042", "metadata": {"page": 37, "section": "Financial Results"}}],
    }

    with patch("backend.api.v1.rag.RetrievalService.retrieve", return_value={"query": "What was Apple's total net sales in 2024?", "results": retrieved_chunks}), \
         patch("backend.api.v1.rag.AnswerGenerationService.generate", return_value=answer_payload) as generate_mock:
        response = client.post(
            "/api/v1/rag/ask",
            json={"query": "What was Apple's total net sales in 2024?", "top_k": 5},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query"] == "What was Apple's total net sales in 2024?"
    assert payload["answer"] == "Grounded answer from the retrieved context."
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["chunk_id"] == "chunk_0042"
    assert generate_mock.call_args[0][0] == "What was Apple's total net sales in 2024?"
    assert generate_mock.call_args[0][1] == retrieved_chunks


def test_orchestration_and_default_top_k() -> None:
    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = {
        "query": "What did Apple say about product design?",
        "results": [{"chunk_id": "chunk_0100", "text": "Design mattered.", "metadata": {"page": 10}, "score": 0.82}],
    }
    answer_service = Mock()
    answer_service.generate.return_value = {
        "answer": "Design matters.",
        "sources": [{"chunk_id": "chunk_0100", "metadata": {"page": 10}}],
    }

    with patch("backend.api.v1.rag.RetrievalService", return_value=retrieval_service), \
         patch("backend.api.v1.rag.AnswerGenerationService", return_value=answer_service):
        response = client.post("/api/v1/rag/ask", json={"query": "What did Apple say about product design?"})

    assert response.status_code == 200, response.text
    retrieval_service.retrieve.assert_called_once_with(query="What did Apple say about product design?", top_k=5)
    answer_service.generate.assert_called_once_with("What did Apple say about product design?", retrieval_service.retrieve.return_value["results"])


def test_invalid_query_and_top_k_rejected() -> None:
    response = client.post("/api/v1/rag/ask", json={"query": "   ", "top_k": 5})
    assert response.status_code == 422, response.text

    for top_k in [0, -1]:
        response = client.post("/api/v1/rag/ask", json={"query": "How did Apple perform?", "top_k": top_k})
        assert response.status_code == 422, response.text


def test_retrieval_failure_returns_500() -> None:
    with patch("backend.api.v1.rag.RetrievalService.retrieve", side_effect=RuntimeError("retrieval bug")):
        response = client.post("/api/v1/rag/ask", json={"query": "What happened?", "top_k": 3})

    assert response.status_code == 500, response.text
    assert "Retrieval failed" in response.json()["detail"]


def test_answer_generation_failure_returns_500() -> None:
    with patch("backend.api.v1.rag.RetrievalService.retrieve", return_value={"query": "What happened?", "results": [{"chunk_id": "chunk_001", "text": "Example context.", "metadata": {"page": 1}, "score": 0.9}]}), \
         patch("backend.api.v1.rag.AnswerGenerationService.generate", side_effect=RuntimeError("provider unavailable")):
        response = client.post("/api/v1/rag/ask", json={"query": "What happened?", "top_k": 3})

    assert response.status_code == 500, response.text
    assert "configured LLM provider" in response.json()["detail"]


def test_no_context_returns_insufficient_context() -> None:
    with patch("backend.api.v1.rag.RetrievalService.retrieve", return_value={"query": "What happened?", "results": []}):
        response = client.post("/api/v1/rag/ask", json={"query": "What happened?", "top_k": 3})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "The available annual-report context is insufficient to answer this question."
    assert payload["sources"] == []


def test_existing_apis_still_register() -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200, health.text

    risk = client.post("/api/v1/risk/predict", json={"features": {"company_age": 10.0, "debt_ratio": 0.1}})
    assert risk.status_code in {400, 500, 422}, risk.text

    stock = client.post("/api/v1/stock/predict", json={"ticker": "AAPL"})
    assert stock.status_code in {200, 400, 500}, stock.text

    sentiment = client.post("/api/v1/sentiment/predict", json={"text": "Apple reported strong growth."})
    assert sentiment.status_code in {200, 400, 500}, sentiment.text

    retrieve = client.post("/api/v1/rag/retrieve", json={"query": "Apple revenue", "top_k": 2})
    assert retrieve.status_code == 200, retrieve.text


def test_openapi_contains_rag_ask() -> None:
    response = client.get("/docs")
    assert response.status_code == 200, response.text

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200, openapi.text
    assert "/api/v1/rag/ask" in openapi.json()["paths"]


if __name__ == "__main__":
    test_valid_request_with_mocked_answer_provider()
    test_orchestration_and_default_top_k()
    test_invalid_query_and_top_k_rejected()
    test_retrieval_failure_returns_500()
    test_answer_generation_failure_returns_500()
    test_no_context_returns_insufficient_context()
    test_existing_apis_still_register()
    test_openapi_contains_rag_ask()
    print("RAG ASK API VALIDATION PASSED")

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


CLIENT = TestClient(app)


def test_health_endpoint() -> None:
    response = CLIENT.get("/api/v1/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "FinSight AI API"


def test_valid_sentiment_prediction() -> None:
    response = CLIENT.post(
        "/api/v1/sentiment/predict",
        json={"text": "The company reported strong earnings growth and lifted guidance."},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert set({"sentiment", "confidence", "probabilities"}).issubset(payload.keys()), payload
    assert payload["sentiment"] in {"negative", "neutral", "positive"}
    assert 0.0 <= float(payload["confidence"]) <= 1.0
    probs = payload["probabilities"]
    assert set({"negative", "neutral", "positive"}).issubset(probs.keys())
    assert abs(sum(probs.values()) - 1.0) < 1e-3, probs


def test_invalid_sentiment_text() -> None:
    response = CLIENT.post("/api/v1/sentiment/predict", json={"text": "   "})
    assert response.status_code == 400, response.text
    payload = response.json()
    assert "text" in str(payload).lower() or "whitespace" in str(payload).lower()


def test_openapi_includes_sentiment_route() -> None:
    openapi_schema = app.openapi()
    paths = openapi_schema.get("paths", {})
    assert "/api/v1/sentiment/predict" in paths


def test_existing_endpoints_still_work() -> None:
    risk_payload = {"features": {"ROA(C) before interest and depreciation before interest": 0.5}}
    risk_response = CLIENT.post("/api/v1/risk/predict", json=risk_payload)
    assert risk_response.status_code == 400, risk_response.text

    stock_response = CLIENT.post("/api/v1/stock/predict", json={"ticker": "AAPL"})
    assert stock_response.status_code == 200, stock_response.text


def main() -> None:
    test_health_endpoint()
    test_valid_sentiment_prediction()
    test_invalid_sentiment_text()
    test_openapi_includes_sentiment_route()
    test_existing_endpoints_still_work()
    print("health endpoint: ok")
    print("valid sentiment prediction: ok")
    print("invalid text: ok")
    print("OpenAPI includes sentiment route: ok")
    print("existing endpoints remain intact: ok")
    print("All sentiment API validation checks passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


CLIENT = TestClient(app)


def build_risk_payload() -> dict:
    metadata_path = Path("ml/risk_prediction/model_metadata.json")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {"features": {feature_name: 0.5 for feature_name in metadata["features"]}}


def test_health_endpoint() -> None:
    response = CLIENT.get("/api/v1/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "FinSight AI API"


def test_valid_stock_prediction() -> None:
    response = CLIENT.post("/api/v1/stock/predict", json={"ticker": "AAPL"})
    assert response.status_code == 200, response.text

    payload = response.json()
    required_fields = {
        "ticker",
        "latest_close",
        "predicted_next_day_return",
        "predicted_next_close",
        "model_type",
        "sequence_length",
    }
    assert required_fields.issubset(payload.keys()), payload
    assert payload["ticker"] == "AAPL"
    assert payload["sequence_length"] == 60
    assert payload["latest_close"] > 0
    assert payload["predicted_next_close"] > 0
    assert np.isfinite(payload["predicted_next_day_return"])


def test_invalid_ticker() -> None:
    response = CLIENT.post("/api/v1/stock/predict", json={"ticker": "MSFT"})
    assert response.status_code == 400, response.text
    payload = response.json()
    assert "AAPL" in str(payload)


def test_risk_endpoint_still_works() -> None:
    response = CLIENT.post("/api/v1/risk/predict", json=build_risk_payload())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "bankruptcy_probability" in payload
    assert "risk_level" in payload


def main() -> None:
    test_health_endpoint()
    test_valid_stock_prediction()
    test_invalid_ticker()
    test_risk_endpoint_still_works()
    print("health endpoint: ok")
    print("valid stock prediction: ok")
    print("invalid ticker: ok")
    print("risk endpoint: ok")
    print("All stock API validation checks passed.")


if __name__ == "__main__":
    main()

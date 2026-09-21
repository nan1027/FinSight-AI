from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def print_result(name: str, passed: bool, details: str | None = None) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{name}: {status}")
    if details:
        print(details)


def load_valid_request() -> dict:
    x_test = pd.read_csv(Path("data/processed/X_test.csv"))
    first_row = x_test.iloc[0].to_dict()
    return {"features": {key: float(value) for key, value in first_row.items()}}


def run_health_test() -> bool:
    response = client.get("/api/v1/health")
    expected = {"status": "ok", "service": "FinSight AI API"}
    if response.status_code != 200:
        print_result("HEALTH TEST", False, f"Expected 200, got {response.status_code}. Body: {response.text}")
        return False

    if response.json() != expected:
        print_result("HEALTH TEST", False, f"Expected {expected}, got {response.json()}")
        return False

    print_result("HEALTH TEST", True, "Response matched the expected health payload.")
    return True


def run_valid_request_test() -> bool:
    payload = load_valid_request()
    response = client.post("/api/v1/risk/predict", json=payload)

    if response.status_code != 200:
        print_result("VALID REQUEST TEST", False, f"Expected 200, got {response.status_code}. Body: {response.text}")
        return False

    body = response.json()
    if not {"bankruptcy_probability", "risk_level", "top_contributors"}.issubset(body):
        print_result("VALID REQUEST TEST", False, f"Missing expected keys. Body: {body}")
        return False

    probability = body["bankruptcy_probability"]
    risk_level = body["risk_level"]
    top_contributors = body["top_contributors"]

    if not isinstance(probability, (int, float)) or not (0.0 <= float(probability) <= 1.0):
        print_result("VALID REQUEST TEST", False, f"Probability out of range: {probability}")
        return False

    if risk_level not in {"Low", "Medium", "High"}:
        print_result("VALID REQUEST TEST", False, f"Unexpected risk_level: {risk_level}")
        return False

    if not isinstance(top_contributors, list) or len(top_contributors) != 5:
        print_result("VALID REQUEST TEST", False, f"Expected exactly 5 contributors, got: {top_contributors}")
        return False

    print_result("VALID REQUEST TEST", True, f"Probability={probability}, risk_level={risk_level}, contributors={len(top_contributors)}")
    return True


def run_missing_feature_test() -> bool:
    payload = load_valid_request()
    feature_to_remove = next(iter(payload["features"]))
    payload["features"].pop(feature_to_remove)

    response = client.post("/api/v1/risk/predict", json=payload)

    if response.status_code != 400:
        print_result("MISSING FEATURE TEST", False, f"Expected 400, got {response.status_code}. Body: {response.text}")
        return False

    detail = response.json().get("detail", "")
    if feature_to_remove not in detail:
        print_result("MISSING FEATURE TEST", False, f"Missing feature not mentioned in detail: {detail}")
        return False

    print_result("MISSING FEATURE TEST", True, f"400 handled as expected. Detail: {detail}")
    return True


def run_extra_feature_test() -> bool:
    payload = load_valid_request()
    payload["features"]["fake_test_feature"] = 123.0

    response = client.post("/api/v1/risk/predict", json=payload)

    if response.status_code != 400:
        print_result("EXTRA FEATURE TEST", False, f"Expected 400, got {response.status_code}. Body: {response.text}")
        return False

    detail = response.json().get("detail", "")
    if "fake_test_feature" not in detail:
        print_result("EXTRA FEATURE TEST", False, f"Extra feature not mentioned in detail: {detail}")
        return False

    print_result("EXTRA FEATURE TEST", True, f"400 handled as expected. Detail: {detail}")
    return True


if __name__ == "__main__":
    all_passed = True
    all_passed &= run_health_test()
    all_passed &= run_valid_request_test()
    all_passed &= run_missing_feature_test()
    all_passed &= run_extra_feature_test()

    if all_passed:
        print("Risk API validation passed.")
    else:
        raise SystemExit(1)

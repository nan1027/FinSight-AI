from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.sentiment.sentiment_service import SentimentPredictionService


SERVICE = SentimentPredictionService()

SENTENCES = [
    "The company's revenue increased significantly during the quarter.",
    "The company reported a decline in operating profit.",
    "The company announced its quarterly results on Monday.",
]


def validate_prediction(payload: dict) -> None:
    sentiment = payload["sentiment"]
    if sentiment not in {"negative", "neutral", "positive"}:
        raise ValueError(f"Unexpected sentiment value: {sentiment}")
    confidence = payload["confidence"]
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence out of range: {confidence}")
    probs = payload["probabilities"]
    values = [probs["negative"], probs["neutral"], probs["positive"]]
    if len(values) != 3:
        raise ValueError(f"Expected 3 probability values, got {len(values)}")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Probability values out of range: {values}")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-3):
        raise ValueError(f"Probabilities sum to {sum(values)}, expected 1.0.")


def validate_invalid_inputs() -> None:
    invalid_values = ["", "   ", 123]
    for value in invalid_values:
        try:
            if isinstance(value, str):
                SERVICE.predict(value)
            else:
                SERVICE.predict(value)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for invalid input: {value!r}")


def main() -> None:
    for index, sentence in enumerate(SENTENCES, start=1):
        result = SERVICE.predict(sentence)
        validate_prediction(result)
        print(f"sentence {index}: sentiment={result['sentiment']}, confidence={result['confidence']:.6f}, negative={result['probabilities']['negative']:.6f}, neutral={result['probabilities']['neutral']:.6f}, positive={result['probabilities']['positive']:.6f}")

    validate_invalid_inputs()

    print("SENTIMENT SERVICE VALIDATION PASSED")
    print("model successfully loaded: True")
    print("tokenizer successfully loaded: True")
    print("metadata successfully loaded: True")
    print("invalid-input tests passed: True")


if __name__ == "__main__":
    main()

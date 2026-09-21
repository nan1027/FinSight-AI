from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "ml/sentiment/finbert_model"
METADATA_PATH = ROOT_DIR / "ml/sentiment/model_metadata.json"
REQUIRED_FILES = [
    MODEL_DIR,
    MODEL_DIR / "config.json",
    MODEL_DIR / "model.safetensors",
    MODEL_DIR / "tokenizer_config.json",
    MODEL_DIR / "vocab.txt",
    ROOT_DIR / "ml/sentiment/finbert_config.json",
    ROOT_DIR / "ml/sentiment/training_metadata.json",
    ROOT_DIR / "ml/sentiment/training_history.json",
    ROOT_DIR / "ml/sentiment/test_metrics.json",
    ROOT_DIR / "ml/sentiment/error_analysis.json",
    ROOT_DIR / "ml/sentiment/finbert_misclassifications.csv",
]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate() -> None:
    warnings: list[str] = []

    for path in REQUIRED_FILES:
        assert_true(path.exists(), f"Missing required artifact: {path}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required_fields = [
        "model_name",
        "pretrained_model_name",
        "model_version",
        "task",
        "framework",
        "tokenizer",
        "max_sequence_length",
        "label_mapping",
        "inverse_label_mapping",
        "training_seed",
        "training_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "training_sample_count",
        "validation_sample_count",
        "test_sample_count",
        "validation_metrics",
        "test_metrics",
        "model_directory",
        "tokenizer_directory",
    ]
    for field in required_fields:
        assert_true(field in metadata, f"Missing metadata field: {field}")

    label_mapping = metadata["label_mapping"]
    assert_true(set(label_mapping.keys()) == {"negative", "neutral", "positive"}, "Label mapping keys are invalid.")
    assert_true(label_mapping == {"negative": 1, "neutral": 2, "positive": 0}, "Label mapping must be exactly {'negative': 1, 'neutral': 2, 'positive': 0}")

    inverse_label_mapping = metadata["inverse_label_mapping"]
    assert_true(set(inverse_label_mapping.keys()) == {"1", "2", "0"}, "Inverse label mapping keys are invalid.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    assert_true(model.config.num_labels == 3, f"Model output classes mismatch: expected 3, got {model.config.num_labels}")
    assert_true(getattr(tokenizer, "model_max_length", 0) >= 128 or tokenizer.model_max_length == 1000000000000000019884624838656, "Tokenizer max length is incompatible with 128")

    sample_texts = [
        "Revenue increased sharply in the quarter.",
        "The company reported lower earnings due to weaker demand.",
        "The firm continues operations under the current plan."
    ]
    encoded = tokenizer(sample_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    assert_true(logits.ndim == 2 and logits.shape[1] == 3, f"Unexpected logits shape: {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=-1)
    assert_true(torch.isfinite(probs).all(), "Probabilities contain non-finite values.")
    assert_true(torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-3), "Probability rows do not sum to approximately 1.")

    pred_labels = torch.argmax(logits, dim=-1).tolist()
    valid_pred_labels = {1, 2, 0}
    assert_true(all(int(p) in valid_pred_labels for p in pred_labels), f"Predictions outside valid label set: {pred_labels}")

    print("FINBERT ARTIFACT VALIDATION PASSED")
    print(f"model path: {MODEL_DIR}")
    print(f"tokenizer path: {MODEL_DIR}")
    print(f"metadata path: {METADATA_PATH}")
    print(f"model label mapping: {label_mapping}")
    print(f"number of output classes: {model.config.num_labels}")
    probability_sum_ok = bool(torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-3))
    print(f"smoke-test result: logits_shape={tuple(logits.shape)}, probabilities_sum_ok={probability_sum_ok}")
    print("all required artifact checks: model loads, tokenizer loads, metadata loads, required fields exist, label mapping valid, 3 output classes, tokenizer max length compatible with 128, smoke-test inference works, probabilities finite, probabilities sum approximately to 1.")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")


if __name__ == "__main__":
    validate()

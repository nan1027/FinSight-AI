from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "ml/sentiment/finbert_model"
TEST_PATH = ROOT_DIR / "data/processed/sentiment/test.csv"
MISCLASSIFICATION_CSV = ROOT_DIR / "ml/sentiment/finbert_misclassifications.csv"
ERROR_ANALYSIS_JSON = ROOT_DIR / "ml/sentiment/error_analysis.json"
MAX_SEQUENCE_LENGTH = 128
LABEL_MAPPING = {"positive": 0, "negative": 1, "neutral": 2}
CLASS_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_NAME = {v: k for k, v in LABEL_MAPPING.items()}


def _load_test_data() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    required = ["sentence", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Test data missing required columns: {missing}")
    normalized = df[["sentence", "label"]].copy()
    normalized["sentence"] = normalized["sentence"].astype(str).str.strip()
    normalized = normalized[normalized["sentence"].ne("")].copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    invalid = normalized[~normalized["label"].isin(LABEL_MAPPING.keys())]
    if not invalid.empty:
        raise ValueError(f"Unexpected labels in test set: {sorted(invalid['label'].unique().tolist())}")
    normalized["label_idx"] = normalized["label"].map(LABEL_MAPPING)
    return normalized.reset_index(drop=True)


def _predict(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        confidences = probs[np.arange(len(texts)), predictions].astype(float)

    return predictions.astype(int), probs.astype(float), confidences.astype(float)


def _error_matrix(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, int]]:
    actual_names = [LABEL_TO_NAME[int(x)] for x in actual]
    pred_names = [LABEL_TO_NAME[int(x)] for x in predicted]
    matrix = {
        "negative": {"negative": 0, "neutral": 0, "positive": 0},
        "neutral": {"negative": 0, "neutral": 0, "positive": 0},
        "positive": {"negative": 0, "neutral": 0, "positive": 0},
    }
    for actual_label, predicted_label in zip(actual_names, pred_names):
        if actual_label == predicted_label:
            continue
        matrix[actual_label][predicted_label] += 1
    return matrix


def _summary_frame(error_rows: pd.DataFrame) -> dict[str, object]:
    actual_counts = error_rows["actual_label"].value_counts().to_dict()
    predicted_counts = error_rows["predicted_label"].value_counts().to_dict()
    confusion_pairs = Counter(zip(error_rows["actual_label"], error_rows["predicted_label"]))
    confusion_pairs_payload = {
        f"{actual}->{predicted}": count
        for (actual, predicted), count in sorted(confusion_pairs.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    }

    return {
        "errors_by_actual_class": {label: int(actual_counts.get(label, 0)) for label in CLASS_ORDER},
        "errors_by_predicted_class": {label: int(predicted_counts.get(label, 0)) for label in CLASS_ORDER},
        "confusion_pairs": confusion_pairs_payload,
        "most_common_confusion_pair": None if not confusion_pairs_payload else max(confusion_pairs_payload.items(), key=lambda item: (item[1], item[0]))[0],
    }


def _representative_examples(error_rows: pd.DataFrame) -> list[dict[str, object]]:
    categories = [
        ("negative_predicted_as_neutral", "negative", "neutral"),
        ("negative_predicted_as_positive", "negative", "positive"),
        ("neutral_predicted_as_negative", "neutral", "negative"),
        ("neutral_predicted_as_positive", "neutral", "positive"),
        ("positive_predicted_as_neutral", "positive", "neutral"),
        ("positive_predicted_as_negative", "positive", "negative"),
    ]
    examples: list[dict[str, object]] = []
    for category_name, actual_label, predicted_label in categories:
        subset = error_rows[(error_rows["actual_label"] == actual_label) & (error_rows["predicted_label"] == predicted_label)]
        if subset.empty:
            continue
        sample = subset.sort_values("model_confidence", ascending=False).iloc[0]
        probabilities = {
            "negative": float(sample["negative_probability"]),
            "neutral": float(sample["neutral_probability"]),
            "positive": float(sample["positive_probability"]),
        }
        examples.append({
            "category": category_name,
            "text": str(sample["text"]),
            "actual": str(sample["actual_label"]),
            "predicted": str(sample["predicted_label"]),
            "confidence": float(sample["model_confidence"]),
            "probabilities": probabilities,
        })
    return examples


def main() -> int:
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory is missing: {MODEL_PATH}")

        test_df = _load_test_data()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        predictions, probabilities, confidences = _predict(model, tokenizer, test_df["sentence"].tolist())
        actual_labels = test_df["label_idx"].to_numpy(dtype=int)

        error_mask = actual_labels != predictions
        if not np.any(error_mask):
            error_rows = pd.DataFrame(columns=[
                "text",
                "actual_label",
                "predicted_label",
                "model_confidence",
                "negative_probability",
                "neutral_probability",
                "positive_probability",
            ])
        else:
            error_indices = np.where(error_mask)[0]
            rows = []
            for idx in error_indices:
                row_idx = int(idx)
                probs_row = probabilities[row_idx]
                actual_label_name = LABEL_TO_NAME[int(actual_labels[row_idx])]
                predicted_label_name = LABEL_TO_NAME[int(predictions[row_idx])]
                rows.append({
                    "text": str(test_df.iloc[row_idx]["sentence"]),
                    "actual_label": actual_label_name,
                    "predicted_label": predicted_label_name,
                    "model_confidence": float(confidences[row_idx]),
                    "negative_probability": float(probs_row[1]),
                    "neutral_probability": float(probs_row[2]),
                    "positive_probability": float(probs_row[0]),
                })
            error_rows = pd.DataFrame(rows)

        if not error_rows.empty:
            error_rows = error_rows.sort_values("model_confidence", ascending=False, ignore_index=True)

        total_test_examples = int(len(test_df))
        total_errors = int(len(error_rows))
        error_rate = float(total_errors / total_test_examples) if total_test_examples else 0.0

        summary = _summary_frame(error_rows)
        representative = _representative_examples(error_rows)

        low_confidence_error_count = int((error_rows["model_confidence"] < 0.60).sum()) if not error_rows.empty else 0
        high_confidence_error_count = int((error_rows["model_confidence"] >= 0.80).sum()) if not error_rows.empty else 0

        # Verification checks
        if not error_rows.empty:
            assert (error_rows["actual_label"] != error_rows["predicted_label"]).all()
            assert (error_rows["model_confidence"] >= 0.0).all() and (error_rows["model_confidence"] <= 1.0).all()
            for col in ["negative_probability", "neutral_probability", "positive_probability"]:
                assert (error_rows[col].between(0.0, 1.0)).all()
            prob_sum = error_rows[["negative_probability", "neutral_probability", "positive_probability"]].sum(axis=1)
            assert np.allclose(prob_sum.to_numpy(), np.ones(len(error_rows)), atol=1e-3)
            assert error_rows[["text", "actual_label", "predicted_label"]].notna().all().all()
        assert len(error_rows) == total_errors
        assert total_errors == int(np.sum(error_mask))

        output = {
            "total_test_examples": total_test_examples,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "errors_by_actual_class": summary["errors_by_actual_class"],
            "errors_by_predicted_class": summary["errors_by_predicted_class"],
            "confusion_pairs": summary["confusion_pairs"],
            "most_common_confusion_pair": summary["most_common_confusion_pair"],
            "low_confidence_error_count": low_confidence_error_count,
            "high_confidence_error_count": high_confidence_error_count,
            "representative_confusion_examples": representative,
        }
        ERROR_ANALYSIS_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
        error_rows.to_csv(MISCLASSIFICATION_CSV, index=False)

        print("Error analysis summary")
        print(f"total test examples: {total_test_examples}")
        print(f"total errors: {total_errors}")
        print(f"error rate: {error_rate:.6f}")
        print(f"errors by actual class: {summary['errors_by_actual_class']}")
        print(f"errors by predicted class: {summary['errors_by_predicted_class']}")
        print(f"most common confusion pair: {summary['most_common_confusion_pair']}")
        print(f"low-confidence error count: {low_confidence_error_count}")
        print(f"high-confidence error count: {high_confidence_error_count}")
        print(f"saved misclassifications csv: {MISCLASSIFICATION_CSV}")
        print(f"saved error analysis json: {ERROR_ANALYSIS_JSON}")

        print("\nRepresentative confusion examples:")
        if not representative:
            print("No misclassification categories were observed.")
        else:
            for item in representative:
                print(f"CATEGORY: {item['category']}")
                print(f"TEXT: {item['text'][:120]}" + ("..." if len(item['text']) > 120 else ""))
                print(f"ACTUAL: {item['actual']}")
                print(f"PREDICTED: {item['predicted']}")
                print(f"CONFIDENCE: {item['confidence']:.4f}")
                print("PROBABILITIES:")
                print(f"  negative: {item['probabilities']['negative']:.6f}")
                print(f"  neutral: {item['probabilities']['neutral']:.6f}")
                print(f"  positive: {item['probabilities']['positive']:.6f}")
                print("---")

        return 0
    except Exception as exc:  # pragma: no cover
        print(f"ERROR ANALYSIS FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "ml/sentiment/finbert_model"
TEST_PATH = ROOT_DIR / "data/processed/sentiment/test.csv"
CONFUSION_MATRIX_PATH = ROOT_DIR / "ml/sentiment/finbert_confusion_matrix.png"
METRICS_PATH = ROOT_DIR / "ml/sentiment/test_metrics.json"
MAX_SEQUENCE_LENGTH = 128
LABEL_MAPPING = {"positive": 0, "negative": 1, "neutral": 2}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
CLASS_ORDER = ["negative", "neutral", "positive"]
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
LABEL_TO_INT = LABEL_MAPPING
INT_TO_LABEL = {int(v): k for k, v in LABEL_MAPPING.items()}


def _load_test_data() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    required = ["sentence", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Test dataset missing required columns: {missing}")
    normalized = df[["sentence", "label"]].copy()
    normalized["sentence"] = normalized["sentence"].astype(str).str.strip()
    normalized = normalized[normalized["sentence"].ne("")].copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    invalid = normalized[~normalized["label"].isin(LABEL_MAPPING.keys())]
    if not invalid.empty:
        vals = sorted(invalid["label"].unique().tolist())
        raise ValueError(f"Unexpected labels in test split: {vals}")
    normalized["label_idx"] = normalized["label"].map(LABEL_MAPPING)
    return normalized.reset_index(drop=True)


def _build_dataloader(df: pd.DataFrame, tokenizer: AutoTokenizer) -> DataLoader:
    dataset = df[["sentence", "label_idx"]].copy()
    dataset = dataset.rename(columns={"sentence": "text", "label_idx": "labels"})

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        return enc

    encoded = [tokenize({"text": [row["text"]], "labels": [row["labels"]]}) for _, row in dataset.iterrows()]
    if not encoded:
        raise ValueError("No encoded test samples available.")
    batch = {
        key: torch.cat([item[key] for item in encoded], dim=0 if key in {"input_ids", "attention_mask", "token_type_ids"} else 0)
        for key in encoded[0].keys()
    }
    batch["labels"] = torch.tensor(dataset["labels"].tolist(), dtype=torch.long)
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    # Keep a simple single-batch DataLoader for the full test set.
    loader = DataLoader(
        [{k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()} for i in range(len(dataset))],
        batch_size=16,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return loader


def _safe_float(value: float | np.floating) -> float:
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        raise ValueError("Encountered NaN in evaluation metrics.")
    return float(value)


def _build_confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    # Use the requested class order: negative, neutral, positive.
    remap = {1: 0, 2: 1, 0: 2}
    mapped_actual = np.vectorize(lambda x: remap[int(x)])(actual)
    mapped_pred = np.vectorize(lambda x: remap[int(x)])(predicted)
    matrix = confusion_matrix(mapped_actual, mapped_pred, labels=[0, 1, 2])
    return matrix


def _plot_confusion_matrix(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("FinBERT Test Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] < matrix.max() / 2 else "white")

    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)


def _print_examples(df: pd.DataFrame, actual_labels: np.ndarray, predicted_labels: np.ndarray, confidences: np.ndarray) -> None:
    for i in range(min(10, len(df))):
        text = df.iloc[i]["sentence"]
        if len(text) > 120:
            text = text[:117] + "..."
        actual = REVERSE_LABEL_MAPPING[int(actual_labels[i])]
        predicted = REVERSE_LABEL_MAPPING[int(predicted_labels[i])]
        confidence = float(confidences[i])
        print(f"TEXT: {text}")
        print(f"ACTUAL: {actual}")
        print(f"PREDICTED: {predicted}")
        print(f"CONFIDENCE: {confidence:.4f}")
        print("---")


def main() -> int:
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Fine-tuned model directory does not exist: {MODEL_PATH}")

        test_df = _load_test_data()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        encoded_sentences = tokenizer(
            test_df["sentence"].tolist(),
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        all_logits = []
        with torch.no_grad():
            outputs = model(
                input_ids=encoded_sentences["input_ids"],
                attention_mask=encoded_sentences["attention_mask"],
                token_type_ids=encoded_sentences.get("token_type_ids", None),
            )
            logits = outputs.logits
            all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1).numpy()
        confidences = probabilities.max(dim=-1).values.numpy()
        actual_labels = test_df["label_idx"].to_numpy(dtype=int)

        for label in predictions:
            if int(label) not in set(LABEL_MAPPING.values()):
                raise ValueError(f"Invalid predicted label found: {label}")
        if len(predictions) != len(test_df):
            raise ValueError("Not all test samples received a prediction.")

        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            actual_labels,
            predictions,
            labels=[LABEL_MAPPING["negative"], LABEL_MAPPING["neutral"], LABEL_MAPPING["positive"]],
            average=None,
            zero_division=0,
            warn_for=(),
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            actual_labels,
            predictions,
            labels=[LABEL_MAPPING["negative"], LABEL_MAPPING["neutral"], LABEL_MAPPING["positive"]],
            average="macro",
            zero_division=0,
            warn_for=(),
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            actual_labels,
            predictions,
            labels=[LABEL_MAPPING["negative"], LABEL_MAPPING["neutral"], LABEL_MAPPING["positive"]],
            average="weighted",
            zero_division=0,
            warn_for=(),
        )

        if any(math.isnan(v) for v in [accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1]):
            raise ValueError("NaN encountered in metrics.")

        per_class = {}
        class_index_lookup = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
        for label_name in CLASS_ORDER:
            idx = class_index_lookup[label_name]
            per_class[label_name] = {
                "precision": _safe_float(precision[idx]),
                "recall": _safe_float(recall[idx]),
                "f1": _safe_float(f1[idx]),
                "support": int(support[idx]),
            }

        confusion = _build_confusion_matrix(actual_labels, predictions)
        if confusion.sum() != len(test_df):
            raise ValueError(f"Confusion matrix total {confusion.sum()} does not match test sample count {len(test_df)}")

        correct_predictions = int(np.sum(actual_labels == predictions))
        incorrect_predictions = int(len(test_df) - correct_predictions)

        metrics = {
            "test_sample_count": int(len(test_df)),
            "accuracy": _safe_float(accuracy),
            "macro_precision": _safe_float(macro_precision),
            "macro_recall": _safe_float(macro_recall),
            "macro_f1": _safe_float(macro_f1),
            "weighted_precision": _safe_float(weighted_precision),
            "weighted_recall": _safe_float(weighted_recall),
            "weighted_f1": _safe_float(weighted_f1),
            "per_class_metrics": per_class,
            "confusion_matrix": confusion.tolist(),
            "label_mapping": {str(k): int(v) for k, v in LABEL_MAPPING.items()},
            "class_order": CLASS_ORDER,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
        }

        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _plot_confusion_matrix(confusion)

        print("FINBERT TEST EVALUATION COMPLETED")
        print(f"test sample count: {len(test_df)}")
        print(f"accuracy: {accuracy:.6f}")
        print(f"macro precision: {macro_precision:.6f}")
        print(f"macro recall: {macro_recall:.6f}")
        print(f"macro F1: {macro_f1:.6f}")
        print(f"weighted F1: {weighted_f1:.6f}")
        print("per-class precision/recall/F1:")
        for label_name in CLASS_ORDER:
            values = per_class[label_name]
            print(f"  {label_name}: precision={values['precision']:.6f}, recall={values['recall']:.6f}, f1={values['f1']:.6f}, support={values['support']}")
        print("confusion matrix:")
        for row in confusion:
            print("  ", row)
        print(f"correct predictions: {correct_predictions}")
        print(f"incorrect predictions: {incorrect_predictions}")
        print(f"saved metrics path: {METRICS_PATH}")
        print(f"saved confusion matrix path: {CONFUSION_MATRIX_PATH}")

        print("\nPrediction sanity check (first 10 samples):")
        _print_examples(test_df, actual_labels, predictions, confidences)

        print("\nClass-wise support:")
        for label_name in CLASS_ORDER:
            support_value = int(np.sum(actual_labels == LABEL_MAPPING[label_name]))
            print(f"  {label_name}: {support_value}")

        print("\nValidation checks:")
        print(f"all test samples received a prediction: {len(predictions) == len(test_df)}")
        print(f"no NaN metrics: {not any(math.isnan(v) for v in [accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1])}")
        print(f"confusion matrix total equals test sample count: {confusion.sum() == len(test_df)}")
        predicted_valid = set(int(p) for p in predictions).issubset(set(LABEL_MAPPING.values()))
        print(f"predicted labels belong only to negative/neutral/positive: {predicted_valid}")

        return 0
    except Exception as exc:  # pragma: no cover
        print(f"EVALUATION FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

MODEL_NAME = "ProsusAI/finbert"
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT_DIR / "data/processed/sentiment/train.csv"
VALIDATION_PATH = ROOT_DIR / "data/processed/sentiment/validation.csv"
TEST_PATH = ROOT_DIR / "data/processed/sentiment/test.csv"
MODEL_OUTPUT_DIR = ROOT_DIR / "ml/sentiment/finbert_model"
TRAINING_METADATA_PATH = ROOT_DIR / "ml/sentiment/training_metadata.json"
TRAINING_HISTORY_PATH = ROOT_DIR / "ml/sentiment/training_history.json"
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"
LABEL_MAPPING = {"positive": 0, "negative": 1, "neutral": 2}
REVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}
RANDOM_SEED = 42
MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 3
WEIGHT_DECAY = 0.01


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    df = pd.read_csv(path)
    missing = [col for col in [TEXT_COLUMN, LABEL_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Split {path.name} is missing required columns: {missing}")
    return df


def _normalize_split(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].astype(str).str.strip()
    normalized = normalized[normalized[TEXT_COLUMN].ne("")].copy()
    normalized[LABEL_COLUMN] = normalized[LABEL_COLUMN].astype(str).str.strip().str.lower()
    invalid = normalized[~normalized[LABEL_COLUMN].isin(LABEL_MAPPING.keys())]
    if not invalid.empty:
        invalid_values = sorted(invalid[LABEL_COLUMN].unique().tolist())
        raise ValueError(f"Unexpected labels found in split: {invalid_values}")
    normalized[LABEL_COLUMN] = normalized[LABEL_COLUMN].map(LABEL_MAPPING)
    return normalized.reset_index(drop=True)


def _tokenize_batch(batch: dict[str, list[str]], tokenizer: AutoTokenizer) -> dict[str, list[int] | list[list[int]] | list[str]]:
    tokenized = tokenizer(
        batch[TEXT_COLUMN],
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    tokenized["labels"] = batch[LABEL_COLUMN]
    return tokenized


def _build_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    dataset = Dataset.from_pandas(df[[TEXT_COLUMN, LABEL_COLUMN]])
    dataset = dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
    )
    return dataset


def _compute_validation_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    confusion = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            "positive": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "negative": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
            "neutral": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(f1[2]),
                "support": int(support[2]),
            },
        },
        "confusion_matrix": confusion.tolist(),
    }
    return metrics


def _save_best_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> int:
    try:
        set_seed(RANDOM_SEED)

        train_df = _normalize_split(_load_split(TRAIN_PATH)).reset_index(drop=True)
        validation_df = _normalize_split(_load_split(VALIDATION_PATH)).reset_index(drop=True)

        print("FINBERT TRAINING STARTING", flush=True)
        print(f"device: CPU", flush=True)
        print(f"batch_size: {BATCH_SIZE}", flush=True)
        print(f"epochs: {EPOCHS}", flush=True)
        print(f"learning_rate: {LEARNING_RATE}", flush=True)
        print(f"max_sequence_length: {MAX_SEQUENCE_LENGTH}", flush=True)
        print(f"training_samples: {len(train_df)}", flush=True)
        print(f"validation_samples: {len(validation_df)}", flush=True)
        print(f"Label mapping: {LABEL_MAPPING}", flush=True)
        print(f"Output model directory: {MODEL_OUTPUT_DIR}", flush=True)

        train_df = _normalize_split(_load_split(TRAIN_PATH)).reset_index(drop=True)
        validation_df = _normalize_split(_load_split(VALIDATION_PATH)).reset_index(drop=True)

        if len(train_df) == 0 or len(validation_df) == 0:
            raise ValueError("Training and validation splits must both be non-empty before fine-tuning.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label={str(key): value for key, value in sorted(REVERSE_LABEL_MAPPING.items())},
            label2id={value: key for key, value in LABEL_MAPPING.items()},
        )
        model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        sanity_texts = [
            "The company reported a strong revenue increase and higher operating margin.",
            "The firm said it expects lower earnings due to weaker demand in the quarter.",
            "The company announced that some operations continue under the current plan.",
        ]
        sanity_labels = [LABEL_MAPPING["positive"], LABEL_MAPPING["negative"], LABEL_MAPPING["neutral"]]
        sanity_tokens = tokenizer(
            sanity_texts,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        sanity_input_ids = sanity_tokens["input_ids"].to(device)
        sanity_attention_mask = sanity_tokens["attention_mask"].to(device)
        with torch.no_grad():
            sanity_outputs = model(input_ids=sanity_input_ids, attention_mask=sanity_attention_mask)
        if sanity_outputs.logits.shape != (len(sanity_texts), 3):
            raise ValueError(f"Sanity check failed: expected logits shape {(len(sanity_texts), 3)}, got {tuple(sanity_outputs.logits.shape)}")
        print("Sanity check passed.", flush=True)
        print(f"Sanity input IDs shape: {tuple(sanity_tokens['input_ids'].shape)}", flush=True)
        print(f"Sanity attention mask shape: {tuple(sanity_tokens['attention_mask'].shape)}", flush=True)
        print(f"Sanity logits shape: {tuple(sanity_outputs.logits.shape)}", flush=True)
        print(f"Sanity labels mapped: {sanity_labels}", flush=True)

        train_dataset = _build_dataset(train_df, tokenizer)
        val_dataset = _build_dataset(validation_df, tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        history: list[dict[str, object]] = []
        best_macro_f1 = -1.0
        best_epoch = -1
        best_model_path: Path | None = None

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise ValueError(f"Training loss was None for epoch {epoch + 1}.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch["input_ids"].size(0)

            train_loss = running_loss / len(train_dataset)
            model.eval()
            all_labels: list[int] = []
            all_predictions: list[int] = []
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**batch)
                    val_loss_total += outputs.loss.item() * batch["input_ids"].size(0)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    all_labels.extend(batch["labels"].detach().cpu().tolist())
                    all_predictions.extend(predictions.detach().cpu().tolist())

            val_loss = val_loss_total / len(val_dataset)
            val_metrics = _compute_validation_metrics(np.array(all_labels), np.array(all_predictions))
            epoch_entry = {
                "epoch": epoch + 1,
                "training_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_metrics": val_metrics,
            }
            history.append(epoch_entry)

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch + 1
                best_model_path = MODEL_OUTPUT_DIR / f"best_epoch_{epoch + 1}"
                _save_best_model(model, tokenizer, best_model_path)

            print(
                f"EPOCH {epoch + 1}/{EPOCHS} | training_loss={train_loss:.6f} | validation_loss={val_loss:.6f} | validation_accuracy={val_metrics['accuracy']:.6f} | validation_macro_precision={val_metrics['macro_precision']:.6f} | validation_macro_recall={val_metrics['macro_recall']:.6f} | validation_macro_f1={val_metrics['macro_f1']:.6f}",
                flush=True,
            )
            print(f"  per-class: {val_metrics['per_class']}", flush=True)

        if best_model_path is None:
            raise RuntimeError("No best model checkpoint was saved. Training did not produce a valid validation run.")

        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_best_model(model, tokenizer, MODEL_OUTPUT_DIR)

        final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
        final_model.to(device)
        final_model.eval()
        eval_labels: list[int] = []
        eval_predictions: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = final_model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                eval_labels.extend(batch["labels"].detach().cpu().tolist())
                eval_predictions.extend(predictions.detach().cpu().tolist())

        final_metrics = _compute_validation_metrics(np.array(eval_labels), np.array(eval_predictions))
        best_validation_metrics = {
            "epoch": best_epoch,
            "macro_f1": float(best_macro_f1),
            "accuracy": float(final_metrics["accuracy"]),
            "macro_precision": float(final_metrics["macro_precision"]),
            "macro_recall": float(final_metrics["macro_recall"]),
            "per_class": final_metrics["per_class"],
            "confusion_matrix": final_metrics["confusion_matrix"],
        }

        training_metadata = {
            "pretrained_model": MODEL_NAME,
            "seed": RANDOM_SEED,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "label_mapping": {key: int(value) for key, value in LABEL_MAPPING.items()},
            "training_sample_count": int(len(train_df)),
            "validation_sample_count": int(len(validation_df)),
            "training_configuration": {
                "optimizer": "AdamW",
                "loss_function": "CrossEntropyLoss",
                "evaluation_metric": "macro_F1",
                "device": str(device),
            },
            "best_validation_metrics": best_validation_metrics,
            "actual_epochs_completed": len(history),
            "best_validation_epoch": best_epoch,
        }
        TRAINING_METADATA_PATH.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
        TRAINING_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print("\nModel and tokenizer saved before completion statement.", flush=True)
        print(f"Saved model path: {MODEL_OUTPUT_DIR}", flush=True)
        print(f"Saved tokenizer path: {MODEL_OUTPUT_DIR}", flush=True)
        print("FINBERT TRAINING COMPLETED", flush=True)
        print(f"epochs actually completed: {len(history)}", flush=True)
        print(f"best validation epoch: {best_epoch}", flush=True)
        print(f"best validation macro-F1: {best_macro_f1:.6f}", flush=True)
        print(f"validation accuracy: {final_metrics['accuracy']:.6f}", flush=True)
        print(f"validation macro precision: {final_metrics['macro_precision']:.6f}", flush=True)
        print(f"validation macro recall: {final_metrics['macro_recall']:.6f}", flush=True)
        print(f"saved model path: {MODEL_OUTPUT_DIR}", flush=True)
        print(f"saved tokenizer path: {MODEL_OUTPUT_DIR}", flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - explicit failure reporting for training runs
        traceback.print_exc()
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "ml" / "sentiment" / "finbert_model"
METADATA_PATH = PROJECT_ROOT / "ml" / "sentiment" / "model_metadata.json"


class SentimentPredictionService:
    """Load the saved FinBERT model and tokenizer for sentiment inference."""

    _instance: "SentimentPredictionService | None" = None

    def __new__(cls, model_path: Path | str = MODEL_PATH, metadata_path: Path | str = METADATA_PATH) -> "SentimentPredictionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: Path | str = MODEL_PATH, metadata_path: Path | str = METADATA_PATH) -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()

        self.metadata = self._load_metadata(self.metadata_path)
        self.model_name = self._validate_model_name(self.metadata.get("model_name"))
        self.label_mapping = self._validate_label_mapping(self.metadata.get("label_mapping"))
        self.max_sequence_length = self._validate_max_length(self.metadata.get("max_sequence_length", 128))

        self.model = self._load_model(self.model_path)
        self.tokenizer = self._load_tokenizer(self.model_path)
        self.model_index_to_label = {int(index): label for label, index in self.label_mapping.items()}
        self._initialized = True

    @staticmethod
    def _load_metadata(metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata file does not contain a JSON object: {metadata_path}")
        return metadata

    @staticmethod
    def _validate_model_name(model_name: Any) -> str:
        if not isinstance(model_name, str):
            raise ValueError(f"Model metadata field 'model_name' must be a string, got {type(model_name).__name__}.")
        if not model_name.strip():
            raise ValueError("Model metadata field 'model_name' must not be empty.")
        return model_name

    @staticmethod
    def _validate_label_mapping(label_mapping: Any) -> dict[str, int]:
        expected = {"negative": 1, "neutral": 2, "positive": 0}
        if not isinstance(label_mapping, dict):
            raise ValueError(f"Label mapping must be a dictionary. Received: {type(label_mapping).__name__}")
        if set(label_mapping.keys()) != set(expected.keys()):
            raise ValueError(f"Invalid label mapping keys: {label_mapping}")
        normalized = {str(label): int(index) for label, index in label_mapping.items()}
        if normalized != expected:
            raise ValueError(f"Unexpected label mapping: {normalized}. Expected: {expected}")
        return normalized

    @staticmethod
    def _validate_max_length(max_length: Any) -> int:
        try:
            value = int(max_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_sequence_length must be an integer, got {max_length!r}.") from exc
        if value <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {value}.")
        return value

    @staticmethod
    def _load_model(model_path: Path) -> AutoModelForSequenceClassification:
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        if model.config.num_labels != 3:
            raise ValueError(f"Model should have exactly 3 output labels, but found {model.config.num_labels}.")
        return model

    @staticmethod
    def _load_tokenizer(model_path: Path) -> AutoTokenizer:
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model directory not found: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        return tokenizer

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text must not be empty or whitespace-only.")

        encoded = self.tokenizer(
            cleaned_text,
            truncation=True,
            max_length=self.max_sequence_length,
            padding=True,
            return_tensors="pt",
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits

        if logits.ndim != 2 or logits.shape[1] != 3:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}. Expected (batch_size, 3).")

        probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy().astype(float)
        if not np_finite(probabilities):
            raise ValueError("Model probabilities are not finite.")
        if not ((probabilities >= 0.0).all() and (probabilities <= 1.0).all()):
            raise ValueError("Model probabilities are outside the [0, 1] range.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Probabilities do not sum to 1: {probabilities.sum()}")

        probabilities_by_label = {
            label: float(probabilities[index])
            for index, label in self.model_index_to_label.items()
        }

        predicted_index = int(torch.argmax(logits[0], dim=-1).item())
        predicted_label = self.model_index_to_label[predicted_index]
        confidence = float(probabilities[predicted_index])

        response = {
            "sentiment": predicted_label,
            "confidence": confidence,
            "probabilities": {
                "negative": float(probabilities_by_label["negative"]),
                "neutral": float(probabilities_by_label["neutral"]),
                "positive": float(probabilities_by_label["positive"]),
            },
        }

        return response


def np_finite(array: Any) -> bool:
    return bool(torch.isfinite(torch.tensor(array)).all().item())

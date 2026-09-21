from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
CONFIG_PATH = Path("ml/sentiment/finbert_config.json")
PROJECT_LABELS = ["negative", "neutral", "positive"]
SAMPLE_TEXTS = {
    "negative": "The company reported a sharp decline in quarterly earnings and lower revenue guidance.",
    "neutral": "The firm said it expects operations to continue as planned over the next quarter.",
    "positive": "Analysts upgraded the stock after strong sales growth and higher operating margins.",
}


def _normalize_label_map(raw_map: dict[str | int, str]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for key, value in raw_map.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        normalized[index] = str(value).strip().lower()
    return dict(sorted(normalized.items()))


def _summarize_config(config: object) -> dict[str, object]:
    if not hasattr(config, "to_dict"):
        return {"type": type(config).__name__}
    config_dict = config.to_dict()
    selected = {
        "architectures": config_dict.get("architectures"),
        "model_type": config_dict.get("model_type"),
        "num_labels": config_dict.get("num_labels"),
        "id2label": config_dict.get("id2label"),
        "label2id": config_dict.get("label2id"),
        "hidden_size": config_dict.get("hidden_size"),
        "num_hidden_layers": config_dict.get("num_hidden_layers"),
        "num_attention_heads": config_dict.get("num_attention_heads"),
        "max_position_embeddings": config_dict.get("max_position_embeddings"),
    }
    return {key: value for key, value in selected.items() if value is not None}


def _infer_project_label_order(id2label: dict[int, str]) -> list[str]:
    ordered: list[str] = []
    for index in sorted(id2label):
        label = str(id2label[index]).strip().lower()
        if label:
            ordered.append(label)
    if ordered:
        return ordered
    return PROJECT_LABELS[:]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    config = model.config
    id2label = _normalize_label_map(getattr(config, "id2label", {}) or {})
    project_order = _infer_project_label_order(id2label)
    model_label_sequence = project_order if project_order else PROJECT_LABELS[:]

    encoded = tokenizer(list(SAMPLE_TEXTS.values()), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits

    logits_shape = tuple(logits.shape)
    prediction_ids = torch.argmax(logits, dim=-1).tolist()
    predictions = {sample_name: model_label_sequence[prediction_id] for sample_name, prediction_id in zip(SAMPLE_TEXTS.keys(), prediction_ids)}

    metadata = {
        "model_name": MODEL_NAME,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "project_labels": PROJECT_LABELS,
        "model_label_order": model_label_sequence,
        "model_config": _summarize_config(config),
        "label_mapping": {str(index): label for index, label in id2label.items()},
        "tokenizer_smoke_test": {
            "batch_size": encoded["input_ids"].shape[0],
            "sequence_length": encoded["input_ids"].shape[1],
            "logits_shape": list(logits_shape),
            "predictions": predictions,
        },
    }

    CONFIG_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"model: {MODEL_NAME}")
    print(f"tokenizer class: {type(tokenizer).__name__}")
    print(f"model class: {type(model).__name__}")
    print(f"model label order: {model_label_sequence}")
    print(f"input batch size: {encoded['input_ids'].shape[0]}")
    print(f"sequence length: {encoded['input_ids'].shape[1]}")
    print(f"logits shape: {logits_shape}")
    print(f"sample predictions: {predictions}")
    print(f"config saved to: {CONFIG_PATH}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


MODEL_PATH = Path(__file__).resolve().parents[3] / "ml" / "risk_prediction" / "xgboost_risk_model.json"
METADATA_PATH = Path(__file__).resolve().parents[3] / "ml" / "risk_prediction" / "model_metadata.json"


class RiskPredictionService:
    """Load the trained bankruptcy risk model and metadata for inference."""

    def __init__(self, model_path: Path | str = MODEL_PATH, metadata_path: Path | str = METADATA_PATH) -> None:
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        self.model = self._load_model(self.model_path)
        self.metadata = self._load_metadata(self.metadata_path)
        self.required_features = list(self.metadata["features"])

        # TreeExplainer is used to attribute the XGBoost score to individual features.
        self.explainer = shap.TreeExplainer(self.model)

    def _load_model(self, model_path: Path) -> XGBClassifier:
        """Load the trained XGBoost model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = XGBClassifier()
        model.load_model(str(model_path))
        return model

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        """Load model metadata used to validate and order input features."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _risk_level(self, probability: float) -> str:
        """Map a probability to a coarse risk category."""
        if probability < 0.30:
            return "Low"
        if probability < 0.60:
            return "Medium"
        return "High"

    def _extract_shap_contributions(self, shap_values: Any, sample_count: int) -> np.ndarray:
        """Normalize SHAP output across possible versions and return the single-sample feature contribution vector."""
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            elif len(shap_values) == 1:
                shap_values = shap_values[0]

        values = np.asarray(shap_values, dtype=float)

        if values.ndim == 3:
            if values.shape[-1] >= 2:
                values = values[:, :, 1]
            else:
                values = values[:, :, 0]

        if values.ndim == 2 and values.shape[0] == sample_count:
            return values[0]
        if values.ndim > 1:
            return values.reshape(-1)
        return values

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict bankruptcy probability for a single observation using the model metadata order."""
        missing_features = [feature for feature in self.required_features if feature not in features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        extra_features = [feature for feature in features if feature not in self.required_features]
        if extra_features:
            raise ValueError(f"Unexpected extra features: {extra_features}")

        ordered_features = {feature: features[feature] for feature in self.required_features}
        input_df = pd.DataFrame([ordered_features], columns=self.required_features)

        probability = float(self.model.predict_proba(input_df)[0, 1])
        result = {
            "bankruptcy_probability": probability,
            "risk_level": self._risk_level(probability),
        }

        # TreeExplainer is used to measure how each feature pushes the XGBoost positive-class probability.
        shap_values = self.explainer.shap_values(input_df)
        positive_contributions = self._extract_shap_contributions(shap_values, input_df.shape[0])
        positive_contributions = np.asarray(positive_contributions, dtype=float).reshape(-1)

        if positive_contributions.size != len(self.required_features):
            raise ValueError(
                "SHAP output shape does not match the feature count: "
                f"expected {len(self.required_features)}, got {positive_contributions.size}"
            )

        contributions = [
            {
                "feature": feature_name,
                "shap_value": float(value),
            }
            for feature_name, value in zip(self.required_features, positive_contributions)
        ]

        contributions = sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)[:5]
        result["top_contributors"] = contributions
        return result

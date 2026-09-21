from __future__ import annotations

from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    """Input payload for bankruptcy risk prediction."""

    features: dict[str, float] = Field(
        ...,
        description="Dictionary of model input features keyed by the exact feature names required by the trained XGBoost risk model.",
    )


class RiskContributor(BaseModel):
    """A single feature contribution from the SHAP explanation."""

    feature: str = Field(
        ...,
        description="Feature name from the trained model metadata, preserved exactly as configured for the model.",
    )
    shap_value: float = Field(
        ...,
        description="SHAP contribution value for this feature for the current bankruptcy risk prediction.",
    )


class RiskPredictionResponse(BaseModel):
    """Response returned by the risk prediction service."""

    bankruptcy_probability: float = Field(
        ...,
        description="Predicted probability that the financial profile indicates bankruptcy risk.",
    )
    risk_level: str = Field(
        ...,
        description="Risk category inferred from the probability: Low, Medium, or High.",
    )
    top_contributors: list[RiskContributor] = Field(
        default_factory=list,
        description="Top contributing features ranked by absolute SHAP value for the current prediction.",
    )

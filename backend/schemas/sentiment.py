from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentPredictionRequest(BaseModel):
    text: str = Field(
        ...,
        description="A single financial text snippet to classify as negative, neutral, or positive sentiment.",
    )


class SentimentProbabilities(BaseModel):
    negative: float
    neutral: float
    positive: float


class SentimentPredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: SentimentProbabilities

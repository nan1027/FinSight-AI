from __future__ import annotations

from pydantic import BaseModel, Field


class StockPredictionRequest(BaseModel):
    """Request payload for stock return prediction."""

    ticker: str = Field(
        default="AAPL",
        description="Ticker symbol for the stock forecast. The current trained model is specific to AAPL.",
    )


class StockPredictionResponse(BaseModel):
    """Response returned by the stock return prediction service."""

    ticker: str = Field(..., description="Ticker symbol for the prediction.")
    latest_close: float = Field(..., description="Most recent available Close price in the engineered feature dataset.")
    predicted_next_day_return: float = Field(..., description="Predicted next-day return as a decimal fraction.")
    predicted_next_close: float = Field(..., description="Predicted next-day Close price based on the current Close and the predicted return.")
    model_type: str = Field(..., description="Model type used for the forecast.")
    sequence_length: int = Field(..., description="Sequence length used for the LSTM input window.")

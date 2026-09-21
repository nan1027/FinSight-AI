from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "ml" / "stock_prediction" / "return_lstm_stock_model.keras",
]
FEATURE_SCALER_CANDIDATES = [
    PROJECT_ROOT / "ml" / "stock_prediction" / "return_feature_scaler.pkl",
    PROJECT_ROOT / "data" / "processed" / "return_feature_scaler.pkl",
]
TARGET_SCALER_CANDIDATES = [
    PROJECT_ROOT / "ml" / "stock_prediction" / "return_target_scaler.pkl",
    PROJECT_ROOT / "data" / "processed" / "return_target_scaler.pkl",
]
METADATA_CANDIDATES = [
    PROJECT_ROOT / "ml" / "stock_prediction" / "return_metadata.json",
    PROJECT_ROOT / "data" / "processed" / "return_metadata.json",
]
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "AAPL_engineered_features.csv"
MODEL_PATH = MODEL_CANDIDATES[0]
FEATURE_SCALER_PATH = FEATURE_SCALER_CANDIDATES[0]
TARGET_SCALER_PATH = TARGET_SCALER_CANDIDATES[0]
METADATA_PATH = METADATA_CANDIDATES[0]
SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Price_Range",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "EMA_20",
    "Volatility_10",
    "Volume_Change",
]


class StockPredictionService:
    """Load the trained return-target LSTM and produce a next-day close forecast for AAPL."""

    @staticmethod
    def _resolve_artifact_path(default_path: Path | str, candidates: list[Path]) -> Path:
        default = Path(default_path).resolve()
        if default.exists():
            return default
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return default

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        feature_scaler_path: Path | str = FEATURE_SCALER_PATH,
        target_scaler_path: Path | str = TARGET_SCALER_PATH,
        engineered_data_path: Path | str = ENGINEERED_DATA_PATH,
        metadata_path: Path | str = METADATA_PATH,
    ) -> None:
        self.model_path = self._resolve_artifact_path(model_path, MODEL_CANDIDATES)
        self.feature_scaler_path = self._resolve_artifact_path(feature_scaler_path, FEATURE_SCALER_CANDIDATES)
        self.target_scaler_path = self._resolve_artifact_path(target_scaler_path, TARGET_SCALER_CANDIDATES)
        self.engineered_data_path = Path(engineered_data_path).resolve()
        self.metadata_path = self._resolve_artifact_path(metadata_path, METADATA_CANDIDATES)

        self.model = self._load_model(self.model_path)
        self.feature_scaler = self._load_pickle(self.feature_scaler_path)
        self.target_scaler = self._load_pickle(self.target_scaler_path)
        self.metadata = self._load_metadata(self.metadata_path)
        self.feature_columns = list(self.metadata.get("feature_names", FEATURE_COLUMNS))
        self.engineered_df = self._load_engineered_data(self.engineered_data_path)

    def _load_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return load_model(str(model_path))

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _load_metadata(metadata_path: Path) -> dict:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_engineered_data(self, engineered_data_path: Path) -> pd.DataFrame:
        if not engineered_data_path.exists():
            raise FileNotFoundError(f"Engineered feature file not found: {engineered_data_path}")

        df = pd.read_csv(engineered_data_path)
        if df.empty:
            raise ValueError(f"Engineered feature dataset is empty: {engineered_data_path}")

        required_columns = ["Date", *self.feature_columns]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = df.loc[:, required_columns].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for column in self.feature_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["Date", *self.feature_columns]).sort_values("Date", ascending=True).reset_index(drop=True)
        if len(df) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Not enough rows to build a sequence window. Need at least {SEQUENCE_LENGTH} rows, found {len(df)}."
            )
        if df["Date"].duplicated().any():
            raise ValueError("Duplicate dates found in the engineered feature dataset.")
        if not df["Date"].is_monotonic_increasing:
            raise ValueError("Engineered feature data is not sorted chronologically.")
        if not np.isfinite(df[self.feature_columns].to_numpy(dtype=float)).all():
            raise ValueError("Engineered feature data contains NaN or infinite values.")

        return df

    def predict(self, ticker: str = "AAPL") -> dict[str, float | int | str]:
        if ticker != "AAPL":
            raise ValueError("The current trained stock model is specifically configured for AAPL.")

        latest_window = self.engineered_df.iloc[-SEQUENCE_LENGTH:].copy()
        feature_values = latest_window[self.feature_columns].to_numpy(dtype=float)

        if feature_values.shape != (SEQUENCE_LENGTH, len(self.feature_columns)):
            raise ValueError(
                f"Expected latest feature window shape {(SEQUENCE_LENGTH, len(self.feature_columns))}, "
                f"got {feature_values.shape}."
            )
        if not np.isfinite(feature_values).all():
            raise ValueError("Latest feature window contains NaN or infinite values.")

        scaled_features = self.feature_scaler.transform(feature_values)
        model_input = scaled_features.reshape(1, SEQUENCE_LENGTH, len(self.feature_columns))
        scaled_return = float(self.model.predict(model_input, verbose=0).reshape(-1)[0])
        predicted_return = float(self.target_scaler.inverse_transform(np.array([[scaled_return]], dtype=float))[0, 0])

        latest_close = float(latest_window["Close"].iloc[-1])
        predicted_next_close = latest_close * (1.0 + predicted_return)

        return {
            "ticker": ticker,
            "latest_close": latest_close,
            "predicted_next_day_return": predicted_return,
            "predicted_next_close": predicted_next_close,
            "model_type": "multivariate_return_lstm",
            "sequence_length": SEQUENCE_LENGTH,
        }

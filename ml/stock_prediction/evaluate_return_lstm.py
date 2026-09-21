from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

MODEL_PATH = Path("ml/stock_prediction/return_lstm_stock_model.keras")
X_TEST_PATH = Path("data/processed/return_X_test.npy")
Y_TEST_PATH = Path("data/processed/return_y_test.npy")
TARGET_SCALER_PATH = Path("data/processed/return_target_scaler.pkl")
TEST_DATES_PATH = Path("data/processed/return_test_dates.csv")
CURRENT_CLOSE_PATH = Path("data/processed/return_test_current_close.npy")
PREDICTIONS_SAVE_PATH = Path("data/processed/return_lstm_predictions.csv")
PLOT_SAVE_PATH = Path("ml/stock_prediction/return_lstm_predictions.png")
SEQUENCE_LENGTH = 60
FEATURE_COUNT = 13


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return np.load(path)


def load_target_scaler(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def validate_test_data(X_test: np.ndarray, y_test: np.ndarray, current_close: np.ndarray, test_dates: pd.DataFrame) -> None:
    if X_test.ndim != 3:
        raise ValueError(f"X_test must be 3D; got shape {X_test.shape}.")
    if X_test.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"X_test shape mismatch: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {X_test.shape[1:]}.")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test lengths do not match: {len(X_test)} vs {len(y_test)}.")
    if len(current_close) != len(y_test):
        raise ValueError(f"current_close and y_test lengths do not match: {len(current_close)} vs {len(y_test)}.")
    if len(test_dates) != len(y_test):
        raise ValueError(f"test_dates and y_test lengths do not match: {len(test_dates)} vs {len(y_test)}.")
    if np.isnan(X_test).any() or np.isnan(y_test).any() or np.isnan(current_close).any():
        raise ValueError("NaN values found in the test arrays.")
    if np.isinf(X_test).any() or np.isinf(y_test).any() or np.isinf(current_close).any():
        raise ValueError("Infinite values found in the test arrays.")


def main() -> None:
    model = load_model(MODEL_PATH)
    X_test = load_array(X_TEST_PATH)
    y_test = load_array(Y_TEST_PATH)
    current_close = load_array(CURRENT_CLOSE_PATH)
    target_scaler = load_target_scaler(TARGET_SCALER_PATH)
    test_dates = pd.read_csv(TEST_DATES_PATH)

    validate_test_data(X_test, y_test, current_close, test_dates)

    predicted_returns_scaled = model.predict(X_test, verbose=0).reshape(-1)
    if len(predicted_returns_scaled) != len(y_test):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(predicted_returns_scaled)}, y_test={len(y_test)}."
        )

    predicted_returns = target_scaler.inverse_transform(predicted_returns_scaled.reshape(-1, 1)).reshape(-1)
    actual_returns = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    predicted_close = current_close * (1.0 + predicted_returns)
    actual_close = current_close * (1.0 + actual_returns)

    price_mae = mean_absolute_error(actual_close, predicted_close)
    price_rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
    price_mape = np.mean(np.abs((actual_close - predicted_close) / actual_close)) * 100.0

    return_mae = mean_absolute_error(actual_returns, predicted_returns)
    return_rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))

    results_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates["Date"]),
            "Current_Close": current_close,
            "Actual_Return": actual_returns,
            "Predicted_Return": predicted_returns,
            "Actual_Close": actual_close,
            "Predicted_Close": predicted_close,
        }
    )
    results_df = results_df.sort_values("Date").reset_index(drop=True)
    results_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["Date"], results_df["Actual_Close"], label="Actual_Close", linewidth=2)
    ax.plot(results_df["Date"], results_df["Predicted_Close"], label="Predicted_Close", linewidth=2, linestyle="--")
    ax.set_title("Return-target LSTM: Actual vs Predicted Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(fig)

    print(f"number of test samples: {len(actual_close)}")
    print(f"price MAE: {price_mae:.6f}")
    print(f"price RMSE: {price_rmse:.6f}")
    print(f"price MAPE: {price_mape:.6f}%")
    print(f"return MAE: {return_mae:.6f}")
    print(f"return RMSE: {return_rmse:.6f}")

    print("\nFirst 5 actual vs predicted returns:")
    print(results_df[["Date", "Actual_Return", "Predicted_Return"]].head(5).to_string(index=False))

    print("\nFirst 5 actual vs predicted prices:")
    print(results_df[["Date", "Actual_Close", "Predicted_Close"]].head(5).to_string(index=False))

    print("\nLast 5 actual vs predicted returns:")
    print(results_df[["Date", "Actual_Return", "Predicted_Return"]].tail(5).to_string(index=False))

    print("\nLast 5 actual vs predicted prices:")
    print(results_df[["Date", "Actual_Close", "Predicted_Close"]].tail(5).to_string(index=False))

    print("\nReturn-target LSTM evaluation completed.")


if __name__ == "__main__":
    main()

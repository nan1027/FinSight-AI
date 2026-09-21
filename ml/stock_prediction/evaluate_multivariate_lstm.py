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

MODEL_PATH = Path("ml/stock_prediction/multivariate_lstm_stock_model.keras")
X_TEST_PATH = Path("data/processed/multivariate_X_test.npy")
Y_TEST_PATH = Path("data/processed/multivariate_y_test.npy")
TARGET_SCALER_PATH = Path("data/processed/multivariate_target_scaler.pkl")
TEST_DATES_PATH = Path("data/processed/multivariate_test_dates.csv")
PREDICTIONS_SAVE_PATH = Path("data/processed/multivariate_lstm_predictions.csv")
PLOT_SAVE_PATH = Path("ml/stock_prediction/multivariate_lstm_predictions.png")
SEQUENCE_LENGTH = 60
FEATURE_COUNT = 13


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return np.load(path)


def load_target_scaler(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def validate_test_data(X_test: np.ndarray, y_test: np.ndarray) -> None:
    if X_test.ndim != 3:
        raise ValueError(f"X_test must be 3D; got shape {X_test.shape}.")
    if X_test.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"X_test shape mismatch: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {X_test.shape[1:]}.")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test lengths do not match: {len(X_test)} vs {len(y_test)}.")
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        raise ValueError("NaN values found in the test arrays.")
    if np.isinf(X_test).any() or np.isinf(y_test).any():
        raise ValueError("Infinite values found in the test arrays.")


def main() -> None:
    model = load_model(MODEL_PATH)
    X_test = load_array(X_TEST_PATH)
    y_test = load_array(Y_TEST_PATH)
    target_scaler = load_target_scaler(TARGET_SCALER_PATH)
    test_dates = pd.read_csv(TEST_DATES_PATH)

    validate_test_data(X_test, y_test)

    if len(test_dates) != len(y_test):
        raise ValueError(f"Date count mismatch: {len(test_dates)} dates vs {len(y_test)} targets.")

    predictions_scaled = model.predict(X_test, verbose=0).reshape(-1)
    if len(predictions_scaled) != len(y_test):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(predictions_scaled)}, y_test={len(y_test)}."
        )

    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1)
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    if len(actual) != len(predictions):
        raise ValueError(f"Actual/predicted length mismatch after inverse_transform: {len(actual)} vs {len(predictions)}")

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100.0

    results_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates["Date"]),
            "Actual_Close": actual,
            "Predicted_Close": predictions,
        }
    )
    results_df = results_df.sort_values("Date").reset_index(drop=True)
    results_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["Date"], results_df["Actual_Close"], label="Actual_Close", linewidth=2)
    ax.plot(results_df["Date"], results_df["Predicted_Close"], label="Predicted_Close", linewidth=2, linestyle="--")
    ax.set_title("Multivariate LSTM: Actual vs Predicted Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(fig)

    print(f"number of test samples: {len(actual)}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}%")

    print("\nFirst 5 actual vs predicted prices:")
    print(results_df.head(5).to_string(index=False))

    print("\nLast 5 actual vs predicted prices:")
    print(results_df.tail(5).to_string(index=False))

    print("\nMultivariate LSTM evaluation completed.")


if __name__ == "__main__":
    main()

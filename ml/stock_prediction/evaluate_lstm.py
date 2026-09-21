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


MODEL_PATH = Path("ml/stock_prediction/lstm_stock_model.keras")
X_TEST_PATH = Path("data/processed/stock_X_test.npy")
Y_TEST_PATH = Path("data/processed/stock_y_test.npy")
SCALER_PATH = Path("data/processed/stock_scaler.pkl")
TEST_DATES_PATH = Path("data/processed/stock_test_dates.csv")
PREDICTIONS_SAVE_PATH = Path("data/processed/stock_test_predictions.csv")
PLOT_SAVE_PATH = Path("ml/stock_prediction/lstm_predictions.png")
SEQUENCE_LENGTH = 60
FEATURES = 1


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


def load_scaler(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def main() -> None:
    model = load_model(MODEL_PATH)
    X_test = load_array(X_TEST_PATH)
    y_test = load_array(Y_TEST_PATH)
    scaler = load_scaler(SCALER_PATH)
    test_dates = pd.read_csv(TEST_DATES_PATH)

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Prediction/test length mismatch: X_test={X_test.shape[0]}, y_test={y_test.shape[0]}")

    X_test = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, FEATURES)
    predictions_scaled = model.predict(X_test, verbose=0).reshape(-1)

    if predictions_scaled.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"Prediction/target length mismatch after inference: predictions={predictions_scaled.shape[0]}, targets={y_test.shape[0]}"
        )

    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    if len(actual) != len(predictions):
        raise ValueError(f"Actual/predicted length mismatch after inverse transform: {len(actual)} vs {len(predictions)}")

    if len(test_dates) != len(actual):
        raise ValueError(f"Date count mismatch: {len(test_dates)} dates, {len(actual)} predictions.")

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
    results_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["Date"], results_df["Actual_Close"], label="Actual_Close", linewidth=2)
    ax.plot(results_df["Date"], results_df["Predicted_Close"], label="Predicted_Close", linewidth=2, linestyle="--")
    ax.set_title("AAPL Close Price vs LSTM Prediction")
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


if __name__ == "__main__":
    main()

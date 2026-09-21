from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AAPL_5y_daily.csv"
TEST_DATES_PATH = PROJECT_ROOT / "data" / "processed" / "stock_test_dates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_baseline_predictions.csv"


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    if raw_df.empty:
        raise ValueError(f"Raw stock dataset is empty: {RAW_DATA_PATH}")

    required_columns = ["Date", "Close"]
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in raw dataset: {missing_columns}")

    raw_df = raw_df.loc[:, ["Date", "Close"]].copy()
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
    raw_df["Close"] = pd.to_numeric(raw_df["Close"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    test_dates_df = pd.read_csv(TEST_DATES_PATH)
    if test_dates_df.empty:
        raise ValueError(f"Test date list is empty: {TEST_DATES_PATH}")

    test_dates_df = test_dates_df.loc[:, ["Date"]].copy()
    test_dates_df["Date"] = pd.to_datetime(test_dates_df["Date"], errors="coerce")
    test_dates_df = test_dates_df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates().reset_index(drop=True)

    return raw_df, test_dates_df


def build_persistence_baseline(raw_df: pd.DataFrame, test_dates_df: pd.DataFrame) -> pd.DataFrame:
    close_by_date = raw_df.set_index("Date")["Close"]
    rows: list[dict[str, float | pd.Timestamp]] = []

    for test_date in test_dates_df["Date"]:
        prior_dates = close_by_date.index[close_by_date.index < test_date]
        if prior_dates.empty:
            raise ValueError(f"No prior trading day exists before test date: {test_date.date()}")

        prior_date = prior_dates.max()
        actual_close = raw_df.loc[raw_df["Date"] == test_date, "Close"].iloc[0]
        predicted_close = float(close_by_date.loc[prior_date])

        rows.append(
            {
                "Date": pd.Timestamp(test_date),
                "Actual_Close": float(actual_close),
                "Predicted_Close": predicted_close,
            }
        )

    results_df = pd.DataFrame(rows, columns=["Date", "Actual_Close", "Predicted_Close"]).sort_values("Date").reset_index(drop=True)

    if len(results_df) != len(test_dates_df):
        raise ValueError(
            f"Baseline results length mismatch: {len(results_df)} predictions for {len(test_dates_df)} test dates."
        )

    return results_df


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100.0
    return float(mae), float(rmse), float(mape)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df, test_dates_df = load_market_data()
    results_df = build_persistence_baseline(raw_df, test_dates_df)

    actual = results_df["Actual_Close"].to_numpy(dtype=float)
    predicted = results_df["Predicted_Close"].to_numpy(dtype=float)
    mae, rmse, mape = compute_metrics(actual, predicted)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"number of test samples: {len(results_df)}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}%")

    print("\nFirst 5 actual vs predicted values:")
    print(results_df.head(5).to_string(index=False, formatters={"Date": lambda value: value.strftime("%Y-%m-%d")}))

    print("\nLast 5 actual vs predicted values:")
    print(results_df.tail(5).to_string(index=False, formatters={"Date": lambda value: value.strftime("%Y-%m-%d")}))


if __name__ == "__main__":
    main()

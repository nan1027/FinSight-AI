from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/AAPL_5y_daily.csv")
OUTPUT_PATH = Path("data/processed/AAPL_engineered_features.csv")

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


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {path}")

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.loc[:, required_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).sort_values("Date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    engineered["Daily_Return"] = engineered["Close"].pct_change()
    engineered["Price_Range"] = (engineered["High"] - engineered["Low"]) / engineered["Close"]
    engineered["SMA_10"] = engineered["Close"].rolling(window=10, min_periods=10).mean()
    engineered["SMA_20"] = engineered["Close"].rolling(window=20, min_periods=20).mean()
    engineered["EMA_10"] = engineered["Close"].ewm(span=10, adjust=False).mean()
    engineered["EMA_20"] = engineered["Close"].ewm(span=20, adjust=False).mean()
    engineered["Volatility_10"] = engineered["Daily_Return"].rolling(window=10, min_periods=10).std()
    engineered["Volume_Change"] = engineered["Volume"].pct_change()

    engineered = engineered[["Date", *FEATURE_COLUMNS]].copy()
    engineered = engineered.dropna().reset_index(drop=True)
    return engineered


def validate_features(df: pd.DataFrame, original_rows: int) -> None:
    if df["Date"].is_monotonic_increasing is not True:
        raise ValueError("Date column is not sorted ascending.")

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates found in the engineered dataset.")

    if df.isnull().any().any():
        raise ValueError("Missing values remain after feature engineering.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) != len(df.columns) - 1:
        raise ValueError("Not all feature columns are numeric.")

    if len(df) >= original_rows:
        raise ValueError("Row count did not drop after feature warm-up removal.")

    if len(FEATURE_COLUMNS) != 13:
        raise ValueError(f"Expected 13 feature columns, found {len(FEATURE_COLUMNS)}.")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("One or more feature columns contain infinite values.")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(RAW_DATA_PATH)
    original_rows = len(raw_df)
    engineered_df = engineer_features(raw_df)
    validate_features(engineered_df, original_rows)

    engineered_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Final shape: {engineered_df.shape}")
    print("Feature names:")
    print(list(engineered_df.columns))
    print("\nFirst 5 rows:")
    print(engineered_df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(engineered_df.tail(5).to_string(index=False))
    print("\nDescriptive statistics:")
    print(engineered_df[FEATURE_COLUMNS].describe().to_string())
    print("\nFeature engineering validation passed.")


if __name__ == "__main__":
    main()

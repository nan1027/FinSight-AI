from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ENGINEERED_DATA_PATH = Path("data/processed/AAPL_engineered_features.csv")
OUTPUT_DIR = Path("data/processed")
SEQUENCE_LENGTH = 60
TARGET_COLUMN = "Close"
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


def create_sequences(feature_matrix: np.ndarray, target_values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    for index in range(len(feature_matrix) - sequence_length):
        sequences.append(feature_matrix[index : index + sequence_length])
        targets.append(target_values[index + sequence_length])

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def validate_split_dates(train_dates: pd.Series, val_dates: pd.Series, test_dates: pd.Series) -> None:
    if not (train_dates.max() < val_dates.min() and val_dates.max() < test_dates.min()):
        raise ValueError("Date ranges overlap between training, validation, and test splits.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    if df.empty:
        raise ValueError(f"Feature dataset is empty: {ENGINEERED_DATA_PATH}")

    required_columns = ["Date", *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.loc[:, required_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", *FEATURE_COLUMNS]).sort_values("Date", ascending=True).reset_index(drop=True)

    if df["Date"].isna().any():
        raise ValueError("Date column contains missing values after parsing.")
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates found in the engineered dataset.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted chronologically.")

    total_rows = len(df)
    train_end = int(total_rows * 0.8)
    val_end = train_end + int(total_rows * 0.1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more chronological splits are empty.")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_matrix_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    feature_matrix_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    feature_matrix_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    X_train_raw = feature_scaler.fit_transform(feature_matrix_train)
    X_val_raw = feature_scaler.transform(feature_matrix_val)
    X_test_raw = feature_scaler.transform(feature_matrix_test)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_raw = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_val_raw = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test_raw = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(-1)
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    X_train, y_train = create_sequences(X_train_raw, y_train_scaled, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(X_val_raw, y_val_scaled, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(X_test_raw, y_test_scaled, SEQUENCE_LENGTH)

    train_dates = train_df["Date"].iloc[SEQUENCE_LENGTH:train_end].reset_index(drop=True)
    val_dates = val_df["Date"].iloc[SEQUENCE_LENGTH:val_end].reset_index(drop=True)
    test_dates = test_df["Date"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)

    if len(train_dates) != len(y_train):
        raise ValueError(f"Training date count mismatch: {len(train_dates)} dates, {len(y_train)} targets.")
    if len(val_dates) != len(y_val):
        raise ValueError(f"Validation date count mismatch: {len(val_dates)} dates, {len(y_val)} targets.")
    if len(test_dates) != len(y_test):
        raise ValueError(f"Test date count mismatch: {len(test_dates)} dates, {len(y_test)} targets.")

    validate_split_dates(train_dates, val_dates, test_dates)

    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("NaN values found in training arrays.")
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        raise ValueError("NaN values found in validation arrays.")
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        raise ValueError("NaN values found in test arrays.")

    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Infinite values found in training arrays.")
    if np.isinf(X_val).any() or np.isinf(y_val).any():
        raise ValueError("Infinite values found in validation arrays.")
    if np.isinf(X_test).any() or np.isinf(y_test).any():
        raise ValueError("Infinite values found in test arrays.")

    if X_train.shape[-1] != len(FEATURE_COLUMNS) or X_val.shape[-1] != len(FEATURE_COLUMNS) or X_test.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected input feature count {len(FEATURE_COLUMNS)}, found mismatch in sequence arrays.")

    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X and y sample counts do not match within each split.")

    if len(test_dates) != len(y_test):
        raise ValueError("Test date count does not match the number of test targets.")

    feature_min_expected = np.min(feature_matrix_train, axis=0)
    feature_max_expected = np.max(feature_matrix_train, axis=0)
    if not np.allclose(feature_scaler.data_min_, feature_min_expected):
        raise ValueError("Feature scaler was not fitted only on the training data.")
    if not np.allclose(feature_scaler.data_max_, feature_max_expected):
        raise ValueError("Feature scaler max values do not match the training data.")

    target_min_expected = np.min(y_train_raw)
    target_max_expected = np.max(y_train_raw)
    if not np.allclose(target_scaler.data_min_, np.array([target_min_expected])):
        raise ValueError("Target scaler was not fitted only on the training target data.")
    if not np.allclose(target_scaler.data_max_, np.array([target_max_expected])):
        raise ValueError("Target scaler max values do not match the training target data.")

    if len(train_df) >= total_rows:
        raise ValueError("Unexpected training split size; data was not partitioned correctly.")

    np.save(OUTPUT_DIR / "multivariate_X_train.npy", X_train)
    np.save(OUTPUT_DIR / "multivariate_y_train.npy", y_train)
    np.save(OUTPUT_DIR / "multivariate_X_val.npy", X_val)
    np.save(OUTPUT_DIR / "multivariate_y_val.npy", y_val)
    np.save(OUTPUT_DIR / "multivariate_X_test.npy", X_test)
    np.save(OUTPUT_DIR / "multivariate_y_test.npy", y_test)
    save_pickle(feature_scaler, OUTPUT_DIR / "multivariate_feature_scaler.pkl")
    save_pickle(target_scaler, OUTPUT_DIR / "multivariate_target_scaler.pkl")
    test_dates.to_frame(name="Date").to_csv(OUTPUT_DIR / "multivariate_test_dates.csv", index=False)

    metadata = {
        "ticker": "AAPL",
        "target": TARGET_COLUMN,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
    }
    (OUTPUT_DIR / "multivariate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"total rows: {total_rows}")
    print(f"training rows: {len(train_df)}")
    print(f"validation rows: {len(val_df)}")
    print(f"test rows: {len(test_df)}")
    print(f"sequence length: {SEQUENCE_LENGTH}")
    print(f"feature count: {len(FEATURE_COLUMNS)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"first test date: {test_dates.iloc[0]}")
    print(f"last test date: {test_dates.iloc[-1]}")

    print("\nMultivariate preprocessing validation passed.")


if __name__ == "__main__":
    main()

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
TARGET_NAME = "Next_Day_Return"
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

    for index in range(len(feature_matrix) - sequence_length + 1):
        sequences.append(feature_matrix[index : index + sequence_length])
        targets.append(target_values[index + sequence_length - 1])

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def validate_split_ranges(train_dates: pd.Series, val_dates: pd.Series, test_dates: pd.Series) -> None:
    if not (train_dates.max() < val_dates.min() and val_dates.max() < test_dates.min()):
        raise ValueError("Date ranges overlap between training, validation, and test splits.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    if df.empty:
        raise ValueError(f"Engineered dataset is empty: {ENGINEERED_DATA_PATH}")

    required_columns = ["Date", *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.loc[:, ["Date", *FEATURE_COLUMNS]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", *FEATURE_COLUMNS]).sort_values("Date", ascending=True).reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates found in the engineered dataset.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Engineered dates are not sorted ascending.")

    df[TARGET_NAME] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df.iloc[:-1].reset_index(drop=True)

    total_rows = len(df)
    train_end = int(total_rows * 0.8)
    val_end = train_end + int(total_rows * 0.1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more chronological splits are empty.")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_val_raw = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_test_raw = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    y_train_raw = train_df[TARGET_NAME].to_numpy(dtype=float)
    y_val_raw = val_df[TARGET_NAME].to_numpy(dtype=float)
    y_test_raw = test_df[TARGET_NAME].to_numpy(dtype=float)

    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_val_scaled = feature_scaler.transform(X_val_raw)
    X_test_scaled = feature_scaler.transform(X_test_raw)

    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(-1)
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    train_dates = train_df["Date"].iloc[SEQUENCE_LENGTH - 1 : train_end].reset_index(drop=True)
    val_dates = val_df["Date"].iloc[SEQUENCE_LENGTH - 1 : val_end].reset_index(drop=True)
    test_dates = test_df["Date"].iloc[SEQUENCE_LENGTH - 1 :].reset_index(drop=True)

    if len(train_dates) != len(y_train):
        raise ValueError(f"Training date count mismatch: {len(train_dates)} dates vs {len(y_train)} targets.")
    if len(val_dates) != len(y_val):
        raise ValueError(f"Validation date count mismatch: {len(val_dates)} dates vs {len(y_val)} targets.")
    if len(test_dates) != len(y_test):
        raise ValueError(f"Test date count mismatch: {len(test_dates)} dates vs {len(y_test)} targets.")

    current_close_test = test_df["Close"].iloc[SEQUENCE_LENGTH - 1 :].to_numpy(dtype=float)
    if len(current_close_test) != len(y_test):
        raise ValueError(f"Current Close length mismatch: {len(current_close_test)} vs {len(y_test)}.")

    validate_split_ranges(train_dates, val_dates, test_dates)

    for array_name, array in {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }.items():
        if np.isnan(array).any():
            raise ValueError(f"NaN values found in {array_name}.")
        if np.isinf(array).any():
            raise ValueError(f"Infinite values found in {array_name}.")

    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError("X arrays must be 3D.")
    if X_train.shape[-1] != len(FEATURE_COLUMNS) or X_val.shape[-1] != len(FEATURE_COLUMNS) or X_test.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected feature dimension {len(FEATURE_COLUMNS)} for X arrays.")
    if X_train.shape[1] != SEQUENCE_LENGTH or X_val.shape[1] != SEQUENCE_LENGTH or X_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH} for X arrays.")
    if X_train.shape[0] != len(y_train) or X_val.shape[0] != len(y_val) or X_test.shape[0] != len(y_test):
        raise ValueError("X and y lengths do not match for at least one split.")

    if not np.allclose(feature_scaler.data_min_, np.min(X_train_raw, axis=0)):
        raise ValueError("Feature scaler was not fit only on the training portion.")
    if not np.allclose(feature_scaler.data_max_, np.max(X_train_raw, axis=0)):
        raise ValueError("Feature scaler max values do not match the training portion.")

    if not np.allclose(target_scaler.data_min_, np.array([np.min(y_train_raw)])):
        raise ValueError("Target scaler was not fit only on the training target.")
    if not np.allclose(target_scaler.data_max_, np.array([np.max(y_train_raw)])):
        raise ValueError("Target scaler max values do not match the training target.")

    np.save(OUTPUT_DIR / "return_X_train.npy", X_train)
    np.save(OUTPUT_DIR / "return_y_train.npy", y_train)
    np.save(OUTPUT_DIR / "return_X_val.npy", X_val)
    np.save(OUTPUT_DIR / "return_y_val.npy", y_val)
    np.save(OUTPUT_DIR / "return_X_test.npy", X_test)
    np.save(OUTPUT_DIR / "return_y_test.npy", y_test)
    save_pickle(feature_scaler, OUTPUT_DIR / "return_feature_scaler.pkl")
    save_pickle(target_scaler, OUTPUT_DIR / "return_target_scaler.pkl")
    test_dates.to_frame(name="Date").to_csv(OUTPUT_DIR / "return_test_dates.csv", index=False)
    np.save(OUTPUT_DIR / "return_test_current_close.npy", current_close_test)

    metadata = {
        "ticker": "AAPL",
        "target": TARGET_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
    }
    (OUTPUT_DIR / "return_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"total usable rows: {total_rows}")
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
    print(f"first test current Close: {current_close_test[0]}")
    print(f"last test current Close: {current_close_test[-1]}")

    print("\nReturn-target preprocessing validation passed.")


if __name__ == "__main__":
    main()

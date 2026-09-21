from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("data/raw/AAPL_5y_daily.csv")
OUTPUT_DIR = Path("data/processed")
SEQUENCE_LENGTH = 60
TICKER = "AAPL"
TARGET_COLUMN = "Close"
FEATURE_NAME = "Close"


def create_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    for index in range(len(values) - sequence_length):
        window = values[index : index + sequence_length]
        target = values[index + sequence_length]
        sequences.append(window)
        targets.append(target)

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def save_pickle(obj: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {DATA_PATH}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    if df["Date"].isna().any():
        raise ValueError("Date column contains missing values after parsing.")

    required_columns = ["Date", TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    close_values = df[TARGET_COLUMN].astype(float).to_numpy()
    total_rows = len(df)

    train_end = int(total_rows * 0.8)
    val_end = train_end + int(total_rows * 0.1)

    train_close = close_values[:train_end]
    val_close = close_values[train_end:val_end]
    test_close = close_values[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_close.reshape(-1, 1))

    train_scaled = scaler.transform(train_close.reshape(-1, 1)).reshape(-1)
    val_scaled = scaler.transform(val_close.reshape(-1, 1)).reshape(-1)
    test_scaled = scaler.transform(test_close.reshape(-1, 1)).reshape(-1)

    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_scaled, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH)

    train_dates = df["Date"].iloc[SEQUENCE_LENGTH:train_end].reset_index(drop=True)
    val_dates = df["Date"].iloc[train_end + SEQUENCE_LENGTH : val_end].reset_index(drop=True)
    test_dates = df["Date"].iloc[val_end + SEQUENCE_LENGTH :].reset_index(drop=True)

    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("NaN or infinite values found in training arrays.")
    if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
        raise ValueError("NaN or infinite values found in validation arrays.")
    if not np.isfinite(X_test).all() or not np.isfinite(y_test).all():
        raise ValueError("NaN or infinite values found in test arrays.")

    if not np.all(np.diff(df["Date"].sort_values().to_numpy()) >= pd.Timedelta(0)):
        raise ValueError("Chronological ordering is not preserved in the dataset.")

    if len(train_dates) != len(y_train):
        raise ValueError(
            f"Train date count mismatch: {len(train_dates)} dates, {len(y_train)} target values."
        )
    if len(val_dates) != len(y_val):
        raise ValueError(
            f"Validation date count mismatch: {len(val_dates)} dates, {len(y_val)} target values."
        )
    if len(test_dates) != len(y_test):
        raise ValueError(
            f"Test date count mismatch: {len(test_dates)} dates, {len(y_test)} target values."
        )

    np.save(OUTPUT_DIR / "stock_X_train.npy", X_train)
    np.save(OUTPUT_DIR / "stock_y_train.npy", y_train)
    np.save(OUTPUT_DIR / "stock_X_val.npy", X_val)
    np.save(OUTPUT_DIR / "stock_y_val.npy", y_val)
    np.save(OUTPUT_DIR / "stock_X_test.npy", X_test)
    np.save(OUTPUT_DIR / "stock_y_test.npy", y_test)
    save_pickle(scaler, OUTPUT_DIR / "stock_scaler.pkl")
    test_dates.to_frame(name="Date").to_csv(OUTPUT_DIR / "stock_test_dates.csv", index=False)

    metadata = {
        "ticker": TICKER,
        "target": TARGET_COLUMN,
        "sequence_length": SEQUENCE_LENGTH,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "feature_count": 1,
        "feature_name": FEATURE_NAME,
    }
    (OUTPUT_DIR / "stock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"total rows: {total_rows}")
    print(f"train rows: {len(train_close)}")
    print(f"validation rows: {len(val_close)}")
    print(f"test rows: {len(test_close)}")
    print(f"sequence length: {SEQUENCE_LENGTH}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"first test date: {test_dates.iloc[0]}")
    print(f"last test date: {test_dates.iloc[-1]}")

    validation_errors = []
    for array_name, array in {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }.items():
        if np.isnan(array).any():
            validation_errors.append(f"{array_name} contains NaN values.")

    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    if not np.all(np.diff(test_dates.to_numpy()) >= np.timedelta64(0, "D")):
        raise ValueError("Chronological ordering is not preserved in the test dates.")

    scaler_fit_on_train_only = np.allclose(
        scaler.data_min_,
        np.min(train_close.reshape(-1, 1), axis=0),
    ) and np.allclose(
        scaler.data_max_,
        np.max(train_close.reshape(-1, 1), axis=0),
    )
    if not scaler_fit_on_train_only:
        raise ValueError("Scaler was not fit exclusively on training data.")

    print("Preprocessing validation passed.")


if __name__ == "__main__":
    main()

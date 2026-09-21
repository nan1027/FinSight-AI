from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/taiwanese_bankruptcy.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COLUMN = "Bankrupt?"
CONSTANT_FEATURE = "Net Income Flag"


def main() -> None:
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}") from exc

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    original_feature_count = X.shape[1]
    X = X.loc[:, X.columns.str.strip() != CONSTANT_FEATURE]

    if X.shape[1] == original_feature_count:
        print(f"Warning: constant feature '{CONSTANT_FEATURE}' was not found. No feature was removed.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    train_overlap = set(X_train.index).intersection(set(X_test.index))
    if train_overlap:
        raise ValueError(f"Train/test index overlap detected: {len(train_overlap)} overlapping rows.")

    print(f"Original feature count: {original_feature_count}")
    print(f"Features after removing constant features: {X.shape[1]}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("y_train class distribution:")
    print(y_train.value_counts())
    print("y_test class distribution:")
    print(y_test.value_counts())
    print(f"Train bankrupt percentage: {y_train.mean() * 100:.2f}%")
    print(f"Test bankrupt percentage: {y_test.mean() * 100:.2f}%")
    print("Train/test index overlap check: no overlap detected.")

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=True)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=True)


if __name__ == "__main__":
    main()

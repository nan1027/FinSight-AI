from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/raw/taiwanese_bankruptcy.csv")
TARGET_COLUMN = "Bankrupt?"


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path)


def main() -> None:
    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print_section("1. Dataset Shape")
    print(f"Shape: {df.shape}")

    print_section("2. Column Data Types")
    print(df.dtypes)

    print_section("3. Total Missing Values")
    print(f"Total missing values: {int(df.isna().sum().sum())}")

    print_section("4. Missing Values per Column")
    missing_by_column = df.isna().sum()
    missing_columns = missing_by_column[missing_by_column > 0]
    if missing_columns.empty:
        print("No missing values found.")
    else:
        print(missing_columns)

    print_section("5. Duplicate Rows")
    print(f"Number of duplicate rows: {int(df.duplicated().sum())}")

    print_section("6. Target Column Name")
    print(f"Target column: {TARGET_COLUMN}")

    print_section("7. Target Class Counts")
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    print(target_counts)

    print_section("8. Target Class Percentages")
    target_percentages = df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
    print(target_percentages)

    print_section("9. Number of Unique Values for Every Feature")
    unique_values_per_feature = df.nunique(dropna=False)
    print(unique_values_per_feature)

    print_section("10. Constant Features")
    constant_features = unique_values_per_feature[unique_values_per_feature <= 1].index.tolist()
    if not constant_features:
        print("No constant features found.")
    else:
        print(constant_features)

    print_section("11. Basic Descriptive Statistics")
    print(df.describe().T)

    print_section("12. Correlation Matrix and High Correlation Pairs")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print(corr_matrix)

    print("\nHighly correlated feature pairs (|corr| >= 0.90, excluding self-correlation):")
    high_corr_pairs: list[tuple[str, str, float]] = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= 0.90:
                high_corr_pairs.append((feature_1, feature_2, corr_value))

    if not high_corr_pairs:
        print("No highly correlated feature pairs found.")
    else:
        for feature_1, feature_2, corr_value in high_corr_pairs:
            print(f"- {feature_1} | {feature_2} | {corr_value:.6f}")


if __name__ == "__main__":
    main()

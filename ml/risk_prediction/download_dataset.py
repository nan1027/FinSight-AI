from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 572
RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = RAW_DATA_DIR / "taiwanese_bankruptcy.csv"


def main() -> None:
    try:
        dataset = fetch_ucirepo(id=DATASET_ID)
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset ID {DATASET_ID}: {exc}") from exc

    features: pd.DataFrame = dataset.data.features
    target_df: pd.DataFrame = dataset.data.targets

    if features is None or target_df is None:
        raise ValueError("Dataset features or target were not returned.")

    if target_df.shape[1] != 1:
        raise ValueError(f"Expected a single target column, got {target_df.shape[1]} columns.")

    target_column_name = target_df.columns[0]
    target_series = target_df[target_column_name]

    combined_df = pd.concat([features, target_df], axis=1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        combined_df.to_csv(OUTPUT_PATH, index=False)
    except Exception as exc:
        raise OSError(f"Failed to save dataset to {OUTPUT_PATH}: {exc}") from exc

    print(f"Dataset shape: {combined_df.shape}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Target column name: {target_column_name}")
    print("Target value counts:")
    print(target_series.value_counts())
    print(f"Output file path: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

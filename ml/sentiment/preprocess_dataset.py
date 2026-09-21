from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_PATH = Path("data/raw/financial_phrasebank/financial_phrasebank.csv")
OUTPUT_DIR = Path("data/processed/sentiment")
TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
METADATA_PATH = Path("ml/sentiment/split_metadata.json")
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"
EXPECTED_LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _require_dataset() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"The Financial PhraseBank dataset is missing at {SOURCE_PATH}. "
            "Run ml/sentiment/download_dataset.py first."
        )

    df = pd.read_csv(SOURCE_PATH)
    missing = [column for column in [TEXT_COLUMN, LABEL_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {list(df.columns)}")
    return df


def _normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    if label not in EXPECTED_LABELS:
        raise ValueError(f"Unexpected label value {value!r}. Allowed labels: {EXPECTED_LABELS}")
    return label


def _minimal_text_preprocess(value: object) -> str:
    text = str(value)
    text = text.strip()
    text = " ".join(text.split())
    return text


def _find_duplicates(df: pd.DataFrame) -> pd.Series:
    text_series = df[TEXT_COLUMN].astype(str)
    return text_series[text_series.duplicated(keep=False)]


def _build_groups(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(TEXT_COLUMN, sort=False, dropna=False)
    group_rows = []
    for text_value, group_df in grouped:
        if group_df.empty:
            continue
        label_value = group_df[LABEL_COLUMN].iloc[0]
        group_rows.append({"text": text_value, "label": label_value, "group_size": len(group_df)})
    return pd.DataFrame(group_rows)


def _split_groups(group_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_group_ids, temp_group_ids = train_test_split(
        group_df.index.to_list(),
        train_size=SPLIT_RATIOS["train"],
        random_state=RANDOM_SEED,
        stratify=group_df[LABEL_COLUMN],
    )

    temp_df = group_df.loc[temp_group_ids].copy()
    val_group_ids, test_group_ids = train_test_split(
        temp_df.index.to_list(),
        train_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df[LABEL_COLUMN],
    )

    train_df = group_df.loc[train_group_ids].copy()
    val_df = temp_df.loc[val_group_ids].copy()
    test_df = temp_df.loc[test_group_ids].copy()

    return train_df, val_df, test_df


def _expand_group_split(group_df: pd.DataFrame, original_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_group_df, val_group_df, test_group_df = _split_groups(group_df)

    train_texts = set(train_group_df["text"])
    val_texts = set(val_group_df["text"])
    test_texts = set(test_group_df["text"])

    if train_texts & val_texts or train_texts & test_texts or val_texts & test_texts:
        raise ValueError("Cross-split text overlap detected after grouping duplicates.")

    train_rows = original_df[original_df[TEXT_COLUMN].isin(train_texts)].copy()
    val_rows = original_df[original_df[TEXT_COLUMN].isin(val_texts)].copy()
    test_rows = original_df[original_df[TEXT_COLUMN].isin(test_texts)].copy()

    for split_name, split_df in {"train": train_rows, "validation": val_rows, "test": test_rows}.items():
        if split_df.empty:
            raise ValueError(f"The {split_name} split is empty.")
        if split_df[TEXT_COLUMN].isna().any() or split_df[TEXT_COLUMN].astype(str).str.strip().eq("").any():
            raise ValueError(f"The {split_name} split contains empty text values.")

    if len(train_rows) + len(val_rows) + len(test_rows) != len(original_df):
        raise ValueError("Split row counts do not add up to the original row count.")

    return train_rows[[TEXT_COLUMN, LABEL_COLUMN]], val_rows[[TEXT_COLUMN, LABEL_COLUMN]], test_rows[[TEXT_COLUMN, LABEL_COLUMN]]


def _split_summary(df: pd.DataFrame) -> dict[str, object]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    total = len(df)
    percentages = {label: round((counts.get(label, 0) / total) * 100, 4) if total else 0.0 for label in EXPECTED_LABELS}
    return {
        "row_count": int(len(df)),
        "class_counts": {label: int(counts.get(label, 0)) for label in EXPECTED_LABELS},
        "class_percentages": {label: float(percentages.get(label, 0.0)) for label in EXPECTED_LABELS},
    }


def _validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for split_name, split_df in {"train": train_df, "validation": val_df, "test": test_df}.items():
        if split_df.empty:
            raise ValueError(f"{split_name} split is empty.")
        if list(split_df.columns) != [TEXT_COLUMN, LABEL_COLUMN]:
            raise ValueError(f"{split_name} split has incorrect columns: {split_df.columns.tolist()}")
        if split_df[TEXT_COLUMN].isna().any() or split_df[TEXT_COLUMN].astype(str).str.strip().eq("").any():
            raise ValueError(f"{split_name} split contains empty text values.")
        if split_df[LABEL_COLUMN].isin(EXPECTED_LABELS).all() is False:
            raise ValueError(f"{split_name} split contains labels outside {EXPECTED_LABELS}.")
        if len(split_df) != len(split_df.drop_duplicates()):
            # Duplicate rows are allowed as raw data findings, but split outputs should still be valid and readable.
            pass

    train_text = set(train_df[TEXT_COLUMN].astype(str).tolist())
    val_text = set(val_df[TEXT_COLUMN].astype(str).tolist())
    test_text = set(test_df[TEXT_COLUMN].astype(str).tolist())

    if train_text & val_text or train_text & test_text or val_text & test_text:
        raise ValueError("Cross-split text overlap detected.")

    if len(train_df) + len(val_df) + len(test_df) != len(pd.concat([train_df, val_df, test_df], ignore_index=True)):
        # This should always match if the split counts sum correctly.
        pass

    if len(train_df) + len(val_df) + len(test_df) == 0:
        raise ValueError("The combined split totals are zero.")


def main() -> None:
    df = _require_dataset().copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(_minimal_text_preprocess)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(_normalize_label)

    if df[TEXT_COLUMN].isna().any() or df[TEXT_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("The dataset contains empty or missing text values after preprocessing.")

    duplicate_text_series = _find_duplicates(df)
    duplicate_text_count = int(duplicate_text_series.nunique())

    grouped_df = _build_groups(df)
    train_group_df, val_group_df, test_group_df = _split_groups(grouped_df)
    train_df, val_df, test_df = _expand_group_split(grouped_df, df)

    _validate_splits(train_df, val_df, test_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df[[TEXT_COLUMN, LABEL_COLUMN]].to_csv(TRAIN_PATH, index=False)
    val_df[[TEXT_COLUMN, LABEL_COLUMN]].to_csv(VALIDATION_PATH, index=False)
    test_df[[TEXT_COLUMN, LABEL_COLUMN]].to_csv(TEST_PATH, index=False)

    train_summary = _split_summary(train_df)
    val_summary = _split_summary(val_df)
    test_summary = _split_summary(test_df)

    metadata = {
        "random_seed": RANDOM_SEED,
        "split_ratios": SPLIT_RATIOS,
        "original_row_count": int(len(df)),
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(val_df)),
        "test_row_count": int(len(test_df)),
        "class_counts": {
            "train": train_summary["class_counts"],
            "validation": val_summary["class_counts"],
            "test": test_summary["class_counts"],
        },
        "class_percentages": {
            "train": train_summary["class_percentages"],
            "validation": val_summary["class_percentages"],
            "test": test_summary["class_percentages"],
        },
        "duplicate_leakage_checks": {
            "duplicate_text_entries": duplicate_text_count,
            "train_validation_overlap": bool(set(train_df[TEXT_COLUMN]).intersection(val_df[TEXT_COLUMN])),
            "train_test_overlap": bool(set(train_df[TEXT_COLUMN]).intersection(test_df[TEXT_COLUMN])),
            "validation_test_overlap": bool(set(val_df[TEXT_COLUMN]).intersection(test_df[TEXT_COLUMN])),
        },
        "text_column_source": TEXT_COLUMN,
        "label_column_source": LABEL_COLUMN,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Original dataset size: {len(df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train class counts: {train_summary['class_counts']}")
    print(f"Validation class counts: {val_summary['class_counts']}")
    print(f"Test class counts: {test_summary['class_counts']}")
    print(f"Train class percentages: {train_summary['class_percentages']}")
    print(f"Validation class percentages: {val_summary['class_percentages']}")
    print(f"Test class percentages: {test_summary['class_percentages']}")
    print(f"Duplicate texts (exact): {duplicate_text_count}")
    print(f"Cross-split text overlap: {metadata['duplicate_leakage_checks']}")
    print(f"Saved train to: {TRAIN_PATH}")
    print(f"Saved validation to: {VALIDATION_PATH}")
    print(f"Saved test to: {TEST_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()

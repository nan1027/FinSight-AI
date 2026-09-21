from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_PATH = Path("data/raw/financial_phrasebank/financial_phrasebank.csv")
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"
EXPECTED_LABELS = ["positive", "neutral", "negative"]


def _require_dataset() -> pd.DataFrame:
    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(
            "The Financial PhraseBank dataset is missing. "
            f"Expected file: {EXPECTED_PATH}. "
            "Run ml/sentiment/download_dataset.py first."
        )

    df = pd.read_csv(EXPECTED_PATH)

    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected columns {TEXT_COLUMN!r} and {LABEL_COLUMN!r} in {EXPECTED_PATH}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def _normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _label_summary(df: pd.DataFrame) -> dict[str, object]:
    label_series = df[LABEL_COLUMN].map(_normalize_label)
    label_values = sorted({value for value in label_series if value})
    counts = label_series.value_counts(dropna=False)
    total = len(df)
    percentages = {label: round((count / total) * 100, 4) for label, count in counts.items()}
    return {
        "label_values": label_values,
        "label_counts": {str(label): int(count) for label, count in counts.items()},
        "label_percentages": {str(label): float(percentages.get(label, 0.0)) for label in label_values},
    }


def _text_length_stats(series: pd.Series) -> dict[str, float | int]:
    text_lengths = series.fillna("").astype(str).str.len()
    return {
        "min": int(text_lengths.min()) if not text_lengths.empty else 0,
        "max": int(text_lengths.max()) if not text_lengths.empty else 0,
        "average": float(text_lengths.mean()) if not text_lengths.empty else 0.0,
        "median": float(text_lengths.median()) if not text_lengths.empty else 0.0,
    }


def main() -> None:
    df = _require_dataset()

    raw_text = df[TEXT_COLUMN]
    raw_labels = df[LABEL_COLUMN]

    label_series = raw_labels.map(_normalize_label)
    actual_values = sorted({value for value in label_series if value})
    normalized_values = EXPECTED_LABELS if set(actual_values) == set(EXPECTED_LABELS) else actual_values
    normalized_ok = set(actual_values) == set(EXPECTED_LABELS)

    total_missing = int(df.isna().sum().sum())
    text_missing = int(raw_text.isna().sum())
    label_missing = int(raw_labels.isna().sum())

    duplicate_rows = int(df.duplicated().sum())
    duplicated_text_entries = int(raw_text.duplicated().sum())
    conflicting_text_labels = int(
        df.groupby(TEXT_COLUMN, dropna=False)[LABEL_COLUMN].nunique().gt(1).sum()
    )

    empty_text = int(raw_text.fillna("").astype(str).eq("").sum())
    whitespace_only = int(raw_text.fillna("").astype(str).str.strip().eq("").sum())
    text_lengths = raw_text.fillna("").astype(str).str.len()

    malformed_rows = []
    for index, row in df.iterrows():
        text_value = row[TEXT_COLUMN]
        label_value = row[LABEL_COLUMN]
        if pd.isna(text_value) or str(text_value).strip() == "":
            malformed_rows.append({"row_index": int(index), "issue": "empty_text", "text": text_value})
            continue
        if pd.isna(label_value) or _normalize_label(label_value) not in EXPECTED_LABELS:
            malformed_rows.append({"row_index": int(index), "issue": "invalid_label", "label": label_value})

    representative_examples = {}
    for label in EXPECTED_LABELS:
        samples = df.loc[df[LABEL_COLUMN].map(_normalize_label) == label, TEXT_COLUMN].dropna().head(3).tolist()
        representative_examples[label] = samples

    report = {
        "row_count": int(len(df)),
        "column_names": list(df.columns),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "label_values": normalized_values,
        "class_counts": {label: int(label_series.eq(label).sum()) for label in EXPECTED_LABELS},
        "class_percentages": {label: round((label_series.eq(label).sum() / len(df)) * 100, 4) for label in EXPECTED_LABELS},
        "missing_values": {
            "total_missing_values": total_missing,
            "missing_text": text_missing,
            "missing_labels": label_missing,
        },
        "duplicate_counts": {
            "complete_duplicates": duplicate_rows,
            "duplicated_text_entries": duplicated_text_entries,
            "duplicated_text_with_conflicting_labels": conflicting_text_labels,
        },
        "text_length_statistics": {
            **_text_length_stats(raw_text),
            "empty_strings": empty_text,
            "whitespace_only_strings": whitespace_only,
        },
        "labels_normalized_consistently": normalized_ok,
        "expected_three_class_categories": set(EXPECTED_LABELS).issubset(set(normalized_values)) and set(normalized_values).issubset(set(EXPECTED_LABELS)),
        "malformed_records": malformed_rows,
        "representative_examples": representative_examples,
        "dataset_source": {
            "path": str(EXPECTED_PATH),
            "format_used": "CSV",
            "notes": "This project uses the canonical Financial PhraseBank CSV exported from the official archive in data/raw/financial_phrasebank/financial_phrasebank.csv. No additional metadata columns are present in the final project dataset file; only sentence and label are used.",
        },
    }

    output_path = Path("ml/sentiment/dataset_report.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset path: {EXPECTED_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Unique labels: {normalized_values}")
    print("Class distribution:")
    for label in EXPECTED_LABELS:
        count = int(label_series.eq(label).sum())
        percent = (count / len(df)) * 100
        print(f"- {label}: {count} ({percent:.2f}%)")
    print(f"Missing values: total={total_missing}, text={text_missing}, labels={label_missing}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicated text entries: {duplicated_text_entries}")
    print(f"Duplicated text with conflicting labels: {conflicting_text_labels}")
    print(f"Empty strings: {empty_text}")
    print(f"Whitespace-only strings: {whitespace_only}")
    print(f"Text length stats: min={int(text_lengths.min())}, max={int(text_lengths.max())}, avg={text_lengths.mean():.2f}, median={text_lengths.median():.2f}")
    print(f"Label normalization consistent: {normalized_ok}")
    print(f"Contains exactly expected categories: {normalized_ok}")
    print(f"Malformed records: {len(malformed_rows)}")
    if malformed_rows:
        for entry in malformed_rows[:5]:
            print(f"- {entry}")
    print("Representative examples by class:")
    for label in EXPECTED_LABELS:
        print(f"- {label}: {representative_examples[label]}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()

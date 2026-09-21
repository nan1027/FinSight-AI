from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


HF_REPO_ID = "takala/financial_phrasebank"
HF_ARCHIVE_PATH = "data/FinancialPhraseBank-v1.0.zip"
RAW_DIR = Path("data/raw/financial_phrasebank")
ARCHIVE_PATH = RAW_DIR / "FinancialPhraseBank-v1.0.zip"
CSV_PATH = RAW_DIR / "financial_phrasebank.csv"
SOURCE_FILE = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
DATASET_NAME = "Financial PhraseBank"


def _build_dataframe() -> pd.DataFrame:
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        if SOURCE_FILE not in archive.namelist():
            raise FileNotFoundError(f"Expected source file {SOURCE_FILE} was not found in the downloaded archive.")

        lines = archive.read(SOURCE_FILE).decode("utf-8", errors="replace").splitlines()

    rows: list[dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        sentence = sentence.strip()
        label = label.strip().lower()
        if sentence and label in {"positive", "neutral", "negative"}:
            rows.append({"sentence": sentence, "label": label})

    if not rows:
        raise RuntimeError("No valid sentiment rows were found in the Financial PhraseBank archive.")

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    archive_cache_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=HF_ARCHIVE_PATH,
    )
    shutil.copy2(archive_cache_path, ARCHIVE_PATH)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(RAW_DIR)

    df = _build_dataframe()
    df.to_csv(CSV_PATH, index=False)

    metadata = {
        "dataset_name": DATASET_NAME,
        "source": "https://huggingface.co/datasets/takala/financial_phrasebank",
        "archive": str(ARCHIVE_PATH),
        "csv_path": str(CSV_PATH),
        "rows": len(df),
        "columns": list(df.columns),
        "labels": sorted(df["label"].unique().tolist()),
    }
    (RAW_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"dataset: {DATASET_NAME}")
    print(f"rows: {len(df)}")
    print(f"labels: {sorted(df['label'].unique().tolist())}")
    print(f"output path: {CSV_PATH}")


if __name__ == "__main__":
    main()

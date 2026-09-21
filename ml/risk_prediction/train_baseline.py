from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def main() -> None:
    try:
        X_train = load_dataset(X_TRAIN_PATH)
        X_test = load_dataset(X_TEST_PATH)
        y_train = load_dataset(Y_TRAIN_PATH)
        y_test = load_dataset(Y_TEST_PATH)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print_section("Accuracy")
    print(accuracy_score(y_test, y_pred))

    print_section("Precision")
    print(precision_score(y_test, y_pred, zero_division=0))

    print_section("Recall")
    print(recall_score(y_test, y_pred, zero_division=0))

    print_section("F1-score")
    print(f1_score(y_test, y_pred, zero_division=0))

    print_section("ROC-AUC")
    print(roc_auc_score(y_test, y_proba))

    print_section("Confusion Matrix")
    print(cm)

    print_section("Classification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    print_section("Test Set Bankrupt Summary")
    print(f"Number of actual bankrupt companies in the test set: {int(y_test.sum())}")
    print(f"Number of bankrupt companies predicted by the model: {int(y_pred.sum())}")
    print(f"Number of true positives: {tp}")
    print(f"Number of false negatives: {fn}")
    print(f"Number of false positives: {fp}")
    print(f"Number of true negatives: {tn}")


if __name__ == "__main__":
    main()

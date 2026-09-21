from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
PLOT_IMPORTANCE_PATH = Path("ml/risk_prediction/shap_feature_importance.png")
PLOT_SUMMARY_PATH = Path("ml/risk_prediction/shap_summary.png")


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_shap_values(raw_values: object, feature_names: list[str], X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: object
    base_values: np.ndarray
    data: np.ndarray

    if hasattr(raw_values, "values"):
        values = raw_values.values
        base_values = np.asarray(raw_values.base_values if hasattr(raw_values, "base_values") else np.zeros(X_test.shape[0]))
        data = np.asarray(raw_values.data if hasattr(raw_values, "data") else X_test.to_numpy())
    else:
        values = raw_values
        base_values = np.zeros(X_test.shape[0])
        data = X_test.to_numpy()

    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        elif len(values) == 1:
            values = values[0]
        else:
            raise ValueError(f"Unexpected SHAP value structure with {len(values)} outputs.")

    values_array = np.asarray(values)

    if values_array.ndim == 3 and values_array.shape[-1] == 2:
        values_array = values_array[:, :, 1]
    elif values_array.ndim == 1:
        values_array = values_array.reshape(-1, 1)

    if values_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values_array.shape}")

    if values_array.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match the number of input features: "
            f"{values_array.shape[1]} vs {len(feature_names)}"
        )

    return values_array, base_values, data


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

    positive_count = int(y_train.sum())
    negative_count = int((1 - y_train).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    raw_shap_values = explainer.shap_values(X_test)
    shap_values, _, _ = normalize_shap_values(raw_shap_values, X_test.columns.tolist(), X_test)

    print_section("SHAP Summary")
    print(f"Number of test samples explained: {X_test.shape[0]}")
    print(f"Number of features explained: {X_test.shape[1]}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]

    print_section("Top 15 Features by Mean Absolute SHAP Value")
    for idx in top_indices:
        feature_name = X_test.columns[idx]
        print(f"Feature: {feature_name}")
        print(f"Mean absolute SHAP value: {mean_abs_shap[idx]:.8f}")

        if mean_shap[idx] > 0:
            direction = "Positive mean SHAP value = generally pushes prediction toward bankruptcy"
        elif mean_shap[idx] < 0:
            direction = "Negative mean SHAP value = generally pushes prediction toward non-bankruptcy"
        else:
            direction = "Mean SHAP value is zero; neutral effect"
        print(direction)
        print("---")

    top_feature_names = X_test.columns[top_indices].tolist()
    top_feature_values = shap_values[:, top_indices]
    top_explanation = shap.Explanation(
        values=top_feature_values,
        base_values=np.zeros(X_test.shape[0]),
        data=X_test.iloc[:, top_indices].to_numpy(),
        feature_names=top_feature_names,
    )

    print_section("Saving SHAP plots")
    print(f"Saving global importance plot to: {PLOT_IMPORTANCE_PATH}")
    shap.plots.bar(top_explanation, max_display=15)
    plt.tight_layout()
    plt.gcf().savefig(PLOT_IMPORTANCE_PATH, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saving summary plot to: {PLOT_SUMMARY_PATH}")
    shap.plots.beeswarm(top_explanation, max_display=15)
    plt.tight_layout()
    plt.gcf().savefig(PLOT_SUMMARY_PATH, bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

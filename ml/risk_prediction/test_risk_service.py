from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.risk.risk_service import RiskPredictionService


if __name__ == "__main__":
    service = RiskPredictionService()

    x_test_path = Path("data/processed/X_test.csv")
    x_test_df = pd.read_csv(x_test_path)
    first_row = x_test_df.iloc[0].to_dict()

    result = service.predict(first_row)

    print(result)

    probability = result.get("bankruptcy_probability")
    risk_level = result.get("risk_level")

    assert isinstance(probability, float), f"Probability must be float, got {type(probability).__name__}"
    assert 0.0 <= probability <= 1.0, f"Probability out of range: {probability}"
    assert risk_level in {"Low", "Medium", "High"}, f"Invalid risk level: {risk_level}"

    print("Risk service test passed.")

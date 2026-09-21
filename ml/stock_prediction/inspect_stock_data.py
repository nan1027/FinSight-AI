from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/AAPL_5y_daily.csv")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Stock dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print_section("1. Dataset shape")
    print(df.shape)

    print_section("2. First 5 rows")
    print(df.head().to_string(index=True))

    print_section("3. Last 5 rows")
    print(df.tail().to_string(index=True))

    print_section("4. Column names and dtypes")
    print(df.dtypes.to_string())

    print_section("5. Missing values per column")
    print(df.isna().sum().to_string())

    print_section("6. Number of duplicate rows")
    print(df.duplicated().sum())

    print_section("7. Minimum and maximum date")
    if "Date" in df.columns:
        date_series = pd.to_datetime(df["Date"], errors="coerce")
        print(f"Min date: {date_series.min()}")
        print(f"Max date: {date_series.max()}")
    else:
        print("Date column not found; dataset may not have been imported with an index-based date column.")

    print_section("8. Descriptive statistics")
    numeric_columns = [column for column in REQUIRED_COLUMNS if column in df.columns]
    print(df[numeric_columns].describe().to_string())

    print_section("9. Validate dates are sorted in ascending order")
    if "Date" in df.columns:
        try:
            parsed_dates = pd.to_datetime(df["Date"], errors="raise")
            is_sorted = parsed_dates.is_monotonic_increasing
            print(f"Dates sorted ascending: {is_sorted}")
            if not is_sorted:
                print("Validation failed: dates are not sorted in ascending order.")
        except Exception as exc:  # pragma: no cover - reporting validation issue
            print(f"Validation failed: could not parse the Date column. Error: {exc}")
    else:
        print("Validation failed: Date column is missing.")

    print_section("10. Validate Close contains only numeric values")
    if "Close" in df.columns:
        numeric_check = pd.to_numeric(df["Close"], errors="coerce")
        invalid_count = numeric_check.isna().sum()
        print(f"Non-numeric Close values: {invalid_count}")
        if invalid_count > 0:
            print("Validation failed: Close contains non-numeric values.")
    else:
        print("Validation failed: Close column is missing.")

    print_section("11. Validate required columns have no missing values")
    required_missing = df[REQUIRED_COLUMNS].isna().sum()
    print(required_missing.to_string())
    if required_missing.any():
        print("Validation failed: required columns contain missing values.")
    else:
        print("All required columns are complete.")


if __name__ == "__main__":
    main()

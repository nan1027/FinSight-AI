from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


TICKER = "AAPL"
PERIOD = "5y"
INTERVAL = "1d"
OUTPUT_PATH = Path("data/raw/AAPL_5y_daily.csv")


def main() -> None:
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        tickers=TICKER,
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if data is None or data.empty:
        raise RuntimeError(f"No stock data was downloaded for {TICKER} ({PERIOD}, {INTERVAL}).")

    if isinstance(data.columns, pd.MultiIndex):
        if TICKER in data.columns.get_level_values(0):
            data = data[TICKER].copy()
        else:
            data = data.droplevel(0, axis=1)

    standard_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in standard_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(
            f"Downloaded data for {TICKER} is missing expected market columns: {missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.loc[:, [column for column in standard_columns if column in data.columns]]

    # Keep the standard OHLCV set and avoid silently saving malformed output if yfinance changes its API shape.
    data.to_csv(output_path, index=True)

    print(f"ticker: {TICKER}")
    print(f"date range: {data.index.min()} to {data.index.max()}")
    print(f"number of rows: {len(data)}")
    print(f"column names: {list(data.columns)}")
    print(f"output path: {output_path}")


if __name__ == "__main__":
    main()

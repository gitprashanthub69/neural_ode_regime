"""
data.py — Download NSE stock data via yfinance
Downloads 5 years of daily OHLCV data for 5 NSE stocks and NIFTY50 index.
Cleans missing values, aligns dates, and saves to data/raw.csv.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"]
TICKER_NAMES = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY50"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_data():
    """Download 5 years of daily OHLCV data for NSE stocks and NIFTY50."""
    os.makedirs(DATA_DIR, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Tickers: {TICKERS}")

    all_data = {}

    for ticker, name in zip(TICKERS, TICKER_NAMES):
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty:
                print(f"  WARNING: No data for {ticker}, skipping.")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns with ticker prefix
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    all_data[f"{name}_{col}"] = df[col]

            print(f"  ✓ {name}: {len(df)} rows downloaded")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")

    if not all_data:
        raise RuntimeError("No data downloaded for any ticker!")

    # Combine into single DataFrame aligned on dates
    combined = pd.DataFrame(all_data)
    combined.index.name = "Date"

    # Forward-fill missing values (market holidays differ)
    combined = combined.ffill()

    # Drop any remaining rows with NaN (start-of-series alignment)
    combined = combined.dropna()

    # Save
    output_path = os.path.join(DATA_DIR, "raw.csv")
    combined.to_csv(output_path)
    print(f"\n✓ Raw data saved to {output_path}")
    print(f"  Shape: {combined.shape}")
    print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")

    return combined


if __name__ == "__main__":
    download_data()

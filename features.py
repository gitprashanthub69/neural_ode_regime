"""
features.py — Engineer financial features from raw OHLCV data.
Computes: log returns, rolling volatility, RSI, MACD signal, MA ratio.
Normalizes using StandardScaler and saves to data/features.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TICKER_NAMES = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY50"]


def compute_rsi(series, period=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd_signal(series, fast=12, slow=26, signal=9):
    """Compute MACD signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return signal_line


def engineer_features():
    """Compute all features for each stock and normalize."""
    raw_path = os.path.join(DATA_DIR, "raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run data.py first.")

    raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    print(f"Loaded raw data: {raw.shape}")

    feature_frames = []

    for name in TICKER_NAMES:
        close_col = f"{name}_Close"
        if close_col not in raw.columns:
            print(f"  WARNING: {close_col} not found, skipping {name}")
            continue

        close = raw[close_col]
        print(f"  Engineering features for {name}...")

        features = pd.DataFrame(index=raw.index)

        # 1. Daily log returns
        features[f"{name}_log_return"] = np.log(close / close.shift(1))

        # 2. 10-day rolling volatility
        features[f"{name}_volatility_10d"] = features[f"{name}_log_return"].rolling(
            window=10, min_periods=10
        ).std()

        # 3. RSI (14 period)
        features[f"{name}_rsi"] = compute_rsi(close, period=14)

        # 4. MACD signal line
        features[f"{name}_macd_signal"] = compute_macd_signal(close)

        # 5. 20-day / 50-day moving average ratio
        ma_20 = close.rolling(window=20, min_periods=20).mean()
        ma_50 = close.rolling(window=50, min_periods=50).mean()
        features[f"{name}_ma_ratio"] = ma_20 / ma_50

        feature_frames.append(features)

    # Combine all stock features
    all_features = pd.concat(feature_frames, axis=1)

    # Drop rows with NaN (from rolling windows warmup)
    all_features = all_features.dropna()

    print(f"\n  Features computed: {all_features.shape[1]} columns, {len(all_features)} rows")
    print(f"  Date range: {all_features.index[0]} to {all_features.index[-1]}")

    # Normalize using StandardScaler
    scaler = StandardScaler()
    feature_values = scaler.fit_transform(all_features.values)
    all_features_normalized = pd.DataFrame(
        feature_values, index=all_features.index, columns=all_features.columns
    )

    # Save
    output_path = os.path.join(DATA_DIR, "features.csv")
    all_features_normalized.to_csv(output_path)
    print(f"\n✓ Normalized features saved to {output_path}")

    return all_features_normalized


if __name__ == "__main__":
    engineer_features()

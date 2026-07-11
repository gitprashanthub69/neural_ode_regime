"""
train.py — Training loop for Neural ODE Regime Detector.
Implements:
  - Sliding window dataset (30-day windows → regime label)
  - Automatic regime labeling (bull/bear/crisis)
  - Walk-forward train/val/test split (70/15/15, no lookahead)
  - Adam optimizer with CrossEntropyLoss
  - Trains for 100 epochs, saves best model
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralODERegimeDetector, LATENT_DIM


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
WINDOW_SIZE = 30
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

# Regime thresholds
BULL_THRESHOLD = 0.03    # 20-day return > +3%
BEAR_THRESHOLD = -0.03   # 20-day return < -3%
VOLATILITY_STD = 2.0     # Crisis: volatility > 2 std above mean

# Regime labels
REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}


class RegimeDataset(Dataset):
    """Sliding window dataset for regime detection."""

    def __init__(self, features, labels):
        """
        Args:
            features: np.array of shape (num_windows, window_size, num_features)
            labels: np.array of shape (num_windows,) — regime labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_regime_labels(raw_path, features_df):
    """
    Automatically label regimes using NIFTY50 data:
      - Bull: 20-day forward return > +3%
      - Bear: 20-day forward return < -3%
      - Crisis: 10-day rolling volatility > 2 std above its mean
    """
    raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True)

    # Use NIFTY50 Close for labeling
    nifty_close = raw["NIFTY50_Close"]

    # 20-day forward return (we use the return over the past 20 days to avoid lookahead)
    returns_20d = nifty_close.pct_change(periods=20)

    # 10-day rolling volatility of daily returns
    daily_returns = nifty_close.pct_change()
    volatility_10d = daily_returns.rolling(window=10).std()
    vol_mean = volatility_10d.mean()
    vol_std = volatility_10d.std()
    crisis_threshold = vol_mean + VOLATILITY_STD * vol_std

    # Assign labels (aligned to features index)
    labels = pd.Series(index=features_df.index, dtype=int)

    for date in features_df.index:
        if date not in returns_20d.index or date not in volatility_10d.index:
            labels[date] = 0  # Default to bull
            continue

        vol = volatility_10d.get(date, 0)
        ret = returns_20d.get(date, 0)

        if pd.isna(vol) or pd.isna(ret):
            labels[date] = 0
            continue

        # Crisis takes priority
        if vol > crisis_threshold:
            labels[date] = 2  # Crisis
        elif ret > BULL_THRESHOLD:
            labels[date] = 0  # Bull
        elif ret < BEAR_THRESHOLD:
            labels[date] = 1  # Bear
        else:
            labels[date] = 0  # Default to bull (neutral periods)

    return labels.values


def create_sliding_windows(features, labels, window_size=WINDOW_SIZE):
    """Create sliding window samples from time series data."""
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(labels[i + window_size - 1])  # Label at end of window
    return np.array(X), np.array(y)


def walk_forward_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Walk-forward split — no lookahead bias.
    Data is already in chronological order; we split sequentially.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model():
    """Full training pipeline."""
    print("=" * 60)
    print("TRAINING NEURAL ODE REGIME DETECTOR")
    print("=" * 60)

    # Load features
    features_path = os.path.join(DATA_DIR, "features.csv")
    raw_path = os.path.join(DATA_DIR, "raw.csv")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found at {features_path}. Run features.py first.")

    features_df = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    feature_values = features_df.values
    input_dim = feature_values.shape[1]

    print(f"\nFeatures loaded: {feature_values.shape}")
    print(f"Input dimension: {input_dim}")

    # Compute regime labels
    print("\nComputing regime labels...")
    labels = compute_regime_labels(raw_path, features_df)

    # Print regime distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Regime distribution:")
    for u, c in zip(unique, counts):
        print(f"  {REGIME_NAMES.get(u, u)}: {c} ({100*c/len(labels):.1f}%)")

    # Create sliding windows
    print(f"\nCreating {WINDOW_SIZE}-day sliding windows...")
    X, y = create_sliding_windows(feature_values, labels)
    print(f"Total windows: {len(X)}")

    # Walk-forward split
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(X, y)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Handle class imbalance with weighted loss
    train_counts = np.bincount(y_train.astype(int), minlength=3).astype(float)
    train_counts = np.maximum(train_counts, 1.0)  # Avoid division by zero
    class_weights = 1.0 / train_counts
    class_weights = class_weights / class_weights.sum() * 3  # Normalize
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights.numpy()}")

    # Create data loaders
    train_dataset = RegimeDataset(X_train, y_train)
    val_dataset = RegimeDataset(X_val, y_val)
    test_dataset = RegimeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialize model
    model = NeuralODERegimeDetector(input_dim=input_dim, latent_dim=LATENT_DIM)
    model = model.to(device)
    class_weights = class_weights.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    model_path = os.path.join(MODEL_DIR, "model.pth")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping for ODE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"{epoch:>5} | {avg_train_loss:>10.4f} | {train_acc:>8.2f}% | "
                f"{avg_val_loss:>8.4f} | {val_acc:>6.2f}%"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": LATENT_DIM,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                model_path,
            )

    print("-" * 55)
    print(f"\n✓ Best model saved at epoch {best_epoch} with val accuracy {best_val_acc:.2f}%")
    print(f"  Model path: {model_path}")

    # Save test data for evaluate.py
    np.savez(
        os.path.join(DATA_DIR, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
    )
    print(f"  Test data saved for evaluation")

    return model, best_val_acc


if __name__ == "__main__":
    train_model()

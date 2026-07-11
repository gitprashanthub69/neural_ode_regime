"""
plot.py — Visualize detected regimes on NIFTY50 price chart.
Colors background by regime: green=bull, red=bear, grey=crisis.
Saves output as regime_plot.png.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from model import NeuralODERegimeDetector


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

REGIME_COLORS = {
    0: "#2ecc71",  # Bull — green
    1: "#e74c3c",  # Bear — red
    2: "#95a5a6",  # Crisis — grey
}
REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}


def predict_all_regimes():
    """Run Neural ODE on full feature dataset to get regime predictions."""
    features_path = os.path.join(DATA_DIR, "features.csv")
    model_path = os.path.join(MODEL_DIR, "model.pth")

    features_df = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    feature_values = features_df.values
    dates = features_df.index

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]

    model = NeuralODERegimeDetector(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create sliding windows for all data
    window_size = 30
    all_preds = []
    pred_dates = []

    with torch.no_grad():
        for i in range(len(feature_values) - window_size):
            window = feature_values[i : i + window_size]
            x = torch.FloatTensor(window).unsqueeze(0)  # (1, 30, features)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            all_preds.append(pred)
            pred_dates.append(dates[i + window_size - 1])

    return np.array(all_preds), pred_dates


def plot_regimes():
    """Plot NIFTY50 price with regime-colored background."""
    print("\nGenerating regime plot...")

    # Get regime predictions
    preds, pred_dates = predict_all_regimes()

    # Load raw NIFTY50 price
    raw = pd.read_csv(
        os.path.join(DATA_DIR, "raw.csv"), index_col="Date", parse_dates=True
    )
    nifty_close = raw["NIFTY50_Close"]

    # Filter to prediction dates
    nifty_plot = nifty_close.loc[nifty_close.index.isin(pred_dates)]
    dates_plot = nifty_plot.index

    # Align predictions with dates
    pred_series = pd.Series(preds, index=pred_dates)
    pred_aligned = pred_series.loc[pred_series.index.isin(dates_plot)]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)

    # Plot regime bands
    prev_regime = pred_aligned.iloc[0]
    start_idx = 0

    for i in range(1, len(pred_aligned)):
        current_regime = pred_aligned.iloc[i]

        if current_regime != prev_regime or i == len(pred_aligned) - 1:
            end_idx = i if current_regime != prev_regime else i + 1
            color = REGIME_COLORS[prev_regime]
            ax.axvspan(
                dates_plot[start_idx],
                dates_plot[min(end_idx, len(dates_plot) - 1)],
                alpha=0.25,
                color=color,
                linewidth=0,
            )
            start_idx = i
            prev_regime = current_regime

    # Plot price line
    ax.plot(
        dates_plot,
        nifty_plot.values,
        color="#2c3e50",
        linewidth=1.2,
        label="NIFTY50 Close",
        zorder=5,
    )

    # Formatting
    ax.set_title(
        "NIFTY50 Market Regimes — Neural ODE Detection",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (₹)", fontsize=12)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)

    # Legend
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[0], alpha=0.4, label="Bull"),
        Patch(facecolor=REGIME_COLORS[1], alpha=0.4, label="Bear"),
        Patch(facecolor=REGIME_COLORS[2], alpha=0.4, label="Crisis"),
        plt.Line2D([0], [0], color="#2c3e50", linewidth=1.5, label="NIFTY50 Close"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    # Regime distribution annotation
    unique, counts = np.unique(preds, return_counts=True)
    dist_text = " | ".join(
        [f"{REGIME_NAMES[u]}: {100*c/len(preds):.0f}%" for u, c in zip(unique, counts)]
    )
    ax.text(
        0.5,
        -0.12,
        f"Regime Distribution: {dist_text}",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
        color="#555",
    )

    plt.tight_layout()

    # Save
    output_path = os.path.join(MODEL_DIR, "regime_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"* Regime plot saved to {output_path}")
    return output_path


if __name__ == "__main__":
    plot_regimes()

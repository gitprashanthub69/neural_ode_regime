"""
evaluate.py — Evaluate trained Neural ODE vs HMM baseline.
Loads best model, runs on test set, generates classification report.
Compares against GaussianHMM (hmmlearn) baseline.
Saves results to accuracy_report.txt.
"""

import os
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
from hmmlearn.hmm import GaussianHMM

from model import NeuralODERegimeDetector


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
REGIME_NAMES = ["Bull", "Bear", "Crisis"]


def evaluate_neural_ode(X_test, y_test):
    """Evaluate the trained Neural ODE model on test set."""
    model_path = os.path.join(MODEL_DIR, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]

    # Initialize model
    model = NeuralODERegimeDetector(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Predict on test set
    X_tensor = torch.FloatTensor(X_test).to(device)
    all_preds = []

    # Process in batches to avoid OOM
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    ode_accuracy = accuracy_score(y_test, all_preds) * 100

    print("\n" + "=" * 60)
    print("NEURAL ODE — TEST SET RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {ode_accuracy:.2f}%\n")
    print(
        classification_report(
            y_test, all_preds, target_names=REGIME_NAMES, zero_division=0
        )
    )

    return all_preds, ode_accuracy


def evaluate_hmm_baseline(X_test, y_test):
    """
    Train and evaluate GaussianHMM baseline.
    We use the test windows' features (flattened to 2D) to fit an HMM,
    then map HMM states to ground-truth labels using majority voting.
    """
    print("\n" + "=" * 60)
    print("HMM BASELINE — TEST SET RESULTS")
    print("=" * 60)

    # For HMM: use the last time step features from each window
    X_hmm = X_test[:, -1, :]  # (n_samples, n_features)

    # Fit GaussianHMM
    n_components = 3
    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )

    try:
        hmm.fit(X_hmm)
        hmm_states = hmm.predict(X_hmm)
    except Exception as e:
        print(f"  HMM fitting failed: {e}")
        print("  Using random baseline instead.")
        hmm_states = np.random.randint(0, 3, size=len(y_test))

    # Map HMM states to true labels via majority voting
    state_map = {}
    for state in range(n_components):
        mask = hmm_states == state
        if mask.sum() > 0:
            # Most common true label for this HMM state
            true_labels = y_test[mask]
            state_map[state] = np.bincount(true_labels.astype(int), minlength=3).argmax()
        else:
            state_map[state] = 0

    # Resolve conflicts: if multiple HMM states map to same label,
    # assign based on highest count
    used_labels = set()
    sorted_states = sorted(
        state_map.keys(),
        key=lambda s: np.sum(hmm_states == s),
        reverse=True,
    )
    final_map = {}
    for state in sorted_states:
        preferred = state_map[state]
        if preferred not in used_labels:
            final_map[state] = preferred
            used_labels.add(preferred)
        else:
            # Assign next available label
            for label in range(3):
                if label not in used_labels:
                    final_map[state] = label
                    used_labels.add(label)
                    break
            else:
                final_map[state] = state_map[state]

    hmm_preds = np.array([final_map.get(s, 0) for s in hmm_states])
    hmm_accuracy = accuracy_score(y_test, hmm_preds) * 100

    print(f"\nAccuracy: {hmm_accuracy:.2f}%\n")
    print(
        classification_report(
            y_test, hmm_preds, target_names=REGIME_NAMES, zero_division=0
        )
    )

    return hmm_preds, hmm_accuracy


def run_evaluation():
    """Run full evaluation and save results."""
    # Load test data
    test_path = os.path.join(DATA_DIR, "test_data.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Run train.py first.")

    test_data = np.load(test_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print(f"Test set: {len(X_test)} samples")

    # Evaluate Neural ODE
    ode_preds, ode_acc = evaluate_neural_ode(X_test, y_test)

    # Evaluate HMM baseline
    hmm_preds, hmm_acc = evaluate_hmm_baseline(X_test, y_test)

    # Summary
    improvement = ode_acc - hmm_acc

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Neural ODE Accuracy:  {ode_acc:.2f}%")
    print(f"  HMM Baseline Accuracy: {hmm_acc:.2f}%")
    print(f"  Improvement:           {improvement:+.2f}%")
    print("=" * 60)

    # Save results
    report_path = os.path.join(MODEL_DIR, "accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NEURAL ODE MARKET REGIME DETECTOR — ACCURACY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Test set size: {len(X_test)} samples\n\n")

        f.write("--- Neural ODE Results ---\n")
        f.write(f"Accuracy: {ode_acc:.2f}%\n")
        f.write(
            classification_report(
                y_test, ode_preds, target_names=REGIME_NAMES, zero_division=0
            )
        )
        f.write("\n")

        f.write("--- HMM Baseline Results ---\n")
        f.write(f"Accuracy: {hmm_acc:.2f}%\n")
        f.write(
            classification_report(
                y_test, hmm_preds, target_names=REGIME_NAMES, zero_division=0
            )
        )
        f.write("\n")

        f.write("--- Comparison ---\n")
        f.write(f"Neural ODE Accuracy:   {ode_acc:.2f}%\n")
        f.write(f"HMM Baseline Accuracy: {hmm_acc:.2f}%\n")
        f.write(f"Improvement:           {improvement:+.2f}%\n")

    print(f"\n* Accuracy report saved to {report_path}")

    return ode_acc, hmm_acc


if __name__ == "__main__":
    run_evaluation()

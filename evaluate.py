"""
evaluate.py — Evaluate trained Neural ODE vs HMM baseline.
Loads best model, runs on test set, generates classification report.
Compares against GaussianHMM (hmmlearn) baseline.
Saves results to accuracy_report.txt.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM

from model import NeuralODERegimeDetector, LATENT_DIM
from train import compute_regime_labels, create_sliding_windows, walk_forward_split


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
REGIME_NAMES = ["Bull", "Bear", "Crisis"]


def evaluate_neural_ode(X_test, y_test):
    """Evaluate the trained Neural ODE model on test set."""
    model_path = os.path.join(MODEL_DIR, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]

    model = NeuralODERegimeDetector(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tensor = torch.FloatTensor(X_test).to(device)
    all_preds = []

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
    """
    print("\n" + "=" * 60)
    print("HMM BASELINE — TEST SET RESULTS")
    print("=" * 60)

    X_hmm = X_test[:, -1, :]

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
        hmm_states = np.random.randint(0, 3, size=len(y_test))

    state_map = {}
    for state in range(n_components):
        mask = hmm_states == state
        if mask.sum() > 0:
            true_labels = y_test[mask]
            state_map[state] = np.bincount(true_labels.astype(int), minlength=3).argmax()
        else:
            state_map[state] = 0

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


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


def run_ablation_study(ode_acc, ode_preds):
    """Train baseline models and print ablation study."""
    features_path = os.path.join(DATA_DIR, "features.csv")
    raw_path = os.path.join(DATA_DIR, "raw.csv")
    
    features_df = pd.read_csv(features_path, index_col="Date", parse_dates=True)
    feature_values = features_df.values
    labels = compute_regime_labels(raw_path, features_df)
    
    X, y = create_sliding_windows(feature_values, labels)
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(X, y)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    results = {}
    
    print("\nRunning Ablation Study on exact walk-forward split...")
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    lr.fit(X_train_flat, y_train)
    lr_preds = lr.predict(X_test_flat)
    results["Logistic Regression"] = (accuracy_score(y_test, lr_preds)*100, f1_score(y_test, lr_preds, average='macro', zero_division=0))
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_preds = rf.predict(X_test_flat)
    results["Random Forest"] = (accuracy_score(y_test, rf_preds)*100, f1_score(y_test, rf_preds, average='macro', zero_division=0))
    
    # 3. GRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]
    gru = GRUModel(input_dim, LATENT_DIM).to(device)
    
    train_counts = np.bincount(y_train.astype(int), minlength=3).astype(float)
    train_counts = np.maximum(train_counts, 1.0)
    class_weights = 1.0 / train_counts
    class_weights = class_weights / class_weights.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.Adam(gru.parameters(), lr=1e-3)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    gru.train()
    for _ in range(100):
        optimizer.zero_grad()
        logits = gru(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        
    gru.eval()
    with torch.no_grad():
        gru_preds = gru(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    results["GRU"] = (accuracy_score(y_test, gru_preds)*100, f1_score(y_test, gru_preds, average='macro', zero_division=0))
    
    # 4. Neural ODE (Using precomputed predictions)
    ode_f1 = f1_score(y_test, ode_preds, average='macro', zero_division=0)
    results["Neural ODE"] = (ode_acc, ode_f1)
    
    # Generate Table
    table = f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<8}\n"
    table += "-" * 40 + "\n"
    for model in ["Logistic Regression", "Random Forest", "GRU", "Neural ODE"]:
        acc, f1 = results[model]
        table += f"{model:<20} {acc:>5.1f}%     {f1:>4.2f}\n"
        
    print("\n" + "=" * 40)
    print("ABLATION STUDY RESULTS")
    print("=" * 40)
    print(table)
    
    return table


def run_evaluation():
    """Run full evaluation and save results."""
    test_path = os.path.join(DATA_DIR, "test_data.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Run train.py first.")

    test_data = np.load(test_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print(f"Test set: {len(X_test)} samples")

    ode_preds, ode_acc = evaluate_neural_ode(X_test, y_test)
    hmm_preds, hmm_acc = evaluate_hmm_baseline(X_test, y_test)

    improvement = ode_acc - hmm_acc

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Neural ODE Accuracy:  {ode_acc:.2f}%")
    print(f"  HMM Baseline Accuracy: {hmm_acc:.2f}%")
    print(f"  Improvement:           {improvement:+.2f}%")
    print("=" * 60)
    
    # Run ablation study
    ablation_table = run_ablation_study(ode_acc, ode_preds)

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

        f.write("--- HMM Baseline Results (Unsupervised) ---\n")
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
        
        f.write("\n\n" + "=" * 40 + "\n")
        f.write("ABLATION STUDY RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(ablation_table)

    print(f"\n* Accuracy report saved to {report_path}")

    return ode_acc, hmm_acc


if __name__ == "__main__":
    run_evaluation()

"""
main.py — Neural ODE Market Regime Detector
============================================
End-to-end pipeline:
  1. Download NSE stock data (yfinance)
  2. Engineer financial features
  3. Train Neural ODE model
  4. Evaluate vs HMM baseline
  5. Plot regime visualization

Run: python main.py
Outputs: regime_plot.png, accuracy_report.txt, model.pth
"""

import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start_time = time.time()

    print("+" + "=" * 58 + "+")
    print("|   NEURAL ODE MARKET REGIME DETECTOR                     |")
    print("|   Latent ODE dynamics for stock regime classification    |")
    print("+" + "=" * 58 + "+")
    print()

    # Step 1: Download data
    print("-" * 60)
    print("STEP 1/5: Downloading NSE stock data")
    print("-" * 60)
    from data import download_data
    download_data()
    print()

    # Step 2: Engineer features
    print("-" * 60)
    print("STEP 2/5: Engineering financial features")
    print("-" * 60)
    from features import engineer_features
    engineer_features()
    print()

    # Step 3: Train model
    print("-" * 60)
    print("STEP 3/5: Training Neural ODE model")
    print("-" * 60)
    from train import train_model
    model, val_acc = train_model()
    print()

    # Step 4: Evaluate
    print("-" * 60)
    print("STEP 4/5: Evaluating model vs HMM baseline")
    print("-" * 60)
    from evaluate import run_evaluation
    ode_acc, hmm_acc = run_evaluation()
    print()

    # Step 5: Plot
    print("-" * 60)
    print("STEP 5/5: Generating regime visualization")
    print("-" * 60)
    from plot import plot_regimes
    plot_regimes()
    print()

    # Final summary
    elapsed = time.time() - start_time
    improvement = ode_acc - hmm_acc

    print("+" + "=" * 58 + "+")
    print("|   FINAL RESULTS                                         |")
    print("+" + "=" * 58 + "+")
    print(f"|   Neural ODE Accuracy:   {ode_acc:>6.2f}%                       |")
    print(f"|   HMM Baseline Accuracy: {hmm_acc:>6.2f}%                       |")
    print(f"|   Improvement:           {improvement:>+6.2f}%                       |")
    print("+" + "=" * 58 + "+")
    print("|   Outputs:                                              |")
    print("|     * regime_plot.png      -- regime visualization      |")
    print("|     * accuracy_report.txt  -- detailed metrics          |")
    print("|     * model.pth            -- trained model weights     |")
    print("+" + "=" * 58 + "+")
    print(f"|   Total runtime: {elapsed:>6.1f}s                                |")
    print("+" + "=" * 58 + "+")


if __name__ == "__main__":
    main()

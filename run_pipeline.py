"""Master Pipeline Runner for SafeAdapt Fresh Training & Evaluation.

Executes the complete workflow:
1. Training ResNet-18 on Hospital 0 (Center 0 - RadboudUMC) with batch size = 1000.
2. Logging intermediate outputs and saving checkpoints per epoch.
3. Evaluating baseline vs SafeAdapt adaptation on Hospital 1 (Center 1 - CWZ Nijmegen).
4. Generating all publication-quality graphs and clinical diagnostic metrics.
"""

import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    print("=" * 70)
    print("        SafeAdapt Pipeline: Source Training & TTA              ")
    print("=" * 70)
    print(f"Working Directory: {root_dir}")
    print("Hospital 0 (Source Train/Val): Center 0 (RadboudUMC) [10,000 samples]")
    print("Hospital 1 & 2 (Target Adaptation): Center 1 (CWZ) & Center 2 (UMCU)")
    print("Effective Batch Size: 1000")
    print("=" * 70)

    # Step 1: Run Training
    print("\n>>> Step 1: Model Training (Hospital 0, 20 Epochs, Batch Size = 1000)")
    cmd_train = [
        sys.executable, str(root_dir / "train.py"),
        "--epochs", "20",
        "--batch-size", "1000",
        "--micro-batch-size", "100",
        "--train-samples", "10000",
        "--val-samples", "1000",
        "--output-dir", str(root_dir)
    ]
    ret_train = subprocess.run(cmd_train)
    if ret_train.returncode != 0:
        print("Training failed. Halting pipeline.")
        sys.exit(ret_train.returncode)

    # Step 2: Run Evaluation
    print("\n>>> Step 2: Evaluation & SafeAdapt Adaptation (Hospital 1 - CWZ Nijmegen)")
    cmd_eval = [
        sys.executable, str(root_dir / "evaluate.py"),
        "--model-path", str(root_dir / "models" / "best_model.pth"),
        "--output-dir", str(root_dir),
        "--test-samples", "1000",
        "--target-hospital", "1"
    ]
    ret_eval = subprocess.run(cmd_eval)
    if ret_eval.returncode != 0:
        print("Evaluation failed. Halting pipeline.")
        sys.exit(ret_eval.returncode)

    # Step 3: Generate Graphs
    print("\n>>> Step 3: Plotting Evaluation Graphs & Feature Maps")
    cmd_plot = [
        sys.executable, str(root_dir / "plot_graphs.py"),
        "--base-dir", str(root_dir)
    ]
    ret_plot = subprocess.run(cmd_plot)
    if ret_plot.returncode != 0:
        print("Graph plotting failed.")
        sys.exit(ret_plot.returncode)

    print("\n" + "=" * 70)
    print("                   Pipeline Complete                           ")
    print("=" * 70)
    print(f"Checkpoints: {root_dir / 'models'}")
    print(f"Graphs:      {root_dir / 'results' / 'graphs'}")
    print(f"Metrics:     {root_dir / 'results' / 'metrics'}")


if __name__ == "__main__":
    main()


"""Evaluation script for Target Hospital 1 (CWZ Nijmegen) and Hospital 2 (UMC Utrecht).

Calculates clinical diagnostic metrics:
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- Evaluates both Source Model and SafeAdapt Adapted Model on Target Hospital.
- Saves raw predictions to predictions.npz for visualization.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from data_loader import get_dataloaders
from model import ResNet18Binary


def evaluate_model(model, dataloader, device):
    """Runs inference and computes full evaluation metrics."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    return metrics, cm, y_true, y_prob


def run_safeadapt_adaptation(model, dataloader, device, steps_per_batch=1, lr=1e-4):
    """Adapts model on Target Hospital data without using ground-truth labels."""
    print("Running SafeAdapt Test-Time Adaptation on Target Hospital...")
    trainable_params = model.configure_tta()
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    for step, (images, _) in enumerate(dataloader, 1):
        images = images.to(device)
        loss = model.adapt_step(images, optimizer)
        if step % 5 == 0 or step == len(dataloader):
            print(f"  [SafeAdapt Step {step}/{len(dataloader)}] TTA Objective: {loss:.4f}")

    print("SafeAdapt test-time adaptation complete.")


def main():
    base_default = Path(__file__).resolve().parent
    model_default = base_default / "models" / "best_model.pth"

    parser = argparse.ArgumentParser(description="Evaluate on Target Hospital")
    parser.add_argument("--model-path", type=str, default=str(model_default))
    parser.add_argument("--output-dir", type=str, default=str(base_default))
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--target-hospital", type=int, default=1, choices=[1, 2], help="Target domain (1: Hospital 1, 2: Hospital 2)")
    args = parser.parse_args()

    target_name = f"Target Domain {args.target_hospital} (Hospital {args.target_hospital})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SafeAdapt Target Evaluation ===")
    print(f"Device: {device} | Evaluation Site: {target_name}")

    base_dir = Path(args.output_dir)
    metrics_dir = base_dir / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load target test data
    _, _, test_loader, _ = get_dataloaders(
        batch_size=args.test_samples,
        micro_batch_size=100,
        test_samples=args.test_samples,
        target_hospital=args.target_hospital,
        include_train=False,
        include_val=False,
        include_test=True
    )

    # 1. Evaluate Source Model Baseline
    print(f"\n--- Evaluating Baseline Model (Source Only) on Hospital {args.target_hospital} ({target_name}) ---")
    model = ResNet18Binary(pretrained=False).to(device)
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found, evaluating initialized model.")

    proto_file = Path(args.output_dir) / "models" / "source_prototypes.pt"
    if proto_file.exists():
        model.prototypes = torch.load(proto_file, map_location=device)
        print(f"Loaded source class prototypes from {proto_file}")

    base_metrics, base_cm, y_true, base_prob = evaluate_model(model, test_loader, device)

    # 2. Run SafeAdapt Test-Time Adaptation
    print(f"\n--- Running SafeAdapt Adaptation on Hospital {args.target_hospital} ({target_name}) ---")
    run_safeadapt_adaptation(model, test_loader, device)
    safe_metrics, safe_cm, _, safe_prob = evaluate_model(model, test_loader, device)

    # Print Comparison Table
    print("\n" + "=" * 65)
    print(f"       HOSPITAL {args.target_hospital} ({target_name}) EVALUATION COMPARISON       ")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Source Baseline':<18} | {'SafeAdapt (Ours)':<18}")
    print("-" * 65)
    for m in ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]:
        print(f"{m.capitalize():<20} | {base_metrics[m]:<18.4f} | {safe_metrics[m]:<18.4f}")
    print("-" * 65)
    print(f"Confusion Matrix (TN/FP/FN/TP):")
    print(f"  Baseline:  TN={base_metrics['tn']}, FP={base_metrics['fp']}, FN={base_metrics['fn']}, TP={base_metrics['tp']}")
    print(f"  SafeAdapt: TN={safe_metrics['tn']}, FP={safe_metrics['fp']}, FN={safe_metrics['fn']}, TP={safe_metrics['tp']}")
    print("=" * 65)

    # Save predictions array for plotting graphs
    np.savez(metrics_dir / "predictions.npz", y_true=y_true, base_prob=base_prob, safe_prob=safe_prob,
             base_cm=base_cm, safe_cm=safe_cm)
    print(f"Saved raw test predictions for graph generation.")


if __name__ == "__main__":
    main()


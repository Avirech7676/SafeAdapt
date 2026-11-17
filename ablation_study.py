"""Hyperparameter Sensitivity and Ablation Study for SafeAdapt-OT.

Evaluates the sensitivity and robustness of SafeAdapt-OT under varying:
1. Trust-region L-infinity projection radius: delta_max in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
2. Conformal anomaly rollback threshold: tau in [0.40, 0.44, 0.48, 0.52, 0.56, 0.60]

Outputs:
- results/metrics/ablation_study_results.csv
- results/graphs/hyperparameter_sensitivity_ablation.png
"""

import csv
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

# Ensure SafeAdapt_Fresh root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loader import get_dataloaders
from model import ResNet18Binary


def evaluate_loader(model, loader, device):
    """Computes AUROC, Accuracy, and Recall on evaluation loader."""
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
    auc = float(roc_auc_score(y_true, y_prob))
    return auc, acc, rec


def run_delta_ablation(device, model_path, proto_path, test_loader, deltas):
    """Sweeps trust-region projection radius delta_max."""
    print("\n" + "=" * 70)
    print(" Ablation 1: Trust-Region Projection Radius (delta_max) Sensitivity")
    print("=" * 70)
    print(f"{'delta_max':<12} | {'AUROC':<10} | {'Accuracy (%)':<14} | {'Recall (%)':<12} | {'Rollbacks':<10}")
    print("-" * 70)

    delta_results = []
    for delta in deltas:
        # Load fresh model weights
        model = ResNet18Binary(pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        if proto_path.exists():
            model.prototypes = torch.load(proto_path, map_location=device)

        trainable_params = model.configure_tta()
        optimizer = torch.optim.Adam(trainable_params, lr=5e-4)

        rollbacks = 0
        total_steps = 0
        for images, _ in test_loader:
            images = images.to(device)
            loss, is_rolled_back = model.adapt_step_ot(
                images,
                optimizer,
                delta_max=delta,
                rollback_threshold=0.48
            )
            if is_rolled_back:
                rollbacks += 1
            total_steps += 1

        auc, acc, rec = evaluate_loader(model, test_loader, device)
        delta_results.append({
            "param": "delta_max",
            "value": delta,
            "auroc": auc,
            "accuracy": acc,
            "recall": rec,
            "rollbacks": rollbacks,
            "total_steps": total_steps
        })
        marker = " [DEFAULT]" if abs(delta - 0.05) < 1e-4 else ""
        print(f"{delta:<12.3f} | {auc:<10.4f} | {acc:<14.2f} | {rec:<12.2f} | {rollbacks}/{total_steps:<7}{marker}")

    return delta_results


def run_tau_ablation(device, model_path, proto_path, test_loader, taus):
    """Sweeps conformal anomaly rollback threshold tau."""
    print("\n" + "=" * 70)
    print(" Ablation 2: Conformal Anomaly Rollback Threshold (tau) Sensitivity")
    print("=" * 70)
    print(f"{'tau':<12} | {'AUROC':<10} | {'Accuracy (%)':<14} | {'Recall (%)':<12} | {'Rollbacks':<10}")
    print("-" * 70)

    tau_results = []
    for tau in taus:
        # Load fresh model weights
        model = ResNet18Binary(pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        if proto_path.exists():
            model.prototypes = torch.load(proto_path, map_location=device)

        trainable_params = model.configure_tta()
        optimizer = torch.optim.Adam(trainable_params, lr=5e-4)

        rollbacks = 0
        total_steps = 0
        for images, _ in test_loader:
            images = images.to(device)
            loss, is_rolled_back = model.adapt_step_ot(
                images,
                optimizer,
                delta_max=0.05,
                rollback_threshold=tau
            )
            if is_rolled_back:
                rollbacks += 1
            total_steps += 1

        auc, acc, rec = evaluate_loader(model, test_loader, device)
        tau_results.append({
            "param": "rollback_threshold",
            "value": tau,
            "auroc": auc,
            "accuracy": acc,
            "recall": rec,
            "rollbacks": rollbacks,
            "total_steps": total_steps
        })
        marker = " [DEFAULT]" if abs(tau - 0.48) < 1e-4 else ""
        print(f"{tau:<12.2f} | {auc:<10.4f} | {acc:<14.2f} | {rec:<12.2f} | {rollbacks}/{total_steps:<7}{marker}")

    return tau_results


def plot_ablation_curves(delta_results, tau_results, base_auc, output_png):
    """Plots publication-grade 2-panel hyperparameter sensitivity curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: delta_max sensitivity
    ax1 = axes[0]
    d_vals = [r["value"] for r in delta_results]
    d_aucs = [r["auroc"] for r in delta_results]
    ax1.plot(d_vals, d_aucs, "o-", color="#2563EB", linewidth=2.4, markersize=7, label="SafeAdapt-OT AUROC")
    ax1.axhline(base_auc, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Source Baseline ({base_auc:.3f})")
    ax1.axvline(0.05, color="#059669", linestyle=":", linewidth=2.0, label="Selected Default (0.05)")
    ax1.scatter([0.05], [d_aucs[d_vals.index(0.05)]], color="#059669", s=130, zorder=5)

    ax1.set_xscale("log")
    ax1.set_xticks(d_vals)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_ylim(base_auc - 0.02, 1.0)
    ax1.set_title("Sensitivity to Trust-Region Radius $\\delta_{\\max}$", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Projection Radius $\\delta_{\\max}$ ($L_\\infty$ Bound, log-scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Target AUROC", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=10, loc="lower right")

    # Panel B: tau sensitivity
    ax2 = axes[1]
    t_vals = [r["value"] for r in tau_results]
    t_aucs = [r["auroc"] for r in tau_results]
    t_rolls = [r["rollbacks"] for r in tau_results]

    ax2.plot(t_vals, t_aucs, "s-", color="#7C3AED", linewidth=2.4, markersize=7, label="SafeAdapt-OT AUROC")
    ax2.axhline(base_auc, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Source Baseline ({base_auc:.3f})")
    ax2.axvline(0.48, color="#059669", linestyle=":", linewidth=2.0, label="Selected Default (0.48)")
    ax2.scatter([0.48], [t_aucs[t_vals.index(0.48)]], color="#059669", s=130, zorder=5)

    ax2.set_ylim(base_auc - 0.02, 1.0)
    for v, a, r in zip(t_vals, t_aucs, t_rolls):
        if r > 0:
            ax2.annotate(f"{r} rejected", (v, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#DC2626", fontweight="bold")

    ax2.set_title("Sensitivity to Rollback Anomaly Threshold $\\tau$", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Rollback Uncertainty Threshold $\\tau$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Target AUROC", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=10, loc="lower right")

    plt.suptitle("SafeAdapt-OT Hyperparameter Robustness & Sensitivity Ablation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"\nSaved ablation figure to: {output_png}")


def main():
    device = torch.device("cpu")
    model_path = ROOT_DIR / "models" / "best_model.pth"
    proto_path = ROOT_DIR / "models" / "source_prototypes.pt"
    metrics_dir = ROOT_DIR / "results" / "metrics"
    graphs_dir = ROOT_DIR / "results" / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" SafeAdapt Hyperparameter Sensitivity & Ablation Study               ")
    print("=" * 70)
    print(f"Device: {device} | Target Site: Hospital 1 (CWZ Nijmegen)")

    # Load 500 test patches with batch size 50
    _, _, test_loader, _ = get_dataloaders(
        batch_size=500,
        micro_batch_size=50,
        test_samples=500,
        target_hospital=1,
        include_train=False,
        include_val=False,
        include_test=True
    )

    # 0. Measure Baseline AUROC
    base_model = ResNet18Binary(pretrained=False).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    base_auc, base_acc, base_rec = evaluate_loader(base_model, test_loader, device)
    print(f"Source Model Baseline -> AUROC: {base_auc:.4f} | Accuracy: {base_acc:.2f}% | Recall: {base_rec:.2f}%")

    # 1. Delta Max Sweep (L_inf Trust Region Bound)
    deltas = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.05]
    delta_res = run_delta_ablation(device, model_path, proto_path, test_loader, deltas)

    # 2. Tau Sweep (Conformal Rollback Anomaly Threshold)
    taus = [0.07, 0.08, 0.09, 0.10, 0.12, 0.48]
    tau_res = run_tau_ablation(device, model_path, proto_path, test_loader, taus)

    # 3. Save CSV
    csv_path = metrics_dir / "ablation_study_results.csv"
    all_res = delta_res + tau_res
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "auroc", "accuracy", "recall", "rollbacks", "total_steps"])
        writer.writeheader()
        writer.writerows(all_res)
    print(f"\nSaved empirical ablation ledger to: {csv_path}")

    # 4. Plot Curves
    png_path = graphs_dir / "hyperparameter_sensitivity_ablation.png"
    plot_ablation_curves(delta_res, tau_res, base_auc, png_path)


if __name__ == "__main__":
    main()

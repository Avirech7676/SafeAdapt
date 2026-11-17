"""Full 5-Hospital Evaluation Suite for Camelyon17.

Evaluates Source Baseline vs SafeAdapt-OT across all 5 clinical centers:
- Hospital 0: Radboud University Medical Center (RadboudUMC, Nijmegen) [In-Distribution]
- Hospital 1: Canisius-Wilhelmina Hospital (CWZ Nijmegen) [Target OOD 1]
- Hospital 2: University Medical Center Utrecht (UMC Utrecht) [Target OOD 2]
- Hospital 3: Rijnstate Hospital (Arnhem) [Target OOD 3]
- Hospital 4: Laboratorium Pathologie Oost-Nederland (LPON, Enschede) [Target OOD 4]

Outputs:
- results/metrics/multi_hospital_5centers_results.csv
- results/graphs/multi_hospital_5centers_progression.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

from data_loader import load_hospital_data
from model import ResNet18Binary


def evaluate_model(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset.images.to(device))
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    y_true = dataset.labels.numpy()
    y_pred = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) * 100.0

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    spec = float(tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0

    return {"AUROC": auc, "Accuracy": acc, "F1_Score": f1, "Recall": rec, "Precision": prec, "Specificity": spec}


def run_5_hospital_evaluation():
    device = torch.device("cpu")
    root_dir = Path(__file__).resolve().parent
    data_dir = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data"
    model_path = root_dir / "models" / "best_model.pth"
    proto_path = root_dir / "models" / "source_prototypes.pt"
    metrics_dir = root_dir / "results" / "metrics"
    graphs_dir = root_dir / "results" / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    hospital_names = {
        0: ("Hospital 0 (RadboudUMC)", "In-Distribution (Source)", "test"),
        1: ("Hospital 1 (CWZ Nijmegen)", "Target OOD Domain 1", "test"),
        2: ("Hospital 2 (UMC Utrecht)", "Target OOD Domain 2", "test"),
        3: ("Hospital 3 (Rijnstate)", "Target OOD Domain 3", "test"),
        4: ("Hospital 4 (LPON Enschede)", "Target OOD Domain 4", "test"),
    }

    print("=" * 80)
    print(" Complete 5-Hospital Clinical Evaluation Suite (Camelyon17-WILDS) ")
    print("=" * 80)

    results = []

    for h_id, (h_name, domain_type, split) in hospital_names.items():
        print(f"\nEvaluating on {h_name} [{domain_type}]...")
        # Load 500 balanced patches
        ds = load_hospital_data(data_dir, hospital=h_id, split=split, num_samples=500, balanced=True)

        # 1. Source Baseline Model
        base_model = ResNet18Binary(pretrained=False).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        m_base = evaluate_model(base_model, ds, device)

        # 2. SafeAdapt-OT Adaptation
        safe_model = ResNet18Binary(pretrained=False).to(device)
        safe_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        safe_model.prototypes = torch.load(proto_path, map_location=device)
        trainable = safe_model.configure_tta()
        opt = torch.optim.Adam(trainable, lr=5e-4)

        batches = torch.split(ds.images, 50)
        for b in batches:
            safe_model.adapt_step_ot(b, opt, delta_max=0.05, rollback_threshold=0.48)

        m_safe = evaluate_model(safe_model, ds, device)

        delta_auc = m_safe["AUROC"] - m_base["AUROC"]
        delta_acc = m_safe["Accuracy"] - m_base["Accuracy"]
        delta_rec = m_safe["Recall"] - m_base["Recall"]

        print(f"  Source Baseline -> AUROC: {m_base['AUROC']:.4f} | Acc: {m_base['Accuracy']:.1f}% | Rec: {m_base['Recall']:.1f}%")
        print(f"  SafeAdapt-OT    -> AUROC: {m_safe['AUROC']:.4f} | Acc: {m_safe['Accuracy']:.1f}% | Rec: {m_safe['Recall']:.1f}% (ΔAUC = {delta_auc:+.4f})")

        results.append({
            "Hospital_ID": h_id,
            "Hospital_Name": h_name,
            "Domain_Type": domain_type,
            "Base_AUROC": m_base["AUROC"],
            "Safe_AUROC": m_safe["AUROC"],
            "Delta_AUROC": delta_auc,
            "Base_Accuracy": m_base["Accuracy"],
            "Safe_Accuracy": m_safe["Accuracy"],
            "Delta_Accuracy": delta_acc,
            "Base_Recall": m_base["Recall"],
            "Safe_Recall": m_safe["Recall"],
            "Delta_Recall": delta_rec,
            "Base_F1": m_base["F1_Score"],
            "Safe_F1": m_safe["F1_Score"],
            "Base_Specificity": m_base["Specificity"],
            "Safe_Specificity": m_safe["Specificity"],
        })

    df = pd.DataFrame(results)
    csv_path = metrics_dir / "multi_hospital_5centers_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 85)
    print("                         SUMMARY TABLE: ALL 5 HOSPITALS                         ")
    print("=" * 85)
    print(df[["Hospital_Name", "Domain_Type", "Base_AUROC", "Safe_AUROC", "Delta_AUROC", "Base_Accuracy", "Safe_Accuracy"]].to_string(index=False))
    print("=" * 85)
    print(f"Saved 5-hospital CSV ledger to: {csv_path}")

    # Plot Multi-Hospital Progression Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    x = np.arange(len(df))
    width = 0.35

    # Panel A: AUROC Comparison
    ax1.bar(x - width/2, df["Base_AUROC"], width, label="Source Baseline (Frozen)", color="#64748B", edgecolor="black", alpha=0.9)
    ax1.bar(x + width/2, df["Safe_AUROC"], width, label="SafeAdapt-OT (Ours)", color="#10B981", edgecolor="black", alpha=0.9)
    ax1.set_ylabel("Macro AUROC", fontsize=11, fontweight="bold")
    ax1.set_title("A. Multi-Hospital AUROC Generalization", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Hospital {h}\n{df['Hospital_Name'].iloc[h].split('(')[1][:-1]}" for h in range(len(df))], fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0.50, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="lower right", frameon=True, fontsize=10)

    for i in range(len(df)):
        b_val = df["Base_AUROC"].iloc[i]
        s_val = df["Safe_AUROC"].iloc[i]
        ax1.annotate(f"{b_val:.3f}", (i - width/2, b_val), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")
        ax1.annotate(f"{s_val:.3f}", (i + width/2, s_val), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold", color="#065F46")

    # Panel B: Sensitivity (Recall) Comparison
    ax2.bar(x - width/2, df["Base_Recall"], width, label="Source Baseline (Frozen)", color="#64748B", edgecolor="black", alpha=0.9)
    ax2.bar(x + width/2, df["Safe_Recall"], width, label="SafeAdapt-OT (Ours)", color="#2563EB", edgecolor="black", alpha=0.9)
    ax2.set_ylabel("Clinical Sensitivity / Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Tumor Detection Sensitivity Gain", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Hospital {h}\n{df['Hospital_Name'].iloc[h].split('(')[1][:-1]}" for h in range(len(df))], fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    for i in range(len(df)):
        b_rec = df["Base_Recall"].iloc[i]
        s_rec = df["Safe_Recall"].iloc[i]
        ax2.annotate(f"{b_rec:.1f}%", (i - width/2, b_rec), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")
        ax2.annotate(f"{s_rec:.1f}%", (i + width/2, s_rec), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold", color="#1E40AF")

    plt.suptitle("Complete 5-Hospital Clinical Evaluation (Camelyon17-WILDS)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = graphs_dir / "multi_hospital_5centers_progression.png"
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved 5-hospital visualization to: {png_path}")


if __name__ == "__main__":
    run_5_hospital_evaluation()

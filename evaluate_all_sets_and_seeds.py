"""Comprehensive Multi-Hospital Evaluation Suite for ResNet-18.

Clinical Domain Layout:
1. Hospital 0 (Base / Center 0 - RadboudUMC):
   - Training Set (N=2,000)
   - Validation Set (N=500)
   - In-Distribution Testing Set (N=500 from train-00007)
2. Hospital 1 (Target 1 / Center 1 - CWZ Nijmegen):
   - Target Test Set (N=1,000) across all 5 seeds: 42, 123, 2024, 7, 99
   - Baseline (Source Frozen) vs SafeAdapt-OT (Ours)
3. Hospital 2 (Target 2 / Center 2 - UMC Utrecht):
   - Target Test Set (N=1,000) across all 5 seeds: 42, 123, 2024, 7, 99
   - Baseline (Source Frozen) vs SafeAdapt-OT (Ours)

Outputs:
- resnet18_source_train_val_metrics.csv
- resnet18_target_h1_5seeds_metrics.csv
- resnet18_target_h2_5seeds_metrics.csv
- resnet18_target_test_5seeds_metrics.csv
- train_val_vs_target_test_comparison.png
"""

import csv
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from data_loader import load_hospital_data
from model import ResNet18Binary

SEEDS = [42, 123, 2024, 7, 99]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
PROTO_PATH = BASE_DIR / "models" / "source_prototypes.pt"
OUTPUT_DIR = BASE_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred, y_prob, y_source_pred=None):
    """Computes full clinical diagnostic metrics suite."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    harm_rate = 0.0
    if y_source_pred is not None:
        correct_source = (y_source_pred == y_true)
        harm = correct_source & (y_pred != y_true)
        harm_rate = float(harm.sum() / max(len(y_true), 1))

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec,
        "f1_score": f1, "roc_auc": auc, "harm_rate": harm_rate,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def evaluate_dataset(model, dataloader):
    """Evaluates model on dataloader without adaptation."""
    model.eval()
    all_targets, all_probs, all_preds = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels.to(DEVICE))
            total_loss += loss.item() * len(labels)

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_targets.extend(labels.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    y_t = np.array(all_targets)
    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    avg_loss = total_loss / max(len(y_t), 1)

    metrics = compute_metrics(y_t, y_pred, y_p)
    metrics["loss"] = float(avg_loss)
    return metrics, y_pred, y_p


def run_safeadapt_evaluation(model, dataloader, y_true, y_source_pred, lr=1.5e-4, max_adapt_steps=4):
    """Runs SafeAdapt-OT and records post-adaptation metrics."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    all_probs, all_preds = [], []
    for step, (x, _) in enumerate(dataloader, 1):
        x = x.to(DEVICE)
        if step <= max_adapt_steps:
            model.adapt_step_ot(x, optimizer, lambda_ot=2.0, lambda_div=0.5, lambda_anchor=0.02)
            post_logits = getattr(model, "last_post_logits", None)
            if post_logits is None:
                with torch.no_grad():
                    post_logits = model(x)
        else:
            with torch.no_grad():
                post_logits = model(x)

        prob = F.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
        pred = post_logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(prob)
        all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred, y_p, y_source_pred)
    return metrics, y_pred, y_p


def evaluate_target_domain(dataset, hospital_name, model_path, proto_path):
    """Evaluates baseline and SafeAdapt-OT across all 5 seeds on a target domain."""
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        perm = torch.randperm(len(dataset))
        shuffled_images = dataset.images[perm]
        shuffled_labels = dataset.labels[perm]

        sub_dataset = torch.utils.data.TensorDataset(shuffled_images, shuffled_labels)
        loader = torch.utils.data.DataLoader(sub_dataset, batch_size=100, shuffle=False)
        y_true = shuffled_labels.numpy()

        # Baseline
        m_base = ResNet18Binary(pretrained=False).to(DEVICE)
        m_base.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        base_metrics, y_pred_source, _ = evaluate_dataset(m_base, loader)

        # SafeAdapt-OT
        m_safe = ResNet18Binary(pretrained=False).to(DEVICE)
        m_safe.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        if Path(proto_path).exists():
            m_safe.prototypes = torch.load(proto_path, map_location=DEVICE)
        safe_metrics, _, _ = run_safeadapt_evaluation(m_safe, loader, y_true, y_pred_source)

        print(f"  [{hospital_name} | Seed {seed:<4}] Baseline Acc: {base_metrics['accuracy']*100:.1f}%, F1: {base_metrics['f1_score']:.4f}, AUC: {base_metrics['roc_auc']:.4f} | "
              f"SafeAdapt Acc: {safe_metrics['accuracy']*100:.1f}%, F1: {safe_metrics['f1_score']:.4f}, AUC: {safe_metrics['roc_auc']:.4f}", flush=True)

        results.append({
            "seed": seed,
            "base": base_metrics,
            "safe": safe_metrics
        })

        del m_base, m_safe
        gc.collect()

    return results


def main():
    print("=" * 70)
    print("  ResNet-18 Evaluation: Base Site (H0) & Target Sites (H1 & H2)      ")
    print("=" * 70)
    print(f"Device: {DEVICE} | Checkpoint: {MODEL_PATH}")

    metrics_dir = OUTPUT_DIR / "metrics"
    graphs_dir = OUTPUT_DIR / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    model = ResNet18Binary(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    if Path(PROTO_PATH).exists():
        model.prototypes = torch.load(PROTO_PATH, map_location=DEVICE)
        print(f"Loaded source class prototypes from {PROTO_PATH}")

    # ========================================================
    # Part 1: Hospital 0 (RadboudUMC - Base Site) Evaluation
    # ========================================================
    p1_csv = metrics_dir / "resnet18_source_train_val_metrics.csv"
    if p1_csv.exists():
        print(f"\n--- [Part 1] Loading Hospital 0 metrics from {p1_csv} ---", flush=True)
        with open(p1_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        train_metrics = {k.lower(): float(v) for k, v in rows[0].items() if k not in ("Dataset_Split", "Hospital")}
        val_metrics = {k.lower(): float(v) for k, v in rows[1].items() if k not in ("Dataset_Split", "Hospital")}
        h0_test_metrics = {k.lower(): float(v) for k, v in rows[2].items() if k not in ("Dataset_Split", "Hospital")}
        print("\nHospital 0 In-Distribution Results:", flush=True)
        print(f"  Training Set:   Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1_score']:.4f} | AUC: {train_metrics['roc_auc']:.4f} | Loss: {train_metrics['loss']:.4f}", flush=True)
        print(f"  Validation Set: Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1_score']:.4f} | AUC: {val_metrics['roc_auc']:.4f} | Loss: {val_metrics['loss']:.4f}", flush=True)
        print(f"  Testing Set:    Acc: {h0_test_metrics['accuracy']*100:.2f}% | F1: {h0_test_metrics['f1_score']:.4f} | AUC: {h0_test_metrics['roc_auc']:.4f} | Loss: {h0_test_metrics['loss']:.4f}", flush=True)
    else:
        print("\n--- [Part 1] Evaluating on Hospital 0 (Center 0 - RadboudUMC) ---", flush=True)
        train_dataset = load_hospital_data(DATA_DIR, hospital=0, split="train", num_samples=2000, balanced=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=False)

        val_dataset = load_hospital_data(DATA_DIR, hospital=0, split="val", num_samples=500, balanced=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)

        h0_test_dataset = load_hospital_data(DATA_DIR, hospital=0, split="test", num_samples=500, balanced=True)
        h0_test_loader = torch.utils.data.DataLoader(h0_test_dataset, batch_size=50, shuffle=False)

        train_metrics, _, _ = evaluate_dataset(model, train_loader)
        val_metrics, _, _ = evaluate_dataset(model, val_loader)
        h0_test_metrics, _, _ = evaluate_dataset(model, h0_test_loader)

        print("\nHospital 0 In-Distribution Results:", flush=True)
        print(f"  Training Set:   Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1_score']:.4f} | AUC: {train_metrics['roc_auc']:.4f} | Loss: {train_metrics['loss']:.4f}", flush=True)
        print(f"  Validation Set: Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1_score']:.4f} | AUC: {val_metrics['roc_auc']:.4f} | Loss: {val_metrics['loss']:.4f}", flush=True)
        print(f"  Testing Set:    Acc: {h0_test_metrics['accuracy']*100:.2f}% | F1: {h0_test_metrics['f1_score']:.4f} | AUC: {h0_test_metrics['roc_auc']:.4f} | Loss: {h0_test_metrics['loss']:.4f}", flush=True)

        with open(p1_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Dataset_Split", "Hospital", "Samples", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC", "Loss", "TN", "FP", "FN", "TP"])
            w.writerow(["Training Set", "Hospital 0 (RadboudUMC)", len(train_dataset), train_metrics["accuracy"], train_metrics["precision"], train_metrics["recall"], train_metrics["specificity"], train_metrics["f1_score"], train_metrics["roc_auc"], train_metrics["loss"], train_metrics["tn"], train_metrics["fp"], train_metrics["fn"], train_metrics["tp"]])
            w.writerow(["Validation Set", "Hospital 0 (RadboudUMC)", len(val_dataset), val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["specificity"], val_metrics["f1_score"], val_metrics["roc_auc"], val_metrics["loss"], val_metrics["tn"], val_metrics["fp"], val_metrics["fn"], val_metrics["tp"]])
            w.writerow(["Testing Set (In-Dist)", "Hospital 0 (RadboudUMC)", len(h0_test_dataset), h0_test_metrics["accuracy"], h0_test_metrics["precision"], h0_test_metrics["recall"], h0_test_metrics["specificity"], h0_test_metrics["f1_score"], h0_test_metrics["roc_auc"], h0_test_metrics["loss"], h0_test_metrics["tn"], h0_test_metrics["fp"], h0_test_metrics["fn"], h0_test_metrics["tp"]])
        print(f"Saved Hospital 0 metrics CSV to: {p1_csv}", flush=True)

    # ========================================================
    # Part 2: Target Hospital 1 (CWZ Nijmegen) Across 5 Seeds
    # ========================================================
    print("\n--- [Part 2] Evaluating on Target Hospital 1 (Center 1 - CWZ Nijmegen) across 5 seeds ---")
    h1_dataset = load_hospital_data(DATA_DIR, hospital=1, split="test", num_samples=1000, balanced=True)
    h1_results = evaluate_target_domain(h1_dataset, "Hospital 1 (CWZ)", MODEL_PATH, PROTO_PATH)

    # Save H1 CSV
    h1_csv = metrics_dir / "resnet18_target_h1_5seeds_metrics.csv"
    with open(h1_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Seed", "Hospital", "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC", "Harm_Rate", "TN", "FP", "FN", "TP"])
        for r in h1_results:
            s, b, sf = r["seed"], r["base"], r["safe"]
            w.writerow([s, "Hospital 1 (CWZ)", "Source Baseline (Frozen)", f"{b['accuracy']:.4f}", f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['specificity']:.4f}", f"{b['f1_score']:.4f}", f"{b['roc_auc']:.4f}", "0.0000", b["tn"], b["fp"], b["fn"], b["tp"]])
            w.writerow([s, "Hospital 1 (CWZ)", "SafeAdapt-OT (Ours)", f"{sf['accuracy']:.4f}", f"{sf['precision']:.4f}", f"{sf['recall']:.4f}", f"{sf['specificity']:.4f}", f"{sf['f1_score']:.4f}", f"{sf['roc_auc']:.4f}", f"{sf['harm_rate']:.4f}", sf["tn"], sf["fp"], sf["fn"], sf["tp"]])

    # ========================================================
    # Part 3: Target Hospital 2 (UMC Utrecht) Across 5 Seeds
    # ========================================================
    print("\n--- [Part 3] Evaluating on Target Hospital 2 (Center 2 - UMC Utrecht) across 5 seeds ---")
    h2_dataset = load_hospital_data(DATA_DIR, hospital=2, split="test", num_samples=1000, balanced=True)
    h2_results = evaluate_target_domain(h2_dataset, "Hospital 2 (UMCU)", MODEL_PATH, PROTO_PATH)

    # Save H2 CSV
    h2_csv = metrics_dir / "resnet18_target_h2_5seeds_metrics.csv"
    with open(h2_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Seed", "Hospital", "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC", "Harm_Rate", "TN", "FP", "FN", "TP"])
        for r in h2_results:
            s, b, sf = r["seed"], r["base"], r["safe"]
            w.writerow([s, "Hospital 2 (UMCU)", "Source Baseline (Frozen)", f"{b['accuracy']:.4f}", f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['specificity']:.4f}", f"{b['f1_score']:.4f}", f"{b['roc_auc']:.4f}", "0.0000", b["tn"], b["fp"], b["fn"], b["tp"]])
            w.writerow([s, "Hospital 2 (UMCU)", "SafeAdapt-OT (Ours)", f"{sf['accuracy']:.4f}", f"{sf['precision']:.4f}", f"{sf['recall']:.4f}", f"{sf['specificity']:.4f}", f"{sf['f1_score']:.4f}", f"{sf['roc_auc']:.4f}", f"{sf['harm_rate']:.4f}", sf["tn"], sf["fp"], sf["fn"], sf["tp"]])

    # Save Combined Target Test Metrics CSV
    comb_csv = metrics_dir / "resnet18_target_test_5seeds_metrics.csv"
    with open(comb_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Target_Domain", "Seed", "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC", "Harm_Rate", "TN", "FP", "FN", "TP"])
        for r in h1_results:
            s, b, sf = r["seed"], r["base"], r["safe"]
            w.writerow(["Hospital 1 (CWZ)", s, "Source Baseline", f"{b['accuracy']:.4f}", f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['specificity']:.4f}", f"{b['f1_score']:.4f}", f"{b['roc_auc']:.4f}", "0.0000", b["tn"], b["fp"], b["fn"], b["tp"]])
            w.writerow(["Hospital 1 (CWZ)", s, "SafeAdapt-OT", f"{sf['accuracy']:.4f}", f"{sf['precision']:.4f}", f"{sf['recall']:.4f}", f"{sf['specificity']:.4f}", f"{sf['f1_score']:.4f}", f"{sf['roc_auc']:.4f}", f"{sf['harm_rate']:.4f}", sf["tn"], sf["fp"], sf["fn"], sf["tp"]])
        for r in h2_results:
            s, b, sf = r["seed"], r["base"], r["safe"]
            w.writerow(["Hospital 2 (UMCU)", s, "Source Baseline", f"{b['accuracy']:.4f}", f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['specificity']:.4f}", f"{b['f1_score']:.4f}", f"{b['roc_auc']:.4f}", "0.0000", b["tn"], b["fp"], b["fn"], b["tp"]])
            w.writerow(["Hospital 2 (UMCU)", s, "SafeAdapt-OT", f"{sf['accuracy']:.4f}", f"{sf['precision']:.4f}", f"{sf['recall']:.4f}", f"{sf['specificity']:.4f}", f"{sf['f1_score']:.4f}", f"{sf['roc_auc']:.4f}", f"{sf['harm_rate']:.4f}", sf["tn"], sf["fp"], sf["fn"], sf["tp"]])

    print(f"Saved combined target test 5-seed metrics to: {comb_csv}")

    # ========================================================
    # Part 4: Multi-Domain Performance Comparison Plot
    # ========================================================
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    stages = [
        "Source Train\n(Hospital 0)", "Source Val\n(Hospital 0)", "Source Test\n(In-Domain)",
        "Target 1 Base\n(Hospital 1)", "Target 1 SafeAdapt\n(Hospital 1)",
        "Target 2 Base\n(Hospital 2)", "Target 2 SafeAdapt\n(Hospital 2)"
    ]

    h1_base_acc = np.mean([r["base"]["accuracy"] for r in h1_results]) * 100
    h1_base_f1 = np.mean([r["base"]["f1_score"] for r in h1_results]) * 100
    h1_base_auc = np.mean([r["base"]["roc_auc"] for r in h1_results]) * 100

    h1_safe_acc = np.mean([r["safe"]["accuracy"] for r in h1_results]) * 100
    h1_safe_f1 = np.mean([r["safe"]["f1_score"] for r in h1_results]) * 100
    h1_safe_auc = np.mean([r["safe"]["roc_auc"] for r in h1_results]) * 100

    h2_base_acc = np.mean([r["base"]["accuracy"] for r in h2_results]) * 100
    h2_base_f1 = np.mean([r["base"]["f1_score"] for r in h2_results]) * 100
    h2_base_auc = np.mean([r["base"]["roc_auc"] for r in h2_results]) * 100

    h2_safe_acc = np.mean([r["safe"]["accuracy"] for r in h2_results]) * 100
    h2_safe_f1 = np.mean([r["safe"]["f1_score"] for r in h2_results]) * 100
    h2_safe_auc = np.mean([r["safe"]["roc_auc"] for r in h2_results]) * 100

    acc_vals = [train_metrics["accuracy"] * 100, val_metrics["accuracy"] * 100, h0_test_metrics["accuracy"] * 100,
                h1_base_acc, h1_safe_acc, h2_base_acc, h2_safe_acc]
    f1_vals = [train_metrics["f1_score"] * 100, val_metrics["f1_score"] * 100, h0_test_metrics["f1_score"] * 100,
               h1_base_f1, h1_safe_f1, h2_base_f1, h2_safe_f1]
    auc_vals = [train_metrics["roc_auc"] * 100, val_metrics["roc_auc"] * 100, h0_test_metrics["roc_auc"] * 100,
                h1_base_auc, h1_safe_auc, h2_base_auc, h2_safe_auc]

    x = np.arange(len(stages))
    width = 0.25

    r1 = ax.bar(x - width, acc_vals, width, label="Accuracy (%)", color="#3182CE", edgecolor="black")
    r2 = ax.bar(x, f1_vals, width, label="F1-Score (%)", color="#38A169", edgecolor="black")
    r3 = ax.bar(x + width, auc_vals, width, label="ROC-AUC (%)", color="#DD6B20", edgecolor="black")

    ax.set_ylabel("Performance (%)", fontsize=12, fontweight="bold")
    ax.set_title("ResNet-18 Evaluation: Source Domain (Hospital 0) vs Target Test-Time Adaptation\n"
                 "(Source Hospital 0 Baseline vs Target Hospital 1 & Target Hospital 2 Adaptation)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    for r in [r1, r2, r3]:
        for rect in r:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 1),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    g_path = graphs_dir / "train_val_vs_target_test_comparison.png"
    plt.savefig(g_path)
    plt.close()
    print(f"Generated Comparative Progression Graph: {g_path}")
    print("\nFull evaluation suite finished successfully.")


if __name__ == "__main__":
    main()

"""Live Benchmark Execution Engine for SafeAdapt.

Runs fresh empirical benchmark experiments directly on Hospital 1 (CWZ Nijmegen) test patches
using our newly trained 20-epoch ResNet-18 model across 5 clinical random seeds (42, 123, 2024, 7, 99):
1. Source (Frozen ResNet-18)
2. Tent (Ungated Test-Time Entropy Minimization)
3. EATA (Entropy-filtered Adaptation)
4. Random-gated (70% random sample updates)
5. Confidence-gated (Confidence-thresholded updates)
6. SafeAdapt-OT (Ours: Optimal Transport Prototype Alignment + Diversity + L_inf Trust Region)

All metrics are computed live from scratch on real histopathology patches.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

from data_loader import load_hospital_data
from model import ResNet18Binary

SEEDS = [42, 123, 2024, 7, 99]
DATA_DIR = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(y_true, y_prob, y_pred, y_source_pred=None):
    """Computes AUROC, F1, Accuracy, and Harm Rate."""
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    harm_rate = 0.0
    if y_source_pred is not None:
        correct_source = (y_source_pred == y_true)
        harm = correct_source & (y_pred != y_true)
        harm_rate = float(harm.sum() / max(len(y_true), 1))

    return auc, f1, acc, harm_rate


def evaluate_source(model, dataloader):
    """Evaluates frozen source model."""
    model.eval()
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            all_targets.extend(y.numpy())
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_t = np.array(all_targets)
    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, _ = calculate_metrics(y_t, y_p, y_pred)
    return auc, f1, acc, y_pred, y_p


def run_tent(model, dataloader, y_true, y_source_pred, lr=1e-4):
    """Runs unconstrained Tent (entropy minimization) on all batches."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    all_probs, all_preds = [], []
    for x, _ in dataloader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss = -(probs * torch.log(probs + 1e-7)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            post_logits = model(x)
            prob = torch.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
            pred = post_logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, harm = calculate_metrics(y_true, y_p, y_pred, y_source_pred)
    return auc, f1, acc, harm, 1.0, y_p


def run_eata(model, dataloader, y_true, y_source_pred, lr=1e-4, e_margin=0.277):
    """Runs EATA (Entropy-filtered adaptation)."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    adapted_count = 0
    total_count = 0
    all_probs, all_preds = [], []

    for x, _ in dataloader:
        x = x.to(DEVICE)
        total_count += len(x)

        with torch.no_grad():
            pre_logits = model(x)
            probs = torch.softmax(pre_logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-7)).sum(dim=1)
            filter_mask = (entropy < e_margin)

        if filter_mask.sum() > 0:
            adapted_count += int(filter_mask.sum().item())
            optimizer.zero_grad()
            logits = model(x[filter_mask])
            p = torch.softmax(logits, dim=1)
            loss = -(p * torch.log(p + 1e-7)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            post_logits = model(x)
            prob = torch.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
            pred = post_logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, harm = calculate_metrics(y_true, y_p, y_pred, y_source_pred)
    cov = adapted_count / max(total_count, 1)
    return auc, f1, acc, harm, cov, y_p


def run_random_gated(model, dataloader, y_true, y_source_pred, lr=1e-4, gate_prob=0.70):
    """Runs Random-gated adaptation (updates on random 70% subset)."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    adapted_count = 0
    total_count = 0
    all_probs, all_preds = [], []

    for x, _ in dataloader:
        x = x.to(DEVICE)
        total_count += len(x)
        mask = torch.rand(len(x)) < gate_prob

        if mask.sum() > 0:
            adapted_count += int(mask.sum().item())
            optimizer.zero_grad()
            logits = model(x[mask])
            p = torch.softmax(logits, dim=1)
            loss = -(p * torch.log(p + 1e-7)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            post_logits = model(x)
            prob = torch.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
            pred = post_logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, harm = calculate_metrics(y_true, y_p, y_pred, y_source_pred)
    cov = adapted_count / max(total_count, 1)
    return auc, f1, acc, harm, cov, y_p


def run_confidence_gated(model, dataloader, y_true, y_source_pred, lr=1e-4, conf_thresh=0.70):
    """Runs Confidence-gated adaptation (updates on high confidence samples)."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    adapted_count = 0
    total_count = 0
    all_probs, all_preds = [], []

    for x, _ in dataloader:
        x = x.to(DEVICE)
        total_count += len(x)

        with torch.no_grad():
            pre_logits = model(x)
            probs = torch.softmax(pre_logits, dim=1)
            conf = probs.max(dim=1).values
            mask = (conf >= conf_thresh)

        if mask.sum() > 0:
            adapted_count += int(mask.sum().item())
            optimizer.zero_grad()
            logits = model(x[mask])
            p = torch.softmax(logits, dim=1)
            loss = -(p * torch.log(p + 1e-7)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            post_logits = model(x)
            prob = torch.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
            pred = post_logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, harm = calculate_metrics(y_true, y_p, y_pred, y_source_pred)
    cov = adapted_count / max(total_count, 1)
    return auc, f1, acc, harm, cov, y_p


def run_safeadapt_ot(model, dataloader, y_true, y_source_pred, lr=1.5e-4):
    """Runs SafeAdapt-OT with Prototype Optimal Transport and Conformal Risk Control."""
    params = model.configure_tta()
    optimizer = torch.optim.Adam(params, lr=lr)

    accepted_batches = 0
    total_batches = 0
    all_probs, all_preds = [], []

    for x, _ in dataloader:
        x = x.to(DEVICE)
        total_batches += 1

        loss, rolled_back = model.adapt_step_ot(x, optimizer, lambda_ot=2.0, lambda_div=0.5, lambda_anchor=0.02)
        if not rolled_back:
            accepted_batches += 1

        with torch.no_grad():
            post_logits = model(x)
            prob = torch.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
            pred = post_logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob)
            all_preds.extend(pred)

    y_p = np.array(all_probs)
    y_pred = np.array(all_preds)
    auc, f1, acc, harm = calculate_metrics(y_true, y_p, y_pred, y_source_pred)
    cov = accepted_batches / max(total_batches, 1)
    return auc, f1, acc, harm, cov, y_p


def run_benchmark_experiments():
    print("=" * 70)
    print("        SafeAdapt-OT Benchmark Evaluation on Target Domain            ")
    print("=" * 70)
    print(f"Device: {DEVICE} | Checkpoint: {MODEL_PATH}")
    print(f"Target Center: CWZ Nijmegen (Hospital 1) | Seeds: {SEEDS}")

    # 1. Load or Compute Source Class Prototypes from Hospital 0 (RadboudUMC)
    proto_path = BASE_DIR / "models" / "source_prototypes.pt"
    if proto_path.exists():
        print(f"\n--- Loading Source Prototypes from {proto_path} ---")
        source_prototypes = torch.load(proto_path, map_location=DEVICE)
    else:
        print("\n--- Computing Source Prototypes from Hospital 0 (RadboudUMC) ---")
        val_dataset = load_hospital_data(DATA_DIR, hospital=0, split="val", num_samples=500, balanced=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)

        m_proto = ResNet18Binary(pretrained=False).to(DEVICE)
        m_proto.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        source_prototypes = m_proto.compute_source_prototypes(val_loader, DEVICE)
        torch.save(source_prototypes, proto_path)
    sim = torch.dot(source_prototypes[0], source_prototypes[1]).item()
    print(f"Source prototypes computed and saved to: {proto_path}")
    print(f"Cosine separation between Normal and Tumor prototypes: {sim:.4f}")

    # 2. Load Target Hospital Test Data
    print("\nLoading test patches from CWZ Nijmegen (Center 1) parquet data...")
    test_dataset = load_hospital_data(DATA_DIR, hospital="cwz", split="test", num_samples=1000, balanced=True)
    print(f"Loaded {len(test_dataset)} test patches (Balanced Normal & Tumor).")

    methods = ["Source (Frozen)", "Tent (Ungated)", "EATA", "Random-gated", "Confidence-gated", "SafeAdapt-OT (Ours)"]
    per_seed_auroc = {m: {} for m in methods}
    per_seed_f1 = {m: {} for m in methods}
    per_seed_acc = {m: {} for m in methods}
    per_seed_harm = {m: {} for m in methods}
    per_seed_cov = {m: {} for m in methods}
    sample_probs = {}

    for seed in SEEDS:
        print(f"\n>>> Running Experiments for Clinical Seed {seed} <<<")
        torch.manual_seed(seed)
        np.random.seed(seed)

        perm = torch.randperm(len(test_dataset))
        shuffled_images = test_dataset.images[perm]
        shuffled_labels = test_dataset.labels[perm]

        sub_dataset = torch.utils.data.TensorDataset(shuffled_images, shuffled_labels)
        loader = torch.utils.data.DataLoader(sub_dataset, batch_size=50, shuffle=False)
        y_true = shuffled_labels.numpy()

        # 1. Source (Frozen)
        m_source = ResNet18Binary(pretrained=False).to(DEVICE)
        m_source.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        auc_s, f1_s, acc_s, y_pred_source, prob_s = evaluate_source(m_source, loader)
        per_seed_auroc["Source (Frozen)"][seed] = auc_s
        per_seed_f1["Source (Frozen)"][seed] = f1_s
        per_seed_acc["Source (Frozen)"][seed] = acc_s
        per_seed_harm["Source (Frozen)"][seed] = 0.0
        per_seed_cov["Source (Frozen)"][seed] = 0.0
        print(f"  [Source]     AUROC: {auc_s:.4f} | F1: {f1_s:.4f} | Acc: {acc_s*100:.1f}%")

        # 2. Tent
        m_tent = ResNet18Binary(pretrained=False).to(DEVICE)
        m_tent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        auc_t, f1_t, acc_t, harm_t, cov_t, prob_t = run_tent(m_tent, loader, y_true, y_pred_source)
        per_seed_auroc["Tent (Ungated)"][seed] = auc_t
        per_seed_f1["Tent (Ungated)"][seed] = f1_t
        per_seed_acc["Tent (Ungated)"][seed] = acc_t
        per_seed_harm["Tent (Ungated)"][seed] = harm_t
        per_seed_cov["Tent (Ungated)"][seed] = cov_t
        print(f"  [Tent]       AUROC: {auc_t:.4f} | F1: {f1_t:.4f} | Acc: {acc_t*100:.1f}% | Harm: {harm_t*100:.2f}% | Cov: {cov_t*100:.1f}%")

        # 3. EATA
        m_eata = ResNet18Binary(pretrained=False).to(DEVICE)
        m_eata.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        auc_e, f1_e, acc_e, harm_e, cov_e, prob_e = run_eata(m_eata, loader, y_true, y_pred_source)
        per_seed_auroc["EATA"][seed] = auc_e
        per_seed_f1["EATA"][seed] = f1_e
        per_seed_acc["EATA"][seed] = acc_e
        per_seed_harm["EATA"][seed] = harm_e
        per_seed_cov["EATA"][seed] = cov_e
        print(f"  [EATA]       AUROC: {auc_e:.4f} | F1: {f1_e:.4f} | Acc: {acc_e*100:.1f}% | Harm: {harm_e*100:.2f}% | Cov: {cov_e*100:.1f}%")

        # 4. Random-gated
        m_rand = ResNet18Binary(pretrained=False).to(DEVICE)
        m_rand.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        auc_r, f1_r, acc_r, harm_r, cov_r, prob_r = run_random_gated(m_rand, loader, y_true, y_pred_source)
        per_seed_auroc["Random-gated"][seed] = auc_r
        per_seed_f1["Random-gated"][seed] = f1_r
        per_seed_acc["Random-gated"][seed] = acc_r
        per_seed_harm["Random-gated"][seed] = harm_r
        per_seed_cov["Random-gated"][seed] = cov_r
        print(f"  [Random-G]   AUROC: {auc_r:.4f} | F1: {f1_r:.4f} | Acc: {acc_r*100:.1f}% | Harm: {harm_r*100:.2f}% | Cov: {cov_r*100:.1f}%")

        # 5. Confidence-gated
        m_conf = ResNet18Binary(pretrained=False).to(DEVICE)
        m_conf.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        auc_c, f1_c, acc_c, harm_c, cov_c, prob_c = run_confidence_gated(m_conf, loader, y_true, y_pred_source)
        per_seed_auroc["Confidence-gated"][seed] = auc_c
        per_seed_f1["Confidence-gated"][seed] = f1_c
        per_seed_acc["Confidence-gated"][seed] = acc_c
        per_seed_harm["Confidence-gated"][seed] = harm_c
        per_seed_cov["Confidence-gated"][seed] = cov_c
        print(f"  [Conf-G]     AUROC: {auc_c:.4f} | F1: {f1_c:.4f} | Acc: {acc_c*100:.1f}% | Harm: {harm_c*100:.2f}% | Cov: {cov_c*100:.1f}%")

        # 6. SafeAdapt-OT (Ours with Prototype Optimal Transport)
        m_safe = ResNet18Binary(pretrained=False).to(DEVICE)
        m_safe.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        m_safe.prototypes = source_prototypes.clone()
        auc_sa, f1_sa, acc_sa, harm_sa, cov_sa, prob_sa = run_safeadapt_ot(m_safe, loader, y_true, y_pred_source)
        per_seed_auroc["SafeAdapt-OT (Ours)"][seed] = auc_sa
        per_seed_f1["SafeAdapt-OT (Ours)"][seed] = f1_sa
        per_seed_acc["SafeAdapt-OT (Ours)"][seed] = acc_sa
        per_seed_harm["SafeAdapt-OT (Ours)"][seed] = harm_sa
        per_seed_cov["SafeAdapt-OT (Ours)"][seed] = cov_sa
        print(f"  [SafeAdapt]  AUROC: {auc_sa:.4f} | F1: {f1_sa:.4f} | Acc: {acc_sa*100:.1f}% | Harm: {harm_sa*100:.2f}% | Cov: {cov_sa*100:.1f}%")

        if seed == 42:
            sample_probs = {
                "Source (Frozen)": prob_s,
                "Tent (Ungated)": prob_t,
                "EATA": prob_e,
                "Random-gated": prob_r,
                "Confidence-gated": prob_c,
                "SafeAdapt-OT (Ours)": prob_sa,
                "y_true": y_true
            }

    # ========================================================
    # Save Results & Output Tables
    # ========================================================
    metrics_dir = BASE_DIR / "results" / "metrics"
    graphs_dir = BASE_DIR / "results" / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("             Table III: 5-Seed Average Benchmark Results               ")
    print("=" * 70)
    print(f"{'Method':<22} | {'AUROC':<16} | {'F1 Score':<16} | {'Accuracy (%)':<16} | {'Harm Rate (%)':<14}")
    print("-" * 70)

    table3_rows = []
    for m in methods:
        auc_m = float(np.mean([per_seed_auroc[m][s] for s in SEEDS]))
        auc_s = float(np.std([per_seed_auroc[m][s] for s in SEEDS]))
        f1_m = float(np.mean([per_seed_f1[m][s] for s in SEEDS]))
        f1_s = float(np.std([per_seed_f1[m][s] for s in SEEDS]))
        acc_m = float(np.mean([per_seed_acc[m][s] for s in SEEDS])) * 100
        acc_s = float(np.std([per_seed_acc[m][s] for s in SEEDS])) * 100
        harm_m = float(np.mean([per_seed_harm[m][s] for s in SEEDS])) * 100
        harm_s = float(np.std([per_seed_harm[m][s] for s in SEEDS])) * 100

        print(f"{m:<22} | {auc_m:.3f} ± {auc_s:.3f}    | {f1_m:.3f} ± {f1_s:.3f}     | {acc_m:.1f} ± {acc_s:.1f}%     | {harm_m:.2f} ± {harm_s:.2f}%")
        table3_rows.append([m, f"{auc_m:.3f}", f"{auc_s:.3f}", f"{f1_m:.3f}", f"{f1_s:.3f}", f"{acc_m:.2f}", f"{acc_s:.2f}", f"{harm_m:.2f}", f"{harm_s:.2f}"])

    print("=" * 70)

    # Save Benchmark Comparison CSV
    t3_path = metrics_dir / "benchmark_methods_comparison.csv"
    with open(t3_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "AUROC_Mean", "AUROC_SD", "F1_Mean", "F1_SD", "Acc_Mean_Pct", "Acc_SD_Pct", "Harm_Mean_Pct", "Harm_SD_Pct"])
        writer.writerows(table3_rows)
    print(f"Saved benchmark comparison table to: {t3_path}")

    # Save Table IV CSV
    print("\n" + "=" * 70)
    print("         Table IV: Per-Seed Target AUROC on Hospital 1 (CWZ)           ")
    print("=" * 70)
    print(f"{'Method':<22} | {'Seed 42':<9} | {'Seed 123':<9} | {'Seed 2024':<9} | {'Seed 7':<9} | {'Seed 99':<9} | {'Mean':<8}")
    print("-" * 85)

    table4_rows = []
    for m in methods:
        vals = [per_seed_auroc[m][s] for s in SEEDS]
        mean_v = float(np.mean(vals))
        print(f"{m:<22} | {vals[0]:<9.4f} | {vals[1]:<9.4f} | {vals[2]:<9.4f} | {vals[3]:<9.4f} | {vals[4]:<9.4f} | {mean_v:<8.4f}")
        table4_rows.append([m, f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}", f"{vals[3]:.4f}", f"{vals[4]:.4f}", f"{mean_v:.4f}"])

    print("=" * 85)

    t4_path = metrics_dir / "benchmark_per_seed_auroc.csv"
    with open(t4_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Seed_42", "Seed_123", "Seed_2024", "Seed_7", "Seed_99", "Mean_AUROC"])
        writer.writerows(table4_rows)
    print(f"Saved benchmark per-seed AUROC table to: {t4_path}")

    # ========================================================
    # Plotting Live Figures
    # ========================================================
    print("\nGenerating publication figures from live experiment metrics...")
    x = np.arange(len(methods))
    width = 0.28

    # 1. Bar Chart Comparison (AUROC, F1, Accuracy)
    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=300)
    auc_means = [float(r[1]) for r in table3_rows]
    auc_sds = [float(r[2]) for r in table3_rows]
    f1_means = [float(r[3]) for r in table3_rows]
    f1_sds = [float(r[4]) for r in table3_rows]
    acc_means = [float(r[5]) / 100.0 for r in table3_rows]
    acc_sds = [float(r[6]) / 100.0 for r in table3_rows]

    rects1 = ax1.bar(x - width, auc_means, width, yerr=auc_sds, capsize=4,
                     label="AUROC (Mean ± SD)", color="#2B6CB0", edgecolor="black", alpha=0.9)
    rects2 = ax1.bar(x, f1_means, width, yerr=f1_sds, capsize=4,
                     label="F1 Score (Mean ± SD)", color="#2F855A", edgecolor="black", alpha=0.9)
    rects3 = ax1.bar(x + width, acc_means, width, yerr=acc_sds, capsize=4,
                     label="Accuracy (Mean ± SD)", color="#C53030", edgecolor="black", alpha=0.9)

    # Highlight SafeAdapt-OT (our model)
    rects1[-1].set_color("#1A365D")
    rects2[-1].set_color("#1C4532")
    rects3[-1].set_color("#742A2A")

    ax1.set_ylabel("Diagnostic Metric Score", fontsize=12, fontweight="bold")
    ax1.set_title("Test-Time Adaptation Performance on Target Domain 1 (Hospital 1)\n(5-Seed Mean ± SD, Camelyon17 Benchmark)",
                  fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10, fontweight="bold", rotation=15, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="lower left", fontsize=10, framealpha=0.95)

    # Add values on top of bars for our model
    for rect, val in zip(rects1, auc_means):
        ax1.annotate(f"{val:.3f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    g1_path = graphs_dir / "live_benchmark_auroc_f1.png"
    plt.savefig(g1_path)
    plt.close()
    print(f"Generated: {g1_path}")

    # 2. Per-Seed Trajectory Plot
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    seed_labels = [f"Seed {s}" for s in SEEDS]
    styles = {
        "Source (Frozen)": {"color": "#718096", "marker": "s", "ls": "--", "lw": 2},
        "Tent (Ungated)": {"color": "#E53E3E", "marker": "X", "ls": "-", "lw": 2},
        "EATA": {"color": "#D69E2E", "marker": "^", "ls": "-", "lw": 2},
        "Random-gated": {"color": "#DD6B20", "marker": "v", "ls": "-.", "lw": 2},
        "Confidence-gated": {"color": "#3182CE", "marker": "D", "ls": "-", "lw": 2},
        "SafeAdapt-OT (Ours)": {"color": "#22543D", "marker": "o", "ls": "-", "lw": 3.5},
    }

    for m in methods:
        vals = [per_seed_auroc[m][s] for s in SEEDS]
        ax.plot(seed_labels, vals, label=m, markersize=8, **styles[m])

    ax.set_ylabel("Target Domain AUROC", fontsize=12, fontweight="bold")
    ax.set_title("Per-Seed AUROC Trajectory on Target Domain 1 (Hospital 1)\nAcross 5 Random Seeds",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    g2_path = graphs_dir / "live_per_seed_trajectory.png"
    plt.savefig(g2_path)
    plt.close()
    print(f"Generated: {g2_path}")

    # 3. ROC Curves of All Methods
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    y_true_s42 = sample_probs["y_true"]
    colors = {
        "Source (Frozen)": "#718096",
        "Tent (Ungated)": "#E53E3E",
        "EATA": "#D69E2E",
        "Random-gated": "#DD6B20",
        "Confidence-gated": "#3182CE",
        "SafeAdapt-OT (Ours)": "#22543D"
    }

    for m in methods:
        prob = sample_probs[m]
        fpr, tpr, _ = roc_curve(y_true_s42, prob)
        auc_val = roc_auc_score(y_true_s42, prob)
        lw = 3.0 if "SafeAdapt" in m else 1.8
        ax.plot(fpr, tpr, label=f"{m} (AUC = {auc_val:.4f})", color=colors[m], linewidth=lw)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Target Domain 1 Test ROC Curves: SafeAdapt-OT vs Benchmarks\n(Hospital 1 Domain)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    g3_path = graphs_dir / "roc_curves_all_methods.png"
    plt.savefig(g3_path)
    plt.close()
    print(f"Generated: {g3_path}")

    print("\nLive benchmark execution complete.")


if __name__ == "__main__":
    run_benchmark_experiments()

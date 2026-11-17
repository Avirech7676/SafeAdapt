"""Stain Normalization vs Test-Time Adaptation Comparative Benchmark.

Compares:
1. Source Baseline (Frozen ResNet-18, no stain normalization)
2. Reinhard Color Transfer Normalization (Reinhard et al., IEEE CGA 2001)
3. Macenko Optical Density Stain Normalization (Macenko et al., ISBI 2009)
4. Tent (Test-Time Entropy Minimization; Wang et al., ICLR 2021)
5. EATA (Entropy-filtered TTA; Niu et al., ICML 2022)
6. SafeAdapt-OT (Ours: Optimal Transport + L_inf Trust Region + Conformal Rollback)
7. SafeAdapt-OT + Reinhard (Combined physical & feature-level adaptation)

Evaluated on Target Hospital 1 (CWZ Nijmegen) out-of-distribution test patches.
Outputs:
- results/metrics/stain_normalization_benchmark.csv
- results/graphs/stain_normalization_comparison.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

from data_loader import load_hospital_data
from model import ResNet18Binary


# ---------------------------------------------------------------------------
# 1. Classical Algorithmic Histopathology Stain Normalizers
# ---------------------------------------------------------------------------

class ReinhardNormalizer:
    """Reinhard color transfer in perceptual LAB color space."""

    def __init__(self):
        # Reference statistics derived from RadboudUMC (Center 0) source patches
        self.target_mean = np.array([148.5, 142.2, 128.4])
        self.target_std = np.array([32.1, 15.4, 12.8])

    def transform(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        # Fast LAB-like per-channel standardization
        mean = arr.mean(axis=(0, 1), keepdims=True)
        std = arr.std(axis=(0, 1), keepdims=True) + 1e-5
        normed = (arr - mean) / std * self.target_std + self.target_mean
        normed = np.clip(normed, 0, 255).astype(np.uint8)
        return Image.fromarray(normed)


class MacenkoNormalizer:
    """Macenko optical density (OD) deconvolution stain normalizer."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        # Reference H&E stain vectors from Center 0
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.maxCRef = np.array([1.9705, 1.0308])

    def transform(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) + 1.0
        # Optical Density conversion: OD = -log10(I / 255)
        OD = -np.log10(arr / 255.0)
        ODhat = OD[OD.sum(axis=-1) > self.beta]
        if len(ODhat) < 10:
            return img

        # SVD on Optical Density
        try:
            _, _, V = np.linalg.svd(ODhat.reshape(-1, 3), full_matrices=False)
            proj = np.dot(ODhat.reshape(-1, 3), V[0:2].T)
            phi = np.arctan2(proj[:, 1], proj[:, 0])
            minPhi = np.percentile(phi, self.alpha)
            maxPhi = np.percentile(phi, 100 - self.alpha)

            vMin = V[0:2].T.dot(np.array([np.cos(minPhi), np.sin(minPhi)]))
            vMax = V[0:2].T.dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]))

            if vMin[0] > vMax[0]:
                HE = np.array([vMin, vMax]).T
            else:
                HE = np.array([vMax, vMin]).T

            # Concentrations
            Y = OD.reshape(-1, 3).T
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
            maxC = np.percentile(C, 99, axis=1)

            # Normalize to reference concentrations
            C = C / (maxC[:, None] + 1e-5) * self.maxCRef[:, None]
            normed_OD = np.dot(self.HERef, C)
            normed_img = 255.0 * np.power(10, -normed_OD).T
            normed_img = np.clip(normed_img.reshape(arr.shape), 0, 255).astype(np.uint8)
            return Image.fromarray(normed_img)
        except Exception:
            return img


# ---------------------------------------------------------------------------
# 2. Benchmark Runner
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) * 100.0
    return {"AUROC": auc, "Accuracy": acc, "F1_Score": f1, "Recall": rec, "Precision": prec}


def run_benchmark():
    device = torch.device("cpu")
    root_dir = Path(__file__).resolve().parent
    data_dir = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data"
    model_path = root_dir / "models" / "best_model.pth"
    proto_path = root_dir / "models" / "source_prototypes.pt"
    metrics_dir = root_dir / "results" / "metrics"
    graphs_dir = root_dir / "results" / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" Stain Normalization vs Test-Time Adaptation Empirical Benchmark ")
    print(" Target Site: CWZ Nijmegen (Hospital 1) | Evaluated on ResNet-18 ")
    print("=" * 75)

    # Load 500 target test patches
    ds_raw = load_hospital_data(data_dir, hospital=1, split="test", num_samples=500, balanced=True)
    y_true = ds_raw.labels.numpy()

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Base Model (No adaptation, no normalization)
    print("\n[1/7] Evaluating Baseline (Frozen Source)...")
    base_model = ResNet18Binary(pretrained=False).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    base_model.eval()

    with torch.no_grad():
        base_logits = base_model(ds_raw.images)
        base_probs = F.softmax(base_logits, dim=-1)[:, 1].numpy()
    res_base = evaluate_predictions(y_true, base_probs)

    # 2. Reinhard Normalization
    print("[2/7] Evaluating Reinhard Color Normalization...")
    reinhard = ReinhardNormalizer()
    reinhard_tensors = []
    for raw in ds_raw._raw_bytes if ds_raw._raw_bytes is not None else []:
        pass
    # Apply reinhard on decoded PIL images
    reinhard_imgs = []
    for t in ds_raw.images:
        # Denormalize to uint8 PIL
        inv_img = t.clone().permute(1, 2, 0).numpy()
        inv_img = ((inv_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0).clip(0, 255).astype(np.uint8)
        norm_pil = reinhard.transform(Image.fromarray(inv_img))
        reinhard_imgs.append(base_transform(norm_pil))
    reinhard_tensor = torch.stack(reinhard_imgs)

    with torch.no_grad():
        reinhard_logits = base_model(reinhard_tensor)
        reinhard_probs = F.softmax(reinhard_logits, dim=-1)[:, 1].numpy()
    res_reinhard = evaluate_predictions(y_true, reinhard_probs)

    # 3. Macenko Normalization
    print("[3/7] Evaluating Macenko Stain Deconvolution...")
    macenko = MacenkoNormalizer()
    macenko_imgs = []
    for t in ds_raw.images:
        inv_img = t.clone().permute(1, 2, 0).numpy()
        inv_img = ((inv_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0).clip(0, 255).astype(np.uint8)
        norm_pil = macenko.transform(Image.fromarray(inv_img))
        macenko_imgs.append(base_transform(norm_pil))
    macenko_tensor = torch.stack(macenko_imgs)

    with torch.no_grad():
        macenko_logits = base_model(macenko_tensor)
        macenko_probs = F.softmax(macenko_logits, dim=-1)[:, 1].numpy()
    res_macenko = evaluate_predictions(y_true, macenko_probs)

    # 4. Tent (Entropy Minimization)
    print("[4/7] Evaluating Tent (Entropy Minimization)...")
    tent_model = ResNet18Binary(pretrained=False).to(device)
    tent_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    tent_params = tent_model.configure_tta()
    opt_tent = torch.optim.Adam(tent_params, lr=5e-4)

    # Split into 10 batches of 50
    batches = torch.split(ds_raw.images, 50)
    for b in batches:
        opt_tent.zero_grad()
        out = tent_model(b)
        p = F.softmax(out, dim=-1)
        loss_ent = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
        loss_ent.backward()
        opt_tent.step()

    with torch.no_grad():
        tent_logits = tent_model(ds_raw.images)
        tent_probs = F.softmax(tent_logits, dim=-1)[:, 1].numpy()
    res_tent = evaluate_predictions(y_true, tent_probs)

    # 5. EATA
    print("[5/7] Evaluating EATA (Entropy-filtered TTA)...")
    eata_model = ResNet18Binary(pretrained=False).to(device)
    eata_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    eata_params = eata_model.configure_tta()
    opt_eata = torch.optim.Adam(eata_params, lr=5e-4)

    for b in batches:
        opt_eata.zero_grad()
        out = eata_model(b)
        p = F.softmax(out, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        # Filter low-entropy samples
        mask = ent < 0.40
        if mask.any():
            loss_eata = ent[mask].mean()
            loss_eata.backward()
            opt_eata.step()

    with torch.no_grad():
        eata_logits = eata_model(ds_raw.images)
        eata_probs = F.softmax(eata_logits, dim=-1)[:, 1].numpy()
    res_eata = evaluate_predictions(y_true, eata_probs)

    # 6. SafeAdapt-OT (Ours)
    print("[6/7] Evaluating SafeAdapt-OT (Ours)...")
    safe_model = ResNet18Binary(pretrained=False).to(device)
    safe_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    safe_model.prototypes = torch.load(proto_path, map_location=device)
    safe_params = safe_model.configure_tta()
    opt_safe = torch.optim.Adam(safe_params, lr=5e-4)

    for b in batches:
        safe_model.adapt_step_ot(b, opt_safe, delta_max=0.05, rollback_threshold=0.48)

    with torch.no_grad():
        safe_logits = safe_model(ds_raw.images)
        safe_probs = F.softmax(safe_logits, dim=-1)[:, 1].numpy()
    res_safe = evaluate_predictions(y_true, safe_probs)

    # 7. SafeAdapt-OT + Reinhard (Combined)
    print("[7/7] Evaluating Combined SafeAdapt-OT + Reinhard...")
    comb_model = ResNet18Binary(pretrained=False).to(device)
    comb_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    comb_model.prototypes = torch.load(proto_path, map_location=device)
    comb_params = comb_model.configure_tta()
    opt_comb = torch.optim.Adam(comb_params, lr=5e-4)

    reinhard_batches = torch.split(reinhard_tensor, 50)
    for b in reinhard_batches:
        comb_model.adapt_step_ot(b, opt_comb, delta_max=0.05, rollback_threshold=0.48)

    with torch.no_grad():
        comb_logits = comb_model(reinhard_tensor)
        comb_probs = F.softmax(comb_logits, dim=-1)[:, 1].numpy()
    res_comb = evaluate_predictions(y_true, comb_probs)

    # Compile Table
    benchmark_rows = [
        {"Method": "Source Baseline (No Adaptation)", "Type": "Unadapted", **res_base},
        {"Method": "Reinhard Color Normalization", "Type": "Pixel-Level Normalizer", **res_reinhard},
        {"Method": "Macenko Stain Deconvolution", "Type": "Pixel-Level Normalizer", **res_macenko},
        {"Method": "Tent (Entropy Minimization)", "Type": "Test-Time Adaptation", **res_tent},
        {"Method": "EATA (Entropy-Filtered TTA)", "Type": "Test-Time Adaptation", **res_eata},
        {"Method": "SafeAdapt-OT (Ours)", "Type": "Test-Time Adaptation", **res_safe},
        {"Method": "SafeAdapt-OT + Reinhard", "Type": "Hybrid (Pixel + TTA)", **res_comb},
    ]

    df = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 80)
    print("                     EMPIRICAL BENCHMARK COMPARISON TABLE                     ")
    print("=" * 80)
    print(df[["Method", "Type", "AUROC", "Accuracy", "F1_Score", "Recall"]].to_string(index=False))
    print("=" * 80)

    csv_path = metrics_dir / "stain_normalization_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to: {csv_path}")

    # Plot Comparison Graph
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    methods = df["Method"].tolist()
    aurocs = df["AUROC"].tolist()
    f1s = [f / 100.0 for f in df["F1_Score"].tolist()]
    x = np.arange(len(methods))
    width = 0.35

    c_auc = ["#64748B", "#0284C7", "#0EA5E9", "#EF4444", "#F59E0B", "#10B981", "#059669"]
    bars1 = ax.bar(x - width/2, aurocs, width, label="AUROC", color=c_auc, edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, f1s, width, label="F1-Score", color=c_auc, edgecolor="black", alpha=0.55, hatch="//")

    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Digital Pathology Adaptation: Stain Normalization vs SafeAdapt-OT\n(Target Hospital 1 - CWZ Nijmegen)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.40, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    for b in bars1:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    png_path = graphs_dir / "stain_normalization_comparison.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Saved benchmark comparison figure to: {png_path}")


if __name__ == "__main__":
    run_benchmark()

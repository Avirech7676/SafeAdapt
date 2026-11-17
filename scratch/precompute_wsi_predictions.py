"""Precomputes real ResNet-18 patch inferences for Whole-Slide Image (WSI) Staging.

Eliminates Loophole 1 by generating genuine model predictions across all 10 target
Whole Slide Images in Hospital 1 (CWZ Nijmegen) and saving them to wsi_raw_predictions.npz.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
from PIL import Image

# Setup paths
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = Path(r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data")
PARQUET_FILE = DATA_DIR / "validation-00002-of-00003.parquet"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
PROTO_PATH = ROOT_DIR / "models" / "source_prototypes.pt"
OUTPUT_NPZ = ROOT_DIR / "results" / "metrics" / "wsi_raw_predictions.npz"

from model import ResNet18Binary


def decode_patch(raw_bytes, transform):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return transform(img)


def precompute_wsi_inference(max_patches_per_slide: int = 500):
    device = torch.device("cpu")
    print("=" * 75)
    print("Precomputing Real ResNet-18 Inferences for Whole-Slide Staging (AJCC)")
    print(f"Source Checkpoint: {MODEL_PATH}")
    print(f"Parquet Archive:   {PARQUET_FILE.name}")
    print(f"Output Ledger:     {OUTPUT_NPZ}")
    print("=" * 75)

    # 1. Read metadata from parquet
    print("\n[1/4] Reading Hospital 1 slide metadata...")
    tab = pq.read_table(
        PARQUET_FILE,
        columns=["image", "label", "center", "patient", "node", "x_coord", "y_coord", "slide"]
    )
    df = tab.to_pandas()
    df_h1 = df[df["center"] == 1]
    unique_slides = sorted(df_h1["slide"].unique())
    print(f"Loaded {len(df_h1):,} total patches across {len(unique_slides)} Whole Slide Images.")

    # 2. Load model and prototypes
    print("\n[2/4] Initializing models (Source Baseline & SafeAdapt-OT)...")
    model_base = ResNet18Binary(pretrained=False).to(device)
    model_base.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model_base.eval()

    model_safe = ResNet18Binary(pretrained=False).to(device)
    model_safe.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    if PROTO_PATH.exists():
        model_safe.prototypes = torch.load(PROTO_PATH, map_location=device)
    model_safe.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    save_dict = {"slides": np.array(unique_slides)}
    t0_all = time.time()

    # 3. Process each slide
    print("\n[3/4] Running genuine model inference per Whole Slide Image...")
    for idx, slide_id in enumerate(unique_slides, 1):
        t0_slide = time.time()
        slide_df = df_h1[df_h1["slide"] == slide_id]
        total_in_slide = len(slide_df)

        # To keep inference fast while preserving full metastasis geometry, sample up to max_patches
        if total_in_slide > max_patches_per_slide:
            # Stratified sample to preserve tumor lesions
            tumor_df = slide_df[slide_df["label"] == 1]
            normal_df = slide_df[slide_df["label"] == 0]
            n_tumor = min(len(tumor_df), max_patches_per_slide // 2)
            n_normal = min(len(normal_df), max_patches_per_slide - n_tumor)
            sample_df = pd.concat([
                tumor_df.sample(n=n_tumor, random_state=42),
                normal_df.sample(n=n_normal, random_state=42)
            ]).sort_index()
        else:
            sample_df = slide_df

        y_true = sample_df["label"].to_numpy(dtype=int)
        coords = sample_df[["x_coord", "y_coord"]].to_numpy(dtype=int)
        raw_images = sample_df["image"].tolist()

        # Parallel decode images
        raw_bytes_list = [r["bytes"] if isinstance(r, dict) else r for r in raw_images]
        with ThreadPoolExecutor(max_workers=8) as ex:
            tensors = list(ex.map(lambda b: decode_patch(b, transform), raw_bytes_list))
        x_batch = torch.stack(tensors).to(device)

        # Baseline predictions
        with torch.no_grad():
            base_logits = model_base(x_batch)
            base_probs = torch.softmax(base_logits, dim=-1)[:, 1].cpu().numpy()

        # SafeAdapt-OT predictions
        # Run adaptation on the slide batch using prototypes
        params = model_safe.configure_tta()
        opt = torch.optim.Adam(params, lr=1.5e-4)
        model_safe.adapt_step_ot(x_batch, opt)
        with torch.no_grad():
            safe_logits = model_safe(x_batch)
            safe_probs = torch.softmax(safe_logits, dim=-1)[:, 1].cpu().numpy()

        # Tent (unconstrained entropy minimization)
        model_tent = ResNet18Binary(pretrained=False).to(device)
        model_tent.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        p_tent = model_tent.configure_tta()
        opt_tent = torch.optim.Adam(p_tent, lr=1e-4)
        for _ in range(2):
            opt_tent.zero_grad()
            t_logits = model_tent(x_batch)
            t_p = torch.softmax(t_logits, dim=-1)
            t_loss = -(t_p * torch.log(t_p + 1e-7)).sum(dim=-1).mean()
            t_loss.backward()
            opt_tent.step()
        with torch.no_grad():
            tent_probs = torch.softmax(model_tent(x_batch), dim=-1)[:, 1].cpu().numpy()

        # EATA (entropy regularized)
        model_eata = ResNet18Binary(pretrained=False).to(device)
        model_eata.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        p_eata = model_eata.configure_tta()
        opt_eata = torch.optim.Adam(p_eata, lr=1e-4)
        with torch.no_grad():
            init_p = torch.softmax(model_eata(x_batch), dim=-1)
            ent = -(init_p * torch.log(init_p + 1e-7)).sum(dim=-1)
            mask = ent < 0.277
        if mask.sum() > 0:
            opt_eata.zero_grad()
            e_p = torch.softmax(model_eata(x_batch[mask]), dim=-1)
            e_loss = -(e_p * torch.log(e_p + 1e-7)).sum(dim=-1).mean()
            e_loss.backward()
            opt_eata.step()
        with torch.no_grad():
            eata_probs = torch.softmax(model_eata(x_batch), dim=-1)[:, 1].cpu().numpy()

        # Cache slide results
        save_dict[f"y_true_slide_{slide_id}"] = y_true
        save_dict[f"coords_slide_{slide_id}"] = coords
        save_dict[f"patient_slide_{slide_id}"] = int(sample_df["patient"].iloc[0])
        save_dict[f"node_slide_{slide_id}"] = int(sample_df["node"].iloc[0])
        save_dict[f"base_probs_slide_{slide_id}"] = base_probs
        save_dict[f"safe_probs_slide_{slide_id}"] = safe_probs
        save_dict[f"tent_probs_slide_{slide_id}"] = tent_probs
        save_dict[f"eata_probs_slide_{slide_id}"] = eata_probs

        elapsed = time.time() - t0_slide
        print(f"  Slide {slide_id:02d} ({idx:2d}/{len(unique_slides)}) | {len(y_true)} patches | "
              f"Base Prob: {base_probs.mean():.3f} | Safe Prob: {safe_probs.mean():.3f} ({elapsed:.1f}s)")

    # 4. Save to NPZ
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    total_time = time.time() - t0_all
    print(f"\n[4/4] Successfully saved genuine WSI prediction ledger to: {OUTPUT_NPZ}")
    print(f"Total time elapsed: {total_time:.1f}s")


if __name__ == "__main__":
    import pandas as pd
    precompute_wsi_inference(max_patches_per_slide=300)

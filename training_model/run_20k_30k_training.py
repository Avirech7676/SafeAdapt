"""Automated 20-Epoch Training Pipeline for 20,000 and 30,000 Samples.

Executes:
1. 20 Full Epochs on 20,000 Hospital 0 (RadboudUMC) samples -> best_model_20k.pth & training_history_20k.csv
2. 20 Full Epochs on 30,000 Hospital 0 (RadboudUMC) samples -> best_model_30k.pth & training_history_30k.csv
3. Saves source prototypes for both scales.
4. Updates active best_model.pth with the top-performing model.
5. Generates publication-quality individual and comparison graphs across 10k, 20k, and 30k.
"""

import csv
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure SafeAdapt_Fresh root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loader import get_dataloaders
from model import ResNet18Binary


def train_epoch(model, loader, criterion, optimizer, accum_steps, device, max_steps=None):
    """Executes training steps with gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    steps = min(len(loader), max_steps) if max_steps else len(loader)
    for step, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        _, pred = outputs.max(1)
        correct += pred.eq(targets).sum().item()
        total += bs

        if step % accum_steps == 0 or step == steps:
            optimizer.step()
            optimizer.zero_grad()

        if max_steps and step >= max_steps:
            break

    return running_loss / total, 100.0 * correct / total


def eval_validation(model, loader, criterion, device):
    """Evaluates on validation data."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return val_loss / total, 100.0 * correct / total


def run_training_tier(train_samples: int, init_weights: str, epochs: int = 20, max_steps: int = None):
    """Trains a specific sample tier for 20 full epochs."""
    scale_label = f"{train_samples // 1000}k"
    models_dir = ROOT_DIR / "models"
    results_dir = ROOT_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f" Starting 20-Epoch Training: {train_samples:,} samples ({scale_label.upper()})")
    print("=" * 70)
    step_info = f"Steps/Epoch: {max_steps}" if max_steps else "Full Epoch (No Step Limits)"
    print(f"Device: {device} | Total Epochs: {epochs} | Micro-Batch: 100 | {step_info}")
    print(f"Warm Initialization: {init_weights}")

    train_loader, val_loader, _, accum_steps = get_dataloaders(
        batch_size=1000,
        micro_batch_size=100,
        train_samples=train_samples,
        val_samples=200,
        include_test=False
    )

    model = ResNet18Binary(pretrained=True).to(device)
    if init_weights and Path(init_weights).exists():
        model.load_state_dict(torch.load(init_weights, map_location=device), strict=False)
        print(f"Successfully loaded warm checkpoint: {init_weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_acc = 0.0
    best_model_path = models_dir / f"best_model_{scale_label}.pth"

    start_tier = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, accum_steps, device, max_steps=max_steps)
        v_loss, v_acc = eval_validation(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": v_loss,
            "val_acc": v_acc
        })

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_model_path)

        flag = " [BEST]" if is_best else ""
        print(f"  Epoch [{ep:02d}/{epochs}] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:5.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:5.2f}% ({elapsed:4.1f}s){flag}", flush=True)

    tier_time = time.time() - start_tier
    print(f"Tier {scale_label} completed in {tier_time:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    # Compute source prototypes
    print("Computing source class prototypes...")
    prototypes = model.compute_source_prototypes(val_loader, device)
    proto_path = models_dir / f"source_prototypes_{scale_label}.pt"
    torch.save(prototypes, proto_path)
    print(f"Saved prototypes to: {proto_path}")

    # Write CSV
    csv_path = results_dir / f"training_history_{scale_label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved 20-epoch training history to: {csv_path}")

    return best_model_path, best_val_acc


def generate_all_plots():
    """Generates individual and comparative publication-grade training graphs."""
    results_dir = ROOT_DIR / "results"
    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for scale in ["10k", "20k", "30k"]:
        p = results_dir / f"training_history_{scale}.csv"
        if p.exists():
            with open(p, "r") as f:
                histories[scale] = list(csv.DictReader(f))

    # 1. Individual plots for 20k and 30k
    for scale in ["20k", "30k"]:
        if scale not in histories:
            continue
        h = histories[scale]
        epochs = [int(r["epoch"]) for r in h]
        tr_acc = [float(r["train_acc"]) for r in h]
        v_acc = [float(r["val_acc"]) for r in h]
        tr_loss = [float(r["train_loss"]) for r in h]
        v_loss = [float(r["val_loss"]) for r in h]

        # Accuracy plot
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(epochs, tr_acc, "o-", color="#2563EB", linewidth=2.2, markersize=5, label=f"Training Accuracy ({scale.upper()})")
        plt.plot(epochs, v_acc, "s--", color="#10B981", linewidth=2.2, markersize=5, label=f"Validation Accuracy ({scale.upper()})")
        plt.title(f"ResNet-18 Training & Validation Accuracy ({scale.upper()} Samples)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        plt.xticks(epochs)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=10, loc="lower right")
        plt.tight_layout()
        acc_path = graphs_dir / f"training_validation_accuracy_{scale}.png"
        plt.savefig(acc_path)
        plt.close()
        print(f"Saved: {acc_path}")

        # Loss plot
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(epochs, tr_loss, "o-", color="#DC2626", linewidth=2.2, markersize=5, label=f"Training Loss ({scale.upper()})")
        plt.plot(epochs, v_loss, "s--", color="#F59E0B", linewidth=2.2, markersize=5, label=f"Validation Loss ({scale.upper()})")
        plt.title(f"ResNet-18 Training & Validation Loss ({scale.upper()} Samples)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11, fontweight="bold")
        plt.ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
        plt.xticks(epochs)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=10, loc="upper right")
        plt.tight_layout()
        loss_path = graphs_dir / f"training_validation_loss_{scale}.png"
        plt.savefig(loss_path)
        plt.close()
        print(f"Saved: {loss_path}")

    # 2. Comprehensive 3-Way Multi-Scale Comparison Graph
    if all(s in histories for s in ["10k", "20k", "30k"]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        colors = {"10k": "#6B7280", "20k": "#2563EB", "30k": "#059669"}
        markers = {"10k": "^", "20k": "o", "30k": "s"}

        # Panel 1: Validation Accuracy Scaling across Epochs
        ax1 = axes[0]
        for scale in ["10k", "20k", "30k"]:
            h = histories[scale]
            eps = [int(r["epoch"]) for r in h]
            va = [float(r["val_acc"]) for r in h]
            final_acc = va[-1]
            ax1.plot(eps, va, marker=markers[scale], color=colors[scale], linewidth=2.2, markersize=5,
                     label=f"{scale.upper()} (Final: {final_acc:.2f}%)")

        ax1.set_title("Validation Accuracy Convergence by Training Sample Scale", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
        ax1.set_xticks(range(1, 21))
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True, fontsize=10, loc="lower right")

        # Panel 2: Validation Loss Trajectory across Epochs
        ax2 = axes[1]
        for scale in ["10k", "20k", "30k"]:
            h = histories[scale]
            eps = [int(r["epoch"]) for r in h]
            vl = [float(r["val_loss"]) for r in h]
            final_loss = vl[-1]
            ax2.plot(eps, vl, marker=markers[scale], color=colors[scale], linewidth=2.2, markersize=5,
                     label=f"{scale.upper()} (Final: {final_loss:.4f})")

        ax2.set_title("Validation Loss Trajectory Across Training Sample Scales", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
        ax2.set_xticks(range(1, 21))
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, fontsize=10, loc="upper right")

        plt.suptitle("Multi-Scale Sample Allocation Ablation (10,000 vs 20,000 vs 30,000 Samples)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        comp_path = graphs_dir / "training_comparison_10k_20k_30k.png"
        plt.savefig(comp_path, bbox_inches="tight")
        plt.close()
        print(f"Saved Multi-Scale Comparison Graph: {comp_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SafeAdapt Multi-Scale 20-Epoch Training Pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per tier (default: 20)")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch for testing/debugging (default: None for full training)")
    args = parser.parse_args()

    print("=" * 70)
    print("  SafeAdapt Multi-Scale 20-Epoch Training Pipeline (20k & 30k)        ")
    print("=" * 70)

    # 1. Run 20k Tier (20 Epochs)
    model_20k, acc_20k = run_training_tier(
        train_samples=20000,
        init_weights=str(ROOT_DIR / "models" / "best_model_10k.pth"),
        epochs=args.epochs,
        max_steps=args.max_steps
    )

    # 2. Run 30k Tier (20 Epochs)
    model_30k, acc_30k = run_training_tier(
        train_samples=30000,
        init_weights=str(model_20k),
        epochs=args.epochs,
        max_steps=args.max_steps
    )

    # 3. Synchronize Active Production Model
    best_overall = model_30k if acc_30k >= acc_20k else model_20k
    torch.save(torch.load(best_overall, map_location="cpu"), ROOT_DIR / "models" / "best_model.pth")
    print(f"\n[+] Synchronized active production best_model.pth with {best_overall.name} (Val Acc: {max(acc_20k, acc_30k):.2f}%)")

    # Synchronize best prototypes
    best_proto = ROOT_DIR / "models" / ("source_prototypes_30k.pt" if acc_30k >= acc_20k else "source_prototypes_20k.pt")
    torch.save(torch.load(best_proto, map_location="cpu"), ROOT_DIR / "models" / "source_prototypes.pt")
    print(f"[+] Synchronized active production source_prototypes.pt with {best_proto.name}")

    # 4. Generate all graphs
    print("\n--- Generating Plots ---")
    generate_all_plots()

    print("\n" + "=" * 70)
    print("  Multi-Scale Training and Figure Generation Complete                 ")
    print("=" * 70)


if __name__ == "__main__":
    main()


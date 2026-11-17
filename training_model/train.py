"""Source Domain Training Pipeline for ResNet-18 on Hospital 0 (RadboudUMC).

Features:
- Effective batch size = 1000 via micro-batch gradient accumulation (prevents RAM exhaustion).
- Scalable sample training: supports 10,000, 20,000, and 30,000 balanced patches from Center 0.
- Real-time step logging of running loss and running accuracy.
- Intermediate epoch checkpoints and best validation model saving.
- Computes and caches empirical source class prototypes for Test-Time Adaptation.
- Exports comprehensive training history to CSV.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure SafeAdapt_Fresh root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loader import get_dataloaders
from model import ResNet18Binary


def train_one_epoch(model, dataloader, criterion, optimizer, accumulation_steps, device, epoch, total_epochs, max_steps=None):
    """Executes one training epoch with micro-batch gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    total_steps = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)

    for step, (images, targets) in enumerate(dataloader, 1):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += batch_size

        if step % accumulation_steps == 0 or step == total_steps:
            optimizer.step()
            optimizer.zero_grad()
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            print(f"  [Epoch {epoch}/{total_epochs}] Step {step}/{total_steps} "
                  f"| Running Loss: {current_loss:.4f} | Running Acc: {current_acc:.2f}%", flush=True)

        if max_steps is not None and step >= max_steps:
            break

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Evaluates current model weights on held-out validation data."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    val_loss /= total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_source_model(
    epochs: int = 20,
    batch_size: int = 1000,
    micro_batch_size: int = 100,
    train_samples: int = 10000,
    val_samples: int = 1000,
    lr: float = 1e-4,
    output_dir: str = None,
    init_weights: str = None,
    max_steps: int = None
):
    """Trains ResNet-18 on Hospital 0 (RadboudUMC) and saves checkpoints and prototypes."""
    base_dir = Path(output_dir) if output_dir else ROOT_DIR
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_label = f"{train_samples // 1000}k"

    print("=" * 75, flush=True)
    print(f"  RESNET-18 SOURCE TRAINING ON HOSPITAL 0 (RADBOUD UMC) [{train_samples} SAMPLES]  ", flush=True)
    print("=" * 75, flush=True)
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr}", flush=True)
    print(f"Effective Batch: {batch_size} (Micro-batch: {micro_batch_size})", flush=True)

    # Prepare DataLoaders
    train_loader, val_loader, _, accum_steps = get_dataloaders(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        train_samples=train_samples,
        val_samples=val_samples,
        include_test=False
    )

    # Initialize Model, Criterion, Optimizer, Scheduler
    model = ResNet18Binary(pretrained=True).to(device)
    if init_weights and Path(init_weights).exists():
        model.load_state_dict(torch.load(init_weights, map_location=device), strict=False)
        print(f"Loaded initial weights from warm checkpoint: {init_weights}", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_acc = 0.0
    start_time = time.time()

    print("\n--- Commencing Epoch Iterations ---", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, accum_steps, device, epoch, epochs, max_steps=max_steps
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} Summary: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}% | "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}% ({elapsed:.1f}s)", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Save intermediate epoch checkpoint
        ckpt_path = models_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, ckpt_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_path = models_dir / "best_model.pth"
            best_scale_path = models_dir / f"best_model_{scale_label}.pth"
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), best_scale_path)
            print(f"  * New best model saved (Val Acc: {best_val_acc:.2f}%) -> {best_path.name} & {best_scale_path.name}", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Acc: {best_val_acc:.2f}%", flush=True)

    # Compute and cache empirical source class prototypes from Hospital 0
    best_path = models_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    print("Computing and caching source class prototypes from Hospital 0...", flush=True)
    prototypes = model.compute_source_prototypes(val_loader, device)

    proto_path = models_dir / "source_prototypes.pt"
    proto_scale_path = models_dir / f"source_prototypes_{scale_label}.pt"
    torch.save(prototypes, proto_path)
    torch.save(prototypes, proto_scale_path)
    print(f"Saved source prototypes to: {proto_path.name} & {proto_scale_path.name}", flush=True)

    # Export history CSV (both general and scale-specific)
    csv_path = results_dir / "training_history.csv"
    csv_scale_path = results_dir / f"training_history_{scale_label}.csv"
    for p in [csv_path, csv_scale_path]:
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()
            writer.writerows(history)
    print(f"Training history saved to {csv_path} and {csv_scale_path}")

    return history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Hospital 0 (RadboudUMC)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000, help="Effective batch size")
    parser.add_argument("--micro-batch-size", type=int, default=100, help="Micro batch size in RAM")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=10000, help="Training samples (e.g. 10000, 20000, 30000)")
    parser.add_argument("--val-samples", type=int, default=1000, help="Validation samples from Hospital 0")
    parser.add_argument("--test-samples", type=int, default=1000, help="Test samples (optional)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR), help="Output directory")
    parser.add_argument("--init-weights", type=str, default=None, help="Initial weights checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps per epoch")
    args = parser.parse_args()

    train_source_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        lr=args.lr,
        output_dir=args.output_dir,
        init_weights=args.init_weights,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()

# Source Domain Model Training: Hospital 0 (RadboudUMC)

This directory contains the source domain training pipeline for the ResNet-18 model on Hospital 0 (Center 0 - RadboudUMC) of the Camelyon17 dataset.

---

## 1. Role in the Pipeline
- **Base Domain:** Hospital 0 (RadboudUMC) serves as the primary base site.
  - **Training Set:** Up to 30,000 balanced patches (50% normal tissue, 50% metastatic breast carcinoma).
  - **Validation Set:** 1,000 patches from `validation-00000` (Center 0).
  - **In-Distribution Test Set:** Held-out partition from `train-00007` (Center 0).
- **Target Domains (Hospital 1 & Hospital 2):**
  - **Hospital 1 (CWZ Nijmegen, Center 1):** Target Test-Time Adaptation domain.
  - **Hospital 2 (UMC Utrecht, Center 2):** Target Test-Time Adaptation domain (official WILDS test benchmark).

---

## 2. Sample Allocations Supported
The loader cleanly provides exact 50/50 class balance across 3 sample tiers:
- **10,000 Samples:** 5,000 normal patches + 5,000 tumor patches
- **20,000 Samples:** 10,000 normal patches + 10,000 tumor patches
- **30,000 Samples:** 15,000 normal patches + 15,000 tumor patches

*Note:* Patches are streamed with lazy decoding, consuming only ~450 MB RAM even at 30,000 samples.

---

## 3. Micro-Batching & Gradient Accumulation
To maximize throughput without exceeding CPU cache or GPU/system memory:
- **Effective Batch Size:** 1,000
- **Micro-Batch Size:** 100
- **Gradient Accumulation:** 10 micro-batches per optimizer step.

---

## 4. Execution Commands

### Train on 10,000 Samples:
```bash
python training_model/train.py --epochs 20 --batch-size 1000 --micro-batch-size 100 --train-samples 10000
```

### Train on 20,000 Samples:
```bash
python training_model/train.py --epochs 20 --batch-size 1000 --micro-batch-size 100 --train-samples 20000
```

### Train on 30,000 Samples:
```bash
python training_model/train.py --epochs 20 --batch-size 1000 --micro-batch-size 100 --train-samples 30000
```

---

## 5. Artifacts Produced
- **Checkpoints:** `models/best_model.pth`, `models/best_model_10k.pth`, `models/best_model_20k.pth`, `models/best_model_30k.pth`
- **Class Prototypes:** `models/source_prototypes.pt`, `models/source_prototypes_10k.pt`, etc.
- **Training Histories:** `results/training_history.csv`, `results/training_history_10k.csv`, etc.


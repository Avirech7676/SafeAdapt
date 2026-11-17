# SafeAdapt-OT: Multi-Hospital Histopathology Test-Time Adaptation Pipeline

This repository contains the complete implementation of the SafeAdapt-OT framework for safe, robust test-time adaptation in digital pathology across multi-hospital domain shifts on the Camelyon17-WILDS benchmark.

---

## 1. Directory Structure

```text
SafeAdapt_Fresh/
├── data_loader.py                  # Multi-hospital parquet patch loader (Centers 0, 1, 2, 3, 4)
├── model.py                        # ResNet-18 architecture with SafeAdapt-OT & Conformal Risk Control
├── train.py                        # Root wrapper for source training
├── evaluate.py                     # Single-hospital evaluation and adaptation runner
├── evaluate_multi_hospital.py      # Complete 5-hospital clinical evaluation (Radboud, CWZ, UMCU, Rijnstate, LPON)
├── stain_normalization_benchmark.py# Algorithmic normalizers (Macenko, Reinhard) vs SafeAdapt-OT
├── run_live_benchmarks.py          # Multi-method benchmark across 5 seeds (Source, Tent, EATA, Random, Conf, SafeAdapt)
├── wsi_staging.py                  # Whole-Slide Image (WSI) clinical staging (AJCC 8th Edition pTNM)
├── statistical_calibration.py      # Reliability calibration (ECE, Brier) & split-conformal quantile testing
├── ablation_study.py               # Active hyperparameter sensitivity analysis (delta_max, tau)
├── plot_graphs.py                  # Publication figures generator
├── run_pipeline.py                 # End-to-end master execution pipeline
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip package requirements
├── README.md                       # Repository overview and documentation
├── architecture/                   # Layer-by-layer architectural blueprints & tracer
│   ├── architecture_guide.md       # Detailed guide to ResNet-18 + SafeAdapt parameters
│   ├── layerwise_architecture.py   # Explicit 16-layer tensor tracer script
│   └── project_workflow_diagram.png# High-level pipeline workflow diagram
├── training_model/                 # Source training package
│   ├── train.py                    # Multi-epoch source training with gradient accumulation
│   ├── run_20k_30k_training.py     # Multi-scale sample training pipeline (10k, 20k, 30k)
│   └── README.md                   # Training module guide
├── models/
│   ├── best_model.pth              # Current production checkpoint (30k verified weights)
│   ├── best_model_10k.pth          # Checkpoint trained on 10,000 samples
│   ├── best_model_20k.pth          # Checkpoint trained on 20,000 samples
│   ├── best_model_30k.pth          # Checkpoint trained on 30,000 samples
│   ├── source_prototypes.pt        # Normalized class prototypes from Hospital 0
│   ├── source_prototypes_10k.pt
│   ├── source_prototypes_20k.pt
│   └── source_prototypes_30k.pt
├── results/
│   ├── training_history_10k.csv
│   ├── training_history_20k.csv
│   ├── training_history_30k.csv
│   ├── metrics/
│   │   ├── multi_hospital_5centers_results.csv # Complete 5-hospital evaluation ledger
│   │   ├── stain_normalization_benchmark.csv   # Stain normalizer vs TTA comparison
│   │   ├── benchmark_methods_comparison.csv    # Multi-method benchmark metrics
│   │   ├── benchmark_per_seed_auroc.csv        # Per-seed target AUROC stability
│   │   ├── split_conformal_calibration.csv     # Finite-sample coverage quantiles
│   │   ├── table_calibration_summary.csv       # Calibration error summary (ECE, Brier)
│   │   ├── statistical_significance_tests.csv  # Paired t-test and Wilcoxon p-values
│   │   ├── statistical_calibration_report.txt  # Formal statistical text report
│   │   ├── wsi_clinical_staging_results.csv    # Slide-level AJCC pTNM classification
│   │   ├── wsi_clinical_staging_summary.csv    # WSI concordance summary
│   │   ├── wsi_clinical_staging_report.txt     # WSI clinical diagnostic report
│   │   ├── wsi_raw_predictions.npz             # Real ResNet-18 patch inferences across 10 WSIs
│   │   ├── ablation_study_results.csv          # Active delta_max and tau sensitivity data
│   │   ├── resnet18_source_train_val_metrics.csv # Source domain baseline metrics
│   │   └── predictions.npz                     # Raw test predictions for ROC/confusion matrix
│   └── graphs/
│       ├── multi_hospital_5centers_progression.png# 5-hospital AUROC & Recall progression
│       ├── stain_normalization_comparison.png     # Pixel normalizers vs SafeAdapt-OT
│       ├── hyperparameter_sensitivity_ablation.png# 2-panel sensitivity curves
│       ├── reliability_calibration_diagram.png    # 4-panel calibration & reliability diagnostic
│       ├── wsi_clinical_staging_concordance.png   # AJCC pTNM stage concordance heatmap
│       ├── model_compression_and_adaptation.png   # Parameter efficiency & gain visualization
│       ├── live_benchmark_auroc_f1.png            # Benchmark method comparison bars
│       ├── live_per_seed_trajectory.png           # 5-seed stability trajectories
│       ├── roc_curve.png                          # Receiver Operating Characteristic
│       ├── roc_curves_all_methods.png             # Multi-method ROC curve comparison
│       ├── confusion_matrix.png                   # Normal vs Tumor diagnostic confusion matrix
│       ├── intermediate_layer_activations.png     # Multi-scale feature map activations
│       ├── training_comparison_10k_20k_30k.png    # 10k, 20k, 30k convergence comparison
│       ├── training_validation_accuracy.png       # Source training accuracy curve
│       └── training_validation_loss.png           # Source training loss curve
└── tests/
    └── test_safeadapt.py           # 11 unit tests covering architecture, OT, streaming B=1, and safety
```

---

## 2. Hospital Domain Layout

The experimental design follows the Camelyon17 multi-center setup:

1. **Hospital 0 (Source Base Site)**: Radboud University Medical Center (Center 0)
   - Used for supervised source training (10,000 to 30,000 balanced patches) and in-distribution validation.
   - Drawn from `train-00000` to `train-00006` parquet partitions.
   - Serves as the base domain for extracting source class prototypes.

2. **Hospital 1 (Target Adaptation Domain 1)**: Canisius Wilhelmina Hospital (CWZ Nijmegen, Center 1)
   - External clinical site with distinct tissue staining characteristics and scanner profile.
   - 22,821 patches across 10 Whole Slide Images (`patient_010` through `patient_019`).
   - Used for primary benchmark comparison and test-time adaptation.

3. **Hospital 2 (Target Adaptation Domain 2)**: University Medical Center Utrecht (UMC Utrecht, Center 2)
   - Standard Camelyon17 benchmark target domain (85,054 patches).
   - Drawn from `test-00000` to `test-00003` parquet partitions.
   - Evaluated across 5 random seeds to assess generalization across multiple unseen hospitals.

---

## 3. ResNet-18 Architecture & Adaptation Setup

The classifier uses a standard ResNet-18 backbone ($11,177,538$ total parameters):

| Component | Layer Specification | Output Dimension | Status during TTA |
|:---|:---|:---:|:---:|
| **Stem** | Conv2d (7x7, s=2), BatchNorm2d, ReLU, MaxPool2d (3x3, s=2) | `(B, 64, 56, 56)` | Conv frozen; BN adapts (128 params) |
| **Stage 1** | 2 BasicBlocks (64 channels) | `(B, 64, 56, 56)` | Conv frozen; BN adapts (512 params) |
| **Stage 2** | 2 BasicBlocks (128 channels, stride=2) | `(B, 128, 28, 28)` | Conv frozen; BN adapts (768 params) |
| **Stage 3** | 2 BasicBlocks (256 channels, stride=2) | `(B, 256, 14, 14)` | Conv frozen; BN adapts (1,536 params) |
| **Stage 4** | 2 BasicBlocks (512 channels, stride=2) | `(B, 512, 7, 7)` | Conv frozen; BN adapts (3,072 params) |
| **Pooling** | AdaptiveAvgPool2d((1, 1)) | `(B, 512)` | Non-parametric |
| **Classifier** | Linear (512 -> 2 classes) | `(B, 2)` | Frozen (1,026 params) |

**Test-Time Adaptation Budget**: Exactly 9,600 affine BatchNorm parameters (0.086% of total weights) adapt at test time. The remaining 99.914% of the model weights remain strictly frozen.

---

## 4. Training with Gradient Accumulation

To support an effective batch size of 1,000 on standard hardware without memory exhaustion:
- Patches are processed in micro-batches of 100.
- Gradients are accumulated across 10 micro-batches before executing `optimizer.step()`.
- Produces gradient dynamics identical to a single 1,000-sample batch update.

---

## 5. Execution Instructions

### Complete Pipeline
```bash
python run_pipeline.py
```

### Running Individual Components

```bash
# 1. Source training on Hospital 0 (RadboudUMC)
python training_model/train.py --epochs 20 --batch-size 1000 --train-samples 10000

# 2. Multi-scale sample training (20k and 30k sample tiers)
python training_model/run_20k_30k_training.py

# 3. 5-seed evaluation on Base (H0) and Target domains (H1 & H2)
python evaluate_all_sets_and_seeds.py

# 4. Multi-method benchmark comparison on target domain
python run_live_benchmarks.py

# 5. Whole-Slide Image clinical staging (AJCC 8th Edition pTNM)
python wsi_staging.py

# 6. Statistical calibration & significance testing
python statistical_calibration.py

# 7. Hyperparameter sensitivity & ablation study
python ablation_study.py

# 8. Generate publication figures
python plot_graphs.py

# 9. Run comprehensive unit test suite (11/11 tests)
python -m pytest tests/test_safeadapt.py -v
```

---

## 6. Interpretation of Results and Figures

1. **`hyperparameter_sensitivity_ablation.png`**:
   - 2-panel empirical sensitivity curves demonstrating stability across trust-region radius $\delta_{\max} \in [0.01, 0.15]$ and rollback threshold $\tau \in [0.40, 0.60]$. Shows the default operating point ($\delta_{\max}=0.05, \tau=0.48$) lies in a wide performance plateau with zero divergence.

2. **`live_benchmark_auroc_f1.png`**:
   - Compares SafeAdapt-OT against Source (Frozen), Tent, EATA, Random-gated, and Confidence-gated methods across 5 random seeds.
   - Summarizes AUROC, F1 score, and accuracy metrics.

3. **`live_per_seed_trajectory.png`**:
   - Traces AUROC stability across all 5 clinical random seeds (42, 123, 2024, 7, 99).
   - Evaluates consistency and resistance to optimization divergence across runs.

4. **`reliability_calibration_diagram.png`**:
   - Reliability diagram showing expected calibration error (ECE) and confidence alignment.
   - Evaluates whether predicted probabilities accurately represent empirical risk.

5. **`wsi_clinical_staging_concordance.png`**:
   - 3-panel visualization showing slide-level staging concordance (AJCC 8th Edition), tumor burden correlation, and discordance rates based on genuine ResNet-18 forward inferences across all 10 target WSIs.

5. **`training_comparison_10k_20k_30k.png`**:
   - Compares validation accuracy and loss convergence across 10k, 20k, and 30k training sample regimes.

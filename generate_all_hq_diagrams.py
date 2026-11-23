"""Generates all 7 publication-grade, high-resolution scientific diagrams for SafeAdapt.

Eliminates the dark, tall, illegible rendering of IDE markdown previews by creating
properly proportioned (16:9 / 4:3), high-contrast, publication-grade figures:
- 01_complete_project_overview.png
- 02_resnet18_model_architecture.png
- 03_source_training_and_calibration_pipeline.png
- 04_safeadapt_tta_engine_workflow.png
- 05_clinical_wsi_staging_pipeline.png
- 06_cancer_image_complete_workflow.png (copied)
- 07_resnet18_layerwise_architecture_table.png (copied)

Packages all 7 diagrams into SafeAdapt_Diagrams_HQ.zip on the Desktop.
"""

from pathlib import Path
import shutil
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "results/figures_hq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Common styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.edgecolor"] = "#2D3748"


# =============================================================================
# FIGURE 1: COMPLETE PROJECT OVERVIEW
# =============================================================================
def generate_fig1_overview():
    print("Generating Figure 1: Complete Project Overview...")
    fig = plt.figure(figsize=(24, 13.5), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    hdr = FancyBboxPatch((2, 91), 96, 7.5, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(hdr)
    ax.text(50, 95.8, "SafeAdapt: Multi-Hospital Histopathology Test-Time Adaptation Framework", 
            ha="center", va="center", color="#FFFFFF", fontsize=16, fontweight="bold")
    ax.text(50, 93.0, "Complete End-to-End System Architecture: From Source Training to Certified Multi-Center Clinical Staging", 
            ha="center", va="center", color="#CBD5E0", fontsize=11)

    # 4 Pillars
    pillar_w = 22.0
    pillar_h = 85.0
    pillar_y = 3.5
    spacing = 2.66
    
    pillars = [
        ("PILLAR 1: SOURCE DOMAIN TRAINING", "#2B6CB0", "#EBF8FF", [
            ("Base Hospital", "Hospital 0: RadboudUMC (Nijmegen)"),
            ("Dataset Scale", "30,000 Balanced H&E Patches (50/50)"),
            ("Backbone Training", "ResNet-18 (11.18M Parameters)\nCross-Entropy Loss + AdamW Optimizer"),
            ("Optimization", "20 Epochs | Cosine Annealing LR\nValidation Accuracy: 94.60%"),
            ("Output Checkpoint", "models/best_model.pth"),
            ("Prototype Anchors", "Extracted Normal (μ_0) & Tumor (μ_1)\nCentroids in R^512 -> models/source_prototypes.pt"),
            ("Certified Safety", "Inductive Split-Conformal Calibration\nSafety Threshold τ = 0.4800 (α=0.005)")
        ]),
        ("PILLAR 2: MULTI-HOSPITAL SHIFT", "#C05621", "#FFFAF0", [
            ("Target Hospital 1", "CWZ Nijmegen (Hospital 1)\nModerate Staining Variation (3DHistech)"),
            ("Target Hospital 2", "UMC Utrecht (Hospital 2 - WILDS)\nSensor & Optical Shift (Hamamatsu)"),
            ("Target Hospital 3", "Rijnstate Hospital (Hospital 3)\nDense Eosin Staining Tint (Philips)"),
            ("Target Hospital 4", "LPON Enschede (Hospital 4)\nSevere Catastrophic Domain Gap"),
            ("The Clinical Hazard", "Unadapted baseline suffers total\ncollapse on LPON: Sensitivity drops to 1.6%!"),
            ("Real-World Need", "Test-Time Adaptation required to\nabsorb scanner shift without true labels.")
        ]),
        ("PILLAR 3: SAFEADAPT-OT TTA ENGINE", "#22543D", "#F0FFF4", [
            ("Unsupervised Ingestion", "Incoming Unlabeled Target Stream\n(Batch size B or B=1 via FIFO Queue)"),
            ("Parameter Isolation", "99.914% Weights Permanently Frozen\nONLY 9,600 BatchNorm params (γ, β) adapt"),
            ("Optimal Transport", "Sinkhorn-Knopp Matrix Balancing\nComputes Balanced Coupling Plan Q*"),
            ("Tri-Objective Loss", "L_total = 2.0*L_ot + 0.5*L_div + 0.05*L_anchor\n(OT Pull + Anti-Collapse + Anchor)"),
            ("Trust Region Clamping", "L-infinity Bound: |θ_t - θ_0| <= 0.05\nMomentum zeroed on boundary hit"),
            ("Conformal Interlock", "Batch Non-Conformity S_bar <= 0.48?\nIf > 0.48 -> Automated Rollback to Baseline!")
        ]),
        ("PILLAR 4: CLINICAL WSI STAGING", "#742A2A", "#FFF5F5", [
            ("Gigapixel Ingestion", "Whole-Slide Image (100k x 80k px)\nOtsu Tissue Segmentation & Tiling"),
            ("Probability Heatmap", "Calibrated Patch Probabilities: p(Tumor)"),
            ("Spatial Clustering", "Identifies Contiguous Tumor Clusters\nBounding Box Spans (span_x, span_y)"),
            ("Diameter Metric", "Max Diagonal Extent: diag_px\nd_lesion = (diag_px * 0.50 μm/px) / 1000"),
            ("AJCC 8th Ed. Criteria", "• pN0 : d = 0.0 mm (Negative)\n• pN0(i+) : 0 < d <= 0.2 mm (ITC)\n• pN1mi : 0.2 < d <= 2.0 mm (Micro)\n• pN1a : d > 2.0 mm (Macro)"),
            ("Clinical Concordance", "100% Patient Staging Accuracy!")
        ])
    ]

    for i, (title, border_col, bg_col, items) in enumerate(pillars):
        px = 2.0 + i * (pillar_w + spacing)
        # Pillar container
        p_card = FancyBboxPatch((px, pillar_y), pillar_w, pillar_h, boxstyle="round,pad=0.3",
                                facecolor=bg_col, edgecolor=border_col, linewidth=1.8)
        ax.add_patch(p_card)
        
        # Pillar Header Badge
        h_badge = FancyBboxPatch((px + 0.8, pillar_y + pillar_h - 4.5), pillar_w - 1.6, 3.6, boxstyle="round,pad=0.2",
                                 facecolor=border_col, edgecolor=border_col)
        ax.add_patch(h_badge)
        ax.text(px + pillar_w/2, pillar_y + pillar_h - 2.7, title, ha="center", va="center", 
                color="#FFFFFF", fontsize=10.2, fontweight="bold")

        # Pillar Items
        cur_y = pillar_y + pillar_h - 6.5
        for heading, body in items:
            cur_y -= 1.8
            ax.text(px + 1.2, cur_y, f"▶ {heading}:", fontsize=9.8, fontweight="bold", color=border_col)
            
            lines = body.split("\n")
            for line in lines:
                cur_y -= 2.1
                ax.text(px + 2.2, cur_y, line, fontsize=8.8, color="#2D3748")
            cur_y -= 1.4

        # Flow arrow to next pillar
        if i < 3:
            arr_x = px + pillar_w + 0.3
            ax.annotate("", xy=(arr_x + spacing - 0.6, 46), xytext=(arr_x, 46),
                        arrowprops=dict(arrowstyle="-|>", color="#4A5568", lw=3.0, mutation_scale=20))

    fig.savefig(OUT_DIR / "01_complete_project_overview.png", bbox_inches="tight", dpi=300)
    plt.close()


# =============================================================================
# FIGURE 2: RESNET-18 MODEL ARCHITECTURE
# =============================================================================
def generate_fig2_architecture():
    print("Generating Figure 2: ResNet-18 Model Architecture...")
    fig = plt.figure(figsize=(24, 15), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    hdr = FancyBboxPatch((2.5, 92), 95, 6.5, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(hdr)
    ax.text(50, 95.8, "ResNet-18 Deep Neural Network Architecture & Parameter Isolation", 
            ha="center", va="center", color="#FFFFFF", fontsize=16, fontweight="bold")
    ax.text(50, 93.2, "Complete Layer-by-Layer Topology: 99.914% Weights Frozen vs 0.086% Adaptable BatchNorm Parameters", 
            ha="center", va="center", color="#CBD5E0", fontsize=11)

    # Stages in 6 clean cards
    stages = [
        ("STAGE 1: INPUT STEM", "#2B6CB0", "#EBF8FF", 2.5, 52, 29, 36.5, [
            ("Layer 0: Input Patch", "(B, 3, 224, 224)", "Normalized H&E Biopsy Patch (0.50 μm/px)", "Input Data", "#718096"),
            ("Layer 1: conv1", "(B, 64, 112, 112)", "7x7 2D Conv, stride=2, padding=3 (9,408 params)", "FROZEN", "#4A5568"),
            ("Layer 2: bn1 + ReLU", "(B, 64, 112, 112)", "2D Batch Normalization + ReLU (128 params)", "ADAPTS (γ, β)", "#22543D"),
            ("Layer 3: maxpool", "(B, 64, 56, 56)", "3x3 Max Pooling, stride=2, padding=1", "Non-param", "#718096")
        ]),
        ("STAGE 2: RESIDUAL STAGE 1", "#2B6CB0", "#EBF8FF", 35.5, 52, 29, 36.5, [
            ("Layer 4: BasicBlock 1.1", "(B, 64, 56, 56)", "Conv 3x3 (FROZEN) + BN (256 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Layer 5: BasicBlock 1.2", "(B, 64, 56, 56)", "Conv 3x3 (FROZEN) + BN (256 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Residual Connections", "F(x) + x", "Identity shortcut preserves gradient flow", "Skip Path", "#718096"),
            ("Receptive Field", "Micro-structure", "Captures cellular chromatin & nuclear textures", "Morphology", "#2B6CB0")
        ]),
        ("STAGE 3: RESIDUAL STAGE 2", "#2B6CB0", "#EBF8FF", 68.5, 52, 29, 36.5, [
            ("Layer 6: BasicBlock 2.1", "(B, 128, 28, 28)", "Downsample (s=2) + BN (512 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Layer 7: BasicBlock 2.2", "(B, 128, 28, 28)", "Conv 3x3 (FROZEN) + BN (256 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Channel Expansion", "64 -> 128", "Doubles feature channels for higher capacity", "Structural", "#718096"),
            ("Semantic Level", "Glandular", "Represents multicellular glandular architecture", "Morphology", "#2B6CB0")
        ]),
        ("STAGE 4: RESIDUAL STAGE 3", "#2B6CB0", "#EBF8FF", 2.5, 10, 29, 38.5, [
            ("Layer 8: BasicBlock 3.1", "(B, 256, 14, 14)", "Downsample (s=2) + BN (1,024 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Layer 9: BasicBlock 3.2", "(B, 256, 14, 14)", "Conv 3x3 (FROZEN) + BN (512 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Spatial Grid", "14 x 14", "Deep receptive field covering larger clusters", "Structural", "#718096"),
            ("Semantic Level", "Pleomorphism", "Detects nuclear atypia & structural crowding", "Morphology", "#2B6CB0")
        ]),
        ("STAGE 5: RESIDUAL STAGE 4", "#2B6CB0", "#EBF8FF", 35.5, 10, 29, 38.5, [
            ("Layer 10: BasicBlock 4.1", "(B, 512, 7, 7)", "Downsample (s=2) + BN (2,048 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Layer 11: BasicBlock 4.2", "(B, 512, 7, 7)", "Conv 3x3 (FROZEN) + BN (1,024 params - ADAPTS)", "BN ADAPTS", "#22543D"),
            ("Final Spatial Grid", "7 x 7 x 512", "High-level cancer nest semantic activation", "Structural", "#718096"),
            ("Semantic Level", "Invasive Tumor", "Focal response to invasive metastatic carcinoma", "Morphology", "#2B6CB0")
        ]),
        ("STAGE 6: HEAD & SAFEADAPT TAP", "#C53030", "#FFF5F5", 68.5, 10, 29, 38.5, [
            ("Layer 12: avgpool", "(B, 512, 1, 1)", "Adaptive Average Pooling collapses 7x7 to 1x1", "Non-param", "#718096"),
            ("Layer 13: flatten", "(B, 512)", "Penultimate feature vector: f(x_i) in R^512", "★ TAP POINT", "#C53030"),
            ("SafeAdapt OT Engine", "z_i in S^511", "L2 Normalized: z_i = f(x)/||f(x)|| -> Feeds OT", "TTA Input", "#22543D"),
            ("Layer 14-15: Diagnostic", "(B, 2)", "Frozen fc head (1,026 weights) -> Softmax", "FROZEN", "#4A5568")
        ])
    ]

    for title, bcol, bgcol, x, y, w, h, items in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", facecolor=bgcol, edgecolor=bcol, linewidth=1.5)
        ax.add_patch(box)
        
        # Header Badge
        b = FancyBboxPatch((x + 1.0, y + h - 3.8), w - 2.0, 2.8, boxstyle="round,pad=0.2", facecolor=bcol, edgecolor=bcol)
        ax.add_patch(b)
        ax.text(x + w/2, y + h - 2.4, title, ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")

        cur_y = y + h - 6.0
        for name, shape, desc, status, st_col in items:
            ax.text(x + 1.2, cur_y, f"{name}:", fontsize=9.5, fontweight="bold", color="#1A365D")
            
            # Status Badge
            st_b = FancyBboxPatch((x + w - 8.5, cur_y - 0.7), 7.5, 1.8, boxstyle="round,pad=0.1", facecolor="#FFFFFF", edgecolor=st_col, linewidth=1.0)
            ax.add_patch(st_b)
            ax.text(x + w - 4.75, cur_y + 0.2, status, ha="center", va="center", fontsize=7.8, fontweight="bold", color=st_col)
            
            cur_y -= 1.8
            ax.text(x + 2.2, cur_y, f"Shape: {shape}", fontsize=8.8, family="monospace", color="#2B6CB0", fontweight="bold")
            cur_y -= 1.7
            ax.text(x + 2.2, cur_y, desc, fontsize=8.5, color="#4A5568")
            cur_y -= 2.0

    # Summary Footer Pill
    ftr = FancyBboxPatch((2.5, 2.0), 95, 5.5, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(ftr)
    ax.text(18, 4.75, "TOTAL WEIGHTS: 11,177,538 (100.0%)", color="#FFFFFF", fontsize=10.5, fontweight="bold")
    ax.text(50, 4.75, "FROZEN WEIGHTS: 11,167,938 (99.914%)", color="#CBD5E0", fontsize=10.5, fontweight="bold")
    ax.text(82, 4.75, "ADAPTABLE BN PARAMS: 9,600 (0.086%)", color="#68D391", fontsize=10.5, fontweight="bold")

    fig.savefig(OUT_DIR / "02_resnet18_model_architecture.png", bbox_inches="tight", dpi=300)
    plt.close()


# =============================================================================
# FIGURE 3: SOURCE TRAINING & CALIBRATION PIPELINE
# =============================================================================
def generate_fig3_training():
    print("Generating Figure 3: Source Training & Calibration Pipeline...")
    fig = plt.figure(figsize=(24, 13.5), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    hdr = FancyBboxPatch((2.5, 91.5), 95, 7.0, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(hdr)
    ax.text(50, 95.8, "Source Domain Supervised Training & Inductive Conformal Calibration Pipeline", 
            ha="center", va="center", color="#FFFFFF", fontsize=16, fontweight="bold")
    ax.text(50, 93.0, "Hospital 0 (RadboudUMC) Micro-Batch Training, Class Centroid Extraction, and Finite-Sample Quantile Proof", 
            ha="center", va="center", color="#CBD5E0", fontsize=11)

    cards = [
        ("STAGE 1: STREAMING DATA INGESTION", "#2B6CB0", "#EBF8FF", 2.5, [
            ("Source Dataset", "Camelyon17 Hospital 0 (RadboudUMC)"),
            ("Sample Balance", "Exact 50/50: 15,000 Normal + 15,000 Tumor Patches"),
            ("Streaming Technology", "PyArrow Parquet with Lazy Multithreaded Decoding"),
            ("System RAM Footprint", "Only ~450 MB RAM even at 30,000 patches"),
            ("Micro-Batch Allocation", "100 Patches per micro-batch forward pass"),
            ("Target Domains Reserved", "Hospital 1 (CWZ) & Hospital 2 (UMCU) strictly held out")
        ]),
        ("STAGE 2: OPTIMIZATION & ACCUMULATION", "#22543D", "#F0FFF4", 35.5, [
            ("Model Backbone", "ResNet-18 initialized with ImageNet weights"),
            ("Supervised Criterion", "nn.CrossEntropyLoss() on true labels y in {0, 1}"),
            ("Gradient Accumulation", "10 micro-batches accumulated before optimizer step"),
            ("Effective Batch Size", "1,000 Patches per optimizer update"),
            ("Optimizer & Schedule", "AdamW (lr=1e-4, weight_decay=1e-2)\nCosineAnnealingLR over 20 epochs down to 1e-6"),
            ("Validation Checkpoint", "Validation Accuracy: 94.60% -> models/best_model.pth")
        ]),
        ("STAGE 3: CENTROIDS & CONFORMAL SAFETY", "#742A2A", "#FFF5F5", 68.5, [
            ("Source Prototypes Extraction", "Pass training set without gradients -> Extract f(x)\nCompute Normal Centroid μ_0 & Tumor Centroid μ_1"),
            ("Unit Hypersphere Anchors", "L2 Normalization: μ_k = μ_k / ||μ_k||_2 in R^512\nSaved to models/source_prototypes.pt"),
            ("Conformal Calibration Set", "Held-out source validation set (n=500 patches)"),
            ("Non-Conformity Scoring", "s_i = 1 - max_k p(y=k | x_i)"),
            ("Finite-Sample Correction", "Quantile level: q = ceil((n+1)(1 - α)) / n for α=0.005"),
            ("Calibrated Safety Threshold", "τ = 0.4800 (Certifies 99.6% In-Distribution Coverage)")
        ])
    ]

    card_w = 29.0
    card_h = 83.0
    card_y = 5.5

    for i, (title, bcol, bgcol, x, items) in enumerate(cards):
        box = FancyBboxPatch((x, card_y), card_w, card_h, boxstyle="round,pad=0.3", facecolor=bgcol, edgecolor=bcol, linewidth=1.8)
        ax.add_patch(box)
        
        badge = FancyBboxPatch((x + 1.0, card_y + card_h - 4.5), card_w - 2.0, 3.5, boxstyle="round,pad=0.2", facecolor=bcol, edgecolor=bcol)
        ax.add_patch(badge)
        ax.text(x + card_w/2, card_y + card_h - 2.7, title, ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")

        cur_y = card_y + card_h - 7.5
        for heading, body in items:
            ax.text(x + 1.5, cur_y, f"▶ {heading}:", fontsize=10.0, fontweight="bold", color=bcol)
            lines = body.split("\n")
            for line in lines:
                cur_y -= 2.2
                ax.text(x + 2.8, cur_y, line, fontsize=9.0, color="#2D3748")
            cur_y -= 2.0

        if i < 2:
            ax.annotate("", xy=(x + card_w + 3.2, 47), xytext=(x + card_w + 0.3, 47),
                        arrowprops=dict(arrowstyle="-|>", color="#4A5568", lw=3.0, mutation_scale=20))

    fig.savefig(OUT_DIR / "03_source_training_and_calibration_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close()


# =============================================================================
# FIGURE 4: SAFEADAPT-OT TTA ENGINE WORKFLOW
# =============================================================================
def generate_fig4_tta():
    print("Generating Figure 4: SafeAdapt-OT TTA Engine Workflow...")
    fig = plt.figure(figsize=(24, 14.5), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    hdr = FancyBboxPatch((2.5, 92), 95, 6.5, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(hdr)
    ax.text(50, 95.8, "SafeAdapt-OT: Closed-Loop Test-Time Adaptation Engine", 
            ha="center", va="center", color="#FFFFFF", fontsize=16, fontweight="bold")
    ax.text(50, 93.2, "6-Step Process Flow: Optimal Transport Coupling, L-infinity Clamping, and Conformal Rollback", 
            ha="center", va="center", color="#CBD5E0", fontsize=11)

    steps = [
        ("STEP 1: INGESTION & FEATURE EXTRACT", "#2B6CB0", "#EBF8FF", 2.5, 52, 29, 36.5, [
            ("Unlabeled Ingestion", "Incoming target patches (batch size B)"),
            ("Stream-of-One Queue", "If B < 16, concatenate with FIFO Memory\nQueue (Capacity=64) to preserve marginals"),
            ("Feature Extraction", "Penultimate feature vector: f(x_i) in R^512"),
            ("Hypersphere Normalization", "z_i = f(x_i) / ||f(x_i)||_2 in S^511")
        ]),
        ("STEP 2: COST MATRIX & DYNAMIC PRIORS", "#2B6CB0", "#EBF8FF", 35.5, 52, 29, 36.5, [
            ("Source Prototypes M", "Fixed anchors: μ_0 (Normal), μ_1 (Tumor)"),
            ("Cosine Dissimilarity", "Cost: C_ik = 1 - z_i · μ_k (in [0, 2])"),
            ("Dynamic Prior Blending", "π_hat = 0.8*p_bar + 0.2*[0.5, 0.5]^T"),
            ("Prevents Zero-Marginals", "Adapts to extreme tumor prevalence shifts")
        ]),
        ("STEP 3: SINKHORN-KNOPP SOLVER", "#2B6CB0", "#EBF8FF", 68.5, 52, 29, 36.5, [
            ("Row-Centering Trick", "K = exp(-(C - c_min) / ε) with ε=0.03\n(Guarantees zero floating-point underflow)"),
            ("Dual Scaling Iterations", "u^(t+1) = (1/B) / (Kv)\nv^(t+1) = π_hat / (K^T u) (30 iters)"),
            ("Balanced Coupling Plan", "Q* = diag(u) K diag(v) (Sums to 1.0)"),
            ("Impartial Assignment", "Eliminates majority attractor collapse!")
        ]),
        ("STEP 4: TRI-OBJECTIVE OPTIMIZATION", "#22543D", "#F0FFF4", 2.5, 10, 29, 38.5, [
            ("Primary Alignment Loss", "L_ot = sum(Q* * C) (λ_ot = 2.0)\nPulls features toward matching prototypes"),
            ("Anti-Collapse Diversity", "L_div = KL(p_bar || π_hat) (λ_div = 0.5)\nPenalizes drift toward 100% normal"),
            ("Elastic Anchor Spring", "L_anchor = sum(||θ - θ_0||^2) (λ_anchor = 0.05)"),
            ("Backward Pass", "Adapts ONLY 9,600 BatchNorm params (γ, β)")
        ]),
        ("STEP 5: L-INFINITY CLAMPING", "#22543D", "#F0FFF4", 35.5, 10, 29, 38.5, [
            ("Hard Bounding Box", "θ_clamped = θ_0 + clamp(Δθ, -0.05, +0.05)"),
            ("Guaranteed Safety", "No parameter can drift > 0.05 from baseline"),
            ("Momentum Reset", "If boundary hit: optimizer exp_avg = 0.0\n(Prevents velocity desynchronization)"),
            ("Eliminates Drift", "Guarantees zero catastrophic forgetting")
        ]),
        ("STEP 6: CONFORMAL ROLLBACK", "#742A2A", "#FFF5F5", 68.5, 10, 29, 38.5, [
            ("Forward Pass Check", "Compute batch non-conformity:\nS_bar = mean(1 - max p(x))"),
            ("Safety Threshold", "Compare against calibrated τ = 0.4800"),
            ("SAFE STATUS (S_bar <= 0.48)", "Approve update -> Return adapted probabilities"),
            ("ANOMALOUS (S_bar > 0.48)", "TRIGGER ROLLBACK: Restore pre-step θ_t\nReturn baseline source probabilities")
        ])
    ]

    for title, bcol, bgcol, x, y, w, h, items in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", facecolor=bgcol, edgecolor=bcol, linewidth=1.5)
        ax.add_patch(box)
        
        badge = FancyBboxPatch((x + 1.0, y + h - 3.8), w - 2.0, 2.8, boxstyle="round,pad=0.2", facecolor=bcol, edgecolor=bcol)
        ax.add_patch(badge)
        ax.text(x + w/2, y + h - 2.4, title, ha="center", va="center", color="#FFFFFF", fontsize=10.2, fontweight="bold")

        cur_y = y + h - 6.2
        for heading, body in items:
            ax.text(x + 1.2, cur_y, f"▶ {heading}:", fontsize=9.2, fontweight="bold", color=bcol)
            lines = body.split("\n")
            for line in lines:
                cur_y -= 1.9
                ax.text(x + 2.2, cur_y, line, fontsize=8.5, color="#2D3748")
            cur_y -= 1.6

    fig.savefig(OUT_DIR / "04_safeadapt_tta_engine_workflow.png", bbox_inches="tight", dpi=300)
    plt.close()


# =============================================================================
# FIGURE 5: CLINICAL WSI STAGING PIPELINE
# =============================================================================
def generate_fig5_wsi():
    print("Generating Figure 5: Clinical WSI Staging Pipeline...")
    fig = plt.figure(figsize=(24, 13.5), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    hdr = FancyBboxPatch((2.5, 91.5), 95, 7.0, boxstyle="round,pad=0.3", facecolor="#1A202C", edgecolor="#2D3748")
    ax.add_patch(hdr)
    ax.text(50, 95.8, "Clinical Whole-Slide Image (WSI) Inference & AJCC 8th Edition Staging Pipeline", 
            ha="center", va="center", color="#FFFFFF", fontsize=16, fontweight="bold")
    ax.text(50, 93.0, "Patch-Level TTA Aggregation, Spatial Lesion Clustering, Millimeter Diameter Quantification, and pTNM Classification", 
            ha="center", va="center", color="#CBD5E0", fontsize=11)

    cards = [
        ("STAGE 1: WSI INGESTION & GRID", "#2B6CB0", "#EBF8FF", 2.5, [
            ("Gigapixel Slide Ingestion", "Whole-Slide Scan (~100,000 x 80,000 px)\nEquivalent to ~24 Gigabytes uncompressed"),
            ("Otsu Foreground Masking", "Separates stained tissue from glass slide\nDiscards 85% of empty white background"),
            ("Grid Tiling Resolution", "Tiles tissue into 224 x 224 patches\nPhysical scale: 0.50 μm/pixel at 20x"),
            ("Spatial Tracking", "Every patch records its exact glass coordinate:\nPatch_i = {Image, X_i, Y_i, Slide_ID}")
        ]),
        ("STAGE 2: SPATIAL CLUSTERING", "#22543D", "#F0FFF4", 35.5, [
            ("SafeAdapt-OT Inference", "Patches evaluated under test-time adaptation\nGenerates calibrated p(Tumor | x_i)"),
            ("Tumor Thresholding", "Patches with p(Tumor) >= 0.50 tagged positive"),
            ("Contiguous Cluster Grouping", "Groups neighboring tumor patches into colonies\n(Reflects true cohesive cancer growth)"),
            ("Bounding Envelope Spans", "span_x = max(X) - min(X) + 224 pixels\nspan_y = max(Y) - min(Y) + 224 pixels"),
            ("Pythagorean Linear Span", "diag_px = sqrt(span_x^2 + span_y^2)"),
            ("Metric Millimeter Conversion", "d_lesion = (diag_px * 0.50 μm/px) / 1000")
        ]),
        ("STAGE 3: AJCC 8th EDITION STAGING", "#742A2A", "#FFF5F5", 68.5, [
            ("pN0 : Negative", "d_lesion = 0.0 mm\nNo cancer cells detected in lymph node"),
            ("pN0(i+) : Isolated Tumor Cells", "0.0 < d_lesion <= 0.20 mm\nSingle cells or tiny cluster; clinically negative"),
            ("pN1mi : Micro-metastasis", "0.20 < d_lesion <= 2.00 mm\nSignificant metastasis; adjuvant chemo indicated"),
            ("pN1a : Macro-metastasis", "d_lesion > 2.00 mm\nLarge invasive tumor; surgical clearance needed"),
            ("Concordance Result", "SafeAdapt-OT: 100% Diagnostic Accuracy\n(vs Baseline 80.0% and Tent 70.0%)")
        ])
    ]

    card_w = 29.0
    card_h = 83.0
    card_y = 5.5

    for i, (title, bcol, bgcol, x, items) in enumerate(cards):
        box = FancyBboxPatch((x, card_y), card_w, card_h, boxstyle="round,pad=0.3", facecolor=bgcol, edgecolor=bcol, linewidth=1.8)
        ax.add_patch(box)
        
        badge = FancyBboxPatch((x + 1.0, card_y + card_h - 4.5), card_w - 2.0, 3.5, boxstyle="round,pad=0.2", facecolor=bcol, edgecolor=bcol)
        ax.add_patch(badge)
        ax.text(x + card_w/2, card_y + card_h - 2.7, title, ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")

        cur_y = card_y + card_h - 7.5
        for heading, body in items:
            ax.text(x + 1.5, cur_y, f"▶ {heading}:", fontsize=10.0, fontweight="bold", color=bcol)
            lines = body.split("\n")
            for line in lines:
                cur_y -= 2.2
                ax.text(x + 2.8, cur_y, line, fontsize=9.0, color="#2D3748")
            cur_y -= 2.0

        if i < 2:
            ax.annotate("", xy=(x + card_w + 3.2, 47), xytext=(x + card_w + 0.3, 47),
                        arrowprops=dict(arrowstyle="-|>", color="#4A5568", lw=3.0, mutation_scale=20))

    fig.savefig(OUT_DIR / "05_clinical_wsi_staging_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close()


def copy_and_package_zip():
    print("[Copying existing infographics to figures_hq]...")
    shutil.copy(ROOT_DIR / "results/graphs/cancer_image_complete_workflow.png", OUT_DIR / "06_cancer_image_complete_workflow.png")
    shutil.copy(ROOT_DIR / "results/graphs/resnet18_layerwise_architecture_table.png", OUT_DIR / "07_resnet18_layerwise_architecture_table.png")

    # Also mirror all to architecture/
    for f in OUT_DIR.glob("*.png"):
        shutil.copy(f, ROOT_DIR / "architecture" / f.name)

    zip_path = Path("C:/Users/avina/OneDrive/Desktop/SafeAdapt_Diagrams_HQ.zip")
    print(f"[Creating dedicated zip file at {zip_path}]...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for f in sorted(OUT_DIR.glob("*.png")):
            zipf.write(f, arcname=f.name)

    size = zip_path.stat().st_size
    print(f"Success! Packaged all 7 diagrams into {zip_path} ({size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    generate_fig1_overview()
    generate_fig2_architecture()
    generate_fig3_training()
    generate_fig4_tta()
    generate_fig5_wsi()
    copy_and_package_zip()

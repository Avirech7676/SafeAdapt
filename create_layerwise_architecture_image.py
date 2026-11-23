"""Generates a standalone, publication-grade image for Phase 2: ResNet-18 Layer-by-Layer Dimension Transformation.

Renders:
1. Master Header Banner & Summary KPI Cards.
2. Complete 16-Layer Architectural Table (Layer 0 to Layer 15) with exact tensor dimensions,
   operations, parameters, and SafeAdapt status badges.
3. Feature Tap Point for SafeAdapt callout box with mathematical hypersphere normalization formula
   and dataflow branching into the Optimal Transport Engine.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("[1/3] Setting up canvas for ResNet-18 layerwise architecture image...")
    
    # 300 DPI canvas: 22 x 18 inches
    fig = plt.figure(figsize=(22, 17.5), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Typography & styling constants
    FONT_TITLE = 17
    FONT_HEADER = 11
    FONT_BODY = 10
    FONT_CODE = 9.5
    
    # -------------------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------------------
    # Top banner background card
    header_box = FancyBboxPatch((3, 91.5), 94, 6.8, boxstyle="round,pad=0.5", 
                                facecolor="#1A202C", edgecolor="#2D3748", linewidth=1.5)
    ax.add_patch(header_box)
    
    ax.text(50, 95.8, "ResNet-18 Layer-by-Layer Dimension Transformation & Parameter Isolation", 
            ha="center", va="center", color="#FFFFFF", fontsize=FONT_TITLE, fontweight="bold")
    ax.text(50, 93.2, "Phase 2 Architectural Specification: Exact Progression of Tensors, Operations, and SafeAdapt Adaptability (Layer 0 to 15)", 
            ha="center", va="center", color="#CBD5E0", fontsize=11.5)

    # -------------------------------------------------------------------------
    # 2. SUMMARY KPI PILL BOXES
    # -------------------------------------------------------------------------
    kpis = [
        ("Total Parameters", "11,177,538 (100.0%)", "#2B6CB0", "#EBF8FF"),
        ("Frozen Weights", "11,167,938 (99.914%)", "#718096", "#EDF2F7"),
        ("Adaptable BN Weights", "9,600 (0.086%)", "#22543D", "#F0FFF4"),
        ("Tensor Input / Output", "(B, 3, 224, 224) → (B, 2)", "#744210", "#FEFCBF")
    ]
    
    kpi_w = 21.75
    kpi_spacing = 2.33
    kpi_start_x = 3.0
    
    for i, (title, val, border_col, bg_col) in enumerate(kpis):
        x = kpi_start_x + i * (kpi_w + kpi_spacing)
        kpi_rect = FancyBboxPatch((x, 85.2), kpi_w, 4.8, boxstyle="round,pad=0.3",
                                  facecolor=bg_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(kpi_rect)
        ax.text(x + kpi_w/2, 88.4, title.upper(), ha="center", va="center", 
                fontsize=9.5, fontweight="bold", color=border_col)
        ax.text(x + kpi_w/2, 86.4, val, ha="center", va="center", 
                fontsize=11.5, fontweight="bold", color="#1A202C")

    # -------------------------------------------------------------------------
    # 3. ARCHITECTURAL TABLE DEFINITION (Layers 0 to 15)
    # -------------------------------------------------------------------------
    table_data = [
        # (Idx, Module, Operation, Input Shape, Output Shape, Params, Status, StatusType)
        ("0", "Input Patch", "Normalized H&E Tissue Patch", "—", "(B, 3, 224, 224)", "0", "Input Data", "NEUTRAL"),
        ("1", "conv1", "7×7 2D Convolution, s=2, p=3", "(B, 3, 224, 224)", "(B, 64, 112, 112)", "9,408", "FROZEN", "FROZEN"),
        ("2", "bn1", "2D Batch Normalization + ReLU", "(B, 64, 112, 112)", "(B, 64, 112, 112)", "128", "ADAPTS (γ, β)", "ADAPTS"),
        ("3", "maxpool", "3×3 Max Pooling, s=2, p=1", "(B, 64, 112, 112)", "(B, 64, 56, 56)", "0", "Non-parametric", "NEUTRAL"),
        ("4", "layer1[0]", "BasicBlock 1.1 (2× Conv 3×3, s=1)", "(B, 64, 56, 56)", "(B, 64, 56, 56)", "73,984", "BN ADAPTS (256)", "ADAPTS"),
        ("5", "layer1[1]", "BasicBlock 1.2 (2× Conv 3×3, s=1)", "(B, 64, 56, 56)", "(B, 64, 56, 56)", "73,984", "BN ADAPTS (256)", "ADAPTS"),
        ("6", "layer2[0]", "BasicBlock 2.1 (s=2 downsample)", "(B, 64, 56, 56)", "(B, 128, 28, 28)", "230,144", "BN ADAPTS (512)", "ADAPTS"),
        ("7", "layer2[1]", "BasicBlock 2.2 (2× Conv 3×3, s=1)", "(B, 128, 28, 28)", "(B, 128, 28, 28)", "295,424", "BN ADAPTS (256)", "ADAPTS"),
        ("8", "layer3[0]", "BasicBlock 3.1 (s=2 downsample)", "(B, 128, 28, 28)", "(B, 256, 14, 14)", "919,040", "BN ADAPTS (1,024)", "ADAPTS"),
        ("9", "layer3[1]", "BasicBlock 3.2 (2× Conv 3×3, s=1)", "(B, 256, 14, 14)", "(B, 256, 14, 14)", "1,180,672", "BN ADAPTS (512)", "ADAPTS"),
        ("10", "layer4[0]", "BasicBlock 4.1 (s=2 downsample)", "(B, 256, 14, 14)", "(B, 512, 7, 7)", "3,673,088", "BN ADAPTS (2,048)", "ADAPTS"),
        ("11", "layer4[1]", "BasicBlock 4.2 (2× Conv 3×3, s=1)", "(B, 512, 7, 7)", "(B, 512, 7, 7)", "4,720,640", "BN ADAPTS (1,024)", "ADAPTS"),
        ("12", "avgpool", "Adaptive Average Pooling (7×7 → 1×1)", "(B, 512, 7, 7)", "(B, 512, 1, 1)", "0", "Non-parametric", "NEUTRAL"),
        ("13", "flatten", "Reshape 4D tensor to 2D matrix", "(B, 512, 1, 1)", "(B, 512)", "0", "Feature Tap Point", "TAP"),
        ("14", "fc", "Linear Head: y = x W^T + b", "(B, 512)", "(B, 2)", "1,026", "FROZEN", "FROZEN"),
        ("15", "softmax", "Posterior Probability Simplex", "(B, 2)", "(B, 2)", "0", "Non-parametric", "NEUTRAL")
    ]

    # Table Column Specifications
    cols = [
        ("Layer", 6.5, "center"),
        ("Module Name", 11.5, "left"),
        ("Mathematical Operation / Kernel", 28.0, "left"),
        ("Input Tensor Shape", 15.0, "center"),
        ("Output Tensor Shape", 15.0, "center"),
        ("Parameters", 9.0, "right"),
        ("Status in SafeAdapt", 15.0, "center")
    ]
    
    table_x = 3.0
    table_y_top = 83.2
    row_h = 3.10
    total_w = sum(w for _, w, _ in cols) # 100.0% of 94 width -> 94.0

    # Draw Table Header Row
    header_rect = FancyBboxPatch((table_x, table_y_top - 3.2), total_w, 3.2, boxstyle="round,pad=0.2",
                                 facecolor="#2D3748", edgecolor="#1A202C", linewidth=1.2)
    ax.add_patch(header_rect)
    
    cur_x = table_x
    for name, w, align in cols:
        if align == "center":
            tx = cur_x + w/2
        elif align == "left":
            tx = cur_x + 1.2
        else: # right
            tx = cur_x + w - 1.2
            
        ax.text(tx, table_y_top - 1.6, name, ha=align, va="center", 
                color="#FFFFFF", fontsize=FONT_HEADER, fontweight="bold")
        cur_x += w

    # Draw Table Rows
    row_y = table_y_top - 3.2 - row_h
    for row_idx, (idx_str, mod, op, in_s, out_s, params, status, st_type) in enumerate(table_data):
        # Alternating row colors
        bg_color = "#FFFFFF" if row_idx % 2 == 0 else "#F7FAFC"
        if st_type == "TAP":
            bg_color = "#FEFCBF" # Gold highlight for tap point
            
        row_rect = Rectangle((table_x, row_y), total_w, row_h, facecolor=bg_color, edgecolor="#E2E8F0", linewidth=0.8)
        ax.add_patch(row_rect)
        
        # Draw cells
        cur_x = table_x
        row_vals = [idx_str, mod, op, in_s, out_s, params, status]
        
        for c_idx, ((_, w, align), val) in enumerate(zip(cols, row_vals)):
            if c_idx == 0: # Layer index
                ax.text(cur_x + w/2, row_y + row_h/2, val, ha="center", va="center",
                        fontsize=FONT_BODY, fontweight="bold", color="#2D3748")
            elif c_idx == 1: # Module name
                ax.text(cur_x + 1.2, row_y + row_h/2, val, ha="left", va="center",
                        fontsize=FONT_CODE, family="monospace", fontweight="bold", color="#1A365D")
            elif c_idx == 2: # Operation
                ax.text(cur_x + 1.2, row_y + row_h/2, val, ha="left", va="center",
                        fontsize=FONT_BODY - 0.5, color="#2D3748")
            elif c_idx in (3, 4): # Tensor shapes
                ax.text(cur_x + w/2, row_y + row_h/2, val, ha="center", va="center",
                        fontsize=FONT_CODE, family="monospace", color="#2B6CB0", fontweight="bold")
            elif c_idx == 5: # Parameters
                ax.text(cur_x + w - 1.2, row_y + row_h/2, val, ha="right", va="center",
                        fontsize=FONT_CODE, family="monospace", fontweight="bold", color="#1A202C")
            elif c_idx == 6: # Status Badge
                badge_w = w - 2.0
                badge_h = row_h * 0.72
                bx = cur_x + 1.0
                by = row_y + (row_h - badge_h)/2
                
                if st_type == "FROZEN":
                    badge_bg, badge_border, badge_txt = "#EDF2F7", "#4A5568", "#2D3748"
                elif st_type == "ADAPTS":
                    badge_bg, badge_border, badge_txt = "#C6F6D5", "#22543D", "#22543D"
                elif st_type == "TAP":
                    badge_bg, badge_border, badge_txt = "#FEEBC8", "#C05621", "#7B341E"
                else:
                    badge_bg, badge_border, badge_txt = "#E2E8F0", "#718096", "#4A5568"
                    
                b_patch = FancyBboxPatch((bx, by), badge_w, badge_h, boxstyle="round,pad=0.2",
                                         facecolor=badge_bg, edgecolor=badge_border, linewidth=1.0)
                ax.add_patch(b_patch)
                ax.text(bx + badge_w/2, by + badge_h/2, val, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=badge_txt)
                
            cur_x += w
            
        row_y -= row_h

    # -------------------------------------------------------------------------
    # 4. FEATURE TAP POINT FOR SAFEADAPT (Bottom Dedicated Container)
    # -------------------------------------------------------------------------
    box_tap_y = 2.8
    box_tap_h = row_y - box_tap_y - 1.2
    
    tap_box = FancyBboxPatch((table_x, box_tap_y), total_w, box_tap_h, boxstyle="round,pad=0.5",
                             facecolor="#FFFFFF", edgecolor="#2B6CB0", linewidth=2.0)
    ax.add_patch(tap_box)
    
    # Tap Point Header
    ax.text(table_x + 2.5, box_tap_y + box_tap_h - 2.5, 
            "★ FEATURE TAP POINT FOR SAFEADAPT TEST-TIME ADAPTATION", 
            fontsize=12.5, fontweight="bold", color="#2B6CB0")
    
    # Left Column: Detailed Mathematical Formula Card
    left_x = table_x + 2.5
    card_w = 42.0
    card_h = box_tap_h - 5.0
    card_y = box_tap_y + 1.8
    
    f_card = FancyBboxPatch((left_x, card_y), card_w, card_h, boxstyle="round,pad=0.3",
                            facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.2)
    ax.add_patch(f_card)
    
    ax.text(left_x + 1.5, card_y + card_h - 2.2, "1. Latent Feature Extraction (Layer 13 Flatten):", 
            fontsize=10.5, fontweight="bold", color="#1A365D")
    ax.text(left_x + 3.0, card_y + card_h - 4.2, 
            "At Layer 13 (flatten), we extract the penultimate feature embedding vector:\n"
            "            f(x_i) in R^512  (Captures high-level histological tissue semantics)", 
            fontsize=10, family="monospace", color="#2B6CB0")
    
    ax.text(left_x + 1.5, card_y + card_h - 7.5, "2. L2 Hypersphere Projection onto 511-d Sphere:", 
            fontsize=10.5, fontweight="bold", color="#1A365D")
    ax.text(left_x + 3.0, card_y + card_h - 9.8, 
            "            z_i = f(x_i) / ||f(x_i)||_2  in S^511\n"
            "Unit normalization projects features onto the hypersphere, allowing cosine\n"
            "dissimilarity to serve as a valid geometric metric for Optimal Transport.", 
            fontsize=10, family="monospace", color="#22543D")

    # Right Column: Dataflow Routing Card
    right_x = left_x + card_w + 4.0
    r_card_w = total_w - card_w - 9.0
    
    r_card = FancyBboxPatch((right_x, card_y), r_card_w, card_h, boxstyle="round,pad=0.3",
                            facecolor="#F0FFF4", edgecolor="#2F855A", linewidth=1.2)
    ax.add_patch(r_card)
    
    ax.text(right_x + 1.5, card_y + card_h - 2.2, "3. Downstream Dual-Pathway Routing:", 
            fontsize=10.5, fontweight="bold", color="#22543D")
    
    ax.text(right_x + 3.0, card_y + card_h - 4.8, 
            "▶ PATH A (SafeAdapt-OT Engine):\n"
            "  The normalized embedding z_i enters the Sinkhorn-Knopp solver to compute the\n"
            "  Cosine Cost Matrix against frozen source prototypes: C_ik = 1 - z_i · μ_k.\n"
            "  Guides test-time BatchNorm adaptation without pseudo-label collapse.", 
            fontsize=9.8, color="#2D3748")
            
    ax.text(right_x + 3.0, card_y + card_h - 9.6, 
            "▶ PATH B (Diagnostic Head):\n"
            "  The unnormalized feature f(x_i) passes into the frozen Layer 14 linear head (fc)\n"
            "  to produce calibrated output logits and posterior probabilities [p_0, p_1].", 
            fontsize=9.8, color="#2D3748")

    # Save high-resolution images
    out1 = ROOT_DIR / "results/graphs/resnet18_layerwise_architecture_table.png"
    out2 = ROOT_DIR / "architecture/resnet18_layerwise_architecture_table.png"
    
    plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"[3/3] Success! High-resolution layerwise image saved to:\n  -> {out1}\n  -> {out2}")

if __name__ == "__main__":
    main()

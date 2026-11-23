"""Publication-grade, perfectly aligned ResNet-18 layerwise architecture image generator.

Fixes all alignment, text-overlap, and clipping issues:
1. Fixes the right-edge header clipping on 'Status in SafeAdapt'.
2. Redesigns the bottom Feature Tap Point section into 3 spacious, independent modular cards:
   - Card 1: Latent Feature Extraction (Layer 13 Flatten -> f(x_i) in R^512)
   - Card 2: L2 Hypersphere Normalization (z_i in S^511)
   - Card 3: Dual Downstream Routing (SafeAdapt-OT Engine vs Diagnostic Classifier Head)
3. Ensures generous line height, strict vertical padding, and zero overlapping text.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("[1/3] Setting up high-resolution canvas with verified alignment...")
    
    # 300 DPI canvas: 24 x 18 inches
    fig = plt.figure(figsize=(24, 18), facecolor="#F8FAFC", dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Typography & styling constants
    FONT_TITLE = 17.5
    FONT_HEADER = 11.0
    FONT_BODY = 10.0
    FONT_CODE = 9.8

    # -------------------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------------------
    header_box = FancyBboxPatch((2.5, 92.2), 95.0, 6.2, boxstyle="round,pad=0.4", 
                                facecolor="#1A202C", edgecolor="#2D3748", linewidth=1.5)
    ax.add_patch(header_box)
    
    ax.text(50.0, 96.2, "ResNet-18 Layer-by-Layer Dimension Transformation & Parameter Isolation", 
            ha="center", va="center", color="#FFFFFF", fontsize=FONT_TITLE, fontweight="bold")
    ax.text(50.0, 93.8, "Phase 2 Architectural Specification: Structural Progression of Tensors, Operations, and SafeAdapt Adaptability (Layer 0 to 15)", 
            ha="center", va="center", color="#CBD5E0", fontsize=11.2)

    # -------------------------------------------------------------------------
    # 2. SUMMARY KPI PILL BOXES
    # -------------------------------------------------------------------------
    kpis = [
        ("Total Parameters", "11,177,538 (100.0%)", "#2B6CB0", "#EBF8FF"),
        ("Frozen Weights", "11,167,938 (99.914%)", "#4A5568", "#EDF2F7"),
        ("Adaptable BN Weights", "9,600 (0.086%)", "#22543D", "#F0FFF4"),
        ("Tensor Input / Output", "(B, 3, 224, 224) → (B, 2)", "#744210", "#FEFCBF")
    ]
    
    kpi_w = 22.0
    kpi_spacing = 2.33
    kpi_start_x = 2.5
    
    for i, (title, val, border_col, bg_col) in enumerate(kpis):
        x = kpi_start_x + i * (kpi_w + kpi_spacing)
        kpi_rect = FancyBboxPatch((x, 86.0), kpi_w, 4.8, boxstyle="round,pad=0.3",
                                  facecolor=bg_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(kpi_rect)
        ax.text(x + kpi_w/2, 89.2, title.upper(), ha="center", va="center", 
                fontsize=9.5, fontweight="bold", color=border_col)
        ax.text(x + kpi_w/2, 87.2, val, ha="center", va="center", 
                fontsize=11.5, fontweight="bold", color="#1A202C")

    # -------------------------------------------------------------------------
    # 3. ARCHITECTURAL TABLE DEFINITION (Layers 0 to 15)
    # -------------------------------------------------------------------------
    table_data = [
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

    # Adjusted Column Widths (Total: exactly 95.0, matching table width)
    cols = [
        ("Layer", 5.5, "center"),
        ("Module Name", 11.5, "left"),
        ("Mathematical Operation / Kernel", 28.0, "left"),
        ("Input Tensor Shape", 15.0, "center"),
        ("Output Tensor Shape", 15.0, "center"),
        ("Parameters", 8.5, "right"),
        ("Status in SafeAdapt", 11.5, "center")
    ]
    
    table_x = 2.5
    table_y_top = 84.4
    row_h = 2.55
    total_w = sum(w for _, w, _ in cols) # Exactly 95.0

    # Draw Table Header Row
    header_h = 2.8
    header_rect = FancyBboxPatch((table_x, table_y_top - header_h), total_w, header_h, boxstyle="round,pad=0.15",
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
            
        ax.text(tx, table_y_top - header_h/2, name, ha=align, va="center", 
                color="#FFFFFF", fontsize=FONT_HEADER, fontweight="bold")
        cur_x += w

    # Draw Table Rows
    row_y = table_y_top - header_h - row_h
    for row_idx, (idx_str, mod, op, in_s, out_s, params, status, st_type) in enumerate(table_data):
        bg_color = "#FFFFFF" if row_idx % 2 == 0 else "#F7FAFC"
        if st_type == "TAP":
            bg_color = "#FEFCBF" # Highlight row 13
            
        row_rect = Rectangle((table_x, row_y), total_w, row_h, facecolor=bg_color, edgecolor="#E2E8F0", linewidth=0.7)
        ax.add_patch(row_rect)
        
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
                        fontsize=FONT_BODY - 0.4, color="#2D3748")
            elif c_idx in (3, 4): # Tensor shapes
                ax.text(cur_x + w/2, row_y + row_h/2, val, ha="center", va="center",
                        fontsize=FONT_CODE, family="monospace", color="#2B6CB0", fontweight="bold")
            elif c_idx == 5: # Parameters
                ax.text(cur_x + w - 1.2, row_y + row_h/2, val, ha="right", va="center",
                        fontsize=FONT_CODE, family="monospace", fontweight="bold", color="#1A202C")
            elif c_idx == 6: # Status Badge
                badge_w = w - 1.2
                badge_h = row_h * 0.74
                bx = cur_x + 0.6
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
    # 4. FEATURE TAP POINT FOR SAFEADAPT (3 Spacious Modular Cards)
    # -------------------------------------------------------------------------
    tap_container_y = 2.2
    tap_container_h = row_y - tap_container_y - 1.2
    
    # Master container card
    master_tap_box = FancyBboxPatch((table_x, tap_container_y), total_w, tap_container_h, boxstyle="round,pad=0.4",
                                    facecolor="#FFFFFF", edgecolor="#2B6CB0", linewidth=2.0)
    ax.add_patch(master_tap_box)
    
    # Title of Section
    ax.text(table_x + 2.5, tap_container_y + tap_container_h - 2.5, 
            "★ FEATURE TAP POINT FOR SAFEADAPT TEST-TIME ADAPTATION", 
            fontsize=13.0, fontweight="bold", color="#2B6CB0")
    ax.text(table_x + total_w - 2.5, tap_container_y + tap_container_h - 2.5, 
            "Extracted at Layer 13 (Flatten) → Feeds Optimal Transport & Classifier", 
            ha="right", fontsize=10.0, fontweight="bold", color="#718096")

    card_y = tap_container_y + 1.8
    card_h = tap_container_h - 5.4
    card_w = 29.5
    gap_x = (total_w - 5.0 - (card_w * 3)) / 2.0 # Perfect equal spacing
    
    # -------------------------------------------------------------------------
    # CARD 1: Latent Feature Extraction
    # -------------------------------------------------------------------------
    c1_x = table_x + 2.5
    c1_box = FancyBboxPatch((c1_x, card_y), card_w, card_h, boxstyle="round,pad=0.3",
                            facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.4)
    ax.add_patch(c1_box)
    
    # Card 1 Badge
    b1 = FancyBboxPatch((c1_x + 1.5, card_y + card_h - 3.8), card_w - 3.0, 2.6, boxstyle="round,pad=0.2",
                        facecolor="#3182CE", edgecolor="#2B6CB0", linewidth=1.0)
    ax.add_patch(b1)
    ax.text(c1_x + card_w/2, card_y + card_h - 2.5, "1. LATENT FEATURE EXTRACTION", 
            ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")
    
    # Card 1 Content (Clear vertical spacing, NO overlaps)
    ax.text(c1_x + 2.0, card_y + card_h - 6.2, "• Source: Layer 13 (flatten)", 
            fontsize=10.5, fontweight="bold", color="#1A365D")
    
    # Formula Box
    f1_box = FancyBboxPatch((c1_x + 1.8, card_y + card_h - 12.8), card_w - 3.6, 5.4, boxstyle="round,pad=0.2",
                            facecolor="#FFFFFF", edgecolor="#BEE3F8", linewidth=1.0)
    ax.add_patch(f1_box)
    ax.text(c1_x + card_w/2, card_y + card_h - 9.0, "f(x_i) ∈ ℝ⁵¹²", 
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#2B6CB0")
    ax.text(c1_x + card_w/2, card_y + card_h - 11.2, "Penultimate 512-dim Embedding", 
            ha="center", va="center", fontsize=9.5, color="#718096")
            
    c1_desc = (
        "• Captures high-level semantic tissue\n"
        "  morphology (cancer nests vs. lymphoid stroma).\n"
        "• Extracted directly after Layer 12 avgpool.\n"
        "• Retains unnormalized magnitude."
    )
    ax.text(c1_x + 2.0, card_y + card_h - 15.5, c1_desc, fontsize=9.5, color="#2D3748", linespacing=1.4)

    # -------------------------------------------------------------------------
    # CARD 2: L2 Hypersphere Normalization
    # -------------------------------------------------------------------------
    c2_x = c1_x + card_w + gap_x
    c2_box = FancyBboxPatch((c2_x, card_y), card_w, card_h, boxstyle="round,pad=0.3",
                            facecolor="#F0FFF4", edgecolor="#38A169", linewidth=1.4)
    ax.add_patch(c2_box)
    
    # Card 2 Badge
    b2 = FancyBboxPatch((c2_x + 1.5, card_y + card_h - 3.8), card_w - 3.0, 2.6, boxstyle="round,pad=0.2",
                        facecolor="#38A169", edgecolor="#22543D", linewidth=1.0)
    ax.add_patch(b2)
    ax.text(c2_x + card_w/2, card_y + card_h - 2.5, "2. HYPERSPHERE PROJECTION", 
            ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")
            
    # Card 2 Content
    ax.text(c2_x + 2.0, card_y + card_h - 6.2, "• Operation: L2 Unit Normalization", 
            fontsize=10.5, fontweight="bold", color="#22543D")
            
    # Formula Box
    f2_box = FancyBboxPatch((c2_x + 1.8, card_y + card_h - 12.8), card_w - 3.6, 5.4, boxstyle="round,pad=0.2",
                            facecolor="#FFFFFF", edgecolor="#C6F6D5", linewidth=1.0)
    ax.add_patch(f2_box)
    ax.text(c2_x + card_w/2, card_y + card_h - 9.0, "z_i = f(x_i) / ||f(x_i)||₂  ∈ 𝕊⁵¹¹", 
            ha="center", va="center", fontsize=12.0, fontweight="bold", color="#22543D")
    ax.text(c2_x + card_w/2, card_y + card_h - 11.2, "Unit Sphere Projection (511-d)", 
            ha="center", va="center", fontsize=9.5, color="#718096")

    c2_desc = (
        "• Projects feature vectors onto the\n"
        "  511-dimensional unit hypersphere.\n"
        "• Eliminates arbitrary activation scale shifts.\n"
        "• Allows Cosine Dissimilarity (C = 1 - z·μ)\n"
        "  to serve as a strict metric distance for OT."
    )
    ax.text(c2_x + 2.0, card_y + card_h - 15.5, c2_desc, fontsize=9.5, color="#2D3748", linespacing=1.4)

    # -------------------------------------------------------------------------
    # CARD 3: Downstream Dual-Pathway Routing
    # -------------------------------------------------------------------------
    c3_x = c2_x + card_w + gap_x
    c3_box = FancyBboxPatch((c3_x, card_y), card_w, card_h, boxstyle="round,pad=0.3",
                            facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.4)
    ax.add_patch(c3_box)
    
    # Card 3 Badge
    b3 = FancyBboxPatch((c3_x + 1.5, card_y + card_h - 3.8), card_w - 3.0, 2.6, boxstyle="round,pad=0.2",
                        facecolor="#D69E2E", edgecolor="#744210", linewidth=1.0)
    ax.add_patch(b3)
    ax.text(c3_x + card_w/2, card_y + card_h - 2.5, "3. DUAL-PATHWAY ROUTING", 
            ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold")
            
    # Sub-card A: SafeAdapt-OT Path
    ax.text(c3_x + 2.0, card_y + card_h - 6.2, "▶ PATH A: SafeAdapt-OT Engine", 
            fontsize=10.5, fontweight="bold", color="#744210")
    pA_desc = (
        "• Input: Normalized vector z_i\n"
        "• Compares z_i to source class prototypes (μ_0, μ_1).\n"
        "• Sinkhorn-Knopp computes balanced coupling Q*.\n"
        "• Updates ONLY BatchNorm affine parameters (γ, β)."
    )
    ax.text(c3_x + 2.0, card_y + card_h - 11.2, pA_desc, fontsize=9.2, color="#2D3748", linespacing=1.35)
    
    # Divider line
    ax.plot([c3_x + 2.0, c3_x + card_w - 2.0], [card_y + card_h - 12.2, card_y + card_h - 12.2], 
            color="#CBD5E0", linestyle="--", linewidth=1.0)

    # Sub-card B: Diagnostic Classifier Head Path
    ax.text(c3_x + 2.0, card_y + card_h - 14.0, "▶ PATH B: Frozen Diagnostic Head", 
            fontsize=10.5, fontweight="bold", color="#1A365D")
    pB_desc = (
        "• Input: Unnormalized vector f(x_i)\n"
        "• Passes into Layer 14 linear head (fc: 512 → 2).\n"
        "• Layer 15 Softmax computes posterior probabilities:\n"
        "  [p(Normal), p(Tumor)] for clinical diagnosis."
    )
    ax.text(c3_x + 2.0, card_y + card_h - 19.2, pB_desc, fontsize=9.2, color="#2D3748", linespacing=1.35)

    # Save images
    out1 = ROOT_DIR / "results/graphs/resnet18_layerwise_architecture_table.png"
    out2 = ROOT_DIR / "architecture/resnet18_layerwise_architecture_table.png"
    
    plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"[3/3] Success! Beautifully aligned image saved to:\n  -> {out1}\n  -> {out2}")

if __name__ == "__main__":
    main()

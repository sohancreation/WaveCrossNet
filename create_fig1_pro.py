import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_professional_fig1():
    # Set high-DPI publication figure (15.5 x 7.4 inches at 300 DPI)
    fig = plt.figure(figsize=(15.5, 7.4), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.4)
    ax.axis('off')

    # ══════════════════════════════════════════════════════════════════════════
    # COLOR PALETTE (Nature / IEEE Transactions Style)
    # ══════════════════════════════════════════════════════════════════════════
    c_bg_input = "#F8FAFC"
    c_border_input = "#475569"
    
    c_bg_dwt = "#F0FDF4"
    c_border_dwt = "#16A34A"
    
    c_bg_conv = "#FFFBEB"
    c_border_conv = "#D97706"
    
    c_bg_attn = "#F5F3FF"
    c_border_attn = "#7C3AED"
    
    c_bg_fuse = "#F0FDFA"
    c_border_fuse = "#0D9488"
    
    c_bg_head = "#FFF1F2"
    c_border_head = "#E11D48"

    # Helper: draw styled rounded card
    def draw_card(x, y, w, h, bg, border, title="", subtitle="", pad=0.06, lw=1.6):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}", 
                             fc=bg, ec=border, lw=lw, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top', 
                    fontsize=9.2, fontweight='bold', color=border, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.46, subtitle, ha='center', va='top', 
                    fontsize=7.0, fontweight='semibold', color='#64748B', zorder=3)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. INPUT ECG BEAT (x: 0.40 to 2.35, y: 1.95 to 6.95)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(0.40, 1.95, 1.95, 5.00, c_bg_input, c_border_input, 
              "Input ECG Beat", "Lead II (250 Hz) $\\cdot$ MIT-BIH")

    # ECG Waveform Plot
    t = np.linspace(0, 1, 250)
    ecg = (0.16 * np.exp(-((t - 0.24) ** 2) / 0.0020) - 
           0.30 * np.exp(-((t - 0.46) ** 2) / 0.0004) + 
           1.80 * np.exp(-((t - 0.50) ** 2) / 0.0006) - 
           0.55 * np.exp(-((t - 0.54) ** 2) / 0.0004) + 
           0.40 * np.exp(-((t - 0.72) ** 2) / 0.0042))

    ax_ecg = fig.add_axes([0.52/15.5, 3.85/7.4, 1.70/15.5, 2.35/7.4])
    ax_ecg.plot(t, ecg, color="#2563EB", lw=2.0)
    # Annotate clinical waves
    ax_ecg.text(0.23, 0.42, 'P', fontsize=8.5, fontweight='bold', color='#1E40AF', ha='center')
    ax_ecg.text(0.43, -0.60, 'Q', fontsize=8.0, fontweight='bold', color='#1E40AF', ha='center')
    ax_ecg.text(0.50, 1.88, 'R', fontsize=10.0, fontweight='bold', color='#DC2626', ha='center')
    ax_ecg.text(0.57, -0.80, 'S', fontsize=8.0, fontweight='bold', color='#1E40AF', ha='center')
    ax_ecg.text(0.73, 0.65, 'T', fontsize=8.5, fontweight='bold', color='#1E40AF', ha='center')
    ax_ecg.set_ylim(-1.0, 2.3)
    ax_ecg.axis('off')

    # Tensor Dimensions & Preprocessing info
    p_box = FancyBboxPatch((0.52, 2.85), 1.70, 0.68, boxstyle="round,pad=0.03",
                           fc="white", ec="#94A3B8", lw=1.1, zorder=3)
    ax.add_patch(p_box)
    ax.text(1.37, 3.30, r"$\mathbf{x} \in \mathbb{R}^{1 \times 256}$", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color="#0F172A", zorder=4)
    ax.text(1.37, 3.02, r"Window: 1.024 s (Centered R)", ha='center', va='center', 
            fontsize=6.5, fontweight='semibold', color="#475569", zorder=4)

    prep_box = FancyBboxPatch((0.52, 2.12), 1.70, 0.54, boxstyle="round,pad=0.02",
                              fc="#F1F5F9", ec="#CBD5E1", lw=0.9, zorder=3)
    ax.add_patch(prep_box)
    ax.text(1.37, 2.39, "Bandpass: 0.5 - 45 Hz\nZ-Score Standardized", ha='center', va='center', 
            fontsize=6.2, fontweight='medium', color="#334155", linespacing=1.25, zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. BRANCH 1: DWT SPECTRAL QUERY ENCODER (x: 2.85 to 7.10, y: 4.10 to 6.95)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(2.85, 4.10, 4.25, 2.85, c_bg_dwt, c_border_dwt, 
              "Branch 1: DWT Spectral Query Encoder", 
              "3-Level Mallat Wavelet Filter Bank (Daubechies db4)")

    subbands = [
        (r"$cD_1$", "[62.5 - 125 Hz]", "EMG & high-frequency transients", "#DC2626"),
        (r"$cD_2$", "[31.25 - 62.5 Hz]", "QRS complex depolarization dynamics", "#059669"),
        (r"$cD_3$", "[15.6 - 31.25 Hz]", "P/T wave repolarization morphology", "#2563EB"),
        (r"$cA_3$", "[0 - 15.6 Hz]", "Baseline wander & ST segment trends", "#7C3AED")
    ]

    for i, (name, band, desc, color) in enumerate(subbands):
        y_sb = 5.86 - i * 0.44
        sb_box = FancyBboxPatch((3.00, y_sb), 2.35, 0.36, boxstyle="round,pad=0.02", 
                                fc='white', ec=color, lw=1.2, zorder=3)
        ax.add_patch(sb_box)
        # Colored Subband badge
        badge = FancyBboxPatch((3.05, y_sb + 0.05), 0.38, 0.26, boxstyle="round,pad=0.01",
                               fc=color, ec=None, zorder=4)
        ax.add_patch(badge)
        ax.text(3.24, y_sb + 0.18, name, fontsize=7.2, fontweight='bold', color='white', ha='center', va='center', zorder=5)
        
        ax.text(3.52, y_sb + 0.24, band, fontsize=6.8, fontweight='bold', color="#1E293B", va='center', zorder=4)
        ax.text(3.52, y_sb + 0.10, desc, fontsize=5.8, fontweight='medium', color="#64748B", va='center', zorder=4)

    # Subband Projection block (Shared Conv1D + Pool)
    box_proj = FancyBboxPatch((5.52, 4.22), 1.45, 2.00, boxstyle="round,pad=0.03", 
                              fc="#DCFCE7", ec="#16A34A", lw=1.2, zorder=3)
    ax.add_patch(box_proj)
    ax.text(6.245, 6.08, "Subband Proj", ha='center', va='top', 
            fontsize=8.0, fontweight='bold', color="#15803D", zorder=4)
    ax.text(6.245, 5.88, "Shared 1D-Conv", ha='center', va='top', 
            fontsize=6.5, fontweight='semibold', color="#16A34A", zorder=4)
    
    proj_ops = "Conv1D (k=3, c=64)\n+\nBatchNorm1D\n+\nGELU Activation\n+\nAdaptiveAvgPool1D"
    ax.text(6.245, 5.20, proj_ops, ha='center', va='center', 
            fontsize=6.0, fontweight='bold', color="#14532D", linespacing=1.25, zorder=4)
    
    proj_dim = FancyBboxPatch((5.62, 4.30), 1.25, 0.30, boxstyle="round,pad=0.02",
                              fc="white", ec="#16A34A", lw=0.9, zorder=4)
    ax.add_patch(proj_dim)
    ax.text(6.245, 4.45, r"$\mathbf{q}_i \in \mathbb{R}^{64}$ ($i=1..4$)", ha='center', va='center', 
            fontsize=6.8, fontweight='bold', color="#15803D", zorder=5)

    # Connecting bus lines from subbands to Subband Proj
    for i in range(4):
        y_sb = 5.86 - i * 0.44 + 0.18
        ax.plot([5.35, 5.52], [y_sb, 5.20], color="#16A34A", lw=1.0, linestyle="-", zorder=3)

    # Query Tokens Box (Q)
    box_q = FancyBboxPatch((7.35, 4.65), 1.20, 1.45, boxstyle="round,pad=0.04", 
                           fc="#A7F3D0", ec="#059669", lw=1.5, zorder=3)
    ax.add_patch(box_q)
    ax.text(7.95, 5.92, "Queries (Q)", ha='center', va='top', 
            fontsize=8.5, fontweight='bold', color="#065F46", zorder=4)
    ax.text(7.95, 5.40, r"$\mathbf{Q} \in \mathbb{R}^{4 \times 64}$", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color="#064E3B", zorder=4)
    ax.text(7.95, 5.00, r"$N_q = 4$ tokens", ha='center', va='center', 
            fontsize=7.0, fontweight='semibold', color="#047857", zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. BRANCH 2: 1D TEMPORAL KEY-VALUE EXTRACTOR (x: 2.85 to 7.10, y: 0.65 to 3.85)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(2.85, 0.65, 4.25, 3.20, c_bg_conv, c_border_conv, 
              "Branch 2: 1D Temporal Key-Value Extractor", 
              "Strided Multi-Scale Convolutional Pyramid (Downsampling factor = 8)")

    stages = [
        ("Stage 1", "Conv1D (k=7, s=2, c=32) + BN + GELU", r"$\mathbf{H}^{(1)} \in \mathbb{R}^{32 \times 128}$"),
        ("Stage 2", "Conv1D (k=5, s=2, c=48) + BN + GELU", r"$\mathbf{H}^{(2)} \in \mathbb{R}^{48 \times 64}$"),
        ("Stage 3", "Conv1D (k=3, s=2, c=64) + BN + GELU", r"$\mathbf{H}^{(3)} \in \mathbb{R}^{64 \times 32}$")
    ]

    for j, (st_name, st_ops, st_dim) in enumerate(stages):
        y_st = 2.58 - j * 0.74
        st_box = FancyBboxPatch((3.00, y_st), 3.95, 0.56, boxstyle="round,pad=0.03", 
                                fc='white', ec="#D97706", lw=1.1, zorder=3)
        ax.add_patch(st_box)
        
        # Stage Tag (Left pill)
        tag_box = FancyBboxPatch((3.08, y_st + 0.12), 0.72, 0.32, boxstyle="round,pad=0.02",
                                fc="#FEF3C7", ec="#D97706", lw=0.9, zorder=4)
        ax.add_patch(tag_box)
        ax.text(3.44, y_st + 0.28, st_name, fontsize=7.2, fontweight='bold', color="#B45309", ha='center', va='center', zorder=5)
        
        # Operations (Center)
        ax.text(3.92, y_st + 0.28, st_ops, fontsize=6.3, fontweight='semibold', color="#1E293B", ha='left', va='center', zorder=4)
        
        # Output Tensor Badge (Right pill)
        dim_box = FancyBboxPatch((5.85, y_st + 0.12), 1.02, 0.32, boxstyle="round,pad=0.02",
                                fc="#FFFBEB", ec="#F59E0B", lw=0.9, zorder=4)
        ax.add_patch(dim_box)
        ax.text(6.36, y_st + 0.28, st_dim, fontsize=6.8, fontweight='bold', color="#92400E", ha='center', va='center', zorder=5)

    # Key/Value Tokens Box (K, V)
    box_kv = FancyBboxPatch((7.35, 1.45), 1.20, 1.70, boxstyle="round,pad=0.04", 
                            fc="#FEF3C7", ec="#D97706", lw=1.5, zorder=3)
    ax.add_patch(box_kv)
    ax.text(7.95, 2.95, "Keys / Values", ha='center', va='top', 
            fontsize=8.0, fontweight='bold', color="#92400E", zorder=4)
    ax.text(7.95, 2.45, r"$\mathbf{K} \in \mathbb{R}^{32 \times 64}$", ha='center', va='center', 
            fontsize=7.8, fontweight='bold', color="#78350F", zorder=4)
    ax.text(7.95, 2.05, r"$\mathbf{V} \in \mathbb{R}^{32 \times 64}$", ha='center', va='center', 
            fontsize=7.8, fontweight='bold', color="#78350F", zorder=4)
    ax.text(7.95, 1.68, r"$N_{kv} = 32$ tokens", ha='center', va='center', 
            fontsize=6.8, fontweight='semibold', color="#B45309", zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ASYMMETRIC CROSS-ATTENTION MODULE (x: 8.85 to 12.15, y: 2.35 to 6.95)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(8.85, 2.35, 3.30, 4.60, c_bg_attn, c_border_attn, 
              "Asymmetric Cross-Attention", 
              "4 Spectral Queries attending to 32 Temporal Tokens")

    # Mathematical Formula Box
    f_box = FancyBboxPatch((9.02, 5.35), 2.96, 0.88, boxstyle="round,pad=0.03",
                           fc="white", ec="#8B5CF6", lw=1.1, zorder=3)
    ax.add_patch(f_box)
    ax.text(10.50, 5.92, r"$\mathbf{A}_h = \mathrm{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{4 \times 32}$", 
            ha='center', va='center', fontsize=8.0, fontweight='bold', color="#4C1D95", zorder=4)
    ax.text(10.50, 5.56, r"$\mathrm{head}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{4 \times 16} \quad (H=4 \text{ heads})$", 
            ha='center', va='center', fontsize=7.2, fontweight='semibold', color="#5B21B6", zorder=4)

    # Mini Heatmap Box
    heat_bg = FancyBboxPatch((9.02, 3.95), 2.96, 1.25, boxstyle="round,pad=0.02",
                             fc="white", ec="#C4B5FD", lw=1.0, zorder=3)
    ax.add_patch(heat_bg)
    ax.text(10.50, 5.06, r"Cross-Attention Heatmap $\mathbf{A} \in \mathbb{R}^{4 \times 32}$", 
            ha='center', va='center', fontsize=7.0, fontweight='bold', color="#6D28D9", zorder=4)

    # Inset Axes for the Heatmap
    ax_mat = fig.add_axes([9.32/15.5, 4.10/7.4, 2.40/15.5, 0.74/7.4])
    np.random.seed(42)
    sample_attn = np.zeros((4, 32))
    sample_attn[0, 15:18] = [0.35, 0.85, 0.40]                     # cD1 sharp spike at R peak
    sample_attn[1, 13:20] = [0.2, 0.5, 0.95, 0.8, 0.3, 0.1, 0.05] # cD2 QRS depolarization
    sample_attn[2, 6:12] = [0.15, 0.45, 0.80, 0.88, 0.35, 0.1]    # cD3 P wave
    sample_attn[2, 21:27] = [0.1, 0.35, 0.82, 0.85, 0.40, 0.15]   # cD3 T wave
    sample_attn[3, 0:32] = 0.12                                    # cA3 baseline wander
    ax_mat.imshow(sample_attn, cmap='Purples', aspect='auto', interpolation='nearest')
    ax_mat.set_yticks([0, 1, 2, 3])
    ax_mat.set_yticklabels([r'$cD_1$', r'$cD_2$', r'$cD_3$', r'$cA_3$'], fontsize=5.8, fontweight='bold')
    ax_mat.set_xticks([0, 15, 31])
    ax_mat.set_xticklabels(['$t_1$', '$t_{16}$', '$t_{32}$'], fontsize=5.8)
    ax_mat.tick_params(axis='both', which='both', length=2, pad=1)

    # 8x FLOP Reduction Badge
    flop_box = FancyBboxPatch((9.02, 3.08), 2.96, 0.72, boxstyle="round,pad=0.03",
                              fc="#EDE9FE", ec="#7C3AED", lw=1.2, zorder=3)
    ax.add_patch(flop_box)
    ax.text(10.50, 3.54, r"$\mathbf{8\times}$ Computational FLOP Reduction", ha='center', va='center', 
            fontsize=8.0, fontweight='bold', color="#5B21B6", zorder=4)
    ax.text(10.50, 3.25, r"$\mathrm{FLOPs}_{\text{Cross}} = 8{,}192$ vs $\mathrm{FLOPs}_{\text{ViT}} = 65{,}536$ FLOPs/head", 
            ha='center', va='center', fontsize=6.6, fontweight='bold', color="#6D28D9", zorder=4)

    # Attended Residual Output
    res_box = FancyBboxPatch((9.02, 2.48), 2.96, 0.48, boxstyle="round,pad=0.02",
                             fc="white", ec="#8B5CF6", lw=0.9, zorder=3)
    ax.add_patch(res_box)
    ax.text(10.50, 2.72, r"$\mathbf{Z}_{\text{spec}} = \mathrm{LN}\left(\mathbf{Q} + \mathrm{MHA}(\mathbf{Q},\mathbf{K},\mathbf{V})\right) \in \mathbb{R}^{4 \times 64}$", 
            ha='center', va='center', fontsize=6.6, fontweight='bold', color="#374151", zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. GLOBAL TEMPORAL POOLING & FEATURE FUSION (x: 8.85 to 12.15, y: 0.65 to 2.20)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(8.85, 0.65, 3.30, 1.55, c_bg_fuse, c_border_fuse, 
              "Temporal Pooling & Feature Fusion", "", pad=0.04, lw=1.3)
    
    # AvgPool formula box
    ap_box = FancyBboxPatch((9.02, 1.45), 2.96, 0.40, boxstyle="round,pad=0.02",
                            fc="white", ec="#0D9488", lw=0.9, zorder=3)
    ax.add_patch(ap_box)
    ax.text(10.50, 1.65, r"Temporal AvgPool: $\mathbf{g}_{\text{temp}} = \frac{1}{32}\sum_{j=1}^{32} \mathbf{K}[:, j, :] \in \mathbb{R}^{64}$", 
            ha='center', va='center', fontsize=6.8, fontweight='semibold', color="#0F766E", zorder=4)

    # Concatenation Box
    cat_box = FancyBboxPatch((9.02, 0.78), 2.96, 0.56, boxstyle="round,pad=0.02",
                             fc="#CCFBF1", ec="#0D9488", lw=1.2, zorder=3)
    ax.add_patch(cat_box)
    ax.text(10.50, 1.15, r"Concatenation: $\mathbf{c} = [\mathrm{vec}(\mathbf{Z}_{\text{spec}}), \; \mathbf{g}_{\text{temp}}] \in \mathbb{R}^{320}$", 
            ha='center', va='center', fontsize=7.2, fontweight='bold', color="#115E59", zorder=4)
    ax.text(10.50, 0.92, "(256 Attended Spectral + 64 Global Temporal Features)", 
            ha='center', va='center', fontsize=6.0, fontweight='semibold', color="#0F766E", zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. CLASSIFICATION HEAD & CLINICAL DIAGNOSIS (x: 12.60 to 15.10, y: 0.65 to 6.95)
    # ══════════════════════════════════════════════════════════════════════════
    draw_card(12.60, 0.65, 2.50, 6.30, c_bg_head, c_border_head, 
              "Classification Head", "2-Layer Non-Linear MLP")

    mlp_steps = [
        (r"Linear ($320 \to 64$)", "#9F1239"),
        (r"LayerNorm + GELU", "#BE123C"),
        (r"Dropout ($p = 0.20$)", "#E11D48"),
        (r"Linear ($64 \to 5$)", "#9F1239"),
        (r"Softmax Classifier", "#881337")
    ]
    for k, (step_txt, step_col) in enumerate(mlp_steps):
        y_mlp = 5.86 - k * 0.44
        m_box = FancyBboxPatch((12.78, y_mlp), 2.14, 0.33, boxstyle="round,pad=0.02", 
                               fc='white', ec=step_col, lw=0.9, zorder=3)
        ax.add_patch(m_box)
        ax.text(13.85, y_mlp + 0.165, step_txt, ha='center', va='center', 
                fontsize=7.2, fontweight='bold', color=step_col, zorder=4)
        if k < len(mlp_steps) - 1:
            ax.annotate("", xy=(13.85, y_mlp - 0.09), xytext=(13.85, y_mlp - 0.01),
                        arrowprops=dict(arrowstyle="->", color=step_col, lw=1.0, shrinkA=0, shrinkB=0))

    # Divider
    ax.plot([12.80, 14.90], [3.55, 3.55], color="#FDA4AF", lw=1.0, linestyle="--", zorder=3)

    # AAMI EC57 Diagnostic Output Classes
    ax.text(13.85, 3.34, "AAMI EC57 Diagnostic Classes:", ha='center', va='center', 
            fontsize=7.2, fontweight='bold', color="#1E293B", zorder=4)

    aami_classes = [
        ("N", "Normal Sinus Rhythm", "94.5% Acc", "#10B981"),
        ("S", "Supraventricular Ectopic", "48.2% F1", "#3B82F6"),
        ("V", "Ventricular Ectopic", "82.9% Rec", "#EF4444"),
        ("F", "Fusion of Ventricular", "41.6% F1", "#F59E0B"),
        ("Q", "Unknown / Other Beat", "99.8% Spe", "#6B7280")
    ]

    for m, (cls, cdesc, cmetric, ccolor) in enumerate(aami_classes):
        y_cls = 2.84 - m * 0.40
        c_badge = FancyBboxPatch((12.78, y_cls), 0.36, 0.29, boxstyle="round,pad=0.02", 
                                 fc=ccolor, ec=None, zorder=3)
        ax.add_patch(c_badge)
        ax.text(12.96, y_cls + 0.145, cls, ha='center', va='center', fontsize=7.8, fontweight='bold', color='white', zorder=4)
        
        ax.text(13.22, y_cls + 0.19, cdesc, ha='left', va='center', fontsize=6.2, fontweight='bold', color="#1E293B", zorder=4)
        ax.text(13.22, y_cls + 0.07, f"DS2: {cmetric}", ha='left', va='center', fontsize=5.8, fontweight='semibold', color=ccolor, zorder=4)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CONNECTING ROUTING LINES & ARROWS
    # ══════════════════════════════════════════════════════════════════════════
    def draw_arrow(x1, y1, x2, y2, color="#475569", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, 
                                   shrinkA=0, shrinkB=0,
                                   mutation_scale=12))

    # From Input Beat to Branch 1 & Branch 2
    draw_arrow(2.35, 4.80, 2.85, 5.50, color='#16A34A', lw=1.8)
    draw_arrow(2.35, 3.80, 2.85, 2.30, color='#D97706', lw=1.8)

    # Subband Proj to Queries (Q)
    draw_arrow(6.97, 5.38, 7.35, 5.38, color='#16A34A', lw=1.8)

    # Conv Pyramid to K/V Box
    draw_arrow(6.95, 2.30, 7.35, 2.30, color='#D97706', lw=1.8)

    # Queries (Q) to Cross-Attention Module
    draw_arrow(8.55, 5.38, 8.85, 5.38, color='#16A34A', lw=1.8)

    # Keys/Values (K,V) routing:
    # 1. K,V up to Cross-Attention
    ax.plot([8.55, 8.70, 8.70, 8.85], [2.45, 2.45, 4.60, 4.60], color='#D97706', lw=1.6, zorder=3)
    ax.annotate("", xy=(8.85, 4.60), xytext=(8.75, 4.60),
                arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=10))

    # 2. K into Temporal AvgPool
    draw_arrow(8.55, 1.65, 8.85, 1.65, color='#0D9488', lw=1.6)

    # Cross-Attention output down into Feature Fusion
    draw_arrow(10.50, 2.35, 10.50, 2.20, color='#7C3AED', lw=1.8)

    # Concatenation to Classification Head (Orthogonal Circuit Bus Routing)
    ax.plot([12.00, 12.38, 12.38, 12.78], [1.06, 1.06, 6.025, 6.025], color='#E11D48', lw=1.8, zorder=3)
    ax.annotate("", xy=(12.78, 6.025), xytext=(12.68, 6.025),
                arrowprops=dict(arrowstyle="->", color='#E11D48', lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=11))
    
    # Feature Vector Dimension Callout along the bus wire
    f_callout = FancyBboxPatch((12.18, 3.35), 0.40, 0.60, boxstyle="round,pad=0.03",
                               fc="white", ec="#E11D48", lw=1.0, zorder=4)
    ax.add_patch(f_callout)
    ax.text(12.38, 3.72, r"$\mathbf{c}$", ha='center', va='center', fontsize=8.2, fontweight='bold', color="#BE123C", zorder=5)
    ax.text(12.38, 3.50, r"$\mathbb{R}^{320}$", ha='center', va='center', fontsize=6.8, fontweight='bold', color="#BE123C", zorder=5)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. HARDWARE EMBEDDED PROFILE STRIP (Bottom Banner)
    # ══════════════════════════════════════════════════════════════════════════
    strip_box = FancyBboxPatch((0.40, 0.08), 14.70, 0.44, boxstyle="round,pad=0.04", 
                               fc="#0F172A", ec="#334155", lw=1.2, zorder=2)
    ax.add_patch(strip_box)
    
    spec_text = (
        r"$\mathbf{Ultra\text{-}Low\text{-}Power\ Edge\ Deployment\ (ARM\ Cortex\text{-}M4\ @\ 84\ MHz):}$   "
        r"INT8 Flash: $\mathbf{47.5\ KB}$ (9.3% limit)   $\mid$   "
        r"Peak SRAM: $\mathbf{12.4\ KB}$ (12.9% limit)   $\mid$   "
        r"Compute: $\mathbf{1.12\ MFLOPs/beat}$   $\mid$   "
        r"Latency: $\mathbf{14.8\ ms}$ (1.77% duty cycle)   $\mid$   "
        r"Battery Life: $\mathbf{>25\ Days}$ (250 mAh)"
    )
    ax.text(7.75, 0.30, spec_text, ha='center', va='center', 
            fontsize=7.0, fontweight='medium', color="#F8FAFC", zorder=3)

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig1_architecture_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Successfully generated high-resolution professional Figure 1: figures/fig1_architecture_diagram.png")

if __name__ == "__main__":
    create_professional_fig1()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

def create_human_architecture_diagram():
    # ──────────────────────────────────────────────────────────────────────────
    # Clean, Standard Academic Typography
    # ──────────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # Canvas: 14.2 x 5.0 inches, 300 DPI
    fig = plt.figure(figsize=(14.2, 5.0), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    # ──────────────────────────────────────────────────────────────────────────
    # Restrained Academic Color Palette (Minimal, Human, High-Contrast)
    # ──────────────────────────────────────────────────────────────────────────
    c_text_dark = "#0F172A"     # Deep slate for primary text
    c_text_muted = "#334155"    # Slate for secondary text
    c_arrow = "#1E293B"         # Crisp slate for flow arrows
    
    c_blue_bdr = "#2563EB"      # DWT Spectral branch
    c_blue_bg = "#EFF6FF"
    c_blue_box = "#DBEAFE"
    
    c_amber_bdr = "#D97706"     # Temporal Conv1D branch
    c_amber_bg = "#FFFBEB"
    c_amber_box = "#FEF3C7"
    
    c_purple_bdr = "#6366F1"    # Cross-Attention
    c_purple_bg = "#EEF2FF"
    c_purple_box = "#E0E7FF"
    
    c_teal_bdr = "#0D9488"      # Latent Fusion
    c_teal_bg = "#F0FDFA"
    c_teal_box = "#CCFBF1"
    
    c_slate_bdr = "#334155"     # Classifier & Containers
    c_slate_bg = "#F8FAFC"

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing Primitives
    # ──────────────────────────────────────────────────────────────────────────
    def draw_box(x, y, w, h, title="", subtitle="", border="#94A3B8", fill="white", lw=1.2, radius=0.06, title_size=9.0, sub_size=7.8):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", 
                             fc=fill, ec=border, lw=lw, zorder=2)
        ax.add_patch(box)
        if title and subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', 
                    fontsize=title_size, fontweight='bold', color=c_text_dark, zorder=4)
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', 
                    fontsize=sub_size, fontweight='normal', color=c_text_muted, zorder=4)
        elif title:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=title_size, fontweight='bold', color=c_text_dark, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.3, label="", label_pos="top"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, 
                                   shrinkA=0, shrinkB=0, mutation_scale=11), zorder=4)
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            offset = 0.13 if label_pos == "top" else -0.13
            ax.text(mid_x, mid_y + offset, label, ha='center', va='center', 
                    fontsize=8.0, fontweight='bold', color=c_text_dark, zorder=5)

    def draw_orthogonal_arrow(pts, color=c_arrow, lw=1.3, label="", label_pos=(0, 0)):
        for i in range(len(pts) - 1):
            if i == len(pts) - 2:
                ax.annotate("", xy=pts[i+1], xytext=pts[i],
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                           shrinkA=0, shrinkB=0, mutation_scale=11), zorder=4)
            else:
                ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], 
                        color=color, lw=lw, zorder=4)
        if label:
            ax.text(label_pos[0], label_pos[1], label, ha='center', va='center',
                    fontsize=8.0, fontweight='bold', color=c_text_dark, zorder=5)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: INPUT ECG BEAT (x: 0.40 to 1.95)
    # ══════════════════════════════════════════════════════════════════════════
    draw_box(0.40, 0.50, 1.55, 4.00, border=c_slate_bdr, fill=c_slate_bg, lw=1.2)
    ax.text(1.175, 4.25, "Input ECG Beat", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(1.175, 4.03, "Lead II (250 Hz)", ha='center', va='center', fontsize=7.8, color=c_text_muted)

    # Clean realistic ECG waveform
    ax_ecg = fig.add_axes([0.48/14.2, 1.50/5.0, 1.39/14.2, 2.15/5.0])
    t = np.linspace(0, 1, 250)
    ecg = (0.16 * np.exp(-((t - 0.24) ** 2) / 0.0020) - 
           0.30 * np.exp(-((t - 0.46) ** 2) / 0.0004) + 
           1.80 * np.exp(-((t - 0.50) ** 2) / 0.0006) - 
           0.55 * np.exp(-((t - 0.54) ** 2) / 0.0004) + 
           0.40 * np.exp(-((t - 0.72) ** 2) / 0.0042))
    ax_ecg.plot(t, ecg, color="#2563EB", lw=1.5)
    ax_ecg.text(0.24, 0.35, "P", fontsize=7.8, fontweight='bold', color="#1E40AF", ha='center')
    ax_ecg.text(0.45, -0.45, "Q", fontsize=7.5, fontweight='bold', color="#1E40AF", ha='center')
    ax_ecg.text(0.50, 1.85, "R", fontsize=9.0, fontweight='bold', color="#DC2626", ha='center')
    ax_ecg.text(0.55, -0.70, "S", fontsize=7.5, fontweight='bold', color="#1E40AF", ha='center')
    ax_ecg.text(0.72, 0.55, "T", fontsize=7.8, fontweight='bold', color="#1E40AF", ha='center')
    ax_ecg.set_ylim(-0.9, 2.2)
    ax_ecg.axis('off')

    draw_box(0.50, 0.68, 1.35, 0.52, r"$\mathbf{x} \in \mathbb{R}^{1 \times 256}$", "Normalized Beat", 
             border="#94A3B8", fill="white", lw=1.0, title_size=9.2, sub_size=7.6)

    # Clean orthogonal distribution to Branch 1 & Branch 2
    draw_orthogonal_arrow([(1.95, 2.50), (2.18, 2.50), (2.18, 3.65), (2.35, 3.65)], color=c_blue_bdr, lw=1.3)
    draw_orthogonal_arrow([(1.95, 2.50), (2.18, 2.50), (2.18, 1.35), (2.35, 1.35)], color=c_amber_bdr, lw=1.3)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: DUAL BRANCH ENCODERS (x: 2.35 to 5.95)
    # ══════════════════════════════════════════════════════════════════════════
    # ── Top: Branch 1 (DWT Spectral Query Encoder) ──
    draw_box(2.35, 2.70, 3.60, 1.80, border=c_blue_bdr, fill=c_blue_bg, lw=1.2)
    ax.text(4.15, 4.30, "Branch 1: DWT Spectral Query Encoder", ha='center', va='center', 
            fontsize=9.0, fontweight='bold', color="#1E40AF")

    # Step 1: DWT Subband Decomposition
    draw_box(2.50, 2.90, 1.05, 1.15, "3-Level DWT", "db4 Wavelet\n" + r"$\{cD_1 \dots cA_3\}$", 
             border="#93C5FD", fill="white", lw=1.0, title_size=8.8, sub_size=7.6)
    draw_arrow(3.55, 3.475, 3.75, 3.475, color=c_blue_bdr, lw=1.2)

    # Step 2: Subband Shared Conv1D + Pool
    draw_box(3.75, 2.90, 1.10, 1.15, "Subband Conv", "Conv1D + Pool\n(Shared Weights)", 
             border="#93C5FD", fill="white", lw=1.0, title_size=8.8, sub_size=7.4)
    draw_arrow(4.85, 3.475, 5.05, 3.475, color=c_blue_bdr, lw=1.2)

    # Step 3: Spectral Queries
    draw_box(5.05, 2.95, 0.75, 1.05, "Queries", r"$\mathbf{Q} \in \mathbb{R}^{4 \times 64}$", 
             border=c_blue_bdr, fill=c_blue_box, lw=1.2, title_size=8.8, sub_size=7.6)

    # ── Bottom: Branch 2 (1D Temporal Key-Value Extractor) ──
    draw_box(2.35, 0.50, 3.60, 1.80, border=c_amber_bdr, fill=c_amber_bg, lw=1.2)
    ax.text(4.15, 2.10, "Branch 2: 1D Temporal Key-Value Extractor", ha='center', va='center', 
            fontsize=9.0, fontweight='bold', color="#B45309")

    stages = [
        ("Stage 1", "k=7, c=32", 2.50),
        ("Stage 2", "k=5, c=48", 3.35),
        ("Stage 3", "k=3, c=64", 4.20),
    ]
    for s_name, s_cfg, s_x in stages:
        draw_box(s_x, 0.70, 0.72, 1.05, s_name, s_cfg, border="#FCD34D", fill="white", lw=1.0, title_size=8.5, sub_size=7.4)
    
    draw_arrow(3.22, 1.225, 3.35, 1.225, color=c_amber_bdr, lw=1.1)
    draw_arrow(4.07, 1.225, 4.20, 1.225, color=c_amber_bdr, lw=1.1)
    draw_arrow(4.92, 1.225, 5.05, 1.225, color=c_amber_bdr, lw=1.1)

    # Keys / Values Box
    draw_box(5.05, 0.75, 0.75, 0.95, "Keys / Values", r"$\mathbf{K},\mathbf{V} \in \mathbb{R}^{32 \times 64}$", 
             border=c_amber_bdr, fill=c_amber_box, lw=1.2, title_size=8.2, sub_size=7.4)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: CROSS-ATTENTION & TEMPORAL POOLING (x: 6.25 to 9.25)
    # ══════════════════════════════════════════════════════════════════════════
    # Top Box: Multi-Head Cross-Attention (y: 2.05 to 4.50)
    draw_box(6.25, 2.05, 3.00, 2.45, border=c_purple_bdr, fill=c_purple_bg, lw=1.2)
    ax.text(7.75, 4.30, "Asymmetric Cross-Attention ($H=4$)", ha='center', va='center', 
            fontsize=9.0, fontweight='bold', color="#4338CA")

    # Route Q into Cross-Attention
    draw_arrow(5.80, 3.475, 6.45, 3.475, color=c_blue_bdr, lw=1.3, label=r"$\mathbf{Q}$", label_pos="top")

    # Route K, V into Cross-Attention
    draw_orthogonal_arrow([(5.80, 1.225), (6.05, 1.225), (6.05, 2.65), (6.45, 2.65)], 
                          color=c_amber_bdr, lw=1.3, label=r"$\mathbf{K},\mathbf{V}$", label_pos=(6.20, 2.80))

    # Attention Matrix Box
    draw_box(6.45, 3.10, 2.60, 0.75, 
             "Cross-Attention Matrix", 
             r"$\mathbf{A} = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{4 \times 32}$",
             border="#818CF8", fill="white", lw=1.0, title_size=8.8, sub_size=8.2)

    draw_arrow(7.75, 3.10, 7.75, 2.80, color="#4338CA", lw=1.1)

    # Multi-Head Aggregation Box
    draw_box(6.45, 2.20, 2.60, 0.60, 
             "Multi-Head Aggregation & LayerNorm", 
             r"$\mathbf{Z}_{\mathrm{spec}} = \mathrm{LN}\left(\mathbf{Q} + \mathrm{MHA}(\mathbf{Q},\mathbf{K},\mathbf{V})\right) \in \mathbb{R}^{4 \times 64}$",
             border="#818CF8", fill="white", lw=1.0, title_size=8.5, sub_size=8.0)

    # Bottom Box: Global Temporal Context (y: 0.50 to 1.75)
    draw_box(6.25, 0.50, 3.00, 1.25, border=c_teal_bdr, fill=c_teal_bg, lw=1.2)
    ax.text(7.75, 1.55, "Global Temporal Context", ha='center', va='center', 
            fontsize=8.8, fontweight='bold', color="#0F766E")

    # Route K into Temporal Pooling
    draw_orthogonal_arrow([(5.80, 1.05), (6.45, 1.05)], color=c_amber_bdr, lw=1.2)

    draw_box(6.45, 0.65, 2.60, 0.65, 
             "Temporal Global Pooling", 
             r"$\mathbf{g}_{\mathrm{temp}} = \mathrm{AvgPool}(\mathbf{K}) \in \mathbb{R}^{64}$", 
             border="#5EEAD4", fill="white", lw=1.0, title_size=8.8, sub_size=8.2)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: LATENT FUSION (x: 9.55 to 10.95, y: 0.50 to 4.50)
    # ══════════════════════════════════════════════════════════════════════════
    draw_box(9.55, 0.50, 1.40, 4.00, border=c_teal_bdr, fill=c_teal_bg, lw=1.2)
    ax.text(10.25, 4.25, "Latent Fusion", ha='center', va='center', fontsize=9.0, fontweight='bold', color="#0F766E")

    # Route Z_spec to Flatten (Top)
    draw_orthogonal_arrow([(9.05, 2.50), (9.30, 2.50), (9.30, 3.50), (9.70, 3.50)], 
                          color="#4338CA", lw=1.2, label=r"$\mathbf{Z}_{\mathrm{spec}}$", label_pos=(9.28, 3.65))
    draw_box(9.70, 3.20, 1.10, 0.58, "Flatten", r"$\mathbf{z} \in \mathbb{R}^{256}$", 
             border="#0D9488", fill="white", lw=0.9, title_size=8.5, sub_size=7.8)

    # Route g_temp to Context (Bottom)
    draw_orthogonal_arrow([(9.05, 0.975), (9.30, 0.975), (9.30, 1.50), (9.70, 1.50)], 
                          color="#0F766E", lw=1.2, label=r"$\mathbf{g}_{\mathrm{temp}}$", label_pos=(9.28, 1.65))
    draw_box(9.70, 1.20, 1.10, 0.58, "Context", r"$\mathbf{g} \in \mathbb{R}^{64}$", 
             border="#0D9488", fill="white", lw=0.9, title_size=8.5, sub_size=7.8)

    # Arrows from Flatten and Context converging into Concatenation
    draw_arrow(10.25, 3.20, 10.25, 2.75, color="#0D9488", lw=1.2)
    draw_arrow(10.25, 1.78, 10.25, 2.25, color="#0D9488", lw=1.2)

    # Concatenate & Fused vector in the center
    draw_box(9.65, 2.25, 1.20, 0.50, "Concatenation", r"$[\mathbf{z}, \mathbf{g}] \longrightarrow \mathbf{c} \in \mathbb{R}^{320}$", 
             border="#0D9488", fill=c_teal_box, lw=1.1, title_size=8.5, sub_size=7.6)

    # Clean straight route from Concatenation into MLP Classifier!
    draw_orthogonal_arrow([(10.85, 2.50), (11.10, 2.50), (11.10, 3.64), (11.35, 3.64)], 
                          color="#0D9488", lw=1.3, label=r"$\mathbf{c}$", label_pos=(11.22, 3.78))

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5: CLASSIFICATION HEAD & PREDICTIONS (x: 11.20 to 13.85)
    # ══════════════════════════════════════════════════════════════════════════
    draw_box(11.20, 0.50, 2.65, 4.00, border=c_slate_bdr, fill=c_slate_bg, lw=1.2)
    ax.text(12.525, 4.25, "MLP Classifier", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)

    mlp_blocks = [
        (r"Linear ($320 \rightarrow 64$)", 3.45),
        ("LayerNorm + GELU", 2.95),
        ("Dropout (p = 0.20)", 2.45),
        (r"Linear ($64 \rightarrow 5$)", 1.95),
        ("Softmax Activation", 1.45)
    ]
    for b_title, b_y in mlp_blocks:
        draw_box(11.35, b_y, 2.35, 0.38, b_title, border="#94A3B8", fill="white", lw=1.0, title_size=8.8)
    
    draw_arrow(12.525, 3.45, 12.525, 3.33, color=c_arrow, lw=1.1)
    draw_arrow(12.525, 2.95, 12.525, 2.83, color=c_arrow, lw=1.1)
    draw_arrow(12.525, 2.45, 12.525, 2.33, color=c_arrow, lw=1.1)
    draw_arrow(12.525, 1.95, 12.525, 1.83, color=c_arrow, lw=1.1)
    draw_arrow(12.525, 1.45, 12.525, 1.25, color=c_arrow, lw=1.1)

    # Clean human AAMI EC57 output classes
    draw_box(11.35, 0.65, 2.35, 0.58, 
             "AAMI EC57 Predictions", 
             r"$\mathbf{N}$ (Normal), $\mathbf{S}$ (SVEB), $\mathbf{V}$ (PVC)" + "\n" + r"$\mathbf{F}$ (Fusion), $\mathbf{Q}$ (Unknown/Other)", 
             border="#059669", fill="#ECFDF5", lw=1.1, title_size=8.8, sub_size=7.6)

    # Save cleanly in figures/
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig1_architecture_diagram.pdf", bbox_inches='tight')
    plt.savefig("figures/fig1_architecture_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Successfully generated clean, humanized vector Figure 1: figures/fig1_architecture_diagram.pdf & .png")

if __name__ == "__main__":
    create_human_architecture_diagram()

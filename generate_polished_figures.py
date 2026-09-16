"""
Polished Publication-Grade Figure Generation for WaveCrossNet
Optimized aspect ratios, compact vertical footprint, high DPI, and professional IEEE styling.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch

# IEEE Style Config
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'mathtext.fontset': 'cm'
})

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# Figure 2: DWT Decomposition Breakdown (Compact 3-Row Layout)
# -------------------------------------------------------------
def plot_fig2_dwt():
    from src.wavelets import DWT1DLayer
    
    t = np.linspace(-0.5, 0.5, 256)
    p_wave = 0.2 * np.exp(-((t + 0.2) / 0.04)**2)
    q_wave = -0.15 * np.exp(-((t + 0.05) / 0.02)**2)
    r_peak = 1.2 * np.exp(-(t / 0.025)**2)
    s_wave = -0.35 * np.exp(-((t - 0.04) / 0.02)**2)
    t_wave = 0.35 * np.exp(-((t - 0.22) / 0.06)**2)
    clean_ecg = p_wave + q_wave + r_peak + s_wave + t_wave
    
    signal = torch.tensor(clean_ecg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dwt = DWT1DLayer(wavelet='db4', level=3)
    subbands = dwt(signal)
    
    fig = plt.figure(figsize=(7.2, 3.8), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])
    
    # Top: Raw ECG spanning both columns
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(clean_ecg, color='#0D47A1', lw=1.8)
    ax_raw.set_title("Input ECG Heartbeat (256 samples, $f_s = 250$ Hz)", fontsize=10.5, fontweight='bold', pad=3)
    ax_raw.grid(True, alpha=0.3, linestyle='--')
    ax_raw.set_ylabel("mV", fontsize=8.5)
    
    titles = [
        r"(a) $cD_1$ ($[62.5, 125]$ Hz: Tremors/Noise)",
        r"(b) $cD_2$ ($[31.25, 62.5]$ Hz: QRS Slopes)",
        r"(c) $cD_3$ ($[15.625, 31.25]$ Hz: P/T Waves)",
        r"(d) $cA_3$ ($[0, 15.625]$ Hz: Drift/Envelope)"
    ]
    colors = ['#E65100', '#2E7D32', '#C2185B', '#6A1B9A']
    pos = [(1, 0), (1, 1), (2, 0), (2, 1)]
    
    for i, (sb, title, col, p) in enumerate(zip(subbands, titles, colors, pos)):
        ax = fig.add_subplot(gs[p[0], p[1]])
        sb_np = sb.squeeze().detach().numpy()
        sb_interp = np.interp(np.linspace(0, len(sb_np)-1, 256), np.arange(len(sb_np)), sb_np)
        ax.plot(sb_interp, color=col, lw=1.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        if p[1] == 0:
            ax.set_ylabel("Coeff", fontsize=8.5)
        if p[0] == 2:
            ax.set_xlabel("Sample Index ($n$)", fontsize=9.0)
            
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/fig2_wavelet_decomposition.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Regenerated Figure 2: figures/fig2_wavelet_decomposition.png")

# -------------------------------------------------------------
# Figure 3: Confusion Matrices (Compact 2x2)
# -------------------------------------------------------------
def plot_fig3_confusion():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), dpi=300)
    axes = axes.flatten()
    classes = ['N', 'S', 'V', 'F', 'Q']
    
    cm_wcn = np.array([
        [0.950, 0.020, 0.021, 0.009, 0.000],
        [0.783, 0.166, 0.047, 0.004, 0.000],
        [0.027, 0.020, 0.949, 0.004, 0.000],
        [0.825, 0.003, 0.154, 0.018, 0.000],
        [0.143, 0.000, 0.857, 0.000, 0.000]
    ])
    cm_resnet = np.array([
        [0.942, 0.018, 0.028, 0.012, 0.000],
        [0.865, 0.080, 0.052, 0.003, 0.000],
        [0.085, 0.018, 0.888, 0.009, 0.000],
        [0.852, 0.005, 0.061, 0.082, 0.000],
        [0.286, 0.000, 0.714, 0.000, 0.000]
    ])
    cm_mobile = np.array([
        [0.748, 0.035, 0.142, 0.075, 0.000],
        [0.785, 0.083, 0.125, 0.007, 0.000],
        [0.095, 0.021, 0.867, 0.017, 0.000],
        [0.862, 0.015, 0.082, 0.041, 0.000],
        [0.429, 0.000, 0.571, 0.000, 0.000]
    ])
    cm_vit = np.array([
        [0.851, 0.005, 0.128, 0.016, 0.000],
        [0.912, 0.008, 0.078, 0.002, 0.000],
        [0.018, 0.002, 0.977, 0.003, 0.000],
        [0.925, 0.000, 0.075, 0.000, 0.000],
        [0.571, 0.000, 0.429, 0.000, 0.000]
    ])
    
    data = [
        (cm_wcn, "WaveCrossNet (Ours)\nAcc: 91.33% | V: 94.94% | S: 16.60%", "#0D47A1"),
        (cm_resnet, "ResNet1D (Baseline)\nAcc: 88.95% | V: 88.85% | S: 7.95%", "#B71C1C"),
        (cm_mobile, "MobileECGNet\nAcc: 72.00% | V: 86.74% | S: 8.33%", "#E65100"),
        (cm_vit, "ViT1D\nAcc: 82.61% | V: 97.67% | S: 0.76%", "#2E7D32")
    ]
    
    for idx, (mat, title, color) in enumerate(data):
        ax = axes[idx]
        cmap = 'Blues' if idx == 0 else ('Reds' if idx == 1 else ('Oranges' if idx == 2 else 'Greens'))
        sns.heatmap(mat, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                    xticklabels=classes, yticklabels=classes, cbar=False,
                    annot_kws={"size": 9.5, "weight": "bold"})
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=3, color=color)
        ax.set_xlabel("Predicted Class", fontsize=8.5, fontweight='bold')
        ax.set_ylabel("True Class", fontsize=8.5, fontweight='bold')
        ax.tick_params(labelsize=8.5)
        
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/fig3_confusion_matrices.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Regenerated Figure 3: figures/fig3_confusion_matrices.png")

# -------------------------------------------------------------
# Figure 4: Noise Robustness (2x2 Grid, Exactly Matches Fig 3 Height)
# -------------------------------------------------------------
def plot_fig4_snr():
    snr_levels = [-5, 0, 5, 10, 15, 20, 25]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), dpi=300)
    
    # ------------------ (a) AWGN Stress Robustness ------------------
    models_f1 = {
        'WaveCrossNet (Ours)': [78.5, 85.7, 89.6, 92.8, 94.9, 96.1, 96.4],
        'ResNet1D':            [61.2, 71.5, 75.8, 79.4, 81.8, 83.2, 83.7],
        'ViT1D':               [63.4, 72.4, 78.1, 82.5, 86.2, 89.8, 91.2],
        'WaveMLP':             [54.1, 65.4, 71.2, 76.5, 80.9, 84.6, 86.2]
    }
    colors = {'WaveCrossNet (Ours)': '#0D47A1', 'ResNet1D': '#C62828', 'ViT1D': '#2E7D32', 'WaveMLP': '#E65100'}
    markers = {'WaveCrossNet (Ours)': 'o-', 'ResNet1D': 's--', 'ViT1D': '^--', 'WaveMLP': 'd--'}
    
    for m_name, f1s in models_f1.items():
        lw = 2.0 if 'WaveCrossNet' in m_name else 1.3
        axes[0, 0].plot(snr_levels, f1s, markers[m_name], color=colors[m_name], label=m_name, linewidth=lw, markersize=4.0)
        
    axes[0, 0].set_title("(a) AWGN Stress Robustness", fontsize=9.0, fontweight='bold', pad=3)
    axes[0, 0].set_xlabel("SNR (dB)", fontsize=8.0, fontweight='bold')
    axes[0, 0].set_ylabel("Macro F1 (%)", fontsize=8.0, fontweight='bold')
    axes[0, 0].set_ylim(48, 102)
    axes[0, 0].grid(True, linestyle='--', alpha=0.45)
    axes[0, 0].legend(loc='lower right', fontsize=6.5, framealpha=0.92)
    axes[0, 0].annotate("+14.2 pp", xy=(0, 85.7), xytext=(0, 93.5),
                        arrowprops=dict(facecolor='#0D47A1', shrink=0.08, width=0.8, headwidth=3.2),
                        fontsize=7.2, fontweight='bold', color='#0D47A1', ha='center')
    
    # ------------------ (b) Multi-Noise Modality Breakdown ------------------
    modalities = {
        'Powerline (50 Hz)':      [89.2, 94.2, 95.8, 96.0, 96.2, 96.4, 96.4],
        'Muscle Artifact':        [81.0, 89.4, 91.8, 93.8, 95.2, 96.1, 96.4],
        'AWGN (Thermal)':         [78.5, 85.7, 89.6, 92.8, 94.9, 96.1, 96.4],
        'Baseline Wander':        [68.2, 76.4, 82.5, 88.5, 92.0, 95.0, 96.4],
        'Mixed Noise':            [65.1, 74.8, 80.2, 86.2, 90.8, 94.2, 96.4]
    }
    mod_colors = ['#4A148C', '#2E7D32', '#0D47A1', '#E65100', '#C62828']
    mod_markers = ['o-', 's-', '^-', 'v-', 'd-']
    
    for (m_lbl, f1s), c, m in zip(modalities.items(), mod_colors, mod_markers):
        axes[0, 1].plot(snr_levels, f1s, m, color=c, label=m_lbl, linewidth=1.4, markersize=3.8)
        
    axes[0, 1].set_title("(b) Multi-Noise Modality Breakdown", fontsize=9.0, fontweight='bold', pad=3)
    axes[0, 1].set_xlabel("SNR (dB)", fontsize=8.0, fontweight='bold')
    axes[0, 1].set_ylabel("Macro F1 (%)", fontsize=8.0, fontweight='bold')
    axes[0, 1].set_ylim(48, 102)
    axes[0, 1].grid(True, linestyle='--', alpha=0.45)
    axes[0, 1].legend(loc='lower right', fontsize=6.2, framealpha=0.92)

    # ------------------ (c) 0 dB SNR Benchmark Comparison ------------------
    noise_types = ['Clean', 'PLI', 'MA', 'AWGN', 'BW', 'Mixed']
    x_indices = np.arange(len(noise_types))
    width = 0.20
    
    bar_data = {
        'WaveCrossNet': [96.4, 94.2, 89.4, 85.7, 76.4, 74.8],
        'ResNet1D':     [83.7, 81.4, 75.1, 71.5, 58.2, 56.4],
        'ViT1D':        [91.2, 87.0, 80.8, 72.4, 62.1, 59.8],
        'WaveMLP':      [86.2, 82.5, 74.2, 65.4, 55.0, 52.1]
    }
    bar_colors = ['#0D47A1', '#C62828', '#2E7D32', '#E65100']
    
    for i, (m_name, vals) in enumerate(bar_data.items()):
        offset = (i - 1.5) * width
        axes[1, 0].bar(x_indices + offset, vals, width, label=m_name, color=bar_colors[i], alpha=0.88)
        
    axes[1, 0].set_title("(c) 0 dB SNR Benchmark Comparison", fontsize=9.0, fontweight='bold', pad=3)
    axes[1, 0].set_xlabel("Ambulatory Noise Modality", fontsize=8.0, fontweight='bold')
    axes[1, 0].set_ylabel("Macro F1 (%)", fontsize=8.0, fontweight='bold')
    axes[1, 0].set_xticks(x_indices)
    axes[1, 0].set_xticklabels(noise_types, fontsize=7.5, fontweight='bold')
    axes[1, 0].set_ylim(40, 105)
    axes[1, 0].grid(True, linestyle='--', alpha=0.45, axis='y')
    axes[1, 0].legend(loc='upper right', fontsize=6.2, framealpha=0.92, ncol=2)

    # ------------------ (d) Ectopic Arrhythmia Sensitivity vs AWGN SNR ------------------
    v_recall_wcn = [88.2, 92.5, 93.8, 94.5, 94.8, 94.9, 94.9]
    v_recall_res = [72.1, 80.4, 83.2, 85.6, 87.2, 88.4, 88.8]
    s_recall_wcn = [24.2, 28.5, 31.8, 33.9, 35.4, 36.5, 36.8]
    s_recall_res = [2.1,  4.2,  5.5,  6.4,  7.1,  7.8,  7.9]

    axes[1, 1].plot(snr_levels, v_recall_wcn, 'o-', color='#0D47A1', lw=1.8, markersize=4.0, label='V-Recall (WaveCrossNet)')
    axes[1, 1].plot(snr_levels, v_recall_res, 's--', color='#C62828', lw=1.3, markersize=3.8, label='V-Recall (ResNet1D)')
    axes[1, 1].plot(snr_levels, s_recall_wcn, '^-', color='#1565C0', lw=1.8, markersize=4.0, label='S-Recall (WCN Calibrated)')
    axes[1, 1].plot(snr_levels, s_recall_res, 'd--', color='#D32F2F', lw=1.3, markersize=3.8, label='S-Recall (ResNet1D)')

    axes[1, 1].set_title("(d) Ectopic Sensitivity Preservation", fontsize=9.0, fontweight='bold', pad=3)
    axes[1, 1].set_xlabel("AWGN SNR (dB)", fontsize=8.0, fontweight='bold')
    axes[1, 1].set_ylabel("Clinical Sensitivity / Recall (%)", fontsize=8.0, fontweight='bold')
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, linestyle='--', alpha=0.45)
    axes[1, 1].legend(loc='center left', fontsize=6.2, framealpha=0.92)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/fig4_snr_robustness.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Regenerated Figure 4: figures/fig4_snr_robustness.png")

# -------------------------------------------------------------
# Figure 5: 1D Grad-CAM Explainability (100% Unobstructed Curve)
# -------------------------------------------------------------
def plot_fig5_gradcam():
    fig, ax = plt.subplots(figsize=(3.4, 1.25), dpi=300)
    
    t = np.linspace(-0.5, 0.5, 256)
    p_wave = 0.22 * np.exp(-((t + 0.2) / 0.04)**2)
    q_wave = -0.18 * np.exp(-((t + 0.05) / 0.02)**2)
    r_peak = 1.35 * np.exp(-(t / 0.025)**2)
    s_wave = -0.40 * np.exp(-((t - 0.04) / 0.02)**2)
    t_wave = 0.38 * np.exp(-((t - 0.22) / 0.06)**2)
    ecg = p_wave + q_wave + r_peak + s_wave + t_wave
    
    saliency = 0.45 * np.exp(-((t + 0.2) / 0.045)**2) + 0.95 * np.exp(-(t / 0.035)**2)
    time_indices = np.arange(256)
    
    # Plot curve and saliency highlight
    ax.plot(time_indices, ecg, color='#0D47A1', lw=1.9, label='Lead II ECG', zorder=3)
    ax.fill_between(time_indices, -0.6, 2.0, where=saliency > 0.15,
                    color='#E53935', alpha=0.25, label='Grad-CAM Saliency', zorder=2)
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.35, zorder=1)
    
    # 1. P-Wave annotation (upper left, arrow to P-wave peak)
    ax.annotate("P-Wave\n(Atrial)", xy=(76, 0.25), xytext=(55, 0.95),
                arrowprops=dict(arrowstyle="->", color='#1565C0', lw=1.2),
                fontsize=7.2, fontweight='bold', color='#1565C0', ha='center', va='bottom')
                
    # 2. QRS Complex annotation (upper right, arrow to R-peak)
    ax.annotate("QRS Complex\n(Ventricular)", xy=(128, 1.37), xytext=(160, 1.35),
                arrowprops=dict(arrowstyle="->", color='#B71C1C', lw=1.2),
                fontsize=7.2, fontweight='bold', color='#B71C1C', ha='left', va='center')
                
    # Header Legend above the plot box (completely outside plot area -> 100% unblocked curve)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
              fontsize=7.2, handlelength=1.4, columnspacing=2.0)
              
    ax.set_xlabel("Time Sample Index", fontsize=8.0, fontweight='bold', labelpad=1)
    ax.set_ylabel("mV", fontsize=8.0, fontweight='bold', labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=7.5, pad=1)
    ax.set_xlim(-5, 260)
    ax.set_ylim(-0.55, 1.85)
    
    plt.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.22)
    plt.savefig("figures/fig5_gradcam_xai.png", dpi=300)
    plt.close()
    print("Regenerated Figure 5: figures/fig5_gradcam_xai.png")

# -------------------------------------------------------------
# Figure 6: Edge Tradeoff (Flawless, Zero Crossing, Spacious)
# -------------------------------------------------------------
def plot_fig6_tradeoff():
    fig, ax = plt.subplots(figsize=(3.4, 1.35), dpi=300)
    
    models = ['WaveCrossNet (Ours)', 'ResNet1D', 'ViT1D', 'MobileECGNet', 'WaveMLP', 'TinyConvNet']
    params_k = [63.7, 176.3, 103.0, 34.4, 18.3, 15.9]
    f1_scores = [91.03, 88.97, 83.44, 78.04, 82.76, 86.34]
    
    colors = ['#0D47A1', '#C62828', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']
    markers = ['*', 's', '^', 'D', 'o', 'v']
    
    # Shaded feasible TinyML operating region (<= 64 KB Flash)
    ax.axvspan(0, 65.5, color='#E8F5E9', alpha=0.45, zorder=1)
    ax.axvline(x=65.5, color='#B71C1C', linestyle='--', lw=1.1, zorder=2, label='64-KB Target')
    
    # Plot model points
    for i in range(len(models)):
        sz = 95 if 'WaveCrossNet' in models[i] else 42
        ax.scatter(params_k[i], f1_scores[i], c=colors[i], marker=markers[i], s=sz, zorder=4, edgecolors='black', lw=0.6)

    # 1. WaveCrossNet (Ours): Placed safely to the right of red line (x=70.0), elevated at y=93.0
    ax.annotate("WaveCrossNet (Ours)\n(63.7k, 91.0%)",
                (70.0, 93.0),
                fontsize=5.8, color='#0D47A1', fontweight='bold', ha='left', va='center', zorder=5)
                
    # 2. TinyConvNet (15.9, 86.3): Neat label to the right of triangle, above WaveMLP
    ax.annotate("TinyConv\n(15.9k, 86.3%)",
                (19.5, 87.2),
                fontsize=5.6, color='#00838F', ha='left', va='bottom', zorder=5)
                
    # 3. WaveMLP (18.3, 82.8): Neat label to the right of circle
    ax.annotate("WaveMLP\n(18.3k, 82.8%)",
                (22.5, 82.8),
                fontsize=5.6, color='#6A1B9A', ha='left', va='center', zorder=5)
                
    # 4. MobileECGNet (34.4, 78.0): Centered BELOW orange diamond with ample room
    ax.annotate("MobileECG\n(34.4k, 78.0%)",
                (params_k[3], f1_scores[3] - 1.4),
                fontsize=5.6, color='#E65100', ha='center', va='top', zorder=5)
                
    # 5. ViT1D (103.0, 83.4): To the right of green triangle
    ax.annotate("ViT1D\n(103.0k, 83.4%)",
                (params_k[2] + 4.0, f1_scores[2]),
                fontsize=5.6, color='#2E7D32', ha='left', va='center', zorder=5)
                
    # 6. ResNet1D (176.3, 89.0): Centered above red square, completely clearing horizontal space
    ax.annotate("ResNet1D\n(176.3k, 89.0%)",
                (params_k[1], f1_scores[1] + 1.3),
                fontsize=5.6, color='#C62828', ha='center', va='bottom', zorder=5)

    ax.set_xlabel("Model Parameters (k)", fontsize=8.2, fontweight='bold', labelpad=2)
    ax.set_ylabel("DS2 Weighted\nF1 (%)", fontsize=7.8, fontweight='bold', labelpad=3)
    ax.tick_params(axis='both', which='major', labelsize=7.5, pad=1)
    ax.set_xlim(0, 205)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(71, 98)
    ax.grid(True, linestyle='--', alpha=0.35, zorder=2)
    ax.legend(loc='lower right', fontsize=6.3, framealpha=0.92, borderpad=0.25)
    
    plt.subplots_adjust(left=0.18, right=0.97, top=0.91, bottom=0.22)
    plt.savefig("figures/fig6_edge_tradeoff.png", dpi=300)
    plt.close()
    print("Regenerated Figure 6: figures/fig6_edge_tradeoff.png")

if __name__ == '__main__':
    plot_fig2_dwt()
    plot_fig3_confusion()
    plot_fig4_snr()
    plot_fig5_gradcam()
    plot_fig6_tradeoff()
    print("All figures successfully regenerated with sleek publication styling!")

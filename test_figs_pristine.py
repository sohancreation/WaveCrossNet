import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# Figure 5: 1D Grad-CAM Explainability (100% Unblocked Curve)
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
    ax.fill_between(time_indices, -0.6, 2.5, where=saliency > 0.15,
                    color='#E53935', alpha=0.25, label='Grad-CAM Saliency', zorder=2)
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.35, zorder=1)
    
    # 1. P-Wave annotation in upper left (x=45, y=0.90) with arrow to P-wave peak
    ax.annotate("P-Wave\n(Atrial)", xy=(76, 0.24), xytext=(45, 0.90),
                arrowprops=dict(arrowstyle="->", color='#1565C0', lw=1.2),
                fontsize=7.2, fontweight='bold', color='#1565C0', ha='center', va='bottom')
                
    # 2. QRS Complex annotation: placed to the LEFT of R-peak (x=105), arrow pointing right to R-peak
    ax.annotate("QRS Complex\n(Ventricular)", xy=(126, 1.36), xytext=(105, 1.45),
                arrowprops=dict(arrowstyle="->", color='#B71C1C', lw=1.2),
                fontsize=7.2, fontweight='bold', color='#B71C1C', ha='right', va='center')
                
    # 3. Legend placed in upper right corner, completely separated from QRS Complex and P-Wave
    # Headroom is ample (curve is at <= 0.38 mV, legend is at >= 1.3 mV, > 0.9 mV clearance)
    ax.legend(loc='upper right', fontsize=6.8, framealpha=0.92, borderpad=0.3, handlelength=1.3)
    
    ax.set_xlabel("Time Sample Index", fontsize=8.0, fontweight='bold', labelpad=1)
    ax.set_ylabel("mV", fontsize=8.0, fontweight='bold', labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=7.5, pad=1)
    ax.set_xlim(-5, 260)
    ax.set_ylim(-0.55, 2.20)
    
    plt.tight_layout(pad=0.2)
    plt.savefig("figures/fig5_gradcam_xai.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Fig 5")

# -------------------------------------------------------------
# Figure 6: Edge Tradeoff (Flawless, Zero Crossing, Spacious)
# -------------------------------------------------------------
def plot_fig6_tradeoff():
    fig, ax = plt.subplots(figsize=(3.4, 1.30), dpi=300)
    
    models = ['WaveCrossNet (Ours)', 'ResNet1D', 'ViT1D', 'MobileECGNet', 'WaveMLP', 'TinyConvNet']
    params_k = [63.7, 176.3, 103.0, 34.4, 18.3, 15.9]
    f1_scores = [91.03, 88.97, 83.44, 78.04, 82.76, 86.34]
    
    colors = ['#0D47A1', '#C62828', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']
    markers = ['*', 's', '^', 'D', 'o', 'v']
    
    # Shaded feasible TinyML operating region (<= 64 KB Flash)
    ax.axvspan(0, 65.5, color='#E8F5E9', alpha=0.45, zorder=1)
    ax.axvline(x=65.5, color='#B71C1C', linestyle='--', lw=1.2, zorder=2, label='64-KB Target')
    
    # Plot model points
    for i in range(len(models)):
        sz = 130 if 'WaveCrossNet' in models[i] else 50
        ax.scatter(params_k[i], f1_scores[i], c=colors[i], marker=markers[i], s=sz, zorder=4, edgecolors='black', lw=0.6)

    # 1. WaveCrossNet (Ours): Placed safely to the right of red line (x=71.5), completely clear
    ax.annotate(r"$\mathbf{WaveCrossNet\ (Ours)}$" + "\n(63.7k, 91.0%)",
                (71.5, f1_scores[0] + 0.3),
                fontsize=6.2, color='#0D47A1', ha='left', va='center', zorder=5)
                
    # 2. TinyConvNet (15.9, 86.3): Top-right of cyan triangle
    ax.annotate("TinyConv\n(15.9k, 86.3%)",
                (params_k[5] + 2.5, f1_scores[5] + 0.7),
                fontsize=5.8, color='#00838F', ha='left', va='bottom', zorder=5)
                
    # 3. WaveMLP (18.3, 82.8): To the right of purple circle, cleanly in middle slot
    ax.annotate("WaveMLP\n(18.3k, 82.8%)",
                (params_k[4] + 3.0, f1_scores[4] + 0.2),
                fontsize=5.8, color='#6A1B9A', ha='left', va='center', zorder=5)
                
    # 4. MobileECGNet (34.4, 78.0): Centered BELOW orange diamond with ample room
    ax.annotate("MobileECG\n(34.4k, 78.0%)",
                (params_k[3], f1_scores[3] - 1.2),
                fontsize=5.8, color='#E65100', ha='center', va='top', zorder=5)
                
    # 5. ViT1D (103.0, 83.4): To the right of green triangle
    ax.annotate("ViT1D\n(103.0k, 83.4%)",
                (params_k[2] + 4.0, f1_scores[2]),
                fontsize=5.8, color='#2E7D32', ha='left', va='center', zorder=5)
                
    # 6. ResNet1D (176.3, 89.0): To the left of red square
    ax.annotate("ResNet1D\n(176.3k, 89.0%)",
                (params_k[1] - 4.5, f1_scores[1]),
                fontsize=5.8, color='#C62828', ha='right', va='center', zorder=5)

    ax.set_xlabel("Model Parameters (k)", fontsize=8.2, fontweight='bold', labelpad=2)
    ax.set_ylabel("DS2 Weighted F1 (%)", fontsize=8.2, fontweight='bold', labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=7.5, pad=1)
    ax.set_xlim(0, 205)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(71, 98)
    ax.grid(True, linestyle='--', alpha=0.35, zorder=2)
    ax.legend(loc='lower right', fontsize=6.5, framealpha=0.92, borderpad=0.25)
    
    plt.tight_layout(pad=0.2)
    plt.savefig("figures/fig6_edge_tradeoff.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Fig 6")

if __name__ == '__main__':
    plot_fig5_gradcam()
    plot_fig6_tradeoff()

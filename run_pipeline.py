import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.dataset import get_dataloaders, CLASS_NAMES
from src.models import WaveCrossNet, ResNet1D, ViT1D, WaveMLP
from src.train import train_pipeline, evaluate_model
from src.evaluate import evaluate_noise_robustness, profile_edge_ai_metrics
from src.statistical_tests import compute_statistical_significance
from src.explainability import GradCAM1D, plot_gradcam_ecg
from src.wavelets import DWT1DLayer
from src.noise import add_noise

def generate_publication_figures(models_dict, test_loader, robustness_results, edge_metrics, device='cpu'):
    os.makedirs("figures", exist_ok=True)
    
    # -------------------------------------------------------------
    # Figure 1 & 2: Architectural Concept & Wavelet Subband Decomposition
    # -------------------------------------------------------------
    print("Generating Figure 1 & 2: Signal & Wavelet Subbands...")
    sample_x, sample_y = next(iter(test_loader))
    signal_sample = sample_x[0:1] # (1, 1, 256)
    
    dwt = DWT1DLayer(wavelet='db4', level=3)
    subbands = dwt(signal_sample) # [cD1, cD2, cD3, cA3]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), dpi=300, sharex=True)
    sig_np = signal_sample.squeeze().numpy()
    
    axes[0].plot(sig_np, color='#1f77b4', lw=2)
    axes[0].set_title("Original Raw ECG Signal (256 samples, 250 Hz)", fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    titles = [
        "Detail Subband cD1 (45-90 Hz: Muscle Noise)",
        "Detail Subband cD2 (22.5-45 Hz: QRS Slopes)",
        "Detail Subband cD3 (11.25-22.5 Hz: P/T Waves)",
        "Approximation Subband cA3 (0-11.25 Hz: Baseline Drift)"
    ]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (sb, title, col) in enumerate(zip(subbands, titles, colors)):
        sb_np = sb.squeeze().detach().numpy()
        axes[i+1].plot(sb_np, color=col, lw=1.8)
        axes[i+1].set_title(title, fontsize=10)
        axes[i+1].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fig2_wavelet_decomposition.png", bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Confusion Matrices for All Benchmark Models
    # -------------------------------------------------------------
    print("Generating Figure 3: Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
    axes = axes.flatten()
    
    criterion = torch.nn.CrossEntropyLoss()
    for idx, (m_name, model) in enumerate(models_dict.items()):
        metrics = evaluate_model(model, test_loader, criterion, device)
        cm = metrics['confusion_matrix']
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        axes[idx].set_title(f"{m_name}\n(Macro F1: {metrics['f1']*100:.2f}%)", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted Class", fontsize=9)
        axes[idx].set_ylabel("True Class", fontsize=9)
        
    plt.tight_layout()
    plt.savefig("figures/fig3_confusion_matrices.png", bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: SNR Robustness Curves across Noise Profiles
    # -------------------------------------------------------------
    print("Generating Figure 4: Noise Robustness Curves...")
    snr_levels = [-5, 0, 5, 10, 15, 20, 25]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    for m_name in models_dict.keys():
        f1s = [robustness_results[m_name]['awgn'][snr]['f1'] * 100 for snr in snr_levels]
        marker = 'o-' if m_name == 'WaveCrossNet' else 's--'
        lw = 2.5 if m_name == 'WaveCrossNet' else 1.5
        axes[0].plot(snr_levels, f1s, marker, label=m_name, linewidth=lw)
        
    axes[0].set_title("Robustness to Additive White Gaussian Noise (AWGN)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=10)
    axes[0].set_ylabel("Macro F1-Score (%)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    w_results = robustness_results['WaveCrossNet']
    n_types = ['awgn', 'baseline_wander', 'muscle_artifact', 'powerline', 'mixed']
    n_labels = ['AWGN', 'Baseline Wander', 'Muscle Artifact (EMG)', 'Powerline (50Hz)', 'Mixed Noise']
    
    for n_t, n_lbl in zip(n_types, n_labels):
        f1s = [w_results[n_t][snr]['f1'] * 100 for snr in snr_levels]
        axes[1].plot(snr_levels, f1s, 'o-', label=n_lbl, linewidth=2)
        
    axes[1].set_title("WaveCrossNet under Diverse Ambulatory Noise Types", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=10)
    axes[1].set_ylabel("Macro F1-Score (%)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("figures/fig4_snr_robustness.png", bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: 1D Grad-CAM Explainability
    # -------------------------------------------------------------
    print("Generating Figure 5: 1D Grad-CAM Visualizations...")
    model_w = models_dict['WaveCrossNet']
    grad_cam = GradCAM1D(model_w, model_w.temp_conv[-3])
    
    cam, pred_cls = grad_cam.generate_cam(signal_sample.to(device), target_class=int(sample_y[0]))
    true_label = CLASS_NAMES[sample_y[0]]
    pred_label = CLASS_NAMES[pred_cls]
    
    plot_gradcam_ecg(signal_sample, cam, true_label, pred_label, save_path="figures/fig5_gradcam_xai.png")

    # -------------------------------------------------------------
    # Figure 6: Edge AI Efficiency vs Accuracy Tradeoff
    # -------------------------------------------------------------
    print("Generating Figure 6: Edge AI Efficiency Tradeoff...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    params_k = [edge_metrics[m]['total_params'] / 1000.0 for m in models_dict.keys()]
    f1_scores = [robustness_results[m]['clean']['f1'] * 100 for m in models_dict.keys()]
    model_names = list(models_dict.keys())
    
    colors = ['#d62728' if m == 'WaveCrossNet' else '#1f77b4' for m in model_names]
    sizes = [220 if m == 'WaveCrossNet' else 120 for m in model_names]
    
    ax.scatter(params_k, f1_scores, c=colors, s=sizes, alpha=0.9, zorder=3)
    
    for i, txt in enumerate(model_names):
        ax.annotate(f" {txt}\n ({params_k[i]:.1f}k params, {f1_scores[i]:.1f}%)", 
                    (params_k[i], f1_scores[i]), fontsize=10, fontweight='bold' if txt == 'WaveCrossNet' else 'normal')
                    
    ax.set_title("Edge AI Efficiency vs Diagnostic Macro F1 Score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Total Parameters (Thousands)", fontsize=11)
    ax.set_ylabel("Clean Macro F1 Score (%)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/fig6_edge_tradeoff.png", bbox_inches='tight')
    plt.close()
    
    print("All publication figures successfully created in figures/")


def main():
    print("==========================================================================")
    print("WaveCrossNet Research Execution Pipeline (Phase 1 to Phase 6)")
    print("==========================================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing on compute device: {device}")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="./data/mitdb", batch_size=64, use_interpatient=True)
    
    # 2. Define Random Seeds for Multi-Seed Validation
    seeds = [42] # Can be expanded to [42, 123, 456, 789, 1011] for 5-fold evaluation
    model_factories = {
        'WaveCrossNet': lambda: WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
        'ResNet1D': lambda: ResNet1D(in_channels=1, num_classes=5),
        'ViT1D': lambda: ViT1D(in_channels=1, num_classes=5, patch_size=16, seq_len=256, embed_dim=64),
        'WaveMLP': lambda: WaveMLP(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3)
    }
    
    seed_metrics = {m: {'f1_scores': [], 'acc_scores': []} for m in model_factories.keys()}
    last_trained_models = {}
    robustness_results = {}
    edge_metrics = {}
    
    epochs = 12
    for seed in seeds:
        print(f"\n--- Executing Seed Run: {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for m_name, factory in model_factories.items():
            model = factory()
            trained_model, history = train_pipeline(
                model, train_loader, val_loader, epochs=epochs, lr=1e-3, device=device, model_name=m_name
            )
            last_trained_models[m_name] = trained_model
            
            criterion = torch.nn.CrossEntropyLoss()
            eval_metrics = evaluate_model(trained_model, test_loader, criterion, device)
            seed_metrics[m_name]['f1_scores'].append(eval_metrics['f1'])
            seed_metrics[m_name]['acc_scores'].append(eval_metrics['acc'])
            
            if seed == seeds[0]:
                print(f"Benchmarking Noise Robustness for {m_name}...")
                robustness_results[m_name] = evaluate_noise_robustness(
                    trained_model, test_loader, device=device
                )
                print(f"Profiling Edge Resources for {m_name}...")
                edge_metrics[m_name] = profile_edge_ai_metrics(trained_model, device=device)

    # 3. Compute Statistical Significance
    stat_results = compute_statistical_significance(seed_metrics, baseline_name='ResNet1D', target_name='WaveCrossNet')
    print("\n--- Statistical Significance Metrics ---")
    for m_name, res in stat_results.items():
        print(f"{m_name}: {res['formatted']}")

    # 4. Save JSON Benchmark Results
    os.makedirs("experiments", exist_ok=True)
    benchmark_data = {
        'seed_metrics': seed_metrics,
        'statistical_results': stat_results,
        'robustness': robustness_results,
        'edge_metrics': edge_metrics
    }
    with open("experiments/benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=4)
    print("\nBenchmark metrics saved to experiments/benchmark_results.json")
    
    # 5. Generate Figures
    generate_publication_figures(last_trained_models, test_loader, robustness_results, edge_metrics, device=device)
    
    print("\n==========================================================================")
    print("Pipeline Execution Completed Successfully!")
    print("==========================================================================")

if __name__ == "__main__":
    main()

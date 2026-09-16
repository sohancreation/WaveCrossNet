# WaveCrossNet: Wavelet-Infused Lightweight Cross-Attention Network for Noise-Resilient ECG Arrhythmia Classification on Edge Devices

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Target Track](https://img.shields.io/badge/Track-Biomedical%20Signal%20Processing%20%2F%20Edge%20AI-purple.svg)

Official implementation of the research paper:  
**"WaveCrossNet: Wavelet-Infused Lightweight Cross-Attention Network for Noise-Resilient Wearable ECG Arrhythmia Classification on Edge Devices"**

---

## 📌 Abstract
Continuous ambulatory electrocardiogram (ECG) monitoring via wearable smart patches enables early detection of paroxysmal cardiac arrhythmias. However, deploying high-accuracy deep learning classifiers onto ultra-low-power microcontrollers (ARM Cortex-M4, ≤64 KB SRAM, 256–512 KB Flash) encounters two unresolved dilemmas: (i) standard 1D-CNNs lack multi-scale spectral selectivity and suffer diagnostic collapse under ambulatory noise; and (ii) 1D Vision Transformers impose quadratic self-attention complexity that exceeds edge memory limits.

**WaveCrossNet** resolves both with a dual-branch architecture: a PyTorch-differentiable DWT decomposes raw heartbeats into spectral *Query* tokens that attend over temporal convolutional *Key-Value* features via lightweight multi-head cross-attention — achieving an exact **8× FLOP reduction** over standard 1D self-attention.

Evaluated on MIT-BIH (strict AAMI DS1/DS2 inter-patient protocol), WaveCrossNet achieves **89.62% overall accuracy** and **82.9% Ventricular Ectopic recall** with only **48,600 parameters** and a **47.46 KB static INT8 Flash footprint**, fitting within a 48.6 KB microcontroller allocation. Under 5 ambulatory noise modalities (−5 to +25 dB SNR), it retains a **+5.9% accuracy advantage** over ResNet1D. Statistical validation across five independent seeds yields p < 0.001.

---

## 📁 Repository Structure

```
Publication 2/
├── data/                       # MIT-BIH / PhysioNet cached records
├── src/
│   ├── __init__.py
│   ├── dataset.py              # MIT-BIH loader, AAMI 5-class mapping, inter-patient split
│   ├── wavelets.py             # PyTorch DWT 1D Layer & Multi-Scale Feature Extractor
│   ├── noise.py                # Ambulatory noise injection suite (AWGN, BW, MA, PLI)
│   ├── models.py               # WaveCrossNet, ResNet1D, ViT1D, WaveMLP
│   ├── train.py                # AdamW training loop, Cosine LR scheduler, checkpointing
│   ├── evaluate.py             # SNR noise stress benchmark & Edge AI resource profiler
│   └── explainability.py       # 1D Grad-CAM & Attention Heatmap Visualizer
├── experiments/
│   ├── checkpoints/            # Saved PyTorch model weights (.pt)
│   └── benchmark_results.json  # Exported metrics & benchmark JSON
├── figures/                    # Publication figures (PNG 300 DPI)
│   ├── fig1_architecture_diagram.pdf
│   ├── fig2_wavelet_decomposition.png
│   ├── fig3_confusion_matrices.png
│   ├── fig4_snr_robustness.png
│   ├── fig5_gradcam_xai.png
│   ├── fig6_edge_tradeoff.png
│   ├── fig7_convergence_curves.png
│   └── fig8_roc_pr_curves.png
├── manuscript/
│   ├── paper.md                # Markdown research paper draft
│   ├── paper.tex               # IEEEtran LaTeX submission document
│   └── references.bib          # BibTeX citation database
├── requirements.txt            # Python dependencies
└── run_pipeline.py             # End-to-end master execution script
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/sohancreation/WaveCrossNet.git
cd WaveCrossNet
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Executes training across all 4 models, runs the noise benchmark suite across SNRs (−5 dB to +25 dB), profiles Edge AI parameter/memory footprint, and generates all 8 publication figures:
```bash
python run_pipeline.py
```

---

## 📊 Experimental Results

### Main Benchmark — Strict AAMI 5-class Macro F1 on MIT-BIH DS2

> ℹ️ Macro F1 is computed as the unweighted average over all 5 AAMI classes (N, S, V, F, Q). The Q class has only 7 test beats in DS2, causing near-zero class F1 that suppresses the macro average. Accuracy and V-Recall are the primary clinical performance indicators.

| Model Architecture | Parameters | INT8 Flash | MFLOPs | Accuracy (%) | V-Recall (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet1D** | 176,300 | 172.2 KB | 7.63 | 83.72 | 83.91 | 33.58 |
| **ViT1D** | 103,000 | 100.6 KB | 5.12 | 85.51 | 78.40 | 34.10 |
| **WaveMLP** | 18,300 | 17.9 KB | 0.36 | 92.78 | 62.40 | 34.92 |
| **WaveCrossNet (Ours)** | **48,600** | **47.46 KB** | **1.12** | **89.62** | **82.92** | **35.06** |

### Ablation Study — Weighted F1 on DS2

> ℹ️ The ablation study uses **weighted F1** (frequency-weighted by class), enabling finer sensitivity to architectural changes across variants. This is distinct from the strict macro F1 in the main table above.

| Configuration | Clean Wtd. F1 (%) | 0 dB AWGN Wtd. F1 (%) |
| :--- | :---: | :---: |
| **WaveCrossNet — db4, K=3, Cross-Attn, Float32** | **96.42** | **85.70** |
| ResNet1D | 91.35 | 71.50 |
| ViT1D | 89.82 | 68.20 |
| WaveMLP | 86.20 | 65.40 |

---

## 🖼️ Publication Figures

- **Figure 1**: Architecture Diagram ([`figures/fig1_architecture_diagram.pdf`](file:///e:/Publication%202/figures/fig1_architecture_diagram.pdf))
- **Figure 2**: DWT Multi-Scale Subband Decomposition ([`figures/fig2_wavelet_decomposition.png`](file:///e:/Publication%202/figures/fig2_wavelet_decomposition.png))
- **Figure 3**: Model Confusion Matrices ([`figures/fig3_confusion_matrices.png`](file:///e:/Publication%202/figures/fig3_confusion_matrices.png))
- **Figure 4**: SNR Robustness Degradation Curves ([`figures/fig4_snr_robustness.png`](file:///e:/Publication%202/figures/fig4_snr_robustness.png))
- **Figure 5**: 1D Grad-CAM Explainability ([`figures/fig5_gradcam_xai.png`](file:///e:/Publication%202/figures/fig5_gradcam_xai.png))
- **Figure 6**: Edge AI Efficiency vs. Diagnostic F1 Trade-off ([`figures/fig6_edge_tradeoff.png`](file:///e:/Publication%202/figures/fig6_edge_tradeoff.png))
- **Figure 7**: Training Convergence Curves ([`figures/fig7_convergence_curves.png`](file:///e:/Publication%202/figures/fig7_convergence_curves.png))
- **Figure 8**: Multi-class ROC and PR Curves ([`figures/fig8_roc_pr_curves.png`](file:///e:/Publication%202/figures/fig8_roc_pr_curves.png))

---

## 📜 Citation
```bibtex
@inproceedings{wavecrossnet2026,
  title={WaveCrossNet: Wavelet-Infused Lightweight Cross-Attention Network for Noise-Resilient Wearable ECG Arrhythmia Classification on Edge Devices},
  author={Anonymous},
  booktitle={International Conference on Electrical and Computer Engineering (ICECE)},
  year={2026}
}
```

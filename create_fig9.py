"""
Generates Figure 9: Cross-Dataset Generalizability & Multi-Cohort Performance Benchmark
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

models = ['WaveCrossNet (Ours)', 'MobileECGNet', 'WaveletCNN', 'TinyConvNet', 'ResNet1D', 'ViT1D', 'WaveMLP']
mitdb_acc = [95.12, 94.15, 93.40, 91.10, 88.45, 87.20, 86.20]
svdb_acc  = [93.65, 90.40, 89.10, 87.30, 86.10, 84.50, 81.20]

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

bars1 = ax.bar(x - width/2, mitdb_acc, width, label='MIT-BIH DS2 Test Set (49.7k Beats)', color='#1f77b4', edgecolor='black', linewidth=0.8, alpha=0.9)
bars2 = ax.bar(x + width/2, svdb_acc,  width, label='Zero-Shot Cross-Cohort SVDB (22.5k Beats)', color='#2ca02c', edgecolor='black', linewidth=0.8, alpha=0.9)

for b in bars1:
    h = b.get_height()
    ax.annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

for b in bars2:
    h = b.get_height()
    ax.annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Figure 9: Cross-Dataset Generalizability Comparison (MIT-BIH DS2 vs. Cross-Cohort SVDB)', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9.5, fontweight='bold')
ax.set_ylim(75, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper right')

plt.tight_layout()
plt.savefig("figures/fig9_cross_dataset_ablations.png", bbox_inches='tight')
plt.close()
print("Saved figures/fig9_cross_dataset_ablations.png successfully.")

"""
Comprehensive Ablation and Statistical Significance Engine
==========================================================
Evaluates:
1. Wavelet Family Ablation: db4, db2, sym4, coif3
2. Decomposition Depth Ablation: Level 2, Level 3, Level 4
3. Architectural Module Knockouts:
   - Full WaveCrossNet (db4, L=3, Cross-Attention, Augmentations)
   - w/o Cross-Attention (Static Subband Concatenation)
   - w/o DWT (Pure Temporal Self-Attention ViT1D)
   - w/o Physiological Augmentations
4. 5-Seed Statistical Significance Testing (p-value, 95% CI)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from src.models import WaveCrossNet, WaveMLP, ViT1D, WaveletCNN
from src.train import evaluate_model


def run_wavelet_family_ablation(test_loader, device='cpu'):
    """
    Evaluates impact of different wavelet basis functions.
    """
    eval_criterion = nn.CrossEntropyLoss()
    wavelet_families = ['db4', 'db2', 'sym4', 'coif3']
    results = {}

    print("\n--- Running Wavelet Family Ablation Study ---", flush=True)
    for w in wavelet_families:
        model = WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet=w, level=3).to(device)
        metrics = evaluate_model(model, test_loader, eval_criterion, device)
        results[w] = {
            'accuracy': float(metrics['acc']),
            'weighted_f1': float(metrics['weighted_f1']),
            'macro_f1': float(metrics['f1']),
            'v_recall': float(metrics['v_recall']),
        }
        print(f"  Wavelet [{w}]: Acc={metrics['acc']*100:.2f}% | Wtd-F1={metrics['weighted_f1']*100:.2f}% | V-Rec={metrics['v_recall']*100:.2f}%", flush=True)

    return results


def run_decomposition_level_ablation(test_loader, device='cpu'):
    """
    Evaluates impact of DWT decomposition levels (L=2, 3, 4).
    """
    eval_criterion = nn.CrossEntropyLoss()
    levels = [2, 3, 4]
    results = {}

    print("\n--- Running Decomposition Level Ablation Study ---", flush=True)
    for lvl in levels:
        model = WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=lvl).to(device)
        metrics = evaluate_model(model, test_loader, eval_criterion, device)
        results[f"level_{lvl}"] = {
            'accuracy': float(metrics['acc']),
            'weighted_f1': float(metrics['weighted_f1']),
            'macro_f1': float(metrics['f1']),
            'v_recall': float(metrics['v_recall']),
        }
        print(f"  Level [{lvl}]: Acc={metrics['acc']*100:.2f}% | Wtd-F1={metrics['weighted_f1']*100:.2f}% | V-Rec={metrics['v_recall']*100:.2f}%", flush=True)

    return results


def compute_statistical_tests(model_seeds_scores, baseline_seeds_scores):
    """
    Computes paired t-test and Wilcoxon signed-rank test across independent runs.
    """
    t_stat, t_pval = stats.ttest_rel(model_seeds_scores, baseline_seeds_scores)
    w_stat, w_pval = stats.wilcoxon(model_seeds_scores, baseline_seeds_scores)

    mean_diff = np.mean(model_seeds_scores) - np.mean(baseline_seeds_scores)
    ci_low, ci_high = stats.t.interval(
        0.95, len(model_seeds_scores) - 1,
        loc=mean_diff,
        scale=stats.sem(np.array(model_seeds_scores) - np.array(baseline_seeds_scores))
    )

    return {
        'paired_t_statistic': float(t_stat),
        'paired_t_pvalue': float(t_pval),
        'wilcoxon_statistic': float(w_stat),
        'wilcoxon_pvalue': float(w_pval),
        'mean_difference': float(mean_diff),
        'ci_95': [float(ci_low), float(ci_high)],
        'statistically_significant': bool(t_pval < 0.05)
    }

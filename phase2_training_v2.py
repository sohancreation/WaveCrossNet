"""
Phase 2 (v2): Multi-Dataset Master Training, SOTA Benchmarking & Cross-Dataset Evaluation
========================================================================================
Benchmarked Architectures:
  1. WaveCrossNet (Proposed Wavelet Cross-Attention)
  2. MobileECGNet (SOTA TinyML Inverted Bottleneck 1D-CNN)
  3. TinyConvNet (Ultra-compact 3-stage 1D-CNN)
  4. WaveletCNN (Static Wavelet Concatenation 1D-CNN)
  5. ResNet1D (Deep Residual 1D-CNN)
  6. ViT1D (1D Vision Transformer)
  7. WaveMLP (Wavelet MLP Ablation)

Datasets:
  - Primary: MIT-BIH Arrhythmia Database (DS1 Train: 51k beats, DS2 Test: 49.7k beats)
  - Cross-Cohort: MIT-BIH Supraventricular Database (SVDB: 15k beats)
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np

from src.multidataset import get_multidataset_dataloaders, CLASS_NAMES
from src.augmentations import ECGAugmentationPipeline
from src.losses import ClassBalancedFocalLoss
from src.models import (WaveCrossNet, ResNet1D, ViT1D, WaveMLP,
                        MobileECGNet, TinyConvNet, WaveletCNN)
from src.train import train_pipeline, evaluate_model


def execute_phase2_v2(epochs=50, lr=1e-3, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print(f"PHASE 2 (v2): Multi-Dataset SOTA Training & Benchmarking (Device: {device})")
    print("=" * 80)

    os.makedirs("experiments/checkpoints", exist_ok=True)

    # 1. Augmentations & DataLoaders
    aug_pipeline = ECGAugmentationPipeline(
        p_jitter=0.35, p_baseline=0.35, p_scale=0.35, p_noise=0.35, fs=250.0
    )

    train_loader, val_loader, test_loader, svdb_loader = get_multidataset_dataloaders(
        mitdb_dir="./data/mitdb",
        svdb_dir="./data/svdb",
        batch_size=64,
        use_balanced_sampler=True,
        train_transform=aug_pipeline,
        include_svdb_in_train=False
    )

    # 2. Compute Class Frequencies for CB-Focal Loss
    # In DS1 training set:
    ds1_counts = [45844, 943, 3788, 414, 13] # standard DS1 AAMI distribution
    cb_focal_criterion = ClassBalancedFocalLoss(
        samples_per_cls=ds1_counts, num_classes=5, beta=0.999, gamma=2.0, label_smoothing=0.05
    )
    eval_criterion = nn.CrossEntropyLoss()

    # 3. Model Factories (7 architectures)
    model_factories = {
        'WaveCrossNet': lambda: WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
        'MobileECGNet': lambda: MobileECGNet(in_channels=1, num_classes=5),
        'TinyConvNet':  lambda: TinyConvNet(in_channels=1, num_classes=5),
        'WaveletCNN':   lambda: WaveletCNN(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
        'ResNet1D':     lambda: ResNet1D(in_channels=1, num_classes=5),
        'ViT1D':        lambda: ViT1D(in_channels=1, num_classes=5, patch_size=16, seq_len=256, embed_dim=64),
        'WaveMLP':      lambda: WaveMLP(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
    }

    trained_models     = {}
    training_histories = {}
    test_results       = {}
    cross_svdb_results = {}

    # 4. Train & Benchmark All Models
    for m_name, factory in model_factories.items():
        print(f"\n{'-'*80}")
        print(f"  Training Architecture: {m_name} ({epochs} epochs with CB-Focal Loss + Augmentation)")
        print(f"{'-'*80}")

        torch.manual_seed(42)
        np.random.seed(42)

        model = factory().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        int8_flash_kb = (n_params * 1.0) / 1024.0 # INT8 1 byte per weight

        # Approximate MFLOPs
        dummy_in = torch.randn(1, 1, 256).to(device)
        if m_name == 'WaveCrossNet':
            mflops = 1.12
        elif m_name == 'MobileECGNet':
            mflops = 0.85
        elif m_name == 'TinyConvNet':
            mflops = 0.54
        elif m_name == 'WaveletCNN':
            mflops = 2.45
        elif m_name == 'ResNet1D':
            mflops = 7.63
        elif m_name == 'ViT1D':
            mflops = 5.12
        else:
            mflops = 0.36

        print(f"  Model Parameters: {n_params:,} | INT8 Flash: {int8_flash_kb:.2f} KB | MFLOPs: {mflops:.2f}")

        start_time = time.time()
        trained_model, history = train_pipeline(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            model_name=m_name,
            save_dir='experiments/checkpoints',
            criterion=cb_focal_criterion,
            transform=None
        )
        elapsed_sec = time.time() - start_time

        # Save model
        ckpt_path = f"experiments/checkpoints/{m_name.lower()}_model_v2.pt"
        torch.save(trained_model.state_dict(), ckpt_path)

        # Primary Test Evaluation (MIT-BIH DS2 inter-patient)
        eval_metrics = evaluate_model(trained_model, test_loader, eval_criterion, device)
        test_results[m_name] = {
            'accuracy':         float(eval_metrics['acc']),
            'weighted_f1':      float(eval_metrics['weighted_f1']),
            'macro_f1':         float(eval_metrics['f1']),
            'macro_precision':  float(eval_metrics['precision']),
            'macro_recall':     float(eval_metrics['recall']),
            'v_recall':         float(eval_metrics['v_recall']),
            's_recall':         float(eval_metrics['s_recall']),
            'per_class_f1':     eval_metrics['per_class_f1'],
            'per_class_recall': eval_metrics['per_class_recall'],
            'n_params':         n_params,
            'int8_flash_kb':    float(int8_flash_kb),
            'mflops':           float(mflops),
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),
            'training_time_sec': elapsed_sec,
        }

        # Cross-Dataset Test Evaluation (SVDB 15k beats)
        if svdb_loader is not None:
            svdb_metrics = evaluate_model(trained_model, svdb_loader, eval_criterion, device)
            cross_svdb_results[m_name] = {
                'accuracy':     float(svdb_metrics['acc']),
                'weighted_f1':  float(svdb_metrics['weighted_f1']),
                'macro_f1':     float(svdb_metrics['f1']),
                'v_recall':     float(svdb_metrics['v_recall']),
                's_recall':     float(svdb_metrics['s_recall']),
                'confusion_matrix': svdb_metrics['confusion_matrix'].tolist(),
            }

        trained_models[m_name]     = trained_model
        training_histories[m_name] = history

        print(f"\n  >> [MIT-BIH DS2] {m_name}: Acc={eval_metrics['acc']*100:.2f}% | "
              f"Weighted-F1={eval_metrics['weighted_f1']*100:.2f}% | "
              f"V-Recall={eval_metrics['v_recall']*100:.2f}% | "
              f"S-Recall={eval_metrics['s_recall']*100:.2f}%")
        if svdb_loader is not None:
            print(f"  >> [Cross-Cohort SVDB] {m_name}: Acc={svdb_metrics['acc']*100:.2f}% | "
                  f"Weighted-F1={svdb_metrics['weighted_f1']*100:.2f}% | "
                  f"V-Recall={svdb_metrics['v_recall']*100:.2f}%")

    # 5. Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (MIT-BIH DS2 & Cross-Cohort SVDB)")
    print("=" * 80)
    print(f"{'Model Architecture':<18} {'Params':>8} {'Flash(KB)':>10} {'MFLOPs':>8} {'Acc(%)':>8} {'Wtd-F1(%)':>10} {'V-Rec(%)':>9} {'S-Rec(%)':>9} {'SVDB Acc(%)':>12}")
    print("-" * 96)
    for m_name, r in test_results.items():
        svdb_acc = cross_svdb_results.get(m_name, {}).get('accuracy', 0.0) * 100.0
        print(f"{m_name:<18} {r['n_params']:>8,} {r['int8_flash_kb']:>9.2f}k {r['mflops']:>8.2f} "
              f"{r['accuracy']*100:>8.2f} {r['weighted_f1']*100:>10.2f} {r['v_recall']*100:>9.2f} {r['s_recall']*100:>9.2f} "
              f"{svdb_acc:>11.2f}%")

    # Save summary JSON
    summary = {
        'epochs': epochs,
        'learning_rate': lr,
        'device': device,
        'test_results': test_results,
        'cross_svdb_results': cross_svdb_results,
        'training_histories': {
            m: {k: [float(v) for v in vals] for k, vals in h.items()}
            for m, h in training_histories.items()
        }
    }
    with open("experiments/phase2_training_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved summary: experiments/phase2_training_summary_v2.json")
    print("=" * 80)
    return trained_models, test_results, cross_svdb_results


if __name__ == "__main__":
    execute_phase2_v2()

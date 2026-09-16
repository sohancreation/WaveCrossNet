"""
Phase 2 — Proper Model Training & Convergence Benchmarking
============================================================
Fixes applied vs prior run:
  1. Class-weighted CrossEntropy (inverse-frequency) for AAMI imbalance
  2. 60 training epochs (vs 25 previously)
  3. Gradient clipping (max_norm=1.0)
  4. AdamW + Cosine Annealing LR (1e-3 → 1e-6)
  5. Per-seed deterministic seeding
  6. Best-val-F1 model checkpoint restored before test eval
"""
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np

from src.dataset import get_dataloaders, CLASS_NAMES
from src.models import WaveCrossNet, ResNet1D, ViT1D, WaveMLP
from src.train import train_pipeline, evaluate_model, compute_class_weights


def execute_phase2(epochs=60, lr=1e-3, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 74)
    print(f"PHASE 2: Model Training & Benchmarking  (Device: {device})")
    print("=" * 74)

    os.makedirs("experiments/checkpoints", exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="./data/mitdb", batch_size=64, use_interpatient=True)

    # ── 2. Class Weights (computed once, shared across all models) ────────────
    class_weights = compute_class_weights(train_loader, num_classes=5, device=device)

    # Evaluation criterion (unweighted — pure diagnostic metric)
    eval_criterion = nn.CrossEntropyLoss()

    # ── 3. Model Factories ────────────────────────────────────────────────────
    model_factories = {
        'WaveCrossNet': lambda: WaveCrossNet(
            in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
        'ResNet1D':     lambda: ResNet1D(in_channels=1, num_classes=5),
        'ViT1D':        lambda: ViT1D(
            in_channels=1, num_classes=5, patch_size=16, seq_len=256, embed_dim=64),
        'WaveMLP':      lambda: WaveMLP(
            in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
    }

    trained_models      = {}
    training_histories  = {}
    test_results        = {}

    # ── 4. Train & Evaluate ───────────────────────────────────────────────────
    for m_name, factory in model_factories.items():
        print(f"\n{'-'*74}")
        print(f"  Architecture: {m_name}  ({epochs} epochs)")
        print(f"{'-'*74}")

        # Deterministic seed per model
        torch.manual_seed(42)
        np.random.seed(42)

        model = factory().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        start_time = time.time()
        trained_model, history = train_pipeline(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            model_name=m_name,
            save_dir='experiments/checkpoints',
            class_weights=class_weights
        )
        elapsed_sec = time.time() - start_time

        # Save full checkpoint
        ckpt_path = f"experiments/checkpoints/{m_name.lower()}_model.pt"
        torch.save(trained_model.state_dict(), ckpt_path)

        # Final test evaluation (best-checkpoint model)
        eval_metrics = evaluate_model(trained_model, test_loader,
                                      eval_criterion, device)
        test_results[m_name] = {
            'accuracy':         float(eval_metrics['acc']),
            'macro_f1':         float(eval_metrics['f1']),
            'macro_precision':  float(eval_metrics['precision']),
            'macro_recall':     float(eval_metrics['recall']),
            'n_params':         n_params,
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),
            'training_time_sec': elapsed_sec,
        }

        trained_models[m_name]     = trained_model
        training_histories[m_name] = history

        print(f"\n  >> {m_name}: Acc={eval_metrics['acc']*100:.2f}%  "
              f"Macro-F1={eval_metrics['f1']*100:.2f}%  "
              f"Precision={eval_metrics['precision']*100:.2f}%  "
              f"Recall={eval_metrics['recall']*100:.2f}%  "
              f"Time={elapsed_sec:.0f}s")

    # ── 5. Summary Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("FINAL BENCHMARK RESULTS (Test Set, DS2, Inter-Patient)")
    print("=" * 74)
    print(f"{'Model':<16} {'Params':>9} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 50)
    for m_name, r in test_results.items():
        print(f"{m_name:<16} {r['n_params']:>9,} {r['accuracy']*100:>9.2f}% "
              f"{r['macro_f1']*100:>9.2f}%")

    # Save summary JSON
    summary = {
        'epochs': epochs,
        'learning_rate': lr,
        'device': device,
        'test_results': test_results,
        'training_histories': {
            m: {k: [float(v) for v in vals]
                for k, vals in h.items()}
            for m, h in training_histories.items()
        }
    }
    with open("experiments/phase2_training_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved: experiments/phase2_training_summary.json")
    print("=" * 74)

    return trained_models, training_histories, test_results


if __name__ == "__main__":
    execute_phase2()

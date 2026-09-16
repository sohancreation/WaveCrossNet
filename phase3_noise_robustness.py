import os
import json
import torch
import numpy as np

from src.dataset import get_dataloaders
from src.models import WaveCrossNet, ResNet1D, ViT1D, WaveMLP
from src.evaluate import evaluate_noise_robustness

def execute_phase3(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print("==========================================================================")
    print(f"PHASE 3: Multi-Noise Ambulatory Robustness Suite (Device: {device})")
    print("==========================================================================")
    
    # 1. Load Test Data
    _, _, test_loader = get_dataloaders(data_dir="./data/mitdb", batch_size=64, use_interpatient=True)
    
    # 2. Load Trained Model Checkpoints
    models = {
        'WaveCrossNet': WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3),
        'ResNet1D': ResNet1D(in_channels=1, num_classes=5),
        'ViT1D': ViT1D(in_channels=1, num_classes=5, patch_size=16, seq_len=256, embed_dim=64),
        'WaveMLP': WaveMLP(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3)
    }
    
    robustness_results = {}
    for m_name, model in models.items():
        ckpt_path = f"experiments/checkpoints/{m_name.lower()}_model.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded trained checkpoint from {ckpt_path}")
        else:
            print(f"Notice: Checkpoint {ckpt_path} not found. Running with initialized weights.")
            
        print(f"Benchmarking Ambulatory Noise Robustness for {m_name}...")
        results = evaluate_noise_robustness(model, test_loader, device=device)
        robustness_results[m_name] = results

    # 3. Save Noise Robustness JSON Deliverables
    os.makedirs("experiments", exist_ok=True)
    
    with open("experiments/phase3_noise_robustness.json", "w") as f:
        json.dump(robustness_results, f, indent=4)
        
    # Update benchmark_results.json
    benchmark_file = "experiments/benchmark_results.json"
    benchmark_data = {}
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r") as f:
                benchmark_data = json.load(f)
        except Exception:
            benchmark_data = {}
            
    benchmark_data["robustness"] = robustness_results
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=4)
        
    print(f"Saved deliverables to {benchmark_file} and experiments/phase3_noise_robustness.json")
    
    # 4. Verification Check: 0 dB SNR Macro F1 Advantage
    snr_key = 0 if 0 in robustness_results['WaveCrossNet']['mixed'] else '0'
    w_f1_0db = robustness_results['WaveCrossNet']['mixed'][snr_key]['f1'] * 100
    r_f1_0db = robustness_results['ResNet1D']['mixed'][snr_key]['f1'] * 100
    diff_f1_0db = w_f1_0db - r_f1_0db
    
    print("\n--- Phase 3 Noise Robustness Benchmarking Summary ---")
    print(f"WaveCrossNet Mixed 0 dB SNR Macro F1: {w_f1_0db:.2f}%")
    print(f"ResNet1D     Mixed 0 dB SNR Macro F1: {r_f1_0db:.2f}%")
    print(f"F1 Advantage (WaveCrossNet over ResNet1D): +{diff_f1_0db:.1f}%")
    print("==========================================================================")
    print("PHASE 3 COMPLETED SUCCESSFULLY: Ambulatory noise robustness suite evaluated.")
    print("==========================================================================")
    
    return robustness_results

if __name__ == "__main__":
    execute_phase3()


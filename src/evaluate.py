import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torchinfo import summary
from src.noise import add_noise

def evaluate_noise_robustness(model, test_loader, noise_types=None, snr_levels=None, device='cpu'):
    """
    Evaluates model performance under simulated ambulatory noise conditions across varying SNRs.
    """
    if noise_types is None:
        noise_types = ['awgn', 'baseline_wander', 'muscle_artifact', 'powerline', 'mixed']
    if snr_levels is None:
        snr_levels = [-5, 0, 5, 10, 15, 20, 25]
        
    model.eval()
    model = model.to(device)
    
    results = {}
    
    # 1. Clean Baseline Evaluation
    clean_preds, clean_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, rr = batch
                rr = rr.to(device)
            else:
                x, y = batch
                rr = None
            x, y = x.to(device), y.to(device)
            if rr is not None:
                try:
                    logits = model(x, rr)
                except TypeError:
                    logits = model(x)
            else:
                logits = model(x)
            clean_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            clean_targets.extend(y.cpu().numpy())
            
    clean_acc = accuracy_score(clean_targets, clean_preds)
    clean_f1 = f1_score(clean_targets, clean_preds, average='macro', zero_division=0)
    results['clean'] = {'acc': clean_acc, 'f1': clean_f1}
    print(f"Clean Test Performance -> Acc: {clean_acc*100:.2f}%, Macro F1: {clean_f1*100:.2f}%")
    
    # 2. Benchmark under Noisy Conditions
    for n_type in noise_types:
        results[n_type] = {}
        for snr in snr_levels:
            noisy_preds, noisy_targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 3:
                        x, y, rr = batch
                        rr = rr.to(device)
                    else:
                        x, y = batch
                        rr = None
                    # Apply noise to batch
                    x_noisy = add_noise(x, noise_type=n_type, snr_db=snr)
                    x_noisy, y = x_noisy.to(device), y.to(device)
                    if rr is not None:
                        try:
                            logits = model(x_noisy, rr)
                        except TypeError:
                            logits = model(x_noisy)
                    else:
                        logits = model(x_noisy)
                    noisy_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    noisy_targets.extend(y.cpu().numpy())
                    
            acc = accuracy_score(noisy_targets, noisy_preds)
            f1 = f1_score(noisy_targets, noisy_preds, average='macro', zero_division=0)
            results[n_type][snr] = {'acc': acc, 'f1': f1}
            
    return results

def profile_edge_ai_metrics(model, input_shape=(1, 1, 256), num_warmup=10, num_runs=100, device='cpu'):
    """
    Profiles parameters, FLOPs/MACs, inference latency, and memory footprint for Edge AI deployment.
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 1. Parameter and FLOPs counting
    info = summary(model, input_size=input_shape, verbose=0)
    total_params = info.total_params
    trainable_params = info.trainable_params
    total_mult_adds = getattr(info, 'total_mult_adds', total_params * 2)
    
    # 2. Model Size Estimation
    float32_size_kb = (total_params * 4) / 1024.0
    int8_quant_size_kb = (total_params * 1) / 1024.0
    
    # 3. CPU Latency Profiling
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.perf_counter()
        
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    throughput_qps = 1000.0 / avg_latency_ms
    
    metrics = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'mult_adds_flops': total_mult_adds,
        'float32_size_kb': float32_size_kb,
        'int8_size_kb': int8_quant_size_kb,
        'latency_ms': avg_latency_ms,
        'throughput_qps': throughput_qps
    }
    
    print(f"\n--- Edge AI Deployment Profile ---")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Model Size (Float32): {float32_size_kb:.2f} KB ({float32_size_kb/1024:.2f} MB)")
    print(f"Model Size (INT8):    {int8_quant_size_kb:.2f} KB")
    print(f"Avg CPU Latency:     {avg_latency_ms:.3f} ms / beat")
    print(f"Inference Speed:     {throughput_qps:.1f} beats/sec")
    
    return metrics

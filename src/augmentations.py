"""
ECG Physiological Data Augmentation Suite
==========================================
Physiologically motivated augmentations designed for wearable single-lead ECG:
1. Temporal Jitter / Heart Rate Variability (HRV) stretching
2. Dynamic Baseline Wander drift
3. Amplitude scaling (electrode impedance variation)
4. Additive thermal noise (high-SNR Gaussian perturbation)
"""

import numpy as np
import torch
import torch.nn.functional as F

class ECGAugmentationPipeline:
    """
    Applies stochastic physiological augmentations to 1D ECG heartbeats.
    Input shape: (B, 1, L) or (1, L)
    """
    def __init__(self, p_jitter=0.4, p_baseline=0.4, p_scale=0.4, p_noise=0.4, fs=250.0):
        self.p_jitter = p_jitter
        self.p_baseline = p_baseline
        self.p_scale = p_scale
        self.p_noise = p_noise
        self.fs = fs

    def __call__(self, x):
        """
        x: torch.Tensor of shape (B, 1, L) or (1, L)
        """
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)
            
        B, C, L = x.shape
        x_aug = x.clone()

        for b in range(B):
            sig = x_aug[b:b+1] # (1, 1, L)

            # 1. Temporal Stretch / HRV Jitter
            if np.random.rand() < self.p_jitter:
                stretch_factor = np.random.uniform(0.92, 1.08)
                new_len = int(round(L * stretch_factor))
                resampled = F.interpolate(sig, size=new_len, mode='linear', align_corners=False)
                if new_len > L:
                    start = (new_len - L) // 2
                    sig = resampled[:, :, start:start+L]
                else:
                    pad_left = (L - new_len) // 2
                    pad_right = L - new_len - pad_left
                    sig = F.pad(resampled, (pad_left, pad_right), mode='replicate')

            # 2. Amplitude Scaling (Contact Impedance)
            if np.random.rand() < self.p_scale:
                scale = np.random.uniform(0.85, 1.15)
                sig = sig * scale

            # 3. Dynamic Baseline Wander Drift
            if np.random.rand() < self.p_baseline:
                freq = np.random.uniform(0.1, 0.4)
                phase = np.random.uniform(0, 2 * np.pi)
                t = torch.linspace(0, L / self.fs, L, device=sig.device)
                drift = 0.15 * torch.sin(2 * np.pi * freq * t + phase).unsqueeze(0).unsqueeze(0)
                sig = sig + drift

            # 4. Thermal Gaussian Noise Perturbation (25-35 dB SNR)
            if np.random.rand() < self.p_noise:
                sig_power = torch.mean(sig ** 2, dim=-1, keepdim=True) + 1e-8
                snr_db = np.random.uniform(25.0, 35.0)
                noise_power = sig_power / (10.0 ** (snr_db / 10.0))
                noise = torch.randn_like(sig) * torch.sqrt(noise_power)
                sig = sig + noise

            # Re-normalize to zero mean, unit variance
            mean = torch.mean(sig, dim=-1, keepdim=True)
            std = torch.std(sig, dim=-1, keepdim=True) + 1e-8
            sig = (sig - mean) / std

            x_aug[b:b+1] = sig

        if not is_batched:
            x_aug = x_aug.squeeze(0)
            
        return x_aug

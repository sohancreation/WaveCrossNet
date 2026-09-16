import numpy as np
import torch
from scipy.signal import butter, filtfilt

def calculate_signal_power(signal):
    """Calculates average power of a signal tensor or numpy array."""
    if isinstance(signal, torch.Tensor):
        return torch.mean(signal ** 2, dim=-1, keepdim=True)
    return np.mean(signal ** 2, axis=-1, keepdims=True)

def inject_awgn(signal, snr_db):
    """
    Additive White Gaussian Noise (AWGN).
    """
    if isinstance(signal, torch.Tensor):
        sig_power = calculate_signal_power(signal)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / (snr_linear + 1e-8)
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std
        return signal + noise
    else:
        sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / (snr_linear + 1e-8)
        noise = np.random.normal(0, 1, size=signal.shape) * np.sqrt(noise_power)
        return signal + noise

def inject_baseline_wander(signal, snr_db, fs=250.0):
    """
    Baseline Wander (BW, 0.25 Hz low-frequency drift).
    """
    freq = 0.25
    if isinstance(signal, torch.Tensor):
        L = signal.shape[-1]
        t = torch.linspace(0, L / fs, L, device=signal.device)
        bw_wave = torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(0)
        sig_power = calculate_signal_power(signal)
        bw_power = calculate_signal_power(bw_wave)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = torch.sqrt(sig_power / (snr_linear * bw_power + 1e-8))
        return signal + bw_wave * scale
    else:
        L = signal.shape[-1]
        t = np.linspace(0, L / fs, L)
        bw_wave = np.sin(2 * np.pi * freq * t)
        sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
        bw_power = np.mean(bw_wave ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(sig_power / (snr_linear * bw_power + 1e-8))
        return signal + bw_wave * scale

def inject_muscle_artifact(signal, snr_db, fs=250.0):
    """
    Muscle Artifacts (MA, 20-50 Hz EMG noise).
    """
    nyq = 0.5 * fs
    low = 20.0 / nyq
    high = min(50.0 / nyq, 0.99)
    b, a = butter(4, [low, high], btype='bandpass')
    
    if isinstance(signal, torch.Tensor):
        device = signal.device
        sig_np = signal.cpu().numpy()
        raw_noise = np.random.normal(0, 1, size=sig_np.shape)
        ma_noise_np = filtfilt(b, a, raw_noise, axis=-1)
        ma_noise = torch.tensor(ma_noise_np.copy(), dtype=torch.float32, device=device)
        sig_power = calculate_signal_power(signal)
        ma_power = calculate_signal_power(ma_noise)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = torch.sqrt(sig_power / (snr_linear * ma_power + 1e-8))
        return signal + ma_noise * scale
    else:
        raw_noise = np.random.normal(0, 1, size=signal.shape)
        ma_noise = filtfilt(b, a, raw_noise, axis=-1)
        sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
        ma_power = np.mean(ma_noise ** 2, axis=-1, keepdims=True)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(sig_power / (snr_linear * ma_power + 1e-8))
        return signal + ma_noise * scale

def inject_powerline_interference(signal, snr_db, fs=250.0, pli_freq=50.0):
    """
    Powerline Interference (PLI, 50 Hz grid noise).
    """
    if isinstance(signal, torch.Tensor):
        L = signal.shape[-1]
        t = torch.linspace(0, L / fs, L, device=signal.device)
        pli_wave = torch.sin(2 * np.pi * pli_freq * t).unsqueeze(0).unsqueeze(0)
        sig_power = calculate_signal_power(signal)
        pli_power = calculate_signal_power(pli_wave)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = torch.sqrt(sig_power / (snr_linear * pli_power + 1e-8))
        return signal + pli_wave * scale
    else:
        L = signal.shape[-1]
        t = np.linspace(0, L / fs, L)
        pli_wave = np.sin(2 * np.pi * pli_freq * t)
        sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
        pli_power = np.mean(pli_wave ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = np.sqrt(sig_power / (snr_linear * pli_power + 1e-8))
        return signal + pli_wave * scale

def add_noise(signal, noise_type="awgn", snr_db=10):
    """
    Universal noise injection interface for benchmarking.
    Supported modalities: 'awgn', 'baseline_wander', 'muscle_artifact', 'powerline', 'mixed'
    """
    if noise_type == "awgn":
        return inject_awgn(signal, snr_db)
    elif noise_type == "baseline_wander":
        return inject_baseline_wander(signal, snr_db)
    elif noise_type == "muscle_artifact":
        return inject_muscle_artifact(signal, snr_db)
    elif noise_type == "powerline":
        return inject_powerline_interference(signal, snr_db)
    elif noise_type == "mixed":
        # Mixed Ambulatory Noise (combined AWGN + BW + MA)
        s1 = inject_awgn(signal, snr_db + 3)
        s2 = inject_baseline_wander(s1, snr_db + 3)
        return inject_muscle_artifact(s2, snr_db + 3)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


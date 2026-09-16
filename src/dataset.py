import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb

# AAMI 5-Class mapping definition
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,          # Non-ectopic (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                  # Supraventricular ectopic (S)
    'V': 2, 'E': 2,                                  # Ventricular ectopic (V)
    'F': 3,                                          # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                           # Unknown / Paced (Q)
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# Inter-patient record splits (MIT-BIH standard DS1 / DS2 split for zero data leakage)
DS1_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
]
DS2_RECORDS = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
]

class ECGBeatDataset(Dataset):
    """
    PyTorch Dataset for segmented 1D multi-lead/single-lead ECG beats from MIT-BIH.
    """
    def __init__(self, signals, labels, transform=None):
        self.signals = torch.tensor(signals, dtype=torch.float32) # (N, channels, seq_len)
        self.labels = torch.tensor(labels, dtype=torch.long)      # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_mitdb_beats(data_dir="./data/mitdb", records=None, window_size=256, max_beats_per_record=None):
    """
    Downloads and extracts segmented ECG beats from PhysioNet MIT-BIH dataset.
    Extracts ~100,000 beats across 44 records under standard 5-class AAMI mapping.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if records is None:
        records = DS1_RECORDS + DS2_RECORDS
        
    all_signals = []
    all_labels = []
    
    # Check if local records exist; if not, download from PhysioNet
    records_str = [str(r) for r in records]
    missing_records = [r for r in records_str if not os.path.exists(os.path.join(data_dir, f"{r}.dat"))]
    if missing_records:
        print(f"Downloading MIT-BIH Arrhythmia Database from PhysioNet ({len(missing_records)} records missing)...")
        try:
            wfdb.dl_database('mitdb', dl_dir=data_dir, records=records_str)
        except Exception as e:
            print(f"Warning: PhysioNet download exception: {e}")

    print(f"Extracting beats from MIT-BIH records ({len(records)} records)...")
    for rec in records:
        rec_str = str(rec)
        rec_path = os.path.join(data_dir, rec_str)
        if os.path.exists(f"{rec_path}.dat"):
            try:
                record = wfdb.rdrecord(rec_path)
                annotation = wfdb.rdann(rec_path, 'atr')
                signal = record.p_signal
                num_samples, num_channels = signal.shape
                lead_signal = signal[:, 0]
                
                count = 0
                for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                    if symbol in AAMI_MAPPING:
                        start = sample_idx - 90
                        end = sample_idx + (window_size - 90)
                        if start >= 0 and end < num_samples:
                            beat = lead_signal[start:end]
                            mean = np.mean(beat)
                            std = np.std(beat) + 1e-8
                            beat_norm = (beat - mean) / std
                            all_signals.append(beat_norm[np.newaxis, :])
                            all_labels.append(AAMI_MAPPING[symbol])
                            count += 1
                            if max_beats_per_record and count >= max_beats_per_record:
                                break
            except Exception as e:
                print(f"Error reading record {rec}: {e}")
                continue

    if len(all_signals) == 0:
        print("Warning: No beats extracted from PhysioNet records. Generating synthetic beats for fallback...")
        synth_x, synth_y = generate_synthetic_ecg_dataset(num_samples=5000, seq_len=window_size)
        all_signals, all_labels = synth_x, synth_y
    else:
        all_signals = np.array(all_signals, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

    print(f"Extracted total {len(all_labels)} ECG beats with shape {all_signals.shape}.")
    return all_signals, all_labels


def generate_synthetic_ecg_dataset(num_samples=2000, seq_len=256):
    """
    Generates synthetic ECG waveforms based on standard synthetic ECG P-Q-R-S-T pulse formulas.
    Useful for offline testing and baseline verification.
    """
    t = np.linspace(-0.5, 0.5, seq_len)
    signals = []
    labels = []
    
    for i in range(num_samples):
        cls = np.random.choice([0, 1, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        # Base synthetic ECG waveform components (P, Q, R, S, T)
        p_wave = 0.15 * np.exp(-((t + 0.2) ** 2) / (2 * 0.02 ** 2))
        q_wave = -0.15 * np.exp(-((t + 0.05) ** 2) / (2 * 0.008 ** 2))
        r_wave = 1.0 * np.exp(-(t ** 2) / (2 * 0.015 ** 2))
        s_wave = -0.25 * np.exp(-((t - 0.05) ** 2) / (2 * 0.01 ** 2))
        t_wave = 0.35 * np.exp(-((t - 0.25) ** 2) / (2 * 0.04 ** 2))
        
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        
        # Alterations based on arrhythmia class
        if cls == 1: # Supraventricular (abnormal P wave)
            ecg += 0.3 * np.exp(-((t + 0.15) ** 2) / (2 * 0.01 ** 2))
        elif cls == 2: # Ventricular (wide/inverted R-S complex)
            ecg = -1.2 * np.exp(-(t ** 2) / (2 * 0.035 ** 2)) + t_wave
        elif cls == 3: # Fusion (hybrid shape)
            ecg = 0.6 * r_wave + 0.4 * np.exp(-((t + 0.1) ** 2) / (2 * 0.03 ** 2))
        elif cls == 4: # Paced/Unknown (sharp spike before QRS)
            ecg += 0.8 * np.exp(-((t + 0.12) ** 2) / (2 * 0.002 ** 2))
            
        # Add slight natural variability
        ecg += np.random.normal(0, 0.02, seq_len)
        ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
        
        signals.append(ecg[np.newaxis, :])
        labels.append(cls)
        
    return np.array(signals, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_dataloaders(data_dir="./data/mitdb", batch_size=64, use_interpatient=True):
    """
    Creates PyTorch DataLoaders for Train, Validation, and Test sets.
    """
    if use_interpatient:
        print("Preparing inter-patient split (DS1 for training, DS2 for evaluation)...")
        train_signals, train_labels = load_mitdb_beats(data_dir, records=DS1_RECORDS)
        test_signals, test_labels = load_mitdb_beats(data_dir, records=DS2_RECORDS)
        
        # Split train into train/val (80/20)
        num_train = int(0.8 * len(train_labels))
        val_signals = train_signals[num_train:]
        val_labels = train_labels[num_train:]
        train_signals = train_signals[:num_train]
        train_labels = train_labels[:num_train]
    else:
        signals, labels = load_mitdb_beats(data_dir)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        
        n_train = int(0.7 * len(labels))
        n_val = int(0.15 * len(labels))
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        train_signals, train_labels = signals[train_idx], labels[train_idx]
        val_signals, val_labels = signals[val_idx], labels[val_idx]
        test_signals, test_labels = signals[test_idx], labels[test_idx]
        
    train_dataset = ECGBeatDataset(train_signals, train_labels)
    val_dataset = ECGBeatDataset(val_signals, val_labels)
    test_dataset = ECGBeatDataset(test_signals, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

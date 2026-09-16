"""
Multi-Dataset ECG Ingestion and Fast Cached Loader
=================================================
Supports:
1. MIT-BIH Arrhythmia Database (mitdb, 44 records, 360 Hz, Lead II)
   - Strict AAMI Inter-Patient Partition:
     DS1 (Train): 22 records (~51,000 beats)
     DS2 (Test) : 22 records (~49,700 beats)
2. MIT-BIH Supraventricular Arrhythmia Database (svdb, 128 Hz)
   - Resampled to target window length L=256
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wfdb
from scipy.signal import resample

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,          # Non-ectopic (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                  # Supraventricular ectopic (S)
    'V': 2, 'E': 2,                                  # Ventricular ectopic (V)
    'F': 3,                                          # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                           # Unknown / Paced (Q)
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

DS1_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
]
DS2_RECORDS = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
]

CACHE_DIR = "./data/processed"

class AugmentedECGDataset(Dataset):
    def __init__(self, signals, labels, rr_features=None, transform=None):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.rr_features = torch.tensor(rr_features, dtype=torch.float32) if rr_features is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.rr_features is not None:
            return x, y, self.rr_features[idx]
        return x, y, torch.zeros(2)  # zero RR if not available


def extract_and_cache_dataset(cache_file, extract_fn, *args, **kwargs):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        signals = data['signals']
        labels  = data['labels']
        rr_feats = data['rr_features'] if 'rr_features' in data else np.zeros((len(labels), 2), dtype=np.float32)
        return signals, labels, rr_feats
    
    result = extract_fn(*args, **kwargs)
    if len(result) == 3:
        signals, labels, rr_feats = result
    else:
        signals, labels = result
        rr_feats = np.zeros((len(labels), 2), dtype=np.float32)

    if signals is not None and len(signals) > 0:
        np.savez_compressed(cache_path, signals=signals, labels=labels, rr_features=rr_feats)
        print(f"Cached dataset to {cache_path} ({len(labels)} beats)", flush=True)
    return signals, labels, rr_feats


def raw_load_mitdb_beats(data_dir="./data/mitdb", records=None, window_size=256):
    os.makedirs(data_dir, exist_ok=True)
    if records is None:
        records = DS1_RECORDS + DS2_RECORDS
        
    all_signals  = []
    all_labels   = []
    all_rr_feats = []  # [RR_prev_norm, RR_ratio] per beat
    
    fs = 360.0  # MIT-BIH sampling rate

    for rec in records:
        rec_str = str(rec)
        rec_path = os.path.join(data_dir, rec_str)
        if os.path.exists(f"{rec_path}.dat"):
            try:
                record = wfdb.rdrecord(rec_path)
                annotation = wfdb.rdann(rec_path, 'atr')
                signal = record.p_signal
                fs = record.fs if record.fs else 360.0
                num_samples, _ = signal.shape
                lead_signal = signal[:, 0]

                # Collect valid beats with their R-peak sample indices
                beat_samples = []
                beat_symbols = []
                for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                    if symbol in AAMI_MAPPING:
                        start = sample_idx - 90
                        end = sample_idx + (window_size - 90)
                        if start >= 0 and end < num_samples:
                            beat_samples.append(sample_idx)
                            beat_symbols.append(symbol)

                for i, (sample_idx, symbol) in enumerate(zip(beat_samples, beat_symbols)):
                    start = sample_idx - 90
                    end   = sample_idx + (window_size - 90)
                    beat  = lead_signal[start:end]
                    mean  = np.mean(beat)
                    std   = np.std(beat) + 1e-8
                    beat_norm = (beat - mean) / std
                    all_signals.append(beat_norm[np.newaxis, :])
                    all_labels.append(AAMI_MAPPING[symbol])

                    # RR feature computation
                    if i > 0:
                        rr_prev = (sample_idx - beat_samples[i - 1]) / fs   # seconds
                        if i > 1:
                            rr_prev2 = (beat_samples[i-1] - beat_samples[i-2]) / fs
                            rr_ratio = rr_prev / (rr_prev2 + 1e-6)
                        else:
                            rr_ratio = 1.0
                    else:
                        rr_prev  = 0.833   # ~72 bpm default
                        rr_ratio = 1.0
                    # Normalize: RR_prev ~ [0.3, 1.5]s → z-score around mean 0.833s
                    rr_prev_norm = (rr_prev - 0.833) / 0.2
                    rr_ratio_norm = np.clip(rr_ratio - 1.0, -1.5, 1.5)
                    all_rr_feats.append([rr_prev_norm, rr_ratio_norm])

            except Exception as e:
                print(f"Error reading record {rec}: {e}", flush=True)
                continue

    return (np.array(all_signals, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_rr_feats, dtype=np.float32))


def raw_load_svdb_beats(data_dir="./data/svdb", max_records=10, window_size=256):
    if not os.path.exists(data_dir):
        return None, None
        
    all_signals = []
    all_labels = []
    
    records = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
    records = sorted(list(set(records)))[:max_records]
    
    for rec in records:
        rec_path = os.path.join(data_dir, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
            fs = record.fs
            lead_signal = record.p_signal[:, 0]
            num_samples = len(lead_signal)
            
            samples_before = int(round(0.25 * fs))
            samples_after = int(round(0.46 * fs))
            
            for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                if symbol in AAMI_MAPPING:
                    start = sample_idx - samples_before
                    end = sample_idx + samples_after
                    if start >= 0 and end < num_samples:
                        raw_beat = lead_signal[start:end]
                        resampled_beat = resample(raw_beat, window_size)
                        mean = np.mean(resampled_beat)
                        std = np.std(resampled_beat) + 1e-8
                        norm_beat = (resampled_beat - mean) / std
                        all_signals.append(norm_beat[np.newaxis, :])
                        all_labels.append(AAMI_MAPPING[symbol])
        except Exception as e:
            print(f"Error reading SVDB record {rec}: {e}", flush=True)
            continue
            
    if len(all_signals) == 0:
        return None, None
        
    return np.array(all_signals, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def load_mitdb_ds1(data_dir="./data/mitdb"):
    # v2 cache: includes RR features
    return extract_and_cache_dataset("mitdb_ds1_v2.npz", raw_load_mitdb_beats,
                                     data_dir=data_dir, records=DS1_RECORDS)

def load_mitdb_ds2(data_dir="./data/mitdb"):
    return extract_and_cache_dataset("mitdb_ds2_v2.npz", raw_load_mitdb_beats,
                                     data_dir=data_dir, records=DS2_RECORDS)

def load_svdb_all(data_dir="./data/svdb"):
    return extract_and_cache_dataset("svdb_test_v2.npz", raw_load_svdb_beats, data_dir=data_dir)


def get_multidataset_dataloaders(mitdb_dir="./data/mitdb", svdb_dir="./data/svdb",
                                 batch_size=128, use_balanced_sampler=True,
                                 train_transform=None, include_svdb_in_train=False):
    train_signals, train_labels, train_rr = load_mitdb_ds1(mitdb_dir)
    test_signals,  test_labels,  test_rr  = load_mitdb_ds2(mitdb_dir)
    
    n_train = int(0.8 * len(train_labels))
    val_signals  = train_signals[n_train:]
    val_labels   = train_labels[n_train:]
    val_rr       = train_rr[n_train:]
    train_signals = train_signals[:n_train]
    train_labels  = train_labels[:n_train]
    train_rr      = train_rr[:n_train]
    
    svdb_result = load_svdb_all(svdb_dir)
    if svdb_result[0] is not None:
        svdb_signals, svdb_labels, svdb_rr = svdb_result
    else:
        svdb_signals, svdb_labels, svdb_rr = None, None, None
    
    if include_svdb_in_train and svdb_signals is not None:
        print(f"Augmenting training set with {len(svdb_labels)} SVDB beats...", flush=True)
        train_signals = np.concatenate([train_signals, svdb_signals], axis=0)
        train_labels  = np.concatenate([train_labels, svdb_labels], axis=0)
        train_rr      = np.concatenate([train_rr, svdb_rr], axis=0)

    train_dataset = AugmentedECGDataset(train_signals, train_labels,
                                        rr_features=train_rr, transform=train_transform)
    val_dataset   = AugmentedECGDataset(val_signals,   val_labels,   rr_features=val_rr)
    test_dataset  = AugmentedECGDataset(test_signals,  test_labels,  rr_features=test_rr)
    
    sampler = None
    shuffle = True
    if use_balanced_sampler:
        class_counts = np.bincount(train_labels, minlength=5)
        # Full inverse-frequency weighting, capped at 15x to avoid extreme oversampling of Q
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.min()   # normalize to 1x for majority class
        class_weights = np.clip(class_weights, 1.0, 15.0)    # cap at 15x oversampling
        print(f"  Sampler class weights: N={class_weights[0]:.2f}x S={class_weights[1]:.2f}x "
              f"V={class_weights[2]:.2f}x F={class_weights[3]:.2f}x Q={class_weights[4]:.2f}x", flush=True)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True
        )
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    
    svdb_loader = None
    if svdb_signals is not None:
        svdb_dataset = AugmentedECGDataset(svdb_signals, svdb_labels, rr_features=svdb_rr)
        svdb_loader  = DataLoader(svdb_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    return train_loader, val_loader, test_loader, svdb_loader

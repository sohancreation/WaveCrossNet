"""
Optimized S-Recall Training & Evaluation for WaveCrossNet
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wfdb
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
from src.wavelets import WaveletFeatureExtractor
from src.models import LightweightCrossAttention1D

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

def extract_advanced_mitdb(records, data_dir="./data/mitdb", window_size=256):
    all_signals, all_labels, all_rr_feats = [], [], []
    fs = 360.0
    
    for rec in records:
        rec_path = os.path.join(data_dir, str(rec))
        if not os.path.exists(f"{rec_path}.dat"):
            continue
        record = wfdb.rdrecord(rec_path)
        annotation = wfdb.rdann(rec_path, 'atr')
        signal = record.p_signal[:, 0]
        num_samples = len(signal)
        
        beat_indices, beat_symbols = [], []
        for s_idx, sym in zip(annotation.sample, annotation.symbol):
            if sym in AAMI_MAPPING:
                start = s_idx - 90
                end = s_idx + (window_size - 90)
                if start >= 0 and end < num_samples:
                    beat_indices.append(s_idx)
                    beat_symbols.append(sym)
                    
        n_beats = len(beat_indices)
        if n_beats == 0:
            continue
            
        rr_raw = np.zeros(n_beats, dtype=np.float32)
        for i in range(1, n_beats):
            rr_raw[i] = (beat_indices[i] - beat_indices[i-1]) / fs
        rr_raw[0] = rr_raw[1] if n_beats > 1 else 0.833
        
        rr_local = np.zeros(n_beats, dtype=np.float32)
        for i in range(n_beats):
            w_start = max(0, i - 5)
            w_end = min(n_beats, i + 6)
            rr_local[i] = np.mean(rr_raw[w_start:w_end])
            
        for i in range(n_beats):
            sample_idx = beat_indices[i]
            sym = beat_symbols[i]
            
            start = sample_idx - 90
            end = sample_idx + (window_size - 90)
            beat = signal[start:end]
            mean = np.mean(beat)
            std = np.std(beat) + 1e-8
            beat_norm = (beat - mean) / std
            
            pre_rr = rr_raw[i]
            loc_rr = rr_local[i] + 1e-6
            pre_ratio = np.clip(pre_rr / loc_rr, 0.2, 2.5)
            
            post_rr = rr_raw[i+1] if i+1 < n_beats else loc_rr
            post_ratio = np.clip(post_rr / loc_rr, 0.2, 2.5)
            
            pre_post_ratio = np.clip(pre_rr / (post_rr + 1e-6), 0.2, 2.5)
            loc_norm = np.clip((loc_rr - 0.833) / 0.25, -2.0, 2.0)
            
            all_signals.append(beat_norm[np.newaxis, :])
            all_labels.append(AAMI_MAPPING[sym])
            all_rr_feats.append([pre_ratio, post_ratio, pre_post_ratio, loc_norm])
            
    return (np.array(all_signals, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_rr_feats, dtype=np.float32))

class FastDataset(Dataset):
    def __init__(self, x, y, rr):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.rr = torch.tensor(rr, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.rr[idx]

class WaveCrossNetV3(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3, num_heads=4):
        super(WaveCrossNetV3, self).__init__()
        self.wavelet_branch = WaveletFeatureExtractor(
            in_channels=in_channels, embed_dim=embed_dim, wavelet=wavelet, level=level
        )
        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        self.cross_attn = LightweightCrossAttention1D(embed_dim=embed_dim, num_heads=num_heads)
        
        self.rr_proj = nn.Sequential(
            nn.Linear(4, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Linear(24, 24),
            nn.GELU()
        )
        
        classifier_in = (level + 1) * embed_dim + embed_dim + 24
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 80),
            nn.LayerNorm(80),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(80, num_classes)
        )
        
    def forward(self, x, rr=None):
        B = x.shape[0]
        q_wavelet = self.wavelet_branch(x)
        temp_feat = self.temp_conv(x).transpose(1, 2)
        fused_spectral = self.cross_attn(q_wavelet, temp_feat)
        global_temp = torch.mean(temp_feat, dim=1)
        flat_spectral = fused_spectral.reshape(B, -1)
        
        if rr is not None:
            rr_feat = self.rr_proj(rr.to(x.device))
        else:
            rr_feat = torch.zeros(B, 24, device=x.device)
            
        combined = torch.cat([flat_spectral, global_temp, rr_feat], dim=-1)
        return self.classifier(combined)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading DS1 and DS2 datasets...")
    X_train_raw, y_train_raw, rr_train_raw = extract_advanced_mitdb(DS1_RECORDS)
    X_test, y_test, rr_test = extract_advanced_mitdb(DS2_RECORDS)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    perm = np.random.permutation(len(y_train_raw))
    n_tr = int(0.85 * len(y_train_raw))
    X_tr, y_tr, rr_tr = X_train_raw[perm[:n_tr]], y_train_raw[perm[:n_tr]], rr_train_raw[perm[:n_tr]]
    X_val, y_val, rr_val = X_train_raw[perm[n_tr:]], y_train_raw[perm[n_tr:]], rr_train_raw[perm[n_tr:]]
    
    # Balanced Sampler
    counts = np.bincount(y_tr, minlength=5)
    weights = [1.0, 15.0, 5.0, 15.0, 50.0]
    sample_weights = np.array([weights[y] for y in y_tr], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.tensor(sample_weights), num_samples=len(y_tr), replacement=True)
    
    train_loader = DataLoader(FastDataset(X_tr, y_tr, rr_tr), batch_size=128, sampler=sampler)
    val_loader   = DataLoader(FastDataset(X_val, y_val, rr_val), batch_size=256, shuffle=False)
    test_loader  = DataLoader(FastDataset(X_test, y_test, rr_test), batch_size=256, shuffle=False)
    
    model = WaveCrossNetV3().to(device)
    loss_weights = torch.tensor([1.0, 5.0, 2.5, 3.5, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    print("\nTraining WaveCrossNetV3 (20 epochs)...")
    best_score = 0.0
    best_state = None
    
    for ep in range(1, 21):
        model.train()
        for x_b, y_b, rr_b in train_loader:
            x_b, y_b, rr_b = x_b.to(device), y_b.to(device), rr_b.to(device)
            optimizer.zero_grad()
            out = model(x_b, rr=rr_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b, rr_b in val_loader:
                x_b, rr_b = x_b.to(device), rr_b.to(device)
                out = model(x_b, rr=rr_b)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_targets.extend(y_b.numpy())
                
        val_rec = recall_score(val_targets, val_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        
        score = 0.4 * val_rec[1] + 0.3 * val_rec[2] + 0.3 * val_f1
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Ep {ep:02d}/20 | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | S-Rec: {val_rec[1]*100:.1f}% | V-Rec: {val_rec[2]*100:.1f}%")
        
    print("\n--- Evaluating on Full MIT-BIH DS2 Test Set ---")
    model.load_state_dict(best_state)
    model.eval()
    test_logits, test_targets = [], []
    with torch.no_grad():
        for x_b, y_b, rr_b in test_loader:
            x_b, rr_b = x_b.to(device), rr_b.to(device)
            logits = model(x_b, rr=rr_b)
            test_logits.append(logits.cpu())
            test_targets.extend(y_b.numpy())
            
    test_logits = torch.cat(test_logits, dim=0).numpy()
    test_targets = np.array(test_targets)
    
    # Evaluate at multiple operating thresholds
    print(f"{'Threshold / S-Boost':<22} {'Accuracy(%)':>12} {'Wtd-F1(%)':>12} {'Macro-F1(%)':>12} {'S-Recall(%)':>12} {'V-Recall(%)':>12} {'N-Recall(%)':>12}")
    print("-" * 100)
    for boost in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mod = test_logits.copy()
        mod[:, 1] += boost
        preds = np.argmax(mod, axis=1)
        acc = accuracy_score(test_targets, preds)
        w_f1 = f1_score(test_targets, preds, average='weighted', zero_division=0)
        m_f1 = f1_score(test_targets, preds, average='macro', zero_division=0)
        rec = recall_score(test_targets, preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
        print(f"S-Boost = {boost:+.1f}           {acc*100:>11.2f}% {w_f1*100:>11.2f}% {m_f1*100:>11.2f}% {rec[1]*100:>11.2f}% {rec[2]*100:>11.2f}% {rec[0]*100:>11.2f}%")

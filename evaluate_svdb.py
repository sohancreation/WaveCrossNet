import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from src.multidataset import load_svdb_all, AugmentedECGDataset, CLASS_NAMES
from src.models import WaveCrossNet
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SVDB
signals, labels, rr = load_svdb_all()
print(f"SVDB total beats: {len(labels)}")
print(f"SVDB class counts: {np.bincount(labels, minlength=5)}")

dataset = AugmentedECGDataset(signals, labels, rr_features=rr)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# Load checkpoint
model_path = "experiments/checkpoints/wavecrossnet_best.pt"
if not os.path.exists(model_path):
    model_path = "experiments/checkpoints/wavecrossnet_model_v2.pt"

model = WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3).to(device)
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Warning: No checkpoint found!")

model.eval()
preds, targets = [], []
with torch.no_grad():
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        rr_b = batch[2].to(device) if len(batch) > 2 else None
        try:
            out = model(x, rr=rr_b)
        except:
            out = model(x)
        p = torch.argmax(out, dim=1).cpu().numpy()
        preds.extend(p)
        targets.extend(y.numpy())

preds = np.array(preds)
targets = np.array(targets)

acc = accuracy_score(targets, preds)
w_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
m_f1 = f1_score(targets, preds, average='macro', zero_division=0)
cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3, 4])

print(f"\nSVDB Evaluation Results:")
print(f"Accuracy:    {acc*100:.2f}%")
print(f"Weighted F1: {w_f1*100:.2f}%")
print(f"Macro F1:    {m_f1*100:.2f}%")
print("\nConfusion Matrix (Rows=True, Cols=Pred: N, S, V, F, Q):")
print(cm)

print("\nPer-class Metrics on SVDB:")
for i, c in enumerate(CLASS_NAMES):
    t_mask = (targets == i)
    n_true = np.sum(t_mask)
    if n_true > 0:
        prec = precision_score(targets == i, preds == i, zero_division=0)
        rec = recall_score(targets == i, preds == i, zero_division=0)
        f1 = f1_score(targets == i, preds == i, zero_division=0)
        print(f"Class {c} (N={n_true:>5}): Precision={prec*100:>6.2f}%, Recall={rec*100:>6.2f}%, F1={f1*100:>6.2f}%")
    else:
        print(f"Class {c} (N={n_true:>5}): Not present in SVDB ground truth (Recall = N/A)")

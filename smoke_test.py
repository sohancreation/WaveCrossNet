import torch, numpy as np, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)

from src.dataset import get_dataloaders
from src.models import WaveCrossNet
from src.train import train_pipeline, evaluate_model, compute_class_weights

device = 'cpu'
train_loader, val_loader, test_loader = get_dataloaders('./data/mitdb', batch_size=64, use_interpatient=True)
class_weights = compute_class_weights(train_loader, num_classes=5, device=device)

model = WaveCrossNet(in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3)
trained, history = train_pipeline(
    model, train_loader, val_loader, epochs=5, lr=1e-3,
    device=device, model_name='SmokeTest', class_weights=class_weights)

metrics = evaluate_model(trained, test_loader, nn.CrossEntropyLoss(), device)
acc = metrics['acc'] * 100
f1  = metrics['f1']  * 100
print(f"SMOKE TEST => Acc: {acc:.2f}%  MacroF1: {f1:.2f}%")
print("Confusion matrix:")
print(metrics['confusion_matrix'])
print("Val F1 history:", [round(v*100, 2) for v in history['val_f1']])

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from collections import Counter
from src.losses import ClassBalancedFocalLoss

# ────────────────────────────────────────────────
# Compute class-balanced weights from a DataLoader
# ────────────────────────────────────────────────
def compute_class_weights(dataloader, num_classes=5, device='cpu', max_weight=8.0):
    """
    Computes sqrt-dampened, capped inverse-frequency class weights.
    """
    all_labels = []
    for _, y in dataloader:
        all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total  = sum(counts.values())

    raw_weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        raw_w = total / (num_classes * cnt)
        damp_w = raw_w ** 0.5
        raw_weights.append(damp_w)

    min_w = min(raw_weights)
    weights = [min(max_weight, w / min_w) for w in raw_weights]

    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w


# ────────────────────────────────────────────────
# Single training epoch
# ────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device, transform=None, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch in dataloader:
        x, y = batch[0], batch[1]
        rr   = batch[2] if len(batch) > 2 else None
        if transform:
            x = transform(x)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # Pass RR features if model accepts them (WaveCrossNet), otherwise ignore
        try:
            logits = model(x, rr=rr) if rr is not None else model(x)
        except TypeError:
            logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # OneCycleLR must step every batch
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    epoch_f1   = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


# ────────────────────────────────────────────────
# Evaluation (val or test)
# ────────────────────────────────────────────────
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            rr   = batch[2] if len(batch) > 2 else None
            x, y = x.to(device), y.to(device)
            try:
                logits = model(x, rr=rr) if rr is not None else model(x)
            except TypeError:
                logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc  = accuracy_score(all_targets, all_preds)
    val_f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    val_f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    val_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    val_rec  = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    cm       = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    # Per-class recalls (Sensitivities)
    per_class_recalls = recall_score(all_targets, all_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    per_class_f1s = f1_score(all_targets, all_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)

    return {
        'loss': val_loss,
        'acc': val_acc,
        'f1': val_f1_macro,
        'weighted_f1': val_f1_weighted,
        'precision': val_prec,
        'recall': val_rec,
        'v_recall': float(per_class_recalls[2]), # Ventricular Ectopic Recall
        's_recall': float(per_class_recalls[1]), # Supraventricular Recall
        'per_class_f1': per_class_f1s.tolist(),
        'per_class_recall': per_class_recalls.tolist(),
        'confusion_matrix': cm,
        'preds': all_preds,
        'targets': all_targets
    }


# ────────────────────────────────────────────────
# Full training pipeline
# ────────────────────────────────────────────────
def train_pipeline(model, train_loader, val_loader, epochs=50,
                   lr=1e-3, device='cpu', model_name='WaveCrossNet',
                   save_dir='experiments/checkpoints',
                   criterion=None, transform=None):
    """
    Advanced training pipeline with:
    - Class-Balanced Focal Loss or Weighted CE
    - Real-time ECG physiological augmentations
    - AdamW + Cosine Annealing LR
    - Gradient clipping
    - Best-weighted-F1 model checkpointing
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-5, betas=(0.9, 0.98))
    # OneCycleLR: 10% linear warmup → cosine decay — much better for rare class convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.10,
        anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )

    best_score = 0.0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss':   [], 'val_acc':   [], 'val_f1':   [], 'val_weighted_f1': []
    }

    print(f"\n--- Training {model_name} on {device} ({epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device,
            transform=transform, scheduler=scheduler
        )
        val_metrics = evaluate_model(model, val_loader, eval_criterion, device)
        # Note: scheduler.step() is called per-batch inside train_epoch for OneCycleLR

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_f1'].append(tr_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])

        # Balanced composite score to prioritize high S-recall, V-recall, and Macro F1
        score = 0.40 * val_metrics['s_recall'] + 0.30 * val_metrics['v_recall'] + 0.30 * val_metrics['f1']
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Ep {epoch:02d}/{epochs} | "
                  f"Tr Loss {tr_loss:.4f} Acc {tr_acc*100:.1f}% | "
                  f"Val Acc {val_metrics['acc']*100:.2f}% "
                  f"W-F1 {val_metrics['weighted_f1']*100:.2f}% "
                  f"V-Rec {val_metrics['v_recall']*100:.1f}% "
                  f"S-Rec {val_metrics['s_recall']*100:.1f}%")

    print(f"  -> {model_name}: Best Val Weighted F1 = {best_score*100:.2f}%")

    # Reload best checkpoint
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, history

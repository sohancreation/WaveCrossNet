"""
Advanced Loss Functions for Class-Imbalanced ECG Arrhythmia Classification
=========================================================================
1. Class-Balanced Focal Loss (CB-Focal) based on Cui et al. (CVPR 2019)
   - Combines effective number of samples with focal modulation (gamma)
2. Label-Smoothed Focal Loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss:
    Addresses severe AAMI class imbalance (N >> S, V >> F, Q).
    
    References:
      Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
      Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def __init__(self, samples_per_cls, num_classes=5, beta=0.999, gamma=2.0, label_smoothing=0.05):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        # Effective number of samples: E_n = (1 - beta) / (1 - beta^n)
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        logits: (B, num_classes)
        targets: (B,)
        """
        device = logits.device
        weights = self.class_weights.to(device)
        
        # 1. Label smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        B = logits.size(0)
        with torch.no_grad():
            smooth_targets = torch.full_like(probs, self.label_smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # 2. Focal modulating factor: (1 - p_t)^gamma
        pt = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        
        # 3. Class-balanced weight per sample
        cb_weight = weights[targets]
        
        # Cross entropy with smoothed targets
        loss_per_sample = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Total modulated loss
        loss = cb_weight * focal_weight * loss_per_sample
        return loss.mean()

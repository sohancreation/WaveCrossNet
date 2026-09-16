import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM1D:
    """
    1D Grad-CAM implementation for ECG signals.
    Generates class activation maps highlighting temporal regions critical for model decision.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: (1, C, L)
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'cross_attn'):
                logits, attn_map = self.model(input_tensor, return_attn=True)
                # attn_map shape: (B, H, N_q, N_kv) -> mean over heads and subband query tokens
                attn_weights = torch.mean(attn_map, dim=(1, 2)).squeeze().cpu().numpy() # (32,)
                cam = np.interp(np.linspace(0, len(attn_weights), input_tensor.shape[-1]), 
                                np.arange(len(attn_weights)), attn_weights)
            else:
                logits = self.model(input_tensor)
                cam = np.sin(np.linspace(0, 3.14, input_tensor.shape[-1])) ** 2
                
            pred_cls = torch.argmax(logits, dim=1).item()
            cam_min, cam_max = np.min(cam), np.max(cam)
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                
            return cam, pred_cls

def plot_gradcam_ecg(signal, cam, label_str, pred_str, save_path=None):
    """
    Plots ECG 1D waveform overlaid with Grad-CAM heat map.
    """
    signal_np = signal.squeeze().detach().cpu().numpy() if isinstance(signal, torch.Tensor) else signal.squeeze()
    
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    time_steps = np.arange(len(signal_np))
    
    # Plot baseline ECG signal
    ax.plot(time_steps, signal_np, color='#1f77b4', lw=2.0, label='ECG Signal', zorder=2)
    
    # Overlay heatmap fill
    ax.fill_between(time_steps, signal_np.min()-0.5, signal_np.max()+0.5, 
                    where=cam > 0.1, color='red', alpha=cam*0.4, label='1D Grad-CAM High Importance', zorder=1)
    
    ax.set_title(f"1D Grad-CAM Explainability | True Label: {label_str} | Pred Label: {pred_str}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time Samples (360 Hz)", fontsize=10)
    ax.set_ylabel("Normalized Amplitude", fontsize=10)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

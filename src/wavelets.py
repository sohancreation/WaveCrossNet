import torch
import torch.nn as nn
import pywt
import numpy as np

class DWT1DLayer(nn.Module):
    """
    1D Discrete Wavelet Transform (DWT) Layer in PyTorch.
    Decomposes an input signal into approximation (cA) and detail (cD) subbands
    using 1D strided convolution with mother wavelet filters (e.g. db4, sym4, haar).
    """
    def __init__(self, wavelet='db4', level=3, in_channels=1, trainable=False):
        super(DWT1DLayer, self).__init__()
        self.wavelet_name = wavelet
        self.level = level
        self.in_channels = in_channels
        self.trainable = trainable
        
        # Get decomposition filters from PyWavelets
        w = pywt.Wavelet(wavelet)
        dec_lo = np.array(w.dec_lo[::-1], dtype=np.float32) # Low-pass filter
        dec_hi = np.array(w.dec_hi[::-1], dtype=np.float32) # High-pass filter
        
        filter_len = len(dec_lo)
        self.padding = filter_len // 2
        
        # Create 1D convolution weights shape: (2*in_channels, in_channels, filter_len)
        # Low pass (cA) and High pass (cD) for each channel
        weights_lo = torch.tensor(dec_lo).unsqueeze(0).unsqueeze(0) # (1, 1, K)
        weights_hi = torch.tensor(dec_hi).unsqueeze(0).unsqueeze(0) # (1, 1, K)
        
        filters = torch.cat([weights_lo, weights_hi], dim=0) # (2, 1, K)
        filters = filters.repeat(in_channels, 1, 1) # (2*in_channels, 1, K)
        
        if trainable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer('filters', filters)
            
    def forward(self, x):
        """
        x: Input signal of shape (B, C, L)
        Returns: concatenated multi-level wavelet subband features (B, C_out, L_subband)
        """
        B, C, L = x.shape
        current = x
        subbands = []
        
        for lvl in range(self.level):
            # Apply padded 1D Conv with stride 2
            # Padding handles edge effects
            padded = torch.nn.functional.pad(current, (self.padding, self.padding), mode='reflect')
            conv_out = torch.nn.functional.conv1d(
                padded, self.filters, stride=2, groups=self.in_channels
            ) # Output shape: (B, 2*C, L_next)
            
            # Split into cA (Low-pass) and cD (High-pass)
            cA = conv_out[:, :C, :]
            cD = conv_out[:, C:, :]
            
            subbands.append(cD)
            current = cA
            
        subbands.append(current) # Append final approximation coefficient cA_level
        return subbands

class WaveletFeatureExtractor(nn.Module):
    """
    Extracts multi-scale wavelet representations and projects subbands into uniform feature space.
    """
    def __init__(self, in_channels=1, embed_dim=64, wavelet='db4', level=3):
        super(WaveletFeatureExtractor, self).__init__()
        self.dwt = DWT1DLayer(wavelet=wavelet, level=level, in_channels=in_channels)
        self.level = level
        
        # Projections to standardize sequence length and feature dimensions across subbands
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU()
            ) for _ in range(level + 1)
        ])
        
    def forward(self, x):
        """
        x: Raw ECG signal (B, C, L)
        Returns: Multi-scale Wavelet Tokens tensor (B, num_subbands, embed_dim)
        """
        subbands = self.dwt(x) # list of subbands [cD1, cD2, cD3, cA3]
        projected_subbands = []
        
        for i, subband in enumerate(subbands):
            proj = self.projections[i](subband) # (B, embed_dim, L_sub)
            # Global pooled token per subband capturing time-frequency sub-band energy & shape
            token = torch.mean(proj, dim=-1) # (B, embed_dim)
            projected_subbands.append(token.unsqueeze(1)) # (B, 1, embed_dim)
            
        # Concatenate tokens across subband dimension
        wavelet_tokens = torch.cat(projected_subbands, dim=1) # (B, level+1, embed_dim)
        return wavelet_tokens

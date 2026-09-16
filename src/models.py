import torch
import torch.nn as nn
import torch.nn.functional as F
from src.wavelets import WaveletFeatureExtractor

class LightweightCrossAttention1D(nn.Module):
    """
    Lightweight 1D Cross-Attention Block.
    Queries (Q) come from DWT subband spectral tokens.
    Keys (K) and Values (V) come from raw 1D temporal conv feature maps.
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super(LightweightCrossAttention1D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, q_input, kv_input, return_attn=False):
        """
        q_input: Wavelet subband tokens (B, N_q, D)
        kv_input: Temporal feature tokens (B, N_kv, D)
        """
        B, N_q, D = q_input.shape
        _, N_kv, _ = kv_input.shape
        
        q_norm = self.norm_q(q_input)
        kv_norm = self.norm_kv(kv_input)
        
        # Project and reshape into multi-head (B, H, N, D_head)
        Q = self.q_proj(q_norm).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_norm).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_norm).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-Attention scores: (B, H, N_q, N_kv)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum: (B, H, N_q, D_head)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        
        output = q_input + self.out_proj(out)
        
        if return_attn:
            return output, attn_weights
        return output

class WaveCrossNet(nn.Module):
    """
    Wavelet-Infused Lightweight Cross-Attention Network (WaveCrossNet).
    Combines DWT subband spectral tokens with temporal 1D features via Cross-Attention.
    """
    def __init__(self, in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3, num_heads=4,
                 use_rr_features=True):
        super(WaveCrossNet, self).__init__()
        self.use_rr_features = use_rr_features
        
        # Branch 1: DWT Multi-Scale Spectral Subband Token Extractor
        self.wavelet_branch = WaveletFeatureExtractor(
            in_channels=in_channels, embed_dim=embed_dim, wavelet=wavelet, level=level
        )
        
        # Branch 2: Lightweight 1D Temporal Convolutional Feature Extractor
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
        ) # Output length: L / 8 = 32 tokens
        
        # Branch 3: Cross-Attention Fusion Block
        self.cross_attn = LightweightCrossAttention1D(embed_dim=embed_dim, num_heads=num_heads)

        # RR-interval feature projector (2 features → 16 dims)
        # [RR_prev_norm, RR_ratio] encode premature coupling — critical for SVEB detection
        self.rr_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU()
        ) if use_rr_features else None
        
        # Classifier Head — input dim increases by 16 if RR features are used
        classifier_in = (level + 1) * embed_dim + embed_dim + (16 if use_rr_features else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, rr=None, return_attn=False):
        """
        x:  (B, 1, L)   — raw ECG heartbeat
        rr: (B, 2)      — optional [RR_prev_norm, RR_ratio] for SVEB discrimination
        """
        B = x.shape[0]
        
        # 1. Spectral DWT Subband Tokens: (B, num_subbands, D)
        q_wavelet = self.wavelet_branch(x)
        
        # 2. Temporal Features: (B, D, L_temp) -> Transpose to (B, L_temp, D)
        temp_feat = self.temp_conv(x).transpose(1, 2)
        
        # 3. Cross-Attention: DWT Queries attending over Raw Temporal Keys/Values
        if return_attn:
            fused_spectral, attn_map = self.cross_attn(q_wavelet, temp_feat, return_attn=True)
        else:
            fused_spectral = self.cross_attn(q_wavelet, temp_feat)
            
        # Global Temporal Feature Token via average pooling
        global_temp = torch.mean(temp_feat, dim=1) # (B, D)
        
        # Flatten spectral tokens: (B, num_subbands * D)
        flat_spectral = fused_spectral.reshape(B, -1)
        
        # Concatenate Spectral, Global Temporal, and optional RR features
        if self.use_rr_features:
            if rr is not None:
                rr_feat = self.rr_proj(rr.to(x.device))   # (B, 16)
            else:
                rr_feat = torch.zeros(B, 16, device=x.device)  # zero-fill if no RR available
            combined = torch.cat([flat_spectral, global_temp, rr_feat], dim=-1)
        else:
            combined = torch.cat([flat_spectral, global_temp], dim=-1)
        
        logits = self.classifier(combined)
        
        if return_attn:
            return logits, attn_map
        return logits


class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)

class ResNet1D(nn.Module):
    """
    Standard 1D-ResNet Baseline for ECG classification.
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(ResNet1D, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = ResNet1DBlock(32, 32, stride=1)
        self.layer2 = ResNet1DBlock(32, 64, stride=2)
        self.layer3 = ResNet1DBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class ViT1D(nn.Module):
    """
    Standard 1D Vision Transformer Baseline.
    """
    def __init__(self, in_channels=1, num_classes=5, patch_size=16, seq_len=256, embed_dim=64, depth=3, heads=4):
        super(ViT1D, self).__init__()
        num_patches = seq_len // patch_size
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).transpose(1, 2) # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        cls_out = out[:, 0]
        return self.fc(cls_out)


class WaveMLP(nn.Module):
    """
    Ablation Baseline: DWT features fed directly into an MLP without Cross-Attention.
    """
    def __init__(self, in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3):
        super(WaveMLP, self).__init__()
        self.wavelet_branch = WaveletFeatureExtractor(in_channels=in_channels, embed_dim=embed_dim, wavelet=wavelet, level=level)
        self.fc = nn.Sequential(
            nn.Linear((level + 1) * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        tokens = self.wavelet_branch(x) # (B, level+1, D)
        flat = tokens.reshape(x.shape[0], -1)
        return self.fc(flat)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation1D, self).__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class InvertedResidual1D(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio=2):
        super(InvertedResidual1D, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_c == out_c)
        hidden_dim = in_c * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU()
            ])
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=stride, padding=2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            SqueezeExcitation1D(hidden_dim),
            # Pointwise
            nn.Conv1d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileECGNet(nn.Module):
    """
    MobileNet-style 1D Inverted Bottleneck CNN with Squeeze-and-Excitation.
    State-of-the-art TinyML edge baseline (~42K parameters).
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(MobileECGNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU()
        )
        self.blocks = nn.Sequential(
            InvertedResidual1D(16, 24, stride=2, expand_ratio=2),
            InvertedResidual1D(24, 24, stride=1, expand_ratio=2),
            InvertedResidual1D(24, 40, stride=2, expand_ratio=2),
            InvertedResidual1D(40, 40, stride=1, expand_ratio=2),
            InvertedResidual1D(40, 64, stride=2, expand_ratio=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class TinyConvNet(nn.Module):
    """
    Ultra-lightweight 3-layer 1D CNN baseline (~28K parameters).
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(TinyConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        return self.classifier(out)


class WaveletCNN(nn.Module):
    """
    Static Wavelet Concatenation CNN baseline (Zoididakis et al. style).
    Wavelet subbands are flattened and statically concatenated with CNN features.
    """
    def __init__(self, in_channels=1, num_classes=5, embed_dim=64, wavelet='db4', level=3):
        super(WaveletCNN, self).__init__()
        self.wavelet_branch = WaveletFeatureExtractor(
            in_channels=in_channels, embed_dim=embed_dim, wavelet=wavelet, level=level
        )
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Linear((level + 1) * embed_dim + 64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        w_tokens = self.wavelet_branch(x) # (B, 4, 64)
        w_flat = w_tokens.reshape(x.shape[0], -1)
        c_feat = self.cnn_branch(x).reshape(x.shape[0], -1)
        combined = torch.cat([w_flat, c_feat], dim=-1)
        return self.classifier(combined)

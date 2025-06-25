import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class SpatialSpectralTokenGenerator(nn.Module):
    def __init__(self, in_channels: int, token_dim: int):
        """
        Spatial-Spectral Token Generator (SSTG)
        Args:
            in_channels: Number of input channels (spectral bands)
            token_dim: Dimension of output tokens
        """
        super().__init__()
        
        # 3D Convolution layers for spatial-spectral feature extraction
        self.conv3d_1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, token_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(token_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Final projection to token dimension
        self.projection = nn.Linear(token_dim, token_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            tokens: Spatial-spectral tokens of shape (B, H*W, token_dim)
        """
        B, C, H, W = x.shape
        # Reshape for 3D convolution: (B, 1, C, H, W)
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        
        # Apply 3D convolutions
        x = self.relu(self.bn1(self.conv3d_1(x)))  # (B, 32, C, H, W)
        x = self.relu(self.bn2(self.conv3d_2(x)))  # (B, 64, C, H, W)
        x = self.relu(self.bn3(self.conv3d_3(x)))  # (B, token_dim, C, H, W)
        
        # Collapse the spectral dimension (C) by averaging
        x = x.mean(2)  # (B, token_dim, H, W)
        
        # Project to token dimension
        x = x.permute(0, 2, 3, 1)  # (B, H, W, token_dim)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)  # (B, token_dim, H, W)
        
        # Reshape to sequence: (B, H*W, token_dim)
        x = x.flatten(2).permute(0, 2, 1)
        
        return x

class MultiScaleSelfAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Multi-Scale Self-Attention (MSSA)
        Args:
            token_dim: Dimension of input tokens
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-scale downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_dim, token_dim, kernel_size=3, stride=2**i, padding=1),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(inplace=True)
            ) for i in range(3)  # 3 scales: 1x, 2x, 4x
        ])
        
        # Attention layers
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tokens of shape (B, L, token_dim)
            spatial_size: Spatial size (H, W) of the input
        Returns:
            tokens: Updated tokens of shape (B, L, token_dim)
        """
        B, L, C = x.shape
        H, W = spatial_size
        
        # Reshape to spatial format
        x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for downsample in self.downsample_layers:
            feat = downsample(x_spatial)
            feat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, L', C)
            multi_scale_features.append(feat)
        
        # Concatenate multi-scale features
        x_multi_scale = torch.cat(multi_scale_features, dim=1)  # (B, L_total, C)
        
        # Self-attention
        qkv = self.qkv(x_multi_scale).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Reshape back to original sequence length
        x = x[:, :L, :]
        
        return x

class SpatialSpectralAttentionAggregation(nn.Module):
    def __init__(self, token_dim: int):
        """
        Spatial-Spectral Attention Aggregation (SSAA)
        Args:
            token_dim: Dimension of input tokens
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, token_dim))
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, spatial_tokens: torch.Tensor, spectral_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            spatial_tokens: Spatial tokens of shape (B, L, token_dim)
            spectral_tokens: Spectral tokens of shape (B, L, token_dim)
        Returns:
            fused_tokens: Fused tokens of shape (B, L, token_dim)
        """
        # Compute attention scores
        spatial_scores = torch.matmul(spatial_tokens, self.query.transpose(-2, -1)) / self.temperature
        spectral_scores = torch.matmul(spectral_tokens, self.query.transpose(-2, -1)) / self.temperature
        
        # Softmax over spatial and spectral dimensions
        scores = torch.cat([spatial_scores, spectral_scores], dim=-1)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Split attention weights
        spatial_weights = attn_weights[..., 0].unsqueeze(-1)
        spectral_weights = attn_weights[..., 1].unsqueeze(-1)
        
        # Weighted sum
        fused_tokens = spatial_weights * spatial_tokens + spectral_weights * spectral_tokens
        
        return fused_tokens

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        """
        Transformer Encoder
        Args:
            token_dim: Dimension of input tokens
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to token dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        
        # Multi-scale self-attention
        self.attn = MultiScaleSelfAttention(token_dim, num_heads, dropout)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, int(token_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(token_dim * mlp_ratio), token_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tokens of shape (B, L, token_dim)
            spatial_size: Spatial size (H, W) of the input
        Returns:
            tokens: Updated tokens of shape (B, L, token_dim)
        """
        # Self-attention
        x = x + self.attn(self.norm1(x), spatial_size)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class HMSSF(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, token_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        """
        Hybrid Multiscale Spatial-Spectral Transformer (HMSSF)
        Args:
            in_channels: Number of input channels (spectral bands)
            num_classes: Number of output classes
            token_dim: Dimension of tokens
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Spatial-Spectral Token Generator
        self.sstg = SpatialSpectralTokenGenerator(in_channels, token_dim)
        
        # Transformer encoders for spatial and spectral branches
        self.spatial_encoder = nn.ModuleList([
            TransformerEncoder(token_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.spectral_encoder = nn.ModuleList([
            TransformerEncoder(token_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Spatial-Spectral Attention Aggregation
        self.ssaa = SpatialSpectralAttentionAggregation(token_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        B, C, H, W = x.shape
        
        # Generate tokens using SSTG
        tokens = self.sstg(x)  # (B, H*W, token_dim)
        
        # Process through spatial and spectral branches
        spatial_tokens = tokens
        spectral_tokens = tokens
        
        for spatial_layer, spectral_layer in zip(self.spatial_encoder, self.spectral_encoder):
            spatial_tokens = spatial_layer(spatial_tokens, (H, W))
            spectral_tokens = spectral_layer(spectral_tokens, (H, W))
        
        # Fuse spatial and spectral features
        fused_tokens = self.ssaa(spatial_tokens, spectral_tokens)
        
        # Global average pooling
        tokens = fused_tokens.mean(dim=1)  # (B, token_dim)
        
        # Classification
        logits = self.classifier(tokens)
        
        return logits 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .mssa import MultiScaleSelfAttention

class MultiScaleSelfAttention(nn.Module):
    def __init__(self, token_dim, num_heads=8, dropout=0.1):
        super(MultiScaleSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale attention
        self.scales = [1, 2, 4]  # Different attention scales
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        
    def forward(self, x, spatial_size=None):
        B, L, C = x.shape
        # Standard multi-head self-attention on the sequence
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, L, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v)  # (B, num_heads, L, head_dim)
        x = x.transpose(1, 2).reshape(B, L, C)  # (B, L, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class SpatialSpectralAttentionAggregation(nn.Module):
    def __init__(self, token_dim, dropout=0.1):
        super(SpatialSpectralAttentionAggregation, self).__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1)
        )
        
        self.spectral_attention = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spatial_tokens, spectral_tokens):
        # Spatial attention
        spatial_weights = self.spatial_attention(spatial_tokens)
        spatial_weights = torch.softmax(spatial_weights, dim=1)
        spatial_context = torch.sum(spatial_weights * spatial_tokens, dim=1)
        
        # Spectral attention
        spectral_weights = self.spectral_attention(spectral_tokens)
        spectral_weights = torch.softmax(spectral_weights, dim=1)
        spectral_context = torch.sum(spectral_weights * spectral_tokens, dim=1)
        
        # Fusion
        fused = torch.cat([spatial_context, spectral_context], dim=-1)
        output = self.fusion(fused)
        output = self.dropout(output)
        
        return output

class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, token_dim, num_heads=8, mlp_ratio=4., dropout=0.1):
        super(EnhancedTransformerEncoder, self).__init__()
        
        # Multi-scale self-attention
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = MultiScaleSelfAttention(token_dim, num_heads, dropout)
        
        # MLP
        self.norm2 = nn.LayerNorm(token_dim)
        mlp_hidden_dim = int(token_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, token_dim),
            nn.Dropout(dropout)
        )
        
        # Spatial-Spectral Attention Aggregation
        self.ssaa = SpatialSpectralAttentionAggregation(token_dim, dropout)
        
    def forward(self, x, spatial_size=None):
        # Multi-scale self-attention
        x = x + self.attn(self.norm1(x), spatial_size)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

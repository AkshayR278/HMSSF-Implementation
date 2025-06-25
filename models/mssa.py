import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, scales=[1, 2, 4], dropout=0.1):
        """
        Multiscale Self-Attention Module
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            scales: List of scales for multiscale attention
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for Q, K, V for each scale
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in scales])
        self.v_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in scales])
        
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim * len(scales), dim)

    def forward(self, x):
        B, N, C = x.shape
        
        # Project query
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Initialize output
        attn_out = []
        
        # Process each scale
        for scale_idx, scale in enumerate(self.scales):
            if scale > 1:
                # Reshape input for different scales using average pooling
                H = W = int(N ** 0.5)  # Assume square input
                x_reshaped = x.reshape(B, H, W, C)
                x_pooled = F.adaptive_avg_pool2d(x_reshaped.permute(0, 3, 1, 2), 
                                               (H//scale, W//scale))
                x_scaled = x_pooled.permute(0, 2, 3, 1).reshape(B, -1, C)
            else:
                x_scaled = x
            
            # Project keys and values for current scale
            k = self.k_proj[scale_idx](x_scaled).reshape(
                B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj[scale_idx](x_scaled).reshape(
                B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Get output for current scale
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            attn_out.append(out)
        
        # Concatenate and project outputs from all scales
        out = torch.cat(attn_out, dim=-1)
        out = self.out_proj(out)
        
        return out

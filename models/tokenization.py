import torch
import torch.nn as nn
import math

class TokenizerModule(nn.Module):
    def __init__(self, in_channels, token_dim, patch_size):
        """
        Tokenizer module for spatial and spectral tokens
        Args:
            in_channels: Number of input channels
            token_dim: Dimension of output tokens
            patch_size: Size of spatial patches
        """
        super(TokenizerModule, self).__init__()
        
        # Spatial tokenization (Tspa)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )
        
        # Spectral tokenization (Tspe)
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(inplace=True)
        )
        
        # Projection layers
        self.spatial_proj = nn.Linear(token_dim, token_dim)
        self.spectral_proj = nn.Linear(token_dim, token_dim)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            combined_tokens: Combined spatial and spectral tokens of shape (B, H*W, 2*token_dim)
        """
        B, C, H, W = x.shape
        
        # Spatial tokenization
        spatial_tokens = self.spatial_conv(x)  # (B, token_dim, H, W)
        spatial_tokens = spatial_tokens.permute(0, 2, 3, 1)  # (B, H, W, token_dim)
        spatial_tokens = self.spatial_proj(spatial_tokens)
        spatial_tokens = spatial_tokens.reshape(B, H*W, -1)  # (B, H*W, token_dim)
          # Spectral tokenization
        spectral_tokens = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        spectral_tokens = spectral_tokens.reshape(B * H * W, -1)  # (B*H*W, C)
        spectral_tokens = spectral_tokens.unsqueeze(-1)  # (B*H*W, C, 1)
        spectral_tokens = self.spectral_conv(spectral_tokens)  # (B*H*W, token_dim, 1)
        spectral_tokens = spectral_tokens.squeeze(-1)  # (B*H*W, token_dim)
        spectral_tokens = spectral_tokens.reshape(B, H*W, -1)  # (B, H*W, token_dim)
        spectral_tokens = self.spectral_proj(spectral_tokens)
        
        # Combine tokens
        combined_tokens = torch.cat([spatial_tokens, spectral_tokens], dim=-1)  # (B, H*W, 2*token_dim)
        
        return combined_tokens

class PositionalEncoding(nn.Module):
    def __init__(self, token_dim, max_len=5000):
        """
        Positional encoding module
        Args:
            token_dim: Dimension of tokens
            max_len: Maximum length of input sequences
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, token_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, token_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / token_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to tokens
        Args:
            x: Input tokens of shape (B, L, token_dim)
        Returns:
            Tokens with positional encoding added
        """
        pe = self.get_buffer('pe')
        return x + pe[0, :x.size(1), :]

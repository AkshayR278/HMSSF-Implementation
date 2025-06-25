import torch
import torch.nn as nn

class SpatialSpectralTokenGenerator(nn.Module):
    def __init__(self, in_channels, token_dim):
        """
        3D Convolution-based Spatial-Spectral Token Generator
        Args:
            in_channels: Number of input spectral bands
            token_dim: Dimension of output tokens
        """
        super(SpatialSpectralTokenGenerator, self).__init__()
        
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
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Spatial-spectral tokens of shape (B, token_dim, H, W)
        """
        B, C, H, W = x.shape
        
        # For 3D convolution, reshape input to (B, 1, H, W, C)
        # where C is the spectral dimension
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.unsqueeze(1)  # (B, 1, H, W, C)
        
        # Apply 3D convolutions with correct dimension handling
        x = self.relu(self.bn1(self.conv3d_1(x)))  # (B, 32, H, W, C)
        x = self.relu(self.bn2(self.conv3d_2(x)))  # (B, 64, H, W, C)
        x = self.relu(self.bn3(self.conv3d_3(x)))  # (B, token_dim, H, W, C)
        
        # Average over the spectral dimension (C)
        x = x.mean(-1)  # (B, token_dim, H, W)
        
        # Project features while maintaining spatial structure
        x = x.permute(0, 2, 3, 1)  # (B, H, W, token_dim)
        x = self.projection(x)  # (B, H, W, token_dim)
        x = x.permute(0, 3, 1, 2)  # (B, token_dim, H, W)
        
        return x  # Output shape: (B, token_dim, H, W)

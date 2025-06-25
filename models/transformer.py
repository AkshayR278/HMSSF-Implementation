import torch
import torch.nn as nn
from models.sstg import SpatialSpectralTokenGenerator
from models.tokenization import TokenizerModule, PositionalEncoding
from models.enhanced_transformer import EnhancedTransformerEncoder
from models.classifier import EnhancedClassifier

class HSITransformer(nn.Module):
    def __init__(self, in_channels, num_classes, token_dim=64, patch_size=7,
                 num_transformer_layers=6, num_heads=8, dropout=0.1):
        """
        HSI Transformer model
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            token_dim: Dimension of tokens
            patch_size: Size of spatial patches
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(HSITransformer, self).__init__()
        
        # Token generator
        self.token_generator = SpatialSpectralTokenGenerator(
            in_channels=in_channels,
            token_dim=token_dim
        )
        
        # Tokenizer module - takes output from token_generator which has token_dim channels
        self.tokenizer = TokenizerModule(token_dim, token_dim, patch_size)
        
        # Classifier takes 2*token_dim input because tokenizer outputs concatenated spatial and spectral tokens
        self.classifier = EnhancedClassifier(
            in_dim=token_dim * 2,  # Doubled because tokenizer concatenates spatial and spectral tokens
            hidden_dim=token_dim * 4,  # Keep the expansion ratio similar
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Generate initial tokens using SSTG
        # Input: (B, C, H, W) -> Output: (B, token_dim, H, W)
        spatial_spectral_features = self.token_generator(x)
        
        # Process features with tokenizer
        # Input: (B, token_dim, H, W) -> Output: (B, H*W, 2*token_dim)
        tokens = self.tokenizer(spatial_spectral_features)
        
        # Classify tokens
        # Input: (B, H*W, 2*token_dim) -> Output: (B, num_classes)
        output = self.classifier(tokens)
        
        return output

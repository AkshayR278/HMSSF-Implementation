import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.transformer import HSITransformer

if __name__ == '__main__':
    # Create model instance
    model = HSITransformer(
        in_channels=30,  # PCA components
        num_classes=16,  # Number of classes in Indian Pines
        token_dim=64,
        patch_size=7,
        num_transformer_layers=6,
        num_heads=8
    )
    print("Model created successfully!") 
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.transformer import HSITransformer
from train import load_dataset, apply_pca

def predict_full_image(model, data, patch_size, device):
    """Predict labels for the entire image"""
    h, w, c = data.shape
    pad = patch_size // 2
    padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Initialize prediction map
    pred_map = np.zeros((h, w), dtype=np.int32)
    
    # Process patches
    for i in tqdm(range(h), desc='Processing rows'):
        for j in range(w):
            # Extract patch
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            patch = torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(patch)
                _, predicted = torch.max(output.data, 1)
                pred_map[i, j] = predicted.item() + 1  # Convert back to 1-based indexing
    
    return pred_map

def plot_classification_map(pred_map, dataset_name):
    """Plot and save classification map"""
    plt.figure(figsize=(10, 10))
    plt.imshow(pred_map, cmap='tab20')
    plt.colorbar(label='Class')
    plt.title(f'Classification Map - {dataset_name}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/classification_map_{dataset_name}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run inference on full image')
    parser.add_argument('--dataset', type=str, default='indian_pines',
                      choices=['indian_pines', 'pavia_university'],
                      help='Dataset to use')
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--pca_components', type=int, default=30)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    print(f'Loading {args.dataset} dataset...')
    data, labels = load_dataset(args.dataset)
    
    print('Applying PCA...')
    data_pca, pca, scaler = apply_pca(data, args.pca_components)
    
    # Load model
    num_classes = len(np.unique(labels)) - 1  # Exclude background
    model = HSITransformer(
        in_channels=args.pca_components,
        num_classes=num_classes,
        token_dim=64,
        patch_size=args.patch_size
    ).to(device)
    
    model.load_state_dict(torch.load('outputs/model_weights.pt'))
    model.eval()
    
    # Run inference
    print('Running inference on full image...')
    pred_map = predict_full_image(model, data_pca, args.patch_size, device)
    
    # Plot and save results
    print('Generating classification map...')
    plot_classification_map(pred_map, args.dataset)
    
    print('Inference completed!')

if __name__ == '__main__':
    main() 
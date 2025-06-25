import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.decomposition
import sklearn.preprocessing
import importlib

from models.transformer import HSITransformer
from train import load_dataset, apply_pca, extract_patches

# Allowlist required classes for torch.load (PyTorch 2.6+)
multiarray = importlib.import_module("numpy.core.multiarray")
torch.serialization.add_safe_globals([
    sklearn.decomposition._pca.PCA,
    sklearn.preprocessing._data.StandardScaler,
    multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.float32().dtype.__class__,
    multiarray.scalar,
    np.int64().dtype.__class__,
    np.float64().dtype.__class__
])

def compute_metrics(y_true, y_pred):
    """Compute OA, AA, and Kappa"""
    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred)
    
    # Average Accuracy
    cm = confusion_matrix(y_true, y_pred)
    aa = np.mean(cm.diagonal() / cm.sum(axis=1))
    
    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Class-wise accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return oa, aa, kappa, class_acc, cm

def plot_confusion_matrix(cm, class_names, dataset_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix_{dataset_name}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test HMSSF model')
    parser.add_argument('--dataset', type=str, default='indian_pines',
                      choices=['indian_pines', 'pavia_university', 'salinas_a', 'ksc'],
                      help='Dataset to use (indian_pines, pavia_university, salinas_a, ksc)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--pca_components', type=int, default=30)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model and preprocessing objects')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    print(f'Loading {args.dataset} dataset...')
    data, labels = load_dataset(args.dataset)
    
    print('Applying PCA...')
    # Load PCA and scaler from model_dir if available
    preproc_path = os.path.join(args.model_dir, 'preprocessing.pt')
    if os.path.exists(preproc_path):
        with torch.serialization.safe_globals([
            sklearn.decomposition._pca.PCA,
            sklearn.preprocessing._data.StandardScaler
        ]):
            preproc = torch.load(preproc_path)
        pca = preproc['pca']
        scaler = preproc['scaler']
        # Ensure the spectral dimension is last (H, W, C) where C = 200
        if data.shape[2] != scaler.n_features_in_:
            # Try to transpose if needed
            if data.shape[0] == scaler.n_features_in_:
                data = data.transpose(1, 2, 0)
            elif data.shape[1] == scaler.n_features_in_:
                data = data.transpose(0, 2, 1)
            elif data.shape[2] != scaler.n_features_in_:
                raise ValueError(f"Cannot match data shape {data.shape} to expected features {scaler.n_features_in_}")

        h, w, c = data.shape
        data_2d = data.reshape(-1, c)
        print(f"Shape of loaded data (after possible transpose): {data.shape}")
        print(f"Shape of data_2d before scaling: {data_2d.shape}")
        print(f"Scaler expects n_features_in_: {scaler.n_features_in_}")
        data_scaled = scaler.transform(data_2d)
        data_pca = pca.transform(data_scaled)
        data_pca = data_pca.reshape(h, w, args.pca_components)
    else:
        data_pca, pca, scaler = apply_pca(data, args.pca_components)
    
    print('Extracting patches...')
    patches, patch_labels = extract_patches(data_pca, labels, args.patch_size)
    
    # Create test dataset
    test_dataset = TensorDataset(
        torch.FloatTensor(patches).permute(0, 3, 1, 2),
        torch.LongTensor(patch_labels) - 1  # Convert to 0-based indexing
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Load model
    num_classes = len(np.unique(labels)) - 1  # Exclude background
    model = HSITransformer(
        in_channels=args.pca_components,
        num_classes=num_classes,
        token_dim=64,
        patch_size=args.patch_size
    ).to(device)
    
    model_path = os.path.join(args.model_dir, 'model_weights.pt')
    if not os.path.exists(model_path):
        # Try without subdirectory
        alt_model_path = os.path.join(os.path.dirname(args.model_dir), 'model_weights.pt')
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise FileNotFoundError(f"Model weights not found in {args.model_dir} or its parent directory.")

    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Test model
    print('Running inference...')
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            if outputs.ndim == 3:  # (B, N, C)
                outputs = outputs.mean(dim=1)  # (B, C)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Compute metrics
    all_labels = np.array(all_labels).reshape(-1)
    all_preds = np.array(all_preds).reshape(-1)
    oa, aa, kappa, class_acc, cm = compute_metrics(all_labels, all_preds)
    
    # Print results
    print(f'\nTest Results for {args.dataset}:')
    print(f'Overall Accuracy: {oa:.4f}')
    print(f'Average Accuracy: {aa:.4f}')
    print(f'Kappa: {kappa:.4f}')
    print('\nClass-wise Accuracy:')
    for i, acc in enumerate(class_acc):
        print(f'Class {i+1}: {acc:.4f}')
    
    # Save metrics
    metrics = {
        'Metric': ['OA', 'AA', 'Kappa'],
        'Value': [oa, aa, kappa]
    }
    # Add class-wise accuracy
    for i, acc in enumerate(class_acc):
        metrics['Metric'].append(f'Class {i+1} Accuracy')
        metrics['Value'].append(acc)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'outputs/test_metrics_{args.dataset}.csv', index=False)
    
    # Plot confusion matrix
    class_names = [f'Class {i+1}' for i in range(num_classes)]
    plot_confusion_matrix(cm, class_names, args.dataset)
    
    print('Testing completed!')

    # Save model weights
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main() 
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat

from models.transformer import HSITransformer

class DummyLogger:
    def info(self, msg): print(msg)
    def warning(self, msg): print(msg)
    def error(self, msg): print(msg)

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_name):
    """Load dataset from .mat file"""
    data_dir = os.path.join('data')
    
    if dataset_name == 'indian_pines':
        # Load the corrected spectral data
        data_path = os.path.join(data_dir, 'Indian_pines_corrected.mat')
        gt_path = os.path.join(data_dir, 'Indian_pines_gt.mat')
        
        data = loadmat(data_path)['indian_pines_corrected']  # Hyperspectral data
        labels = loadmat(gt_path)['indian_pines_gt']  # Ground truth labels
        
    elif dataset_name == 'pavia_university':
        data_path = os.path.join(data_dir, 'Pavia.mat')
        gt_path = os.path.join(data_dir, 'Pavia_gt.mat')
        
        data = loadmat(data_path)['pavia']  # Hyperspectral data
        labels = loadmat(gt_path)['pavia_gt']  # Ground truth labels
        
    elif dataset_name == 'salinas_a':
        data_path = os.path.join(data_dir, 'SalinasA.mat')
        gt_path = os.path.join(data_dir, 'SalinasA_gt.mat')
        
        data = loadmat(data_path)['salinasA']  # Hyperspectral data
        labels = loadmat(gt_path)['salinasA_gt']  # Ground truth labels
        
    elif dataset_name == 'ksc':
        data_path = os.path.join(data_dir, 'KSC.mat')
        gt_path = os.path.join(data_dir, 'KSC_gt.mat')
        
        data = loadmat(data_path)['KSC']  # Hyperspectral data
        labels = loadmat(gt_path)['KSC_gt']  # Ground truth labels
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Convert data to float32 for memory efficiency
    data = data.astype(np.float32)
    
    # Transpose data from (H, W, C) to (H, W, C) if needed
    if len(data.shape) == 3 and data.shape[0] < data.shape[2]:
        data = data.transpose(1, 2, 0)
    
    return data, labels

def extract_patches(data, labels, patch_size=15):
    """Extract overlapping patches from the image
    
    Args:
        data: Hyperspectral data cube of shape (H, W, C)
        labels: Ground truth labels of shape (H, W)
        patch_size: Size of patches to extract (must be odd)
        
    Returns:
        patches: Array of patches of shape (N, patch_size, patch_size, C)
        patch_labels: Array of labels for each patch
    """
    # Ensure data is in correct shape (H, W, C)
    if len(data.shape) != 3:
        raise ValueError(f"Data must be 3D (H, W, C), got shape {data.shape}")
      # If data dimensions don't match labels, try to fix the format
    if data.shape[0] != labels.shape[0] or data.shape[1] != labels.shape[1]:
        print(f"Data shape {data.shape} doesn't match labels shape {labels.shape}, attempting to fix...")
        
        # Find the spectral dimension (should be 30 after PCA)
        spec_dim = min(data.shape)  # After PCA, spectral dimension should be smallest
        
        # Get current order of dimensions
        dims = list(range(len(data.shape)))
        spec_dim_idx = data.shape.index(spec_dim)
        
        # Move spectral dimension to last position
        dims.remove(spec_dim_idx)
        dims.append(spec_dim_idx)
        
        # Transpose data to (H, W, C) format
        data = data.transpose(*dims)
        print(f"Transposed data to shape: {data.shape}")
    
    h, w, c = data.shape
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Ensure labels match spatial dimensions of data
    if labels.shape != (h, w):
        raise ValueError(f"Labels shape {labels.shape} doesn't match data spatial dimensions ({h}, {w})")
    
    patches = []
    patch_labels = []
    
    pad = patch_size // 2
    padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_labels = np.pad(labels, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    # Extract patches only for labeled pixels (non-zero labels)
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            if padded_labels[i, j] != 0:  # Skip background
                patch = padded_data[i-pad:i+pad+1, j-pad:j+pad+1, :]
                if patch.shape[:2] == (patch_size, patch_size):  # Verify patch size
                    patches.append(patch)
                    patch_labels.append(padded_labels[i, j])
    
    if len(patches) == 0:
        raise ValueError("No valid patches found. Check if labels contain non-zero values.")
    
    patches = np.array(patches)
    patch_labels = np.array(patch_labels)
    
    print(f"Extracted {len(patches)} patches of shape {patches.shape}")
    print(f"Label range: {patch_labels.min()}-{patch_labels.max()}")
    
    return patches, patch_labels

def apply_pca(data, n_components=30):
    """Apply PCA to reduce spectral dimensions
    
    Args:
        data: Input hyperspectral data
        n_components: Number of PCA components
        
    Returns:
        data_pca: PCA transformed data of shape (H, W, n_components)
        pca: Fitted PCA object
        scaler: Fitted StandardScaler object
    """
    print(f"Input data shape: {data.shape}")
    
    # Find the spectral dimension (should be largest for Indian Pines)
    if len(data.shape) != 3:
        raise ValueError(f"Data must be 3D, got shape {data.shape}")
        
    # Identify spatial dimensions from shape
    h, w = None, None
    c = max(data.shape)  # Spectral dimension should be largest
    spec_dim = data.shape.index(c)
    
    if spec_dim == 0:
        h, w = data.shape[1:]
    elif spec_dim == 2:
        h, w = data.shape[:2]
    else:
        h, w = (data.shape[0], data.shape[2])
    
    # Reshape to 2D array (pixels x bands)
    if spec_dim == 2:
        # If bands are last, simple reshape
        data_2d = data.reshape(-1, c)
    else:
        # Otherwise, transpose first
        perm = list(range(len(data.shape)))
        perm.remove(spec_dim)
        perm.append(spec_dim)
        data_2d = data.transpose(*perm).reshape(-1, c)
    
    print(f"Reshaped data for PCA: {data_2d.shape}")
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_2d)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Reshape back to 3D (H, W, n_components)
    data_pca = data_pca.reshape(h, w, n_components)
    print(f"Output data shape: {data_pca.shape}")
    
    return data_pca, pca, scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, logger, output_dir, num_classes):
    """Train the model"""
    if logger is None:
        logger = DummyLogger()
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    val_aas = []  # Average Accuracy
    val_kappas = []  # Kappa coefficient
    
    # Validate input shapes
    sample_batch = next(iter(train_loader))
    batch_x, batch_y = sample_batch
    logger.info(f"Training batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
    
    if len(batch_x.shape) != 4:
        raise ValueError(f"Expected 4D input tensor (B,C,H,W), got shape {batch_x.shape}")
    
    # Reset train_loader iterator
    train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # Shape: [B, S, C]
            
            # Average over sequence dimension
            outputs = outputs.mean(dim=1)  # Shape: [B, C]
            
            # Print shapes for debugging
            if epoch == 0 and len(train_preds) == 0:
                logger.info(f"Model output shape (before mean): {outputs.shape}")
                logger.info(f"Model output shape (after mean): {outputs.shape}")
                logger.info(f"Target shape: {batch_y.shape}")
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        # Calculate training metrics
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                outputs = outputs.mean(dim=1)  # Average over sequence dimension
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_kappa = cohen_kappa_score(val_labels, val_preds)
        
        # Calculate class-wise accuracy (AA)
        conf_matrix = np.zeros((num_classes, num_classes))
        for t, p in zip(val_labels, val_preds):
            conf_matrix[t][p] += 1
        class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        val_aa = np.mean(class_acc)
        
        # Store metrics
        val_accs.append(val_acc)
        val_aas.append(val_aa)
        val_kappas.append(val_kappa)
        
        # Log metrics
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
        logger.info(f'Validation - OA: {val_acc:.4f}, AA: {val_aa:.4f}, Kappa: {val_kappa:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, 'model_weights.pt')
            torch.save(model.state_dict(), model_path)
            logger.info('Best model saved!')
        
        # Save current metrics
        metrics_df = pd.DataFrame({
            'epoch': range(1, epoch + 2),
            'train_loss': train_losses,
            'val_oa': val_accs,
            'val_aa': val_aas,
            'val_kappa': val_kappas
        })
        metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    
    return {
        'train_losses': train_losses,
        'val_accs': val_accs,
        'val_aas': val_aas,
        'val_kappas': val_kappas,
        'best_val_acc': best_val_acc
    }

def setup_output_dir(dataset_name):
    """Create output directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('outputs', f'{dataset_name}_{timestamp}')
    dirs = {
        'weights': os.path.join(base_dir, 'weights'),
        'logs': os.path.join(base_dir, 'logs'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'visualization': os.path.join(base_dir, 'visualization')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def save_training_curves(metrics_dict, save_dir):
    """Save training curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics_dict['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(metrics_dict['val_accs'], label='OA')
    plt.plot(metrics_dict['val_aas'], label='AA')
    plt.title('Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Kappa
    plt.subplot(1, 3, 3)
    plt.plot(metrics_dict['val_kappas'])
    plt.title('Validation Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train HMSSF model')
    parser.add_argument('--dataset', type=str, default='indian_pines',
                      choices=['indian_pines', 'pavia_university', 'salinas_a', 'ksc'],
                      help='Dataset to use (indian_pines, pavia_university, salinas_a, ksc)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--pca_components', type=int, default=30)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model and preprocessing objects')
    args = parser.parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dir(args.dataset)
    
    # Setup logging
    logger = setup_logging(output_dirs['logs'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    pd.Series(config).to_json(os.path.join(output_dirs['logs'], 'config.json'))
      # Load and preprocess data
    logger.info(f'Loading {args.dataset} dataset...')
    data, labels = load_dataset(args.dataset)
    logger.info(f'Raw data shape: {data.shape}, Labels shape: {labels.shape}')
    logger.info(f'Label range: {labels.min()}-{labels.max()}')
    logger.info(f'Unique labels: {np.unique(labels)}')
    
    logger.info('Applying PCA...')
    data_pca, pca, scaler = apply_pca(data, args.pca_components)
    logger.info(f'Data shape after PCA: {data_pca.shape}')
    
    logger.info('Extracting patches...')
    patches, patch_labels = extract_patches(data_pca, labels, args.patch_size)
    patches = np.array(patches)
    patch_labels = np.array(patch_labels)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        patches, patch_labels, test_size=0.2, random_state=42, stratify=patch_labels
    )
      # Create data loaders with correct tensor format (B, C, H, W)    print(f"Training data shape before loader: {X_train.shape}")
    
    # Convert to torch tensors with correct shape (B, C, H, W)
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Input should be (N, H, W, C), then we'll permute to (N, C, H, W)
    if len(X_train_tensor.shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape {X_train_tensor.shape}")
    
    # Permute to (N, C, H, W) format
    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
    
    print(f"Training tensor shape after permute: {X_train_tensor.shape}")
      # Create class mapping to handle gaps in label indices
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels > 0]  # Exclude background
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_labels)}
    logger.info(f"Label mapping: {label_map}")
    
    # Map the labels to consecutive indices starting from 0
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_test_mapped = np.array([label_map[y] for y in y_test])
    
    train_dataset = TensorDataset(
        X_train_tensor,  # Already in (N, C, H, W) format
        torch.LongTensor(y_train_mapped)
    )
    test_dataset = TensorDataset(
        X_test_tensor,  # Already in (N, C, H, W) format
        torch.LongTensor(y_test_mapped)
    )
    
    sample_input = next(iter(DataLoader(train_dataset, batch_size=args.batch_size)))[0]
    print(f"Sample batch shape: {sample_input.shape}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
      # Initialize model    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels > 0]  # Exclude background
    num_classes = len(valid_labels)  # This will match our mapped labels
    logger.info(f"Original labels in dataset: {unique_labels}")
    logger.info(f"Valid labels (excluding background): {valid_labels}")
    logger.info(f"Number of classes: {num_classes}")
    
    model = HSITransformer(
        in_channels=args.pca_components,
        num_classes=num_classes,
        token_dim=64,
        patch_size=args.patch_size
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Save preprocessing objects
    preproc_path = os.path.join(output_dirs['weights'], 'preprocessing.pt')
    torch.save({
        'pca': pca,
        'scaler': scaler,
    }, preproc_path)
    
    # Train model
    logger.info('Starting training...')
    metrics = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        args.num_epochs, device, logger, output_dirs['weights'],
        num_classes=num_classes
    )
    
    # Save final metrics
    metrics_df = pd.DataFrame({
        'epoch': range(1, args.num_epochs + 1),
        'train_loss': metrics['train_losses'],
        'val_oa': metrics['val_accs'],
        'val_aa': metrics['val_aas'],
        'val_kappa': metrics['val_kappas']
    })
    metrics_df.to_csv(os.path.join(output_dirs['metrics'], 'training_metrics.csv'), index=False)
    
    # Save training curves
    save_training_curves(metrics, output_dirs['visualization'])
    
    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {metrics["best_val_acc"]:.4f}')

if __name__ == '__main__':
    main()

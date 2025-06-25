import numpy as np
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import os

class HSIPatchExtractor:
    def __init__(self, patch_size: int = 15, stride: int = 1):
        """
        Initialize HSI Patch Extractor
        Args:
            patch_size: Size of the patch (patch_size x patch_size)
            stride: Stride for patch extraction
        """
        self.patch_size = patch_size
        self.stride = stride
        self.padding = patch_size // 2
        
    def extract_patches(self, data: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Extract patches from HSI data
        Args:
            data: HSI data (H, W, C)
            gt: Ground truth labels (H, W)
        Returns:
            patches: Extracted patches (N, patch_size, patch_size, C)
            labels: Patch labels (N,)
            positions: List of (x, y) positions for each patch
        """
        h, w, c = data.shape
        patches = []
        labels = []
        positions = []
        
        # Pad the data
        data_padded = np.pad(data, ((self.padding, self.padding), 
                                  (self.padding, self.padding), 
                                  (0, 0)), mode='reflect')
        gt_padded = np.pad(gt, ((self.padding, self.padding), 
                               (self.padding, self.padding)), mode='constant', 
                          constant_values=0)
        
        # Extract patches
        for i in range(0, h, self.stride):
            for j in range(0, w, self.stride):
                # Skip if the center pixel is background (0)
                if gt[i, j] == 0:
                    continue
                    
                # Extract patch
                patch = data_padded[i:i+self.patch_size, j:j+self.patch_size, :]
                label = gt[i, j]
                
                patches.append(patch)
                labels.append(label)
                positions.append((i, j))
        
        return np.array(patches), np.array(labels), positions
    
    def save_patches(self, patches: np.ndarray, labels: np.ndarray, 
                    positions: List[Tuple[int, int]], save_dir: str):
        """
        Save extracted patches
        Args:
            patches: Extracted patches
            labels: Patch labels
            positions: Patch positions
            save_dir: Directory to save patches
        """
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'patches.npy'), patches)
        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        np.save(os.path.join(save_dir, 'positions.npy'), positions)

class HSIDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        """
        Initialize HSI Dataset
        Args:
            patches: Extracted patches (N, patch_size, patch_size, C)
            labels: Patch labels (N,)
        """
        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to (C, H, W) format for PyTorch
        patch = self.patches[idx].permute(2, 0, 1)
        label = self.labels[idx]
        return patch, label

def create_dataloaders(patches: np.ndarray, labels: np.ndarray, 
                      batch_size: int = 32, train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    Args:
        patches: Extracted patches
        labels: Patch labels
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """
    # Create dataset
    dataset = HSIDataset(patches, labels)
    
    # Split dataset
    n_samples = len(dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create dataloaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices.tolist()),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def extract_and_save_patches(data_path: str, gt_path: str, patch_size: int = 15,
                           stride: int = 1, save_dir: str = 'datasets/patches'):
    """
    Extract and save patches from HSI data
    Args:
        data_path: Path to preprocessed HSI data
        gt_path: Path to ground truth
        patch_size: Size of the patch
        stride: Stride for patch extraction
        save_dir: Directory to save patches
    """
    # Load data
    data = np.load(data_path)
    gt = np.load(gt_path)
    
    # Create patch extractor
    extractor = HSIPatchExtractor(patch_size=patch_size, stride=stride)
    
    # Extract patches
    patches, labels, positions = extractor.extract_patches(data, gt)
    
    # Save patches
    extractor.save_patches(patches, labels, positions, save_dir)
    
    print(f"Extracted {len(patches)} patches")
    print(f"Unique labels: {np.unique(labels)}")
    
    return patches, labels, positions

if __name__ == '__main__':
    # Example usage
    data_path = 'datasets/Indian_pines_corrected.mat'  # Original .mat file
    gt_path = 'datasets/Indian_pines_gt.mat'  # Ground truth .mat file
    
    # Check if preprocessed files exist
    pca_path = 'datasets/preprocessed/Indian_pines_pca.npy'
    gt_npy_path = 'datasets/preprocessed/Indian_pines_gt.npy'
    
    if not (os.path.exists(pca_path) and os.path.exists(gt_npy_path)):
        print("Preprocessed files not found. Running PCA preprocessing...")
        # First run PCA preprocessing
        from .pca import preprocess_hsi
        data_pca, gt = preprocess_hsi(data_path, gt_path)
        print("PCA preprocessing complete!")
    
    # Then extract patches
    print("Extracting patches...")
    patches, labels, positions = extract_and_save_patches(pca_path, gt_npy_path) 
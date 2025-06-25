import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, Optional
import h5py
import scipy.io as sio

class HSIPreprocessor:
    def __init__(self, n_components: int = 30):
        """
        Initialize HSI Preprocessor
        Args:
            n_components: Number of PCA components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load HSI data and ground truth
        Args:
            data_path: Path to HSI data (.mat file)
            gt_path: Path to ground truth (.mat file)
        Returns:
            data: HSI data (H, W, C)
            gt: Ground truth labels (H, W)
        """
        # Load data
        if data_path.endswith('.mat'):
            data = sio.loadmat(data_path)
            gt = sio.loadmat(gt_path)
            
            # Get the key that contains the data matrix
            data_key = [k for k in data.keys() if not k.startswith('__')][0]
            gt_key = [k for k in gt.keys() if not k.startswith('__')][0]
            
            data = data[data_key]
            gt = gt[gt_key]
        else:
            raise ValueError("Only .mat files are supported")
            
        return data, gt
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess HSI data using PCA
        Args:
            data: HSI data (H, W, C)
        Returns:
            data_pca: Reduced data (H, W, n_components)
        """
        # Reshape data for PCA
        h, w, c = data.shape
        data_2d = data.reshape(-1, c)
        
        # Standardize the data
        data_2d_scaled = self.scaler.fit_transform(data_2d)
        
        # Apply PCA
        data_2d_pca = self.pca.fit_transform(data_2d_scaled)
        
        # Reshape back to image format
        data_pca = data_2d_pca.reshape(h, w, self.n_components)
        
        return data_pca
    
    def save_preprocessed_data(self, data: np.ndarray, save_path: str):
        """
        Save preprocessed data
        Args:
            data: Preprocessed data
            save_path: Path to save the data
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data)
        
    def load_preprocessed_data(self, load_path: str) -> np.ndarray:
        """
        Load preprocessed data
        Args:
            load_path: Path to load the data
        Returns:
            data: Preprocessed data
        """
        return np.load(load_path)
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Get explained variance ratio of PCA components
        Returns:
            explained_variance: Explained variance ratio
        """
        return self.pca.explained_variance_ratio_
    
    def get_components(self) -> np.ndarray:
        """
        Get PCA components
        Returns:
            components: PCA components
        """
        return self.pca.components_

def preprocess_hsi(data_path: str, gt_path: str, n_components: int = 30, 
                  save_dir: str = 'datasets/preprocessed') -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess HSI data and save results
    Args:
        data_path: Path to HSI data
        gt_path: Path to ground truth
        n_components: Number of PCA components
        save_dir: Directory to save preprocessed data
    Returns:
        data_pca: Preprocessed data
        gt: Ground truth
    """
    # Create preprocessor
    preprocessor = HSIPreprocessor(n_components=n_components)
    
    # Load data
    data, gt = preprocessor.load_data(data_path, gt_path)
    
    # Preprocess data
    data_pca = preprocessor.preprocess(data)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save preprocessed data
    dataset_name = 'Indian_pines'  # Hardcode the name since we know it
    np.save(os.path.join(save_dir, f'{dataset_name}_pca.npy'), data_pca)
    np.save(os.path.join(save_dir, f'{dataset_name}_gt.npy'), gt)
    
    # Print explained variance
    explained_variance = preprocessor.get_explained_variance()
    print(f"Explained variance ratio: {explained_variance.sum():.4f}")
    print(f"Number of components: {n_components}")
    
    return data_pca, gt

if __name__ == '__main__':
    # Example usage
    data_path = 'datasets/Indian_pines_corrected.mat'
    gt_path = 'datasets/Indian_pines_gt.mat'
    data_pca, gt = preprocess_hsi(data_path, gt_path) 
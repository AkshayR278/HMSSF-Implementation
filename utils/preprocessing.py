import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch

def load_data(data_path, gt_path):
    """
    Load hyperspectral image and ground truth data
    """
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    # Get the key that contains the data matrix
    data_key = [k for k in data.keys() if not k.startswith('__')][0]
    gt_key = [k for k in gt.keys() if not k.startswith('__')][0]
    
    return data[data_key], gt[gt_key]

def apply_pca(data, n_components=30):
    """
    Apply PCA to reduce the spectral dimensionality
    Args:
        data: Input hyperspectral image (H x W x C)
        n_components: Number of principal components to keep
    Returns:
        Reduced data (H x W x n_components)
    """
    h, w, c = data.shape
    data_2d = data.reshape(-1, c)
    
    # Standardize the data
    scaler = StandardScaler()
    data_2d_scaled = scaler.fit_transform(data_2d)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    data_2d_pca = pca.fit_transform(data_2d_scaled)
    
    # Reshape back to image format
    data_pca = data_2d_pca.reshape(h, w, n_components)
    
    return data_pca

def extract_patches(data, patch_size=7, stride=1):
    """
    Extract patches from the hyperspectral image
    Args:
        data: Input hyperspectral image (H x W x C)
        patch_size: Size of the patches to extract
        stride: Stride for patch extraction
    Returns:
        patches: Extracted patches (N x C x patch_size x patch_size)
    """
    h, w, c = data.shape
    pad_size = patch_size // 2
    
    # Pad the input data
    padded_data = np.pad(data, ((pad_size, pad_size), 
                               (pad_size, pad_size), 
                               (0, 0)), 
                        mode='reflect')
    
    patches = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    # Convert to torch tensor and reshape to (N, C, H, W)
    patches = np.array(patches)
    patches = np.transpose(patches, (0, 3, 1, 2))
    patches = torch.from_numpy(patches).float()
    
    return patches

def create_pixel_patches(data, patch_size=7):
    """
    Create patches centered around each pixel
    Args:
        data: Input hyperspectral image (H x W x C)
        patch_size: Size of the patches to extract
    Returns:
        patches: Dictionary containing patches for each pixel position
    """
    h, w, c = data.shape
    pad_size = patch_size // 2
    
    # Pad the input data
    padded_data = np.pad(data, ((pad_size, pad_size), 
                               (pad_size, pad_size), 
                               (0, 0)), 
                        mode='reflect')
    
    pixel_patches = {}
    for i in range(h):
        for j in range(w):
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            pixel_patches[(i, j)] = patch
    
    return pixel_patches

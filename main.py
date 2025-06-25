import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.transformer import HSITransformer
from utils.preprocessing import load_data, apply_pca
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class HSIDataset(Dataset):
    def __init__(self, data, labels, patch_size=7):
        """
        Dataset class for hyperspectral images
        Args:
            data: Input hyperspectral image (H x W x C)
            labels: Ground truth labels (H x W)
            patch_size: Size of patches to extract
        """
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        
        # Get valid indices (non-zero labels)
        self.valid_indices = np.argwhere(labels > 0)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get coordinates
        i, j = self.valid_indices[idx]
        
        # Extract patch
        pad = self.patch_size // 2
        padded_data = np.pad(self.data, 
                            ((pad, pad), (pad, pad), (0, 0)),
                            mode='reflect')
        patch = padded_data[i:i+self.patch_size,
                           j:j+self.patch_size, :]
        
        # Convert to tensor
        patch = torch.from_numpy(patch).float()
        patch = patch.permute(2, 0, 1)  # (C, H, W)
        
        # Get label (subtract 1 as labels are 1-based)
        label = self.labels[i, j] - 1
        label = torch.tensor(label, dtype=torch.long)
        
        return patch, label

def train(model, train_loader, val_loader, criterion, optimizer, 
          num_epochs=100, device: torch.device = torch.device('cuda')):
    """
    Training function
    """
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'Indian_pines_corrected.mat')
    gt_path = os.path.join(project_root, 'data', 'Indian_pines_gt.mat')
    
    data, gt = load_data(data_path, gt_path)
    
    # Apply PCA
    data_pca = apply_pca(data, n_components=30)
    
    # Create datasets
    dataset = HSIDataset(data_pca, gt)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, 
                          shuffle=False, num_workers=4)
    
    # Create model
    model = HSITransformer(in_channels=30,  # PCA components
                          num_classes=16,    # Number of classes in Indian Pines
                          token_dim=64,
                          patch_size=7,
                          num_transformer_layers=6,
                          num_heads=8)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, 
                           weight_decay=0.01)
    
    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, 
          num_epochs=100, device=device)

if __name__ == '__main__':
    main()

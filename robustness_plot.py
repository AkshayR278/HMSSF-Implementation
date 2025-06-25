import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from models.transformer import HSITransformer
from train import load_dataset, apply_pca, extract_patches, train_model

def evaluate_with_sample_size(data, labels, sample_sizes, args, device, output_dir):
    """Train and evaluate model with different sample sizes"""
    results = []
    
    num_classes = len(np.unique(labels)) - 1
    for size in sample_sizes:
        print(f'\nTraining with {size}% of samples...')
        
        # Create output subdirectory for this run
        run_dir = os.path.join(output_dir, f'sample_size_{size}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Extract patches
        patches, patch_labels = extract_patches(data, labels, args.patch_size)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            patches, patch_labels,
            test_size=0.2,
            random_state=42,
            stratify=patch_labels
        )
        
        # Reduce training set size
        n_samples = int(len(X_train) * size / 100)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).permute(0, 3, 1, 2),
            torch.LongTensor(y_train) - 1
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test).permute(0, 3, 1, 2),
            torch.LongTensor(y_test) - 1
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Initialize model
        model = HSITransformer(
            in_channels=args.pca_components,
            num_classes=num_classes,
            token_dim=64,
            patch_size=args.patch_size
        ).to(device)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Create logger for this run
        # run_logger = setup_logging(os.path.join(run_dir, 'logs'))
        
        # Train model
        metrics = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            args.num_epochs, device, None, run_dir, num_classes
        )
        
        # Save training curves for this run
        # save_training_curves(metrics, run_dir)
        
        # Record metrics
        results.append({
            'sample_size': size,
            'best_accuracy': metrics['best_val_acc'],
            'final_accuracy': metrics['val_accs'][-1],
            'best_aa': max(metrics['val_aas']),
            'best_kappa': max(metrics['val_kappas'])
        })
        
        print(f'Results for {size}% samples:')
        print(f'Best accuracy: {metrics["best_val_acc"]:.4f}')
        print(f'Best AA: {max(metrics["val_aas"]):.4f}')
        print(f'Best Kappa: {max(metrics["val_kappas"]):.4f}')
    
    return pd.DataFrame(results)

def plot_robustness_curves(results, dataset_name, save_path):
    """Plot and save robustness curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(results['sample_size'], results['best_accuracy'], 'o-', linewidth=2, label='Best')
    plt.plot(results['sample_size'], results['final_accuracy'], 's--', linewidth=2, label='Final')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy vs Sample Size')
    plt.legend()
    plt.grid(True)
    
    # Plot AA
    plt.subplot(1, 3, 2)
    plt.plot(results['sample_size'], results['best_aa'], 'o-', linewidth=2)
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Sample Size')
    plt.grid(True)
    
    # Plot Kappa
    plt.subplot(1, 3, 3)
    plt.plot(results['sample_size'], results['best_kappa'], 'o-', linewidth=2)
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Kappa Coefficient')
    plt.title('Kappa vs Sample Size')
    plt.grid(True)
    
    plt.suptitle(f'Model Robustness Analysis - {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--dataset', type=str, default='indian_pines',
                      choices=['indian_pines', 'pavia_university'],
                      help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--pca_components', type=int, default=30)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', f'robustness_{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    pd.Series(config).to_json(os.path.join(output_dir, 'config.json'))
    
    # Load and preprocess data
    print(f'Loading {args.dataset} dataset...')
    data, labels = load_dataset(args.dataset)
    
    print('Applying PCA...')
    data_pca, pca, scaler = apply_pca(data, args.pca_components)
    
    # Evaluate with different sample sizes
    sample_sizes = [1, 2, 5, 10, 20, 50, 100]
    results = evaluate_with_sample_size(data_pca, labels, sample_sizes, args, device, output_dir)
    
    # Save results
    results.to_csv(os.path.join(output_dir, 'robustness_results.csv'), index=False)
    
    # Plot results
    plot_robustness_curves(results, args.dataset, os.path.join(output_dir, 'robustness_curves.png'))
    
    print(f'Robustness analysis completed! Results saved to {output_dir}')

if __name__ == '__main__':
    main()
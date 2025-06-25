import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
import seaborn as sns

def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calculate classification metrics
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    Returns:
        Dictionary containing metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names)
    }
    return metrics

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training curves
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_predictions(image, gt, predictions, save_path=None):
    """
    Visualize predictions
    Args:
        image: Original hyperspectral image
        gt: Ground truth labels
        predictions: Predicted labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Original image (first band)
    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original Image (First Band)')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='tab20')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Predictions
    plt.subplot(1, 3, 3)
    plt.imshow(predictions, cmap='tab20')
    plt.title('Predictions')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 
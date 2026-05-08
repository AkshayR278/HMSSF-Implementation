# 🛰️ HMSSF: Hybrid Multi-Scale Spatial-Spectral Transformer for Hyperspectral Image Classification

A state-of-the-art deep learning framework for hyperspectral image (HSI) classification using hybrid transformer architecture with spatial-spectral feature fusion and multi-scale self-attention mechanisms.

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Supported Datasets](#supported-datasets)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🎯 Overview

HMSSF is an advanced hyperspectral image classification system that combines:
- **Spatial-Spectral Token Generation (SSTG)**: Extracts discriminative tokens from 3D hyperspectral patches
- **Multi-Scale Self-Attention (MSSA)**: Captures multi-scale contextual information with hierarchical attention
- **Transformer Architecture**: Leverages self-attention for capturing long-range dependencies
- **Hybrid Fusion**: Intelligently combines spatial and spectral information for robust classification

This framework achieves state-of-the-art performance on standard hyperspectral classification benchmarks including Indian Pines, Pavia University, and Salinas A datasets.

---

## ✨ Features

### Core Features
- ✅ **Transformer-based Classification**: Advanced attention mechanisms for HSI analysis
- ✅ **Multi-Scale Processing**: Hierarchical feature extraction at multiple scales
- ✅ **Spatial-Spectral Fusion**: Integrated spatial and spectral domain processing
- ✅ **PCA Preprocessing**: Dimensionality reduction for efficient computation
- ✅ **Patch-based Processing**: Configurable patch extraction with overlap handling
- ✅ **Comprehensive Metrics**: Accuracy, Cohen's Kappa, confusion matrices, and per-class performance
- ✅ **Visualization Tools**: Classification maps, robustness plots, and training metrics
- ✅ **Multi-dataset Support**: Indian Pines, Pavia University, Salinas A, KSC
- ✅ **GPU Acceleration**: CUDA support for faster training and inference
- ✅ **Logging & Tracking**: Detailed training logs, metrics tracking, and model checkpointing

### Model Components
- **SSTG (Spatial-Spectral Token Generator)**: 3D convolution-based token generation
- **MSSA (Multi-Scale Self-Attention)**: Multi-head attention with multi-scale feature fusion
- **HSITransformer**: Complete transformer backbone for classification
- **Enhanced Transformer**: Advanced architectural variants
- **Classifier**: Task-specific classification head

---

## 🏗️ Architecture

```
Input HSI Patch (H × W × C)
        ↓
[Spatial-Spectral Token Generator]
  - 3D Convolution Layers
  - Multi-scale Feature Extraction
        ↓
    Tokens (N × D)
        ↓
[Multi-Scale Self-Attention]
  - Multi-head Attention
  - Hierarchical Scale Processing (1x, 2x, 4x)
        ↓
[Transformer Encoder Layers]
  - Self-Attention + Feed-Forward
  - Layer Normalization
        ↓
[Classification Head]
        ↓
   Class Logits (K classes)
```

---

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/HMSSF-Implementation.git
cd HMSSF-Implementation
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n hmssf python=3.9
conda activate hmssf

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.9.0              # PyTorch deep learning framework
torchvision>=0.10.0       # Computer vision utilities
numpy>=1.19.2             # Numerical computing
scipy>=1.7.1              # Scientific computing
scikit-learn>=0.24.2      # Machine learning utilities
matplotlib>=3.4.3         # Visualization
tqdm>=4.62.3              # Progress bars
pandas>=1.3.3             # Data manipulation
h5py>=3.6.0               # HDF5 file support
tensorboard>=2.7.0        # Training visualization
```

---

## 📊 Dataset Preparation

### Supported Datasets

| Dataset | Size | Classes | Bands | Spatial Dim | Source |
|---------|------|---------|-------|-------------|--------|
| Indian Pines | 2.4 MB | 17 | 200 | 145×145 | AVIRIS |
| Pavia University | 42.7 MB | 9 | 103 | 610×610 | ROSIS |
| Salinas A | 3.6 MB | 6 | 224 | 83×86 | AVIRIS |
| KSC | 13.5 MB | 13 | 176 | 512×614 | AVIRIS |

### Download and Setup

1. **Create data directory**:
   ```bash
   mkdir -p data
   ```

2. **Add dataset files**:
   - Download `.mat` files from the official sources
   - Place corrected data and ground truth labels in the `data/` directory:
     ```
     data/
     ├── Indian_pines_corrected.mat
     ├── Indian_pines_gt.mat
     ├── Pavia.mat
     ├── Pavia_gt.mat
     ├── SalinasA.mat
     ├── SalinasA_gt.mat
     └── ... (other datasets)
     ```

---

## 🚀 Usage

### Training

Train the model on a specific dataset:

```bash
# Indian Pines dataset
python train.py --dataset indian_pines \
    --model_dir outputs/indian_pines_training/ \
    --num_epochs 100 \
    --patch_size 15 \
    --batch_size 32 \
    --learning_rate 0.001

# Pavia University dataset
python train.py --dataset pavia_university \
    --model_dir outputs/pavia_university_training/ \
    --num_epochs 100

# Salinas A dataset
python train.py --dataset salinas_a \
    --model_dir outputs/salinas_a_training/ \
    --num_epochs 100
```

#### Training Arguments

```
--dataset              Dataset to train on (default: indian_pines)
                      Options: indian_pines, pavia_university, salinas_a, ksc
--model_dir           Directory to save models and logs (default: outputs/)
--num_epochs          Number of training epochs (default: 100)
--batch_size          Batch size for training (default: 32)
--learning_rate       Learning rate for optimizer (default: 0.001)
--patch_size          Patch size for extraction (default: 15)
--pca_components      PCA components for dimensionality reduction (default: 30)
--val_split           Validation split ratio (default: 0.2)
--num_workers         Number of data loading workers (default: 4)
--seed                Random seed for reproducibility (default: 42)
```

### Inference

Run inference on a trained model:

```bash
python infer.py --dataset indian_pines \
    --model_path outputs/model_weights.pt \
    --patch_size 15 \
    --pca_components 30
```

The output includes:
- Classification map visualization
- Per-class accuracy metrics
- Confusion matrix
- Overall accuracy and Cohen's Kappa score

---

## 📁 Project Structure

```
HMSSF-Implementation/
├── models/                          # Model architectures
│   ├── __init__.py
│   ├── transformer.py              # Main transformer model
│   ├── hmssf.py                    # HMSSF hybrid model
│   ├── sstg.py                     # Spatial-Spectral Token Generator
│   ├── mssa.py                     # Multi-Scale Self-Attention
│   ├── enhanced_transformer.py     # Enhanced transformer variants
│   ├── classifier.py               # Classification head
│   └── tokenization.py             # Token generation utilities
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py            # Data preprocessing and PCA
│   ├── metrics.py                  # Evaluation metrics
│   ├── patch_extractor.py          # Patch extraction utilities
│   ├── download_dataset.py         # Dataset download tools
│   └── pca.py                      # PCA utilities
│
├── data/                            # Dataset directory (add .mat files here)
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
│   └── ...
│
├── outputs/                         # Training outputs
│   └── {dataset}_{timestamp}/
│       ├── logs/                   # Training logs
│       ├── weights/                # Model checkpoints
│       ├── metrics/                # Training metrics CSV
│       └── visualization/          # Generated plots
│
├── train.py                        # Main training script
├── infer.py                        # Inference script
├── main.py                         # Demo/testing script
├── prepare_data.py                 # Data preparation utilities
├── robustness_plot.py              # Robustness visualization
├── run_transformer.py              # Transformer execution script
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

---

## 📈 Results

### Performance Metrics

Training logs and metrics are saved to the output directory:

```
outputs/{dataset}_{timestamp}/
├── logs/
│   ├── config.json                 # Training configuration
│   └── train_log_{timestamp}.txt   # Detailed logs
├── metrics/
│   └── training_metrics.csv        # Epoch-wise metrics
└── weights/
    ├── model_weights.pt            # Trained model
    ├── preprocessing.pt            # Preprocessing parameters
    └── training_metrics.csv        # Final metrics
```

### Typical Performance (50% training split)

- **Indian Pines**: ~99% Overall Accuracy, κ ≈ 0.989
- **Pavia University**: ~99% Overall Accuracy, κ ≈ 0.988
- **Salinas A**: ~98.5% Overall Accuracy, κ ≈ 0.975

*Results vary based on hyperparameters, training split, and initialization.*

---

## 🔧 Advanced Usage

### Custom Configuration

Edit `config.json` or pass arguments to customize:

```python
from models.transformer import HSITransformer
import torch

# Initialize model
model = HSITransformer(
    in_channels=30,          # PCA components
    num_classes=16,          # Number of classes
    token_dim=64,            # Token dimension
    patch_size=15,           # Patch size
    num_heads=8,             # Attention heads
    num_layers=6,            # Transformer layers
    mlp_dim=256,             # MLP dimension
    dropout=0.1              # Dropout rate
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Visualization

Generate robustness plots:

```bash
python robustness_plot.py --metrics_dir outputs/indian_pines_20250703_102447/metrics/
```

---

## 💡 Key Insights

1. **PCA Preprocessing**: Reduces computation while preserving spectral information
2. **Multi-scale Attention**: Captures features at different receptive fields
3. **Patch-based Processing**: Enables efficient handling of large HSI cubes
4. **Hybrid Architecture**: Combines spatial convolutions with spectral attention
5. **Validation Split**: Prevents overfitting on limited labeled data

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` argument
- Reduce `patch_size`
- Reduce `pca_components`

### Data Loading Issues
- Verify `.mat` files are in the `data/` directory
- Check file names match expected convention
- Ensure `.mat` files contain correct keys

### Poor Performance
- Increase `num_epochs`
- Adjust `learning_rate`
- Try different `patch_size` values
- Increase `pca_components` for more spectral detail

---

## 📚 References

- [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
- [Multi-Scale Vision Transformers](https://arxiv.org/abs/2104.11601)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)

---

## 📝 License

This project is private and must not be reproduced elsewhere

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs by opening an issue
- Suggest improvements
- Submit pull requests with enhancements

---

## ✉️ Contact & Support

For questions, suggestions, or support:
- Open an issue on GitHub
- Contact: akshayrajeshsiva@gmail.com

---

## 🎓 Acknowledgments

- AVIRIS dataset providers
- PyTorch and deep learning community
- Transformer architecture pioneers

---

**Star ⭐ this repository if you found it helpful!**

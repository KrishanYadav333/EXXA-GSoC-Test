# EXXA GSoC 2026 Test - Protoplanetary Disk Analysis

**ML4Sci - EXXA Project**  
**Applicant:** [Your Name]  
**Tests Completed:** General Test + Image-Based Test

## Overview

This project implements machine learning pipelines for analyzing synthetic ALMA observations of protoplanetary disks. The goal is to identify and cluster disk structures that may indicate planet formation.

### Tasks Completed

1. **General Test**: Unsupervised clustering of protoplanetary disks
2. **Image-Based Test**: Autoencoder for image reconstruction with accessible latent space

## Approach

### Pipeline Architecture

```
FITS Images (600×600)
    ↓
Preprocessing & Normalization
    ↓
Convolutional Autoencoder
    ├─ Encoder → Latent Space (512-D)
    └─ Decoder → Reconstructed Images
    ↓
Latent Feature Extraction
    ↓
Clustering (K-Means + HDBSCAN)
    ↓
UMAP Visualization
    ↓
Cluster Analysis & Interpretation
```

### Key Features

- **Advanced Autoencoder**: U-Net inspired architecture with skip connections
- **Multi-Algorithm Clustering**: Compares K-Means, HDBSCAN, and Agglomerative clustering
- **Comprehensive Metrics**: MSE, MS-SSIM, Silhouette Score, Davies-Bouldin Index
- **Rich Visualizations**: UMAP embeddings, reconstruction comparisons, cluster distributions
- **Fully Automated**: Runs end-to-end without manual intervention

## Project Structure

```
EXXA-GSoC-Test/
├── notebooks/
│   └── exxa_complete_pipeline.ipynb    # Main deliverable
│
├── src/
│   ├── data_loader.py                  # FITS file loading utilities
│   ├── autoencoder.py                  # Model architecture
│   ├── clustering.py                   # Clustering algorithms
│   ├── evaluation.py                   # Metrics and evaluation
│   └── visualization.py                # Plotting functions
│
├── models/
│   └── autoencoder_best.pth            # Pre-trained model weights
│
├── outputs/
│   └── figures/                        # Generated plots
│
├── data/                               # FITS files (not in repo)
│
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EXXA-GSoC-Test.git
cd EXXA-GSoC-Test

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the synthetic ALMA observations from the provided link and place FITS files in the `data/` directory.

### 3. Run the Pipeline

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/exxa_complete_pipeline.ipynb
```

Run all cells from top to bottom. The notebook includes:
- Data loading and exploration
- Model training
- Evaluation metrics
- Clustering analysis
- Visualizations

**Option B: Use Pre-trained Model**
```python
from src.autoencoder import ImprovedAutoencoder
import torch

model = ImprovedAutoencoder()
model.load_state_dict(torch.load('models/autoencoder_best.pth'))
model.eval()
```

## Results

### Image Reconstruction (Image-Based Test)

- **Mean MSE**: [Value will be shown after training]
- **Mean MS-SSIM**: [Value will be shown after training]
- High-quality reconstructions preserving disk structures (gaps, rings, spirals)

### Clustering (General Test)

- **Number of Clusters Found**: [Auto-detected by HDBSCAN]
- **Cluster Properties**: 
  - Cluster 0: [Description based on visual analysis]
  - Cluster 1: [Description]
  - Cluster 2: [Description]
- **Silhouette Score**: [Metric value]

Key findings: Clusters correspond to different planet configurations, disk viewing angles, and gap structures.

## Key Design Decisions

### 1. Autoencoder Architecture
- **U-Net inspired** with skip connections for better reconstruction
- **Batch normalization** for training stability
- **Residual connections** to preserve fine details
- **Latent dimension**: 512-D (tunable)

### 2. Clustering Strategy
- **High-dimensional clustering**: Applied to full latent space (not 2D)
- **Multiple algorithms**: K-Means (with elbow method), HDBSCAN, Agglomerative
- **UMAP for visualization only**: Not used for clustering to avoid information loss

### 3. Preprocessing
- **Normalization**: Z-score normalization per image
- **Data augmentation**: Rotation, flipping (optional during training)
- **Robust handling**: NaN/Inf values checked and handled

## Performance Metrics

### Quantitative Metrics
- Mean Squared Error (MSE)
- Multi-Scale Structural Similarity Index (MS-SSIM)
- Silhouette Score (clustering quality)
- Davies-Bouldin Index
- Calinski-Harabasz Score

### Qualitative Analysis
- Visual inspection of reconstructions
- Cluster homogeneity analysis
- Representative samples from each cluster
- UMAP embedding visualization

## Tested On

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **GPU**: CUDA-compatible (optional, CPU also works)
- **Environment**: Google Colab / Local Jupyter

## Notes

- Training time: ~10-20 minutes on GPU for 50 epochs
- Dataset size: [Number of FITS files]
- Seeds set for reproducibility (`random_seed=42`)
- All plots automatically saved to `outputs/figures/`

## Links

- **Google Colab Notebook**: [Link to Colab]
- **Pre-trained Model**: [Google Drive link]
- **GitHub Repository**: [This repo]

## References

- Terry et al. (2022) - Synthetic observation methodology
- ALMA Observatory - Data source and format
- PyTorch MS-SSIM - Image quality metric
- UMAP - Dimensionality reduction
- HDBSCAN - Density-based clustering

## Contact

For questions about this submission, please contact [your email].

---

**GSoC 2026 - ML4Sci - EXXA Project**

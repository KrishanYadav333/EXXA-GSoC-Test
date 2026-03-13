# EXXA GSoC 2026 Test - Protoplanetary Disk Analysis

**ML4Sci - EXXA Project**  
**Applicant:** Krishan Yadav  
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
git clone https://github.com/KrishanYadav333/EXXA-GSoC-Test.git
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

- **Mean MSE**: 0.3219 ± 0.0600
- **Mean MS-SSIM**: 0.9975 ± 0.0015
- High-quality reconstructions preserving disk structures (gaps, rings, spirals)
- Autoencoder successfully learns 184,832-dimensional latent representations

### Clustering (General Test)

#### K-Means Clustering (Recommended)
- **Number of Clusters**: 2 (optimized via silhouette analysis)
- **Silhouette Score**: 0.394
- **Davies-Bouldin Score**: 1.027 (lower is better)
- **Calinski-Harabasz Score**: 231.99 (higher is better)

**Cluster Properties**: 
  - **Cluster 0** (112 disks): Bright, compact disks with strong central emission and minimal gap structures. These represent younger or more massive disks with continuous brightness distributions.
  - **Cluster 1** (188 disks): Fainter, more extended disks with visible ring structures and gaps. These show evidence of planet-disk interactions and more evolved disk morphologies.

#### HDBSCAN Clustering (Alternative)
- **Number of Clusters**: 7 + 76 noise points
- **Silhouette Score**: 0.314
- Identifies more granular substructures but with lower overall cohesion

**Key Findings**: The dominant clustering pattern separates compact vs. structured disks, likely corresponding to different evolutionary stages or planet-disk interaction scenarios.

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

- Training completed successfully on 300 disk images
- Training time: ~69 minutes on NVIDIA RTX 2050 GPU (50 epochs)
- Dataset: 300 synthetic ALMA observations at 1250 μm
- Seeds set for reproducibility (`random_seed=42`)
- All plots automatically saved to `outputs/figures/`
- GPU acceleration recommended but CPU mode also supported

## Links

- **Google Colab Notebook**: https://colab.research.google.com/github/KrishanYadav333/EXXA-GSoC-Test/blob/main/notebooks/exxa_complete_pipeline_colab.ipynb
- **Pre-trained Model**: https://drive.google.com/file/d/1Hx3WM4OYKnIPyvv5Kw6mkTN2cOMWQerH/view?usp=drive_link
- **GitHub Repository**: https://github.com/KrishanYadav333/EXXA-GSoC-Test

## References

- Terry et al. (2022) - Synthetic observation methodology
- ALMA Observatory - Data source and format
- PyTorch MS-SSIM - Image quality metric
- UMAP - Dimensionality reduction
- HDBSCAN - Density-based clustering

## Contact

For questions about this submission, please contact kryshan753@gmail.com.

---

**GSoC 2026 - ML4Sci - EXXA Test**

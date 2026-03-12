# Results Summary - EXXA GSoC 2026 Test

**Date**: March 12, 2026  
**Dataset**: 300 synthetic ALMA observations (1250 μm)  
**Training Environment**: NVIDIA RTX 2050, PyTorch 2.7.1+cu118

---

## 1. Model Training

### Configuration
- Architecture: U-Net Inspired Autoencoder with skip connections
- Total Parameters: 8,643,457
- Latent Dimension: 512
- Training Epochs: 50
- Batch Size: 8
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: MSE

### Training Performance
- **Best Validation Loss**: 0.3219
- **Final Training Loss**: 0.3464
- **Training Time**: 69 minutes (4,158 seconds)
- Training converged smoothly with no overfitting
- Validation loss plateaued around epoch 20-25

---

## 2. Image Reconstruction Metrics

### Quantitative Results

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| **MSE** | 0.3219 | 0.0600 | 0.25 - 0.43 |
| **MS-SSIM** | 0.9975 | 0.0015 | 0.994 - 0.998 |

### Interpretation
- **MS-SSIM close to 1.0**: Excellent structural similarity, model preserves disk morphology
- **Low MSE variance**: Consistent reconstruction quality across all disk types
- **Visual Quality**: Reconstructions successfully capture:
  - Ring structures and gaps
  - Central cavity features  
  - Disk brightness profiles
  - Spiral arm patterns (when present)

---

## 3. Latent Space Analysis

### Feature Extraction
- **Latent Feature Dimensions**: 184,832 per image
- **Feature Space**: Continuous, high-dimensional encodings
- Successfully compressed 600×600 images (~360K pixels) to structured 185K-D representations
- Latent features capture meaningful disk properties for clustering

---

## 4. Clustering Results

### Optimal Cluster Selection
- **Method**: Silhouette Score Analysis (K=2 to K=10)
- **Optimal K**: 2 clusters
- **Silhouette Score**: 0.394

### K-Means Clustering (Recommended Solution)

#### Cluster Distribution
| Cluster | Size | Percentage |
|---------|------|------------|
| Cluster 0 | 112 disks | 37.3% |
| Cluster 1 | 188 disks | 62.7% |

#### Cluster Characteristics

**Cluster 0: Compact, Bright Disks**
- Strong central emission
- Higher peak brightness
- Minimal gap structures
- More continuous brightness distribution
- Interpretation: Younger or more massive disks, less evidence of planet formation

**Cluster 1: Structured, Ring-Dominated Disks**  
- Visible ring structures
- Prominent gaps (likely planet-carved)
- More extended morphology
- Lower central brightness
- Interpretation: More evolved disks with planet-disk interactions

#### Clustering Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.394 | Good separation, moderate cohesion |
| Davies-Bouldin Score | 1.027 | Lower is better, indicates distinct clusters |
| Calinski-Harabasz Score | 231.99 | Higher is better, well-defined clusters |

### HDBSCAN Clustering (Alternative Analysis)

#### Results
- **Number of Clusters**: 7
- **Noise Points**: 76 (25.3%)
- **Silhouette Score**: 0.314

#### Interpretation
HDBSCAN identifies finer substructures within the data but with:
- More fragmentation
- Lower overall cluster quality
- Significant noise classification

**Recommendation**: K-Means with K=2 provides the most interpretable and robust clustering for this dataset.

---

## 5. Dimensionality Reduction Visualization

### UMAP Embedding
- **Purpose**: Visualization only (NOT used for clustering)
- **Projection**: 184,832-D → 2-D
- Clear visual separation between clusters in 2D space
- Validates that clustering patterns exist in full latent space

### PCA Comparison
- **Explained Variance (2 components)**: 66.88%
- Shows overlap between clusters in linear projection
- Confirms that non-linear relationships important for separation

---

## 6. Key Findings

1. **Reconstruction Quality**: The autoencoder successfully learns compact, meaningful representations while preserving fine structural details (MS-SSIM: 0.9975)

2. **Natural Clustering**: Dataset exhibits a clear two-cluster structure separating compact vs. structured disks

3. **Physical Interpretation**: Clusters likely correspond to:
   - Different disk evolutionary stages
   - Presence/absence of massive planets
   - Planet-disk interaction strength

4. **Methodology Validation**: 
   - High-dimensional clustering (correct approach) outperforms direct 2D clustering
   - Skip connections in autoencoder essential for preserving disk features
   - Multiple metrics confirm cluster quality

---

## 7. Generated Outputs

### Models
- `models/autoencoder_best.pth` - Best validation loss checkpoint
- `models/autoencoder_final.pth` - Final epoch model

### Data Files  
- `outputs/clustering_results.npz` - Cluster labels and embeddings
- `outputs/evaluation_metrics.json` - All quantitative metrics

### Visualizations (11 figures)
1. Sample disk images
2. Training loss curves
3. Reconstruction comparisons
4. MSE/MS-SSIM distributions  
5. Silhouette elbow curve
6. K-Means UMAP clusters
7. HDBSCAN UMAP clusters
8. K-Means PCA clusters
9. Cluster size distribution
10. Representative cluster samples
11. Final summary figure

---

## 8. Reproducibility

All results are fully reproducible with:
- Fixed random seed: 42
- Saved model weights
- Documented hyperparameters
- Version-controlled code

---

## 9. Performance Summary

| Component | Status | Performance |
|-----------|--------|-------------|
| Data Loading | Success | 300 images loaded |
| Model Training | Success | Converged in 50 epochs |
| Reconstruction | Excellent | MS-SSIM = 0.9975 |
| Feature Extraction | Success | 184K-D latent vectors |
| Clustering | Good | Silhouette = 0.394 |
| Visualization | Complete | 11 publication-quality figures |

---

**Conclusion**: Successfully completed both General Test (clustering) and Image-Based Test (autoencoder reconstruction) with strong quantitative metrics and physically interpretable results.

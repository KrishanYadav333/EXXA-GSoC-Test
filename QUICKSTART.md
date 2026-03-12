# Quick Start Guide - EXXA GSoC Test

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/EXXA-GSoC-Test.git
cd EXXA-GSoC-Test
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Download Data

1. Download the synthetic ALMA observations from the test instructions link
2. Place all `.fits` files in the `data/` directory

## Run the Complete Pipeline

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/exxa_complete_pipeline.ipynb
```

Then run all cells from top to bottom (Cell → Run All)

### Option 2: Python Scripts

You can also use the modules individually:

```python
from src.data_loader import FITSDataLoader
from src.autoencoder import ImprovedAutoencoder
from src.clustering import DiskClusterer
from src.evaluation import ReconstructionEvaluator
from src.visualization import DiskVisualizer

# Load data
loader = FITSDataLoader('data/')
images, names = loader.load_all_fits()

# Train model
model = ImprovedAutoencoder(latent_dim=512)
# ... (see notebook for complete example)
```

## Expected Runtime

- **With GPU:** 15-20 minutes
- **With CPU:** 45-60 minutes

## Outputs

All results are automatically saved to:
- `models/` - Trained model weights
- `outputs/figures/` - Generated plots
- `outputs/clustering_results.npz` - Clustering data
- `outputs/evaluation_metrics.json` - Performance metrics

## Troubleshooting

### CUDA Out of Memory
Reduce `BATCH_SIZE` in the notebook (default: 8)

### FITS Files Not Found
Ensure files are in `data/` directory and have `.fits` or `.fit` extension

### Import Errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

### Slow Training
Use Google Colab with GPU for faster training

## Google Colab

To run in Colab:

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Change runtime to GPU (Runtime → Change runtime type → GPU)
4. Upload FITS files to Colab or mount Google Drive
5. Run all cells

## Project Structure

```
EXXA-GSoC-Test/
├── notebooks/           # Jupyter notebooks
│   └── exxa_complete_pipeline.ipynb
├── src/                 # Python modules
│   ├── data_loader.py
│   ├── autoencoder.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
├── models/              # Saved model weights
├── outputs/             # Results and figures
├── data/                # FITS files (not in repo)
├── requirements.txt     # Python dependencies
└── README.md           # Full documentation
```

## Key Features

- State-of-the-art U-Net autoencoder with skip connections
- Multiple clustering algorithms (K-Means, HDBSCAN, GMM)
- Comprehensive metrics (MSE, MS-SSIM, Silhouette)
- Rich visualizations (UMAP, cluster analysis)
- Fully automated and reproducible
- Ready for withheld data testing

## Getting Help

If you encounter issues:
1. Check this guide first
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify FITS files are in correct format

## Next Steps

After running the notebook:
1. Review generated figures in `outputs/figures/`
2. Check metrics in `outputs/evaluation_metrics.json`
3. Analyze cluster interpretations
4. Submit results via ML4Sci form

---

**Good luck with your GSoC application!**

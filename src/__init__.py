"""
EXXA GSoC Test - Protoplanetary Disk Analysis
ML4Sci Project

Python package for analyzing synthetic ALMA observations using
deep learning and unsupervised clustering.
"""

__version__ = "1.0.0"
__author__ = "EXXA GSoC Applicant"

# Import main classes for convenience
from .data_loader import FITSDataLoader
from .autoencoder import ImprovedAutoencoder, SimpleAutoencoder
from .clustering import DiskClusterer, DimensionalityReducer
from .evaluation import ReconstructionEvaluator, ClusteringEvaluator
from .visualization import DiskVisualizer

__all__ = [
    'FITSDataLoader',
    'ImprovedAutoencoder',
    'SimpleAutoencoder',
    'DiskClusterer',
    'DimensionalityReducer',
    'ReconstructionEvaluator',
    'ClusteringEvaluator',
    'DiskVisualizer'
]

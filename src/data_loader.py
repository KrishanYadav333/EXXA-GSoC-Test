"""
Data loading utilities for FITS files containing synthetic ALMA observations.

This module provides functions to load, preprocess, and prepare protoplanetary
disk images for machine learning pipelines.
"""

import os
import numpy as np
import torch
from astropy.io import fits
from typing import List, Tuple, Optional
from pathlib import Path
import warnings


class FITSDataLoader:
    """
    Loader for FITS format astronomical images.
    
    Each FITS file contains a data cube with 4 layers (600x600 pixels).
    Only layer 0 (index 0) is used for analysis.
    """
    
    def __init__(self, data_dir: str, normalize: bool = True):
        """
        Initialize the FITS data loader.
        
        Args:
            data_dir: Path to directory containing FITS files
            normalize: Whether to apply z-score normalization
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.file_paths = []
        self.images = None
        self.image_names = []
        
    def load_fits_file(self, file_path: str) -> np.ndarray:
        """
        Load a single FITS file and extract the first layer.
        
        Args:
            file_path: Path to FITS file
            
        Returns:
            2D numpy array (600x600) containing the image data
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdul = fits.open(file_path)
                
                # Extract first layer (index 0) from the data cube
                data = hdul[0].data
                
                # Handle different data cube formats
                if len(data.shape) == 4:
                    image = data[0, 0, :, :]  # 4D: [batch, channel, height, width]
                elif len(data.shape) == 3:
                    image = data[0, :, :]  # 3D: [channel, height, width]
                else:
                    image = data  # 2D: [height, width]
                
                hdul.close()
                
                # Check for NaN or Inf values
                if np.isnan(image).any() or np.isinf(image).any():
                    print(f"Warning: {file_path} contains NaN or Inf values. Replacing with zeros.")
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                
                return image.astype(np.float32)
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    
    def load_all_fits(self, max_files: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load all FITS files from the data directory.
        
        Args:
            max_files: Maximum number of files to load (None = all files)
            
        Returns:
            Tuple of (images array, list of filenames)
        """
        # Find all FITS files
        fits_extensions = ['*.fits', '*.fit', '*.FITS', '*.FIT']
        self.file_paths = []
        
        for ext in fits_extensions:
            self.file_paths.extend(self.data_dir.glob(ext))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No FITS files found in {self.data_dir}")
        
        # Limit number of files if specified
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]
        
        print(f"Found {len(self.file_paths)} FITS files")
        
        # Load all images
        images_list = []
        valid_names = []
        
        for fpath in self.file_paths:
            image = self.load_fits_file(fpath)
            if image is not None:
                images_list.append(image)
                valid_names.append(fpath.name)
        
        self.images = np.array(images_list)
        self.image_names = valid_names
        
        print(f"Successfully loaded {len(images_list)} images")
        print(f"Image shape: {self.images.shape}")
        
        # Normalize if requested
        if self.normalize:
            self.images = self.normalize_images(self.images)
            print("Images normalized")
        
        return self.images, self.image_names
    
    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to images.
        
        Args:
            images: Array of images [N, H, W]
            
        Returns:
            Normalized images
        """
        # Normalize each image independently
        normalized = np.zeros_like(images)
        
        for i in range(len(images)):
            img = images[i]
            mean = img.mean()
            std = img.std()
            
            # Avoid division by zero
            if std > 0:
                normalized[i] = (img - mean) / std
            else:
                normalized[i] = img - mean
        
        return normalized
    
    def to_pytorch_tensors(self, images: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Convert images to PyTorch tensors with channel dimension.
        
        Args:
            images: Optional numpy array. If None, uses self.images
            
        Returns:
            PyTorch tensor of shape [N, 1, H, W]
        """
        if images is None:
            images = self.images
        
        # Add channel dimension: [N, H, W] -> [N, 1, H, W]
        tensors = torch.from_numpy(images).unsqueeze(1).float()
        
        return tensors
    
    def create_dataloaders(
        self, 
        batch_size: int = 8,
        train_split: float = 0.8,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation.
        
        Args:
            batch_size: Batch size for DataLoader
            train_split: Fraction of data for training (rest for validation)
            shuffle: Whether to shuffle the data
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.images is None:
            raise ValueError("No images loaded. Call load_all_fits() first.")
        
        # Convert to tensors
        tensors = self.to_pytorch_tensors()
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(tensors)
        
        # Split into train/val
        n_images = len(dataset)
        n_train = int(n_images * train_split)
        n_val = n_images - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [n_train, n_val],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Train samples: {n_train}, Validation samples: {n_val}")
        
        return train_loader, val_loader
    
    def get_image_statistics(self) -> dict:
        """
        Compute basic statistics of the loaded images.
        
        Returns:
            Dictionary with statistics
        """
        if self.images is None:
            raise ValueError("No images loaded.")
        
        stats = {
            'n_images': len(self.images),
            'shape': self.images.shape,
            'mean': self.images.mean(),
            'std': self.images.std(),
            'min': self.images.min(),
            'max': self.images.max(),
            'median': np.median(self.images)
        }
        
        return stats


def load_fits_data(data_dir: str, normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to quickly load FITS data.
    
    Args:
        data_dir: Path to directory containing FITS files
        normalize: Whether to normalize images
        
    Returns:
        Tuple of (images array, filenames)
    """
    loader = FITSDataLoader(data_dir, normalize=normalize)
    images, names = loader.load_all_fits()
    return images, names


if __name__ == "__main__":
    # Example usage
    print("FITS Data Loader Module")
    print("Usage:")
    print("  from src.data_loader import FITSDataLoader")
    print("  loader = FITSDataLoader('data/')")
    print("  images, names = loader.load_all_fits()")

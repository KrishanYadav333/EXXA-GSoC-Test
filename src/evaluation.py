"""
Evaluation metrics for autoencoder reconstruction and clustering quality.

This module provides quantitative metrics for assessing:
- Image reconstruction quality (MSE, MS-SSIM)
- Clustering performance (Silhouette, Davies-Bouldin, etc.)
"""

import torch
import numpy as np
from pytorch_msssim import ms_ssim, MS_SSIM
from typing import Dict, Tuple, List
import torch.nn.functional as F


class ReconstructionEvaluator:
    """
    Evaluate autoencoder reconstruction quality.
    
    Provides MSE and MS-SSIM metrics as required by the test.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device
        self.ms_ssim_module = MS_SSIM(data_range=2.0, size_average=True, channel=1).to(device)
        
    def compute_mse(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute Mean Squared Error between original and reconstructed images.
        
        Args:
            original: Original images [N, 1, H, W]
            reconstructed: Reconstructed images [N, 1, H, W]
            
        Returns:
            MSE value (scalar)
        """
        mse = F.mse_loss(reconstructed, original).item()
        return mse
    
    def compute_ms_ssim(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute Multi-Scale Structural Similarity Index (MS-SSIM).
        
        Higher values indicate better reconstruction quality.
        
        Args:
            original: Original images [N, 1, H, W]
            reconstructed: Reconstructed images [N, 1, H, W]
            
        Returns:
            MS-SSIM value (scalar, range [0, 1])
        """
        # Ensure tensors are in range for MS-SSIM
        # For normalized images in [-1, 1], we need to shift to [0, 1]
        original_shifted = (original + 1) / 2
        reconstructed_shifted = (reconstructed + 1) / 2
        
        # Clamp to valid range
        original_shifted = torch.clamp(original_shifted, 0, 1)
        reconstructed_shifted = torch.clamp(reconstructed_shifted, 0, 1)
        
        try:
            ms_ssim_value = ms_ssim(
                reconstructed_shifted,
                original_shifted,
                data_range=1.0,
                size_average=True
            ).item()
        except:
            # Fallback if MS-SSIM fails
            ms_ssim_value = 0.0
        
        return ms_ssim_value
    
    def evaluate_batch(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all reconstruction metrics for a batch.
        
        Args:
            original: Original images [N, 1, H, W]
            reconstructed: Reconstructed images [N, 1, H, W]
            
        Returns:
            Dictionary with MSE and MS-SSIM
        """
        original = original.to(self.device)
        reconstructed = reconstructed.to(self.device)
        
        metrics = {
            'mse': self.compute_mse(original, reconstructed),
            'ms_ssim': self.compute_ms_ssim(original, reconstructed)
        }
        
        return metrics
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """
        Evaluate model on entire dataset.
        
        Args:
            model: Autoencoder model
            dataloader: DataLoader with test images
            
        Returns:
            Dictionary with lists of per-batch metrics
        """
        model.eval()
        model = model.to(self.device)
        
        all_mse = []
        all_ms_ssim = []
        
        with torch.no_grad():
            for batch_idx, (images,) in enumerate(dataloader):
                images = images.to(self.device)
                
                # Get reconstructions
                reconstructions = model(images)
                
                # Compute metrics
                metrics = self.evaluate_batch(images, reconstructions)
                
                all_mse.append(metrics['mse'])
                all_ms_ssim.append(metrics['ms_ssim'])
        
        results = {
            'mse_per_batch': all_mse,
            'ms_ssim_per_batch': all_ms_ssim,
            'mean_mse': np.mean(all_mse),
            'std_mse': np.std(all_mse),
            'mean_ms_ssim': np.mean(all_ms_ssim),
            'std_ms_ssim': np.std(all_ms_ssim)
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            results: Dictionary from evaluate_model()
        """
        print("\n" + "="*60)
        print("RECONSTRUCTION EVALUATION SUMMARY")
        print("="*60)
        print(f"\nMean Squared Error (MSE):")
        print(f"  Mean: {results['mean_mse']:.6f}")
        print(f"  Std:  {results['std_mse']:.6f}")
        print(f"\nMulti-Scale SSIM (MS-SSIM):")
        print(f"  Mean: {results['mean_ms_ssim']:.6f}")
        print(f"  Std:  {results['std_ms_ssim']:.6f}")
        print(f"\nInterpretation:")
        print(f"  - Lower MSE = better reconstruction")
        print(f"  - Higher MS-SSIM = better structural similarity")
        print(f"  - MS-SSIM range: [0, 1], where 1 is perfect")
        print("="*60 + "\n")


class ClusteringEvaluator:
    """
    Evaluate clustering quality.
    
    Note: Most clustering metrics are in clustering.py,
    this class provides additional analysis tools.
    """
    
    @staticmethod
    def analyze_cluster_properties(
        images: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Analyze visual properties of each cluster.
        
        Args:
            images: Original images [N, H, W]
            labels: Cluster labels [N,]
            
        Returns:
            Dictionary with statistics for each cluster
        """
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get images in this cluster
            mask = labels == label
            cluster_images = images[mask]
            
            stats = {
                'n_samples': int(np.sum(mask)),
                'mean_intensity': float(cluster_images.mean()),
                'std_intensity': float(cluster_images.std()),
                'max_intensity': float(cluster_images.max()),
                'min_intensity': float(cluster_images.min())
            }
            
            cluster_stats[int(label)] = stats
        
        return cluster_stats
    
    @staticmethod
    def get_cluster_representatives(
        images: np.ndarray,
        latent_features: np.ndarray,
        labels: np.ndarray,
        n_representatives: int = 5
    ) -> Dict[int, np.ndarray]:
        """
        Get representative images from each cluster (closest to centroid).
        
        Args:
            images: Original images [N, H, W]
            latent_features: Latent features [N, D]
            labels: Cluster labels [N,]
            n_representatives: Number of representatives per cluster
            
        Returns:
            Dictionary mapping cluster_id -> representative image indices
        """
        unique_labels = np.unique(labels)
        representatives = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            # Get features for this cluster
            mask = labels == label
            cluster_features = latent_features[mask]
            cluster_indices = np.where(mask)[0]
            
            # Compute centroid
            centroid = cluster_features.mean(axis=0)
            
            # Find closest points to centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_indices = np.argsort(distances)[:n_representatives]
            
            # Map back to original indices
            representatives[int(label)] = cluster_indices[closest_indices]
        
        return representatives
    
    @staticmethod
    def print_cluster_summary(
        labels: np.ndarray,
        cluster_stats: Dict[int, Dict]
    ):
        """
        Print a formatted summary of clustering results.
        
        Args:
            labels: Cluster labels
            cluster_stats: Statistics from analyze_cluster_properties()
        """
        print("\n" + "="*60)
        print("CLUSTERING SUMMARY")
        print("="*60)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"\nTotal Clusters: {n_clusters}")
        if n_noise > 0:
            print(f"Noise Points: {n_noise}")
        
        print(f"\nCluster Sizes:")
        for cluster_id, stats in sorted(cluster_stats.items()):
            print(f"  Cluster {cluster_id}: {stats['n_samples']} images")
        
        print(f"\nCluster Properties:")
        for cluster_id, stats in sorted(cluster_stats.items()):
            print(f"  Cluster {cluster_id}:")
            print(f"    Mean intensity: {stats['mean_intensity']:.4f}")
            print(f"    Std intensity:  {stats['std_intensity']:.4f}")
        
        print("="*60 + "\n")


def evaluate_full_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    images: np.ndarray,
    latent_features: np.ndarray,
    cluster_labels: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Comprehensive evaluation of the entire pipeline.
    
    Args:
        model: Trained autoencoder
        dataloader: DataLoader for reconstruction evaluation
        images: Original images for cluster analysis
        latent_features: Latent features for clustering
        cluster_labels: Assigned cluster labels
        device: Computation device
        
    Returns:
        Dictionary with all evaluation results
    """
    print("Evaluating full pipeline...\n")
    
    # Reconstruction evaluation
    print("1. Evaluating reconstruction quality...")
    recon_evaluator = ReconstructionEvaluator(device=device)
    recon_results = recon_evaluator.evaluate_model(model, dataloader)
    recon_evaluator.print_evaluation_summary(recon_results)
    
    # Clustering evaluation
    print("2. Analyzing clustering results...")
    cluster_evaluator = ClusteringEvaluator()
    cluster_stats = cluster_evaluator.analyze_cluster_properties(images, cluster_labels)
    cluster_evaluator.print_cluster_summary(cluster_labels, cluster_stats)
    
    # Get representative images
    representatives = cluster_evaluator.get_cluster_representatives(
        images, latent_features, cluster_labels, n_representatives=5
    )
    
    # Combine results
    full_results = {
        'reconstruction': recon_results,
        'clustering': {
            'labels': cluster_labels,
            'stats': cluster_stats,
            'representatives': representatives
        }
    }
    
    return full_results


if __name__ == "__main__":
    print("Evaluation Module")
    print("Usage:")
    print("  from src.evaluation import ReconstructionEvaluator")
    print("  evaluator = ReconstructionEvaluator()")
    print("  results = evaluator.evaluate_model(model, dataloader)")

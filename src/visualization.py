"""
Visualization utilities for protoplanetary disk analysis.

This module provides comprehensive plotting functions for:
- Image exploration and comparison
- Reconstruction quality visualization
- Clustering results and embeddings
- Performance metrics and training curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DiskVisualizer:
    """
    Comprehensive visualization toolkit for disk image analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), save_dir: str = 'outputs/figures'):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            save_dir: Directory to save figures
        """
        self.figsize = figsize
        self.save_dir = save_dir
        
    def plot_sample_images(
        self,
        images: np.ndarray,
        n_samples: int = 9,
        titles: Optional[List[str]] = None,
        suptitle: str = "Sample Disk Images",
        cmap: str = 'inferno',
        save_name: Optional[str] = None
    ):
        """
        Plot a grid of sample images.
        
        Args:
            images: Array of images [N, H, W]
            n_samples: Number of samples to display
            titles: Optional list of titles for each image
            suptitle: Overall title
            cmap: Colormap for images
            save_name: Optional filename to save figure
        """
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i in range(n_samples):
            if i < len(images):
                im = axes[i].imshow(images[i], cmap=cmap, origin='lower')
                if titles and i < len(titles):
                    axes[i].set_title(titles[i], fontsize=10)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
            else:
                axes[i].axis('off')
        
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_reconstruction_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        n_samples: int = 6,
        mse_values: Optional[List[float]] = None,
        ssim_values: Optional[List[float]] = None,
        cmap: str = 'inferno',
        save_name: Optional[str] = None
    ):
        """
        Plot original vs reconstructed images side by side.
        
        Args:
            original: Original images [N, H, W]
            reconstructed: Reconstructed images [N, H, W]
            n_samples: Number of comparison pairs
            mse_values: Optional MSE for each sample
            ssim_values: Optional SSIM for each sample
            cmap: Colormap
            save_name: Optional filename to save
        """
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples*2))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            if i >= len(original):
                break
            
            # Original
            im1 = axes[i, 0].imshow(original[i], cmap=cmap, origin='lower')
            axes[i, 0].set_title('Original', fontsize=10)
            axes[i, 0].axis('off')
            
            # Reconstructed
            im2 = axes[i, 1].imshow(reconstructed[i], cmap=cmap, origin='lower')
            axes[i, 1].set_title('Reconstructed', fontsize=10)
            axes[i, 1].axis('off')
            
            # Difference
            diff = np.abs(original[i] - reconstructed[i])
            im3 = axes[i, 2].imshow(diff, cmap='hot', origin='lower')
            axes[i, 2].set_title('Absolute Difference', fontsize=10)
            axes[i, 2].axis('off')
            
            # Add metrics to row label
            row_label = f"Sample {i+1}"
            if mse_values and i < len(mse_values):
                row_label += f"\nMSE: {mse_values[i]:.6f}"
            if ssim_values and i < len(ssim_values):
                row_label += f"\nSSIM: {ssim_values[i]:.4f}"
            
            axes[i, 0].text(-0.3, 0.5, row_label, transform=axes[i, 0].transAxes,
                          fontsize=9, verticalalignment='center', rotation=0)
        
        plt.suptitle('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            history: Dictionary with 'train_loss' and 'val_loss' lists
            save_name: Optional filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        ax.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_clustering_embedding(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        method: str = 'UMAP',
        save_name: Optional[str] = None
    ):
        """
        Plot 2D embedding colored by cluster labels.
        
        Args:
            embedding: 2D embedding [N, 2]
            labels: Cluster labels [N,]
            method: Name of dimensionality reduction method
            save_name: Optional filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create color map
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Handle noise points (label -1)
        if -1 in unique_labels:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters - 1))
            noise_color = np.array([[0.5, 0.5, 0.5, 0.5]])  # Gray for noise
            colors = np.vstack([noise_color, colors])
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f'Cluster {label}' if label != -1 else 'Noise'
            
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[idx]],
                label=label_name,
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
        ax.set_title(f'{method} Embedding - Protoplanetary Disk Clusters', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cluster_grid(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        representatives: Dict[int, np.ndarray],
        n_per_cluster: int = 5,
        cmap: str = 'inferno',
        save_name: Optional[str] = None
    ):
        """
        Plot representative images from each cluster in a grid.
        
        Args:
            images: All images [N, H, W]
            labels: Cluster labels [N,]
            representatives: Dict mapping cluster_id -> representative indices
            n_per_cluster: Number of images per cluster
            cmap: Colormap
            save_name: Optional filename to save
        """
        n_clusters = len(representatives)
        
        fig, axes = plt.subplots(n_clusters, n_per_cluster, 
                                figsize=(n_per_cluster*2.5, n_clusters*2.5))
        
        if n_clusters == 1:
            axes = axes.reshape(1, -1)
        
        for cluster_id, indices in sorted(representatives.items()):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_images = images[indices[:n_per_cluster]]
            
            for j, img in enumerate(cluster_images):
                ax = axes[cluster_id, j] if n_clusters > 1 else axes[j]
                ax.imshow(img, cmap=cmap, origin='lower')
                ax.axis('off')
                
                if j == 0:
                    n_in_cluster = np.sum(labels == cluster_id)
                    ax.text(-0.1, 0.5, f'Cluster {cluster_id}\n({n_in_cluster} disks)',
                          transform=ax.transAxes, fontsize=10,
                          verticalalignment='center', rotation=0,
                          fontweight='bold')
        
        plt.suptitle('Representative Images from Each Cluster', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cluster_size_distribution(
        self,
        labels: np.ndarray,
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of cluster sizes.
        
        Args:
            labels: Cluster labels [N,]
            save_name: Optional filename to save
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        # Remove noise if present
        mask = unique != -1
        unique = unique[mask]
        counts = counts[mask]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax.bar(unique, counts, color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Disks', fontsize=12)
        ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(unique)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_elbow_curve(
        self,
        k_values: List[int],
        scores: List[float],
        metric_name: str = 'Silhouette Score',
        optimal_k: Optional[int] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot elbow curve for optimal K selection.
        
        Args:
            k_values: List of K values tested
            scores: Corresponding scores
            metric_name: Name of the metric
            optimal_k: Optimal K to highlight
            save_name: Optional filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(k_values, scores, 'b-o', linewidth=2, markersize=8)
        
        if optimal_k is not None:
            optimal_idx = k_values.index(optimal_k)
            ax.plot(optimal_k, scores[optimal_idx], 'r*', markersize=20, 
                   label=f'Optimal K = {optimal_k}')
        
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Cluster Selection: {metric_name} vs K', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        
        if optimal_k is not None:
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metric_distributions(
        self,
        mse_values: List[float],
        ssim_values: List[float],
        save_name: Optional[str] = None
    ):
        """
        Plot distributions of MSE and MS-SSIM values.
        
        Args:
            mse_values: List of MSE values
            ssim_values: List of MS-SSIM values
            save_name: Optional filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MSE histogram
        axes[0].hist(mse_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(mse_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(mse_values):.6f}')
        axes[0].set_xlabel('Mean Squared Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('MSE Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MS-SSIM histogram
        axes[1].hist(ssim_values, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(ssim_values):.6f}')
        axes[1].set_xlabel('Multi-Scale SSIM', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('MS-SSIM Distribution', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()


def create_summary_figure(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/figures/summary.png'
):
    """
    Create a comprehensive summary figure with multiple subplots.
    
    Args:
        original_images: Original images
        reconstructed_images: Reconstructed images
        embedding: 2D embedding
        labels: Cluster labels
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Sample original images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_images[0], cmap='inferno', origin='lower')
    ax1.set_title('Sample Original Disk', fontweight='bold')
    ax1.axis('off')
    
    # Reconstructed image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(reconstructed_images[0], cmap='inferno', origin='lower')
    ax2.set_title('Reconstructed Disk', fontweight='bold')
    ax2.axis('off')
    
    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(original_images[0] - reconstructed_images[0])
    ax3.imshow(diff, cmap='hot', origin='lower')
    ax3.set_title('Reconstruction Error', fontweight='bold')
    ax3.axis('off')
    
    # Clustering
    ax4 = fig.add_subplot(gs[1, :])
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        label_name = f'Cluster {label}' if label != -1 else 'Noise'
        ax4.scatter(embedding[mask, 0], embedding[mask, 1], 
                   label=label_name, s=30, alpha=0.6)
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.set_title('Disk Clustering in Latent Space', fontweight='bold', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Visualization Module")
    print("Usage:")
    print("  from src.visualization import DiskVisualizer")
    print("  viz = DiskVisualizer()")
    print("  viz.plot_sample_images(images)")

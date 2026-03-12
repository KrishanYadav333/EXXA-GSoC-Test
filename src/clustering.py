"""
Clustering algorithms and utilities for protoplanetary disk analysis.

This module provides multiple clustering approaches and evaluation metrics
for unsupervised grouping of disk images based on their latent features.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Tuple, Optional
import hdbscan
import umap


class DiskClusterer:
    """
    Comprehensive clustering analysis for protoplanetary disk images.
    
    Supports multiple clustering algorithms and provides evaluation metrics.
    """
    
    def __init__(self, latent_features: np.ndarray):
        """
        Initialize clusterer with latent features.
        
        Args:
            latent_features: Array of shape [N, D] containing latent representations
        """
        self.latent_features = latent_features
        self.n_samples = latent_features.shape[0]
        self.n_features = latent_features.shape[1]
        
        self.labels = None
        self.algorithm_used = None
        self.cluster_centers = None
        
        print(f"Initialized clusterer with {self.n_samples} samples, {self.n_features} features")
    
    def kmeans_clustering(
        self, 
        n_clusters: int = 5,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Perform K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Cluster labels [N,]
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = kmeans.fit_predict(self.latent_features)
        self.cluster_centers = kmeans.cluster_centers_
        self.algorithm_used = f"K-Means (k={n_clusters})"
        
        print(f"K-Means clustering completed: {n_clusters} clusters")
        return self.labels
    
    def hdbscan_clustering(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        Perform HDBSCAN clustering (density-based, automatic cluster detection).
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood
            metric: Distance metric
            
        Returns:
            Cluster labels [N,] (noise points labeled as -1)
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        self.labels = clusterer.fit_predict(self.latent_features)
        self.algorithm_used = "HDBSCAN"
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        print(f"HDBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
        return self.labels
    
    def agglomerative_clustering(
        self,
        n_clusters: int = 5,
        linkage: str = 'ward'
    ) -> np.ndarray:
        """
        Perform Agglomerative (hierarchical) clustering.
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average')
            
        Returns:
            Cluster labels [N,]
        """
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = clusterer.fit_predict(self.latent_features)
        self.algorithm_used = f"Agglomerative (k={n_clusters})"
        
        print(f"Agglomerative clustering completed: {n_clusters} clusters")
        return self.labels
    
    def gmm_clustering(
        self,
        n_components: int = 5,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Perform Gaussian Mixture Model clustering (soft clustering).
        
        Args:
            n_components: Number of mixture components
            random_state: Random seed
            
        Returns:
            Cluster labels [N,]
        """
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels = gmm.fit_predict(self.latent_features)
        self.algorithm_used = f"GMM (k={n_components})"
        
        print(f"GMM clustering completed: {n_components} components")
        return self.labels
    
    def find_optimal_k(
        self,
        k_range: range = range(2, 11),
        method: str = 'silhouette'
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters using elbow method or silhouette analysis.
        
        Args:
            k_range: Range of k values to test
            method: 'silhouette', 'davies_bouldin', or 'calinski_harabasz'
            
        Returns:
            Tuple of (optimal_k, scores_dict)
        """
        scores = {}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.latent_features)
            
            if method == 'silhouette':
                score = silhouette_score(self.latent_features, labels)
                scores[k] = score
            elif method == 'davies_bouldin':
                score = davies_bouldin_score(self.latent_features, labels)
                scores[k] = -score  # Lower is better, so negate
            elif method == 'calinski_harabasz':
                score = calinski_harabasz_score(self.latent_features, labels)
                scores[k] = score
        
        optimal_k = max(scores, key=scores.get)
        
        print(f"Optimal k using {method}: {optimal_k}")
        print(f"Score: {scores[optimal_k]:.4f}")
        
        return optimal_k, scores
    
    def evaluate_clustering(self, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Args:
            labels: Cluster labels (uses self.labels if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if labels is None:
            labels = self.labels
        
        if labels is None:
            raise ValueError("No clustering labels available")
        
        # Filter out noise points (label -1) for metrics
        mask = labels != -1
        features_filtered = self.latent_features[mask]
        labels_filtered = labels[mask]
        
        if len(set(labels_filtered)) < 2:
            print("Warning: Less than 2 clusters, some metrics unavailable")
            return {}
        
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(features_filtered, labels_filtered)
        except:
            metrics['silhouette_score'] = np.nan
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(features_filtered, labels_filtered)
        except:
            metrics['davies_bouldin_score'] = np.nan
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_filtered, labels_filtered)
        except:
            metrics['calinski_harabasz_score'] = np.nan
        
        metrics['n_clusters'] = len(set(labels_filtered))
        metrics['n_noise'] = np.sum(labels == -1)
        
        return metrics
    
    def get_cluster_sizes(self, labels: Optional[np.ndarray] = None) -> Dict[int, int]:
        """
        Get the size of each cluster.
        
        Args:
            labels: Cluster labels (uses self.labels if None)
            
        Returns:
            Dictionary mapping cluster_id -> size
        """
        if labels is None:
            labels = self.labels
        
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


class DimensionalityReducer:
    """
    Dimensionality reduction for visualization.
    
    Note: Used ONLY for visualization, not for clustering.
    """
    
    def __init__(self, latent_features: np.ndarray):
        """
        Initialize with latent features.
        
        Args:
            latent_features: Array of shape [N, D]
        """
        self.latent_features = latent_features
        self.embedding_2d = None
        self.embedding_3d = None
    
    def umap_reduction(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42
    ) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.
        
        Args:
            n_components: Number of dimensions (2 or 3 for visualization)
            n_neighbors: Size of local neighborhood
            min_dist: Minimum distance between points in embedding
            metric: Distance metric
            random_state: Random seed
            
        Returns:
            Reduced embedding [N, n_components]
        """
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        embedding = reducer.fit_transform(self.latent_features)
        
        if n_components == 2:
            self.embedding_2d = embedding
        elif n_components == 3:
            self.embedding_3d = embedding
        
        print(f"UMAP reduction to {n_components}D completed")
        return embedding
    
    def pca_reduction(
        self,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Apply PCA dimensionality reduction (for comparison with UMAP).
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Reduced embedding [N, n_components]
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(self.latent_features)
        
        print(f"PCA reduction to {n_components}D completed")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return embedding


def compare_clustering_algorithms(
    latent_features: np.ndarray,
    n_clusters: int = 5
) -> Dict[str, Dict]:
    """
    Compare multiple clustering algorithms on the same data.
    
    Args:
        latent_features: Latent feature array [N, D]
        n_clusters: Number of clusters for algorithms that require it
        
    Returns:
        Dictionary with results for each algorithm
    """
    clusterer = DiskClusterer(latent_features)
    
    results = {}
    
    # K-Means
    print("\n--- K-Means ---")
    labels_km = clusterer.kmeans_clustering(n_clusters=n_clusters)
    results['kmeans'] = {
        'labels': labels_km,
        'metrics': clusterer.evaluate_clustering(labels_km)
    }
    
    # HDBSCAN
    print("\n--- HDBSCAN ---")
    labels_hdb = clusterer.hdbscan_clustering(min_cluster_size=10)
    results['hdbscan'] = {
        'labels': labels_hdb,
        'metrics': clusterer.evaluate_clustering(labels_hdb)
    }
    
    # Agglomerative
    print("\n--- Agglomerative ---")
    labels_agg = clusterer.agglomerative_clustering(n_clusters=n_clusters)
    results['agglomerative'] = {
        'labels': labels_agg,
        'metrics': clusterer.evaluate_clustering(labels_agg)
    }
    
    # GMM
    print("\n--- Gaussian Mixture Model ---")
    try:
        labels_gmm = clusterer.gmm_clustering(n_components=n_clusters)
        results['gmm'] = {
            'labels': labels_gmm,
            'metrics': clusterer.evaluate_clustering(labels_gmm)
        }
        print("GMM clustering completed")
    except MemoryError:
        print("⚠️  GMM clustering skipped: insufficient memory for full covariance with high-dimensional data")
        print("   (GMM requires ~255 GB for 184K features)")
    
    return results


if __name__ == "__main__":
    print("Clustering Module")
    print("Usage:")
    print("  from src.clustering import DiskClusterer, DimensionalityReducer")
    print("  clusterer = DiskClusterer(latent_features)")
    print("  labels = clusterer.kmeans_clustering(n_clusters=5)")

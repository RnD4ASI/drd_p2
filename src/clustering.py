import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ClusteringModel:
    """Handles clustering of embedding vectors."""
    
    def __init__(self):
        """Initialize the ClusteringModel."""
        self.model = None
        self.clusters = None
    
    def fit(self, embeddings: np.ndarray, max_cluster_size: int = 10, max_clusters: int = 20, 
            random_state: int = 42) -> Dict[str, Any]:
        """Fit K-means clustering model to the embeddings.
        
        Parameters:
            embeddings (np.ndarray): Embedding vectors
            max_cluster_size (int): Maximum size of each cluster
            max_clusters (int): Maximum number of clusters
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, Any]: Dictionary containing clustering results
        """
        try:
            n_samples = embeddings.shape[0]
            
            # Calculate minimum number of clusters needed based on max_cluster_size
            min_clusters = max(2, n_samples // max_cluster_size)
            
            # Limit to max_clusters
            n_clusters = min(min_clusters, max_clusters, n_samples - 1)
            
            logger.info(f"Fitting K-means with {n_clusters} clusters")
            
            # Fit K-means
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            self.clusters = self.model.fit_predict(embeddings)
            
            # Calculate silhouette score
            if n_clusters > 1 and n_samples > n_clusters:
                silhouette_avg = silhouette_score(embeddings, self.clusters)
                logger.info(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                silhouette_avg = None
                logger.warning("Not enough samples or clusters for silhouette score")
            
            # Get cluster centers
            cluster_centers = self.model.cluster_centers_
            
            # Count samples in each cluster
            cluster_counts = np.bincount(self.clusters, minlength=n_clusters)
            
            # Create result dictionary
            result = {
                "n_clusters": n_clusters,
                "clusters": self.clusters.tolist(),
                "cluster_centers": cluster_centers.tolist(),
                "cluster_counts": cluster_counts.tolist(),
                "silhouette_score": silhouette_avg
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fit: {e}")
            raise
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict clusters for new embeddings.
        
        Parameters:
            embeddings (np.ndarray): Embedding vectors
            
        Returns:
            np.ndarray: Cluster assignments
        """
        try:
            if self.model is None:
                raise ValueError("Model not fitted. Call fit first.")
            
            return self.model.predict(embeddings)
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise
    
    def visualize_clusters(self, embeddings: np.ndarray, output_path: Union[str, Path]):
        """Visualize clusters using PCA.
        
        Parameters:
            embeddings (np.ndarray): Embedding vectors
            output_path (Union[str, Path]): Path to save the visualization
        """
        try:
            if self.clusters is None:
                raise ValueError("Model not fitted. Call fit first.")
            
            from sklearn.decomposition import PCA
            
            # Reduce to 2D using PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                                 c=self.clusters, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title('Cluster Visualization (PCA)')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Cluster visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in visualize_clusters: {e}")
            logger.warning("Continuing without visualization")

def run_clustering(embeddings: np.ndarray, attributes_df: pd.DataFrame, 
                  max_cluster_size: int = 10, max_clusters: int = 20,
                  output_dir: Union[str, Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cluster embeddings and assign cluster IDs to attributes.
    
    Parameters:
        embeddings (np.ndarray): Embedding vectors
        attributes_df (pd.DataFrame): DataFrame containing attribute information
        max_cluster_size (int): Maximum size of each cluster
        max_clusters (int): Maximum number of clusters
        output_dir (Union[str, Path]): Directory to save outputs
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: 
            - DataFrame with cluster assignments
            - Dictionary containing clustering results
    """
    try:
        # Initialize clustering model
        clustering_model = ClusteringModel()
        
        # Fit model
        clustering_results = clustering_model.fit(
            embeddings=embeddings,
            max_cluster_size=max_cluster_size,
            max_clusters=max_clusters
        )
        
        # Add cluster IDs to attributes DataFrame
        attributes_with_clusters = attributes_df.copy()
        attributes_with_clusters['cluster_id'] = clustering_results['clusters']
        
        # Save visualization if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Visualize clusters
            try:
                clustering_model.visualize_clusters(
                    embeddings=embeddings,
                    output_path=output_dir / 'cluster_visualization.png'
                )
            except Exception as e:
                logger.warning(f"Failed to create visualization: {e}")
            
            # Save clustering results
            cluster_stats = pd.DataFrame({
                'cluster_id': range(clustering_results['n_clusters']),
                'size': clustering_results['cluster_counts']
            })
            cluster_stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
            
            # Save attributes with clusters
            attributes_with_clusters.to_csv(output_dir / 'attributes_with_clusters.csv', index=False)
        
        return attributes_with_clusters, clustering_results
        
    except Exception as e:
        logger.error(f"Error in cluster_embeddings: {e}")
        raise

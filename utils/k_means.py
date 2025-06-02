# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
#
# Inspired in part by:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# ------------------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class ClusterEvaluator:
    """
    Evaluates KMeans clustering performance over a range of cluster numbers.
    
    Attributes:
        X (np.ndarray): Feature data for clustering.
        cluster_range (Iterable[int]): Range of cluster numbers to evaluate.
        use_silhouette (bool): Whether to compute the silhouette score for each clustering.
    """
    
    def __init__(self, X, cluster_range, use_silhouette=True):
        """
        Args:
            X (np.ndarray): The input data for clustering.
            cluster_range (Iterable[int]): A range or list of integers indicating the number of clusters to try.
            use_silhouette (bool): Whether to compute silhouette scores.
        """
        self.X = X
        self.cluster_range = cluster_range
        self.use_silhouette = use_silhouette

    @staticmethod
    def _safe_silhouette_score(X, labels, sample_size=1000, random_state=0):
        """
        Computes silhouette score safely by ensuring label diversity.

        Args:
            X (np.ndarray): Input data.
            labels (np.ndarray): Cluster labels.
            sample_size (int): Sample size for computing silhouette.
            random_state (int): Random seed.

        Returns:
            float: Silhouette score or 0.0 if not computable.
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels, sample_size=min(sample_size, len(X)), random_state=random_state)

    def evaluate_kmeans(self, n_init=10, max_iter=300, random_state=None):
        """
        Generator that yields evaluation metrics for each cluster count in cluster_range.

        Args:
            n_init (int): Number of initializations for KMeans.
            max_iter (int): Maximum iterations for KMeans.
            random_state (int): Random seed.

        Yields:
            dict: Contains 'n_clusters', 'inertia', 'labels', and optionally 'silhouette_score'.
        """
        for n_clusters in self.cluster_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state
            )
            kmeans.fit(self.X)
            result = {
                'n_clusters': n_clusters,
                'inertia': kmeans.inertia_,
                'labels': kmeans.labels_
            }

            if self.use_silhouette:
                result['silhouette_score'] = self._safe_silhouette_score(self.X, kmeans.labels_)

            yield result

                
                

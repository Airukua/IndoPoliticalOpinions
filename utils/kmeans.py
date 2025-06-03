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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


class ClusterEvaluator:
    """
    Evaluates KMeans clustering performance across different cluster counts.
    
    Attributes:
        X (np.ndarray): Feature matrix for clustering.
        cluster_range (Iterable[int]): Range of cluster counts to evaluate.
        use_silhouette (bool): Flag to compute silhouette score for each configuration.
    """

    def __init__(self, X: np.ndarray, cluster_range, use_silhouette: bool = True):
        """
        Initializes the evaluator with data and configuration.

        Args:
            X (np.ndarray): Feature data for clustering.
            cluster_range (Iterable[int]): Range/list of cluster counts to test.
            use_silhouette (bool): Whether to compute silhouette scores.
        """
        self.X = X
        self.cluster_range = cluster_range
        self.use_silhouette = use_silhouette

    @staticmethod
    def _safe_silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 1000, random_state: int = 0) -> float:
        """
        Computes the silhouette score safely, ensuring there is label diversity.

        Args:
            X (np.ndarray): Input feature matrix.
            labels (np.ndarray): Cluster labels.
            sample_size (int): Number of samples to use.
            random_state (int): Seed for reproducibility.

        Returns:
            float: Silhouette score or 0.0 if not computable.
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels, sample_size=min(sample_size, X.shape[0]), random_state=random_state)

    def evaluate_kmeans(self, n_init: int = 10, max_iter: int = 300, random_state: int = None):
        """
        Evaluates KMeans clustering over a range of cluster counts.

        Args:
            n_init (int): Number of KMeans initializations.
            max_iter (int): Maximum number of iterations.
            random_state (int): Seed for reproducibility.

        Yields:
            dict: {
                'n_clusters': int,
                'inertia': float,
                'labels': np.ndarray,
                'centers': np.ndarray,
                'silhouette_score' (optional): float
            }
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
                'labels': kmeans.labels_,
                'centers': kmeans.cluster_centers_
            }

            if self.use_silhouette:
                result['silhouette_score'] = self._safe_silhouette_score(self.X, kmeans.labels_)

            yield result

    def cluster_inference(self, X_new: np.ndarray, centers: np.ndarray) -> dict:
        """
        Assigns new samples to the closest cluster center and evaluates the clustering.

        Args:
            X_new (np.ndarray): New input data.
            centers (np.ndarray): Precomputed cluster centers.

        Returns:
            dict: {
                'labels': np.ndarray,
                'inertia': float,
                'silhouette_score': float
            }
        """
        distances = cdist(X_new, centers, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        inertia = np.sum((np.linalg.norm(X_new - centers[labels], axis=1)) ** 2)

        silhouette = (
            silhouette_score(X_new, labels)
            if len(np.unique(labels)) > 1 else 0.0
        )

        return {
            'labels': labels,
            'inertia': inertia,
            'silhouette_score': silhouette
        }
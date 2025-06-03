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
from sklearn.metrics.pairwise import euclidean_distances


class ClusterEvaluator:
    """
    Evaluates KMeans clustering performance across different cluster counts.

    Attributes:
        X (np.ndarray): Feature matrix for clustering.
        cluster_range (Iterable[int]): Range of cluster counts to evaluate.
        use_silhouette (bool): Flag to compute silhouette score for each configuration.
    """

    def __init__(self, X: np.ndarray, cluster_range, use_silhouette: bool = True):
        self.X = X
        self.cluster_range = cluster_range
        self.use_silhouette = use_silhouette

    @staticmethod
    def _safe_silhouette_score(
        X: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 1000,
        random_state: int = 0
    ) -> float:
        """
        Computes the silhouette score safely, ensuring cluster diversity and avoiding
        memory issues on large datasets.

        Returns:
            float: Silhouette score, or 0.0 if not computable.
        """
        if len(np.unique(labels)) < 2:
            return 0.0

        try:
            if X.shape[0] > sample_size:
                return silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
            return silhouette_score(X, labels)
        except Exception:
            return 0.0

    def evaluate_kmeans(
        self,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = None
    ):
        """
        Evaluates KMeans clustering over a range of cluster counts.

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
       

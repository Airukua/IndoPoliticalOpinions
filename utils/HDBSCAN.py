import numpy as np
import hdbscan

class HDBSCANClusterer:
    """
    Wrapper class for HDBSCAN clustering algorithm.

    Parameters
    ----------
    min_cluster_size : int, optional
        The minimum size of clusters; single linkage splits that contain fewer points than this will be
        considered points "falling out" of a cluster rather than a cluster splitting into two new clusters.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered a core point.
    cluster_selection_epsilon : float, default=0.0
        A distance threshold for cluster selection. Clusters below this threshold are merged.
    """

    def __init__(self, min_cluster_size=None, min_samples=None, cluster_selection_epsilon=0.0):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = None

    def fit(self, data):
        """
        Fit the HDBSCAN model to the data.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        model : hdbscan.HDBSCAN
            The fitted HDBSCAN model instance.
        """
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon
        ).fit(data)
        return self.model

    def evaluate(self):
        """
        Evaluate clustering results and return cluster statistics.

        Returns
        -------
        dict
            Contains number of clusters, number of noise points,
            cluster sizes, and label assignments.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'evaluate'.")

        labels = self.model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        cluster_sizes = {
            label: np.sum(labels == label)
            for label in set(labels) if label != -1
        }

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print("Cluster sizes:")
        for cluster_label, size in cluster_sizes.items():
            print(f"  Cluster {cluster_label}: {size} points")

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": cluster_sizes,
            "labels": self.model.labels_
        }

    def hyperparameter_search(self, data, min_cluster_sizes, min_samples_list):
        """
        Perform a simple grid search over min_cluster_size and min_samples parameters.
        Evaluation is based on maximizing the number of clusters and minimizing noise.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to perform clustering on.
        min_cluster_sizes : list of int
            List of candidate values for min_cluster_size.
        min_samples_list : list of int
            List of candidate values for min_samples.

        Returns
        -------
        dict
            Best parameter combination found and corresponding cluster statistics.
        """
        best_result = None

        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                try:
                    model = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=self.cluster_selection_epsilon
                    ).fit(data)

                    labels = model.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = np.sum(labels == -1)

                    print(f"min_cluster_size={min_cluster_size}, min_samples={min_samples} => "
                          f"{n_clusters} clusters, {n_noise} noise points")

                    if best_result is None or \
                       n_clusters > best_result['n_clusters'] or \
                       (n_clusters == best_result['n_clusters'] and n_noise < best_result['n_noise']):
                        best_result = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise
                        }
                except Exception as e:
                    print(f"Error with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}")

        print(f"\nBest Parameters: {best_result}")
        return best_result
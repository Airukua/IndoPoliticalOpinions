import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class ClusterPostProcessor:
    """
    Full post-processing pipeline for KMeans clustering:
    - TF-IDF vectorization
    - Fit KMeans
    - Assign cluster labels
    - Split into clusters
    - Save each cluster to CSV
    """

    def __init__(self, best_k: int, df: pd.DataFrame, text_column: str = 'text', output_dir: str = 'clusters'):
        self.best_k = best_k
        self.df = df.copy()
        self.text_column = text_column
        self.output_dir = output_dir
        self.vectorizer = TfidfVectorizer()
        self.X = None
        self.kmeans = None
        self.clustered_df = None
        self.cluster_groups = None

    def run(self):
        """
        Executes the full clustering pipeline and saves the results.

        Returns:
            dict[int, pd.DataFrame]: Dictionary of cluster-separated DataFrames.
        """
        self._vectorize_text()
        self._fit_kmeans()
        self._assign_labels()
        self._split_by_cluster()
        self._save_clusters_to_csv()
        return self.cluster_groups

    def _vectorize_text(self):
        texts = self.df[self.text_column].astype(str).tolist()
        self.X = self.vectorizer.fit_transform(texts)

    def _fit_kmeans(self):
        self.kmeans = KMeans(n_clusters=self.best_k, random_state=42)
        self.kmeans.fit(self.X)

    def _assign_labels(self):
        self.df['cluster'] = self.kmeans.labels_
        self.clustered_df = self.df

    def _split_by_cluster(self):
        self.cluster_groups = {
            cluster: group for cluster, group in self.clustered_df.groupby('cluster')
        }

    def _save_clusters_to_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for cluster_id, group_df in self.cluster_groups.items():
            filename = os.path.join(self.output_dir, f'cluster_{cluster_id}.csv')
            group_df.to_csv(filename, index=False)

    
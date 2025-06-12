# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
from utils.dimention_reduction import reduce_dimensions_umap
from matplotlib import pyplot as plt

class ClusterPostProcessor:
    """
    Full post-processing pipeline for KMeans clustering:
    - Sentence BERT vectorization
    - Dimensionality reduction with UMAP
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
        self.vectorizer = SentenceTransformer('firqaaa/indo-sentence-bert-base')  
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
        self.X = self.vectorizer.encode(texts, show_progress_bar=True)
        self.X = reduce_dimensions_umap(self.X, metric='cosine')

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

    def plot_clusters(self):
        if self.X is None or self.kmeans is None:
            raise RuntimeError("Run the clustering pipeline first.")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.kmeans.labels_, cmap='tab10', s=50)
        plt.title('UMAP projection with KMeans Clusters')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()
    

class ClusterKeywordExtractor:
    """
    Extracts top keywords from text data in clustered CSV files and summarizes them in an output CSV.
    """

    def __init__(self, cluster_folder: str, text_column: str = 'text', top_n: int = 10, output_file: str = 'top_keywords_per_cluster.csv'):
        """
        Initializes the extractor with the directory of cluster CSV files and configuration.

        Args:
            cluster_folder (str): Path to the folder containing clustered CSV files.
            text_column (str): Name of the text column to analyze.
            top_n (int): Number of top keywords to extract per cluster.
            output_file (str): Path for saving the output CSV summary.
        """
        self.cluster_folder = cluster_folder
        self.text_column = text_column
        self.top_n = top_n
        self.output_file = output_file

    def _extract_top_keywords(self, texts: pd.Series) -> list[str]:
        """
        Extracts the top N most frequent words from a pandas Series of text.

        Args:
            texts (pd.Series): Series of textual data.

        Returns:
            List of top N words (without frequency).
        """
        combined_text = ' '.join(texts.astype(str)).lower()
        tokens = re.findall(r'\b\w+\b', combined_text)
        frequency = Counter(tokens)
        return [word for word, _ in frequency.most_common(self.top_n)]

    def extract_and_save(self) -> None:
        """
        Extracts keywords from each cluster file and writes the results to a CSV summary.
        """
        summary_data = []

        for filename in sorted(os.listdir(self.cluster_folder)):
            if not (filename.endswith('.csv') and filename.startswith('cluster_')):
                continue

            cluster_id = self._get_cluster_id(filename)
            filepath = os.path.join(self.cluster_folder, filename)
            df = pd.read_csv(filepath)

            if self.text_column not in df.columns:
                print(f"Skipping '{filename}': Missing column '{self.text_column}'.")
                continue

            top_keywords = self._extract_top_keywords(df[self.text_column])
            summary_data.append(self._format_summary_row(cluster_id, top_keywords))

        pd.DataFrame(summary_data).to_csv(self.output_file, index=False)
        print(f"[✓] Top keywords saved to '{self.output_file}'")

    @staticmethod
    def _get_cluster_id(filename: str) -> int:
        """
        Extracts the cluster ID from a filename.

        Args:
            filename (str): Name of the cluster file.

        Returns:
            Integer cluster ID.
        """
        return int(filename.removeprefix('cluster_').removesuffix('.csv'))

    def _format_summary_row(self, cluster_id: int, keywords: list[str]) -> dict:
        """
        Formats a single row of top keywords for the summary.

        Args:
            cluster_id (int): The ID of the cluster.
            keywords (list): List of keywords.

        Returns:
            Dictionary for one row of the summary DataFrame.
        """
        row = {'cluster': cluster_id}
        for i, word in enumerate(keywords, start=1):
            row[f'keyword_{i}'] = word
        return row

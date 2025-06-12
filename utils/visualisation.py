# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import math
import seaborn as sns

def plot_wordcloud_for_cluster(cluster_df: pd.DataFrame, text_column: str = 'text', cluster_id: int | None = None) -> None:
    """
    Generate and display a word cloud for the specified text column of a DataFrame cluster.

    Args:
        cluster_df (pd.DataFrame): DataFrame containing the clustered data.
        text_column (str, optional): Name of the column containing text data. Defaults to 'text'.
        cluster_id (int or None, optional): ID of the cluster for labeling. Defaults to None.

    Returns:
        None
    """
    # Combine all text entries into a single string
    text_data = cluster_df[text_column].dropna().astype(str)
    combined_text = " ".join(text_data)

    if not combined_text.strip():
        print(f"[INFO] Cluster {cluster_id if cluster_id is not None else '?'}: No text available to generate word cloud.")
        return

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        collocations=False
    ).generate(combined_text)

    # Plot word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    title = f"Word Cloud for Cluster {cluster_id}" if cluster_id is not None else "Word Cloud"
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_cluster_hdbcsan(cluster_df: pd.DataFrame, exclude_cols = ['author', 'text', 'published_at', 'source']) -> None:
    plot_cols = [col for col in cluster_df.columns if col not in exclude_cols]
    n_cols = 2
    n_rows = math.ceil(len(plot_cols) / n_cols)

    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(plot_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        if cluster_df[col].dtype == 'object':
            sns.countplot(y=cluster_df[col], order=cluster_df[col].value_counts().index)
            plt.title(f'Category Distribution {col}')
        else:
            plt.hist(cluster_df[col], bins=30, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    


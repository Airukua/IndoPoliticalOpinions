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

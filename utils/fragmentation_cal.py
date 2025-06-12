# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import re
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TopicFragmentation:
    """
    A class to calculate topic fragmentation in a DataFrame column
    containing lists or strings of topics.
    """

    def __init__(self, dataframe: pd.DataFrame, topic_column: str = 'topics') -> None:
        """
        Initializes the TopicFragmentation class.

        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame containing topic data.
        - topic_column (str): The name of the column containing the topics.
        """
        self.dataframe = dataframe
        self.topic_column = topic_column
        self.topic_words = self._process_topics()

    def _process_topics(self) -> list:
        """
        Cleans and tokenizes the topic text in the specified column.

        Returns:
        - List of lists, where each sublist contains cleaned topic words.
        """

        def clean_topic(text: str) -> list:
            if not isinstance(text, str):
                return []
            text = re.sub(r'Topic', '', text)
            text = re.sub(r'\d+|[^\w\s]', '', text)
            parts = re.split(r'\n', text)
            parts = [re.sub(r'\s+', ' ', part).strip() for part in parts]
            cleaned_text = ' '.join(parts)
            return list(dict.fromkeys(cleaned_text))

        self.dataframe[self.topic_column] = self.dataframe[self.topic_column].apply(clean_topic)
        return self.dataframe[self.topic_column].tolist()

    @staticmethod
    def _jaccard_distance(set1: list, set2: list) -> float:
        """
        Computes the Jaccard distance between two sets of words.

        Parameters:
        - set1 (list): First list of words.
        - set2 (list): Second list of words.

        Returns:
        - Jaccard distance (float)
        """
        set1, set2 = set(set1), set(set2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 0.0 if union == 0 else 1 - intersection / union

    def calculate_fragmentation(self) -> float:
        """
        Calculates the average Jaccard distance between all topic pairs.

        Returns:
        - Average fragmentation score (float)
        """
        distances = [
            self._jaccard_distance(topics1, topics2)
            for topics1, topics2 in combinations(self.topic_words, 2)
        ]
        return 0.0 if not distances else sum(distances) / len(distances)
    
    def plot_jaccard_heatmap(self) -> None:
        """
        Plots a heatmap of the Jaccard distances between topic entries.
        """
        n = len(self.topic_words)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = self._jaccard_distance(self.topic_words[i], self.topic_words[j])

        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    xticklabels=[f'Topic {i}' for i in range(n)],
                    yticklabels=[f'Topic {i}' for i in range(n)])
        plt.title("Jaccard Distance Heatmap Between Topics")
        plt.tight_layout()
        plt.show()
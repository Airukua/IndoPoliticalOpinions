# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import entropy

class FeatureEngineering:
    """
    Performs feature engineering on a DataFrame with textual and temporal data.
    
    Features added:
        - total_sources: Unique sources per author
        - screening_time_min: Time range between first and last post per author (minutes)
        - total_comments: Total number of posts per author
        - source_entropy: Entropy of source distribution per author
    """

    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'published_at',
        source_col: str = 'source',
        author_col: str = 'author'
    ):
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.source_col = source_col
        self.author_col = author_col

    def _compute_source_entropy(self) -> pd.DataFrame:
        """
        Computes entropy of source distribution per author.
        Returns:
            pd.DataFrame: author and source_entropy.
        """
        def calc_entropy(group):
            counts = group[self.source_col].value_counts(normalize=True)
            return entropy(counts)

        entropy_df = (
            self.df.groupby(self.author_col)
            .apply(calc_entropy)
            .reset_index(name='source_entropy')
        )
        return entropy_df

    def add_features(self) -> pd.DataFrame:
        self.df[self.timestamp_col] = pd.to_datetime(
            self.df[self.timestamp_col], errors='coerce', utc=True
        )

        # Total unique sources per author
        author_source_counts = (
            self.df.groupby(self.author_col)[self.source_col]
            .nunique()
            .reset_index(name='total_sources')
        )

        # Total comments per author
        author_total_comments = (
            self.df.groupby(self.author_col)
            .size()
            .reset_index(name='total_comments')
        )

        # Time range per author
        author_time_stats = (
            self.df.groupby(self.author_col)[self.timestamp_col]
            .agg(first_comment='min', last_comment='max')
            .reset_index()
        )

        # Screening time in minutes
        author_time_stats['screening_time_min'] = (
            (author_time_stats['last_comment'] - author_time_stats['first_comment'])
            .dt.total_seconds() / 60
        )

        # Source entropy per author
        entropy_df = self._compute_source_entropy()

        # Merge all features
        self.df = self.df.merge(author_source_counts, on=self.author_col, how='left')
        self.df = self.df.merge(author_total_comments, on=self.author_col, how='left')
        self.df = self.df.merge(
            author_time_stats[[self.author_col, 'screening_time_min']],
            on=self.author_col,
            how='left'
        )
        self.df = self.df.merge(entropy_df, on=self.author_col, how='left')

        return self.df

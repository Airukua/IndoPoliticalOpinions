# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import math
import re
from typing import List, Generator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

class DataLoader:
    """
    DataLoader is responsible for loading, validating, preprocessing,
    and splitting CSV-based datasets for analysis and modeling.
    """

    def __init__(self, file_paths: List[str]):
        """
        Initializes the DataLoader with a list of file paths.

        :param file_paths: List of paths to CSV files.
        """
        self.file_paths = file_paths

    def load_data(self) -> Generator[pd.DataFrame, None, None]:
        """
        Generator that yields DataFrames from the specified file paths.

        :raises ValueError: If any file yields an empty DataFrame.
        :yield: Non-empty DataFrame.
        """
        for file in self.file_paths:
            df = pd.read_csv(file)
            if df.empty:
                raise ValueError(f"DataFrame is empty for file: {file}")
            yield df

    def check_unique_values(self) -> None:
        """
        Plots bar charts showing the number of unique values per column
        for each dataset. Raises an error if any column has fewer than 2 unique values.
        """
        num_files = len(self.file_paths)
        cols = 2
        rows = math.ceil(num_files / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes = axes.flatten()

        for i, (file, ax) in enumerate(zip(self.file_paths, axes)):
            df = pd.read_csv(file)
            if df.empty:
                raise ValueError(f"DataFrame is empty for file: {file}")

            unique_counts = {col: df[col].nunique() for col in df.columns}
            for col, count in unique_counts.items():
                if count < 2:
                    raise ValueError(f"Column '{col}' has less than 2 unique values.")

            data_plot = pd.DataFrame({
                'column': list(unique_counts.keys()),
                'Uniques Values': list(unique_counts.values())
            })

            sns.barplot(data=data_plot, x='column', y='Uniques Values', ax=ax, palette='Set2')
            ax.set_title(file.split('/')[-1])
            ax.set_xlabel('')
            ax.set_ylabel('The number of unique values')
            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_comment_times(self) -> None:
        """
        Plots the number of comments over time using the 'published_at' timestamp column.
        """
        num_files = len(self.file_paths)
        cols = 2
        rows = math.ceil(num_files / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes = axes.flatten()

        for i, (file, ax) in enumerate(zip(self.file_paths, axes)):
            df = pd.read_csv(file)

            if 'published_at' not in df.columns:
                raise ValueError(f"'published_at' column not found in file: {file}")

            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            if df['published_at'].isnull().all():
                raise ValueError(f"All values in 'published_at' could not be parsed in file: {file}")

            df['published_date'] = df['published_at'].dt.date
            comment_counts = df['published_date'].value_counts().sort_index()

            sns.lineplot(x=comment_counts.index, y=comment_counts.values, ax=ax)
            ax.set_title(f"Komentar per Tanggal - {file.split('/')[-1]}")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah Komentar")
            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def data_preprocessor(self, file_path: str, text_col: str) -> pd.DataFrame:
        """
        Preprocesses text data by removing emojis, digits, special characters, and excessive whitespace.

        :param file_path: Path to the CSV file.
        :param text_col: Name of the text column to preprocess.
        :return: DataFrame with cleaned text column.
        :raises ValueError: If the file is not in known files or column does not exist.
        """
        if file_path not in self.file_paths:
            raise ValueError(f"File {file_path} is not in the list of known files.")
        
        stop_words = set(stopwords.words('indonesian'))

        emoji_pattern = re.compile(
            r"["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\u2700-\u27BF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA70-\U0001FAFF"
            u"\u2600-\u26FF"
            u"\u2300-\u23FF"
            r"]+",
            flags=re.UNICODE
        )

        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            raise ValueError(f"'{text_col}' column not found in file: {file_path}")

        df[text_col] = (
            df[text_col].astype(str)
            .str.lower()
            .str.replace(emoji_pattern, '', regex=True)
            .str.replace(r'[\d\W_]+', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
        )

        df[text_col] = df[text_col].apply(
            lambda text: ' '.join([word for word in text.split() if word not in stop_words])
        )
        return df

    def data_splitter(
        self,
        df: pd.DataFrame,
        text_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Splits the dataset into training, validation, and test sets.

        :param df: DataFrame containing the data.
        :param text_col: Column to use for splitting.
        :param test_size: Proportion for the test set.
        :param val_size: Proportion for the validation set (from the remaining after test split).
        :param random_state: Seed for reproducibility.
        :return: Tuple of (train_df, val_df, test_df)
        :raises ValueError: If text_col is not found in df.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        train_val_df, test_df = train_test_split(
            df[[text_col]],
            test_size=test_size,
            random_state=random_state
        )

        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state
        )

        print(f"Train set: {len(train_df)} rows")
        print(f"Validation set: {len(val_df)} rows")
        print(f"Test set: {len(test_df)} rows")

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )
    



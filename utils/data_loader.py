import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from typing import List, Generator

class DataLoader:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def load_data(self) -> Generator[pd.DataFrame, None, None]:
        for file in self.file_paths:
            df = pd.read_csv(file)
            if df.empty:
                raise ValueError(f"DataFrame is empty for file: {file}")
            yield df

    def check_unique_values(self) -> None:
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
            ax.set_title(f"{file.split('/')[-1]}")
            ax.set_xlabel('')
            ax.set_ylabel('The number of unique values')
            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_comment_times(self) -> None:
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

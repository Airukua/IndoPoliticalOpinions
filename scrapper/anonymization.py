# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import pandas as pd
from typing import List, Dict

def build_global_author_mapping(file_paths: List[str]) -> Dict[str, str]:
    """
    Collects all unique authors from the provided list of CSV files
    and assigns each a global anonymized label (e.g., user_1, user_2, etc.).

    Args:
        file_paths (List[str]): A list of paths to CSV files containing an 'author' column.

    Returns:
        Dict[str, str]: A mapping from original author names to anonymized user labels.
    """
    unique_authors = set()

    for path in file_paths:
        df = pd.read_csv(path)
        authors = df['author'].dropna().unique()
        unique_authors.update(authors)

    return {author: f"user_{i+1}" for i, author in enumerate(sorted(unique_authors))}

def apply_anonymization(file_path: str, author_map: Dict[str, str]) -> None:
    """
    Replaces author names in the specified CSV file using the provided anonymization map.

    Args:
        file_path (str): Path to the CSV file to anonymize.
        author_map (Dict[str, str]): Mapping of original author names to anonymized labels.
    """
    df = pd.read_csv(file_path)
    df['author'] = df['author'].map(author_map)
    df.to_csv(file_path, index=False)

def anonymize_authors_globally(file_paths: List[str]) -> Dict[str, str]:
    """
    Orchestrates the global anonymization of authors across multiple CSV files.

    Args:
        file_paths (List[str]): A list of CSV file paths to anonymize.

    Returns:
        Dict[str, str]: The global mapping used for anonymization.
    """
    author_map = build_global_author_mapping(file_paths)

    for path in file_paths:
        apply_anonymization(path, author_map)

    return author_map

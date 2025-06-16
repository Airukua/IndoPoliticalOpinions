from collections import Counter
import pandas as pd

def build_vocabulary(df: pd.DataFrame, text_column: str = 'text'):
    """
    Build a vocabulary from a text column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Name of the column with text. Default is 'text'.

    Returns:
        list: A list of unique words in the vocabulary.
        Counter: A Counter object with word frequencies.
    """
    all_words = []
    for sentence in df[text_column].dropna().astype(str):
        words = sentence.split()
        all_words.extend(words)
    return list(set(all_words)), Counter(all_words)
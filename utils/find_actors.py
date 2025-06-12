from collections import Counter
import numpy as np

def pmi(word1, word2, corpus):
    """
    Calculate the Pointwise Mutual Information (PMI) between two words in a corpus.

    Args:
        word1 (str): First word.
        word2 (str): Second word.
        corpus (list of list of str): Corpus where each document is a list of words.

    Returns:
        float: PMI value between the two words.
    """
    total_docs = len(corpus)
    if total_docs == 0:
        return 0.0
    
    count_word1 = sum(1 for doc in corpus if word1 in doc)
    count_word2 = sum(1 for doc in corpus if word2 in doc)
    count_both = sum(1 for doc in corpus if word1 in doc and word2 in doc)
    if count_word1 == 0 or count_word2 == 0 or count_both == 0:
        return 0.0
    
    p_word1 = count_word1 / total_docs
    p_word2 = count_word2 / total_docs
    p_both = count_both / total_docs
    pmi_value = (p_both / (p_word1 * p_word2)) if (p_word1 * p_word2) > 0 else 0.0
    return np.log2(pmi_value) if pmi_value > 0 else 0.0

def find_key_actors(text, key_actors):
    """
    Identifies key actors mentioned in the given text.

    Args:
        text (str): The text to search within.
        key_actors (dict): A dictionary where keys are actor names and values are lists of aliases.

    Returns:
        list: A list of key actors found in the text.
    """
    text_lower = text.lower()
    found_actors = []

    for actor, aliases in key_actors.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                found_actors.append(actor)
                break

    return found_actors

def filter_actors(dataframe, key_actors, text_column='text'):
    """
    Filters the DataFrame to find key actors based on their mentions in the text.

    Args:
        dataframe (pd.DataFrame): DataFrame containing a text column.
        key_actors (dict): Dictionary of actors and their aliases.
        text_column (str): Name of the text column in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with an additional 'key_actors' column containing lists of found actors.
    """
    dataframe['key_actors'] = dataframe[text_column].apply(lambda x: find_key_actors(x, key_actors))
    filtered_df = dataframe[dataframe['key_actors'].apply(lambda x: len(x) > 0)]
    return filtered_df
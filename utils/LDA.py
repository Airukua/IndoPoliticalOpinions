# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk.download('punkt')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

class LDAProcessor:
    def __init__(self, df, text_column='text', num_topics=10, passes=10, workers=2, custom_stopwords=None):
        """
        Initializes the LDAProcessor with a DataFrame and parameters for topic modeling.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Column name containing the text to process.
            num_topics (int): Number of topics to extract.
            passes (int): Number of passes through the corpus during training.
            workers (int): Number of parallel workers for training.
        """
        self.df = df
        self.text_column = text_column
        self.num_topics = num_topics
        self.passes = passes
        self.workers = workers
        self.lda_model = None
        self.corpus = None
        self.dictionary = None

        if custom_stopwords is not None:
            if not isinstance(custom_stopwords, list):
                raise ValueError('[INFO] file must be a list')
        
        self.custom_stopwords = custom_stopwords or []

    @staticmethod
    def _cleaning(text):
        """
        Cleans the input text by lowercasing, removing non-ASCII characters,
        punctuation, and extra spaces.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ''

        text = text.lower()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\d+','',text)
        text = re.sub(r'[\W_]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _preprocess_text(self):
        """
        Applies full preprocessing pipeline: cleaning, tokenizing,
        stopword removal, and stemming.

        Returns:
            list: List of tokenized and stemmed texts.
        """
        cleaned = self.df[self.text_column].apply(self._cleaning)
        tokens = cleaned.apply(word_tokenize)
        filtered_custom = tokens.apply(lambda words: [w for w in words if w not in self.custom_stopwords])
        filtered = filtered_custom.apply(lambda words: [t for t in words if t.isalpha() and t not in stop_words])
        stemmed = filtered.apply(lambda words: [stemmer.stem(t) for t in words])
        return stemmed

    def prepare_corpus(self):
        """
        Prepares the corpus and dictionary for LDA training.

        Returns:
            list: Preprocessed texts used to generate corpus.
        """
        processed_texts = self._preprocess_text()
        self.dictionary = corpora.Dictionary(processed_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        return processed_texts

    def train_lda(self):
        """
        Trains the LDA model using the prepared corpus and dictionary.
        """
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Corpus and dictionary are not prepared. Call prepare_corpus() first.")
        
        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            workers=self.workers,
            random_state=42
        )

    def get_topics(self, num_words=10):
        """
        Prints the top keywords for each topic.

        Args:
            num_words (int): Number of top keywords to display per topic.
        """
        if self.lda_model is None:
            raise ValueError("LDA model is not trained. Call train_lda() first.")
        
        topics = self.lda_model.print_topics(num_words=num_words)
        for idx, topic in topics:
            print(f"Topic #{idx}: {topic}")

    def compute_coherence(self, processed_texts=None):
        """
        Computes the coherence score of the trained LDA model.

        Args:
            processed_texts (list, optional): Preprocessed texts. If None, it will preprocess again.

        Returns:
            float: Coherence score (c_v).
        """
        if processed_texts is None:
            processed_texts = self._preprocess_text()
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=processed_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()

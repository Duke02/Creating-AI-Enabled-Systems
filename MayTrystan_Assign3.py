import pandas as pd
import os
import typing as tp
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn


class NlpAnalyzer:
    def __init__(self, data_path: tp.Optional[str] = None):
        """
        Initializes the analyzer.

        :param data_path: The path to the CSV to analyze text from. Must contain a summary
            text column. If None (default), analyzer will assign a default data path. Does not check if file exists at
            path!
        """
        if data_path is None:
            self.data_path: str = os.path.join('.', 'data', 'Assign3', 'Musical_instruments_reviews.csv')
        else:
            self.data_path: str = data_path

        self.data: tp.Optional[pd.DataFrame] = None
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stemmer: PorterStemmer = PorterStemmer()

    def load_data(self) -> pd.DataFrame:
        """
        Loads input data at internal data path set at __init__. This step must be done first before any analysis can be
        performed!
        :return: The loaded data
        """
        self.data: pd.DataFrame = pd.read_csv(self.data_path)
        return self.data

    def tokenize(self, do_punctuation: bool = False) -> pd.Series:
        """

        Tokenizes the summary column in the inputted data.

        :param do_punctuation: True - tokenize alphanumeric text separate from punctuation. False (default), treat punctuation as a word.
        :return: The tokenized data.
        """
        if not do_punctuation:
            self.data['tokenized'] = self.data['summary'].apply(lambda s: nltk.word_tokenize(s))
        else:
            self.data['tokenized'] = self.data['summary'].apply(lambda s: nltk.wordpunct_tokenize(s))

        return self.data['tokenized']

    def stem(self, lower_case: bool = True) -> pd.Series:
        """
        Stems the data using a Porter Stemmer.
        :param lower_case: True (default) - convert summary column to lower case before stemming. False - leave case as is.
        :return:
        """
        if lower_case:
            self.data['stemmed'] = self.data['summary'].apply(lambda s: self.stemmer.stem(s.lower()))
        else:
            self.data['stemmed'] = self.data['summary'].apply(lambda s: self.stemmer.stem(s))
        return self.data['stemmed']

    @staticmethod
    def _convert_to_wordnet_pos(nltk_pos: str) -> tp.Optional[str]:
        pos_converter: tp.Dict[str, str] = {'J': wn.ADJ, 'V': wn.VERB, 'N': wn.NOUN, 'R': wn.ADV}

        # The := is the walrus operator. It was introduced in Python 3.8.
        # Here's some docs on it - https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
        if (nltk_pos_key := nltk_pos[0].upper()) in pos_converter.keys():
            return pos_converter[nltk_pos_key]
        else:
            return None

    def _lemmatize_sentence(self, sentence: str) -> str:
        tokenized: tp.List[str] = nltk.wordpunct_tokenize(sentence)
        nltk_poses: tp.List[tp.Tuple[str, str]] = nltk.pos_tag(tokenized)
        wordnet_poses: tp.List[tp.Tuple[str, str]] = [(token, self._convert_to_wordnet_pos(nltk_pos)) for
                                                      token, nltk_pos in
                                                      nltk_poses]

        lemmatized_sentence: tp.List[str] = [
            self.lemmatizer.lemmatize(token, pos=wn_pos) if wn_pos is not None else token
            for token, wn_pos in wordnet_poses]

        return ' '.join(lemmatized_sentence)

    def lemmatize(self, lower_case: bool = True) -> pd.Series:
        """
        Lemmatizes the summary column in the internal dataframe. Takes longer than stemming but tends to be more accurate.
        :param lower_case: True (default) - Convert summary column to lower case before lemmatizing. False - Leave casing as is.
        :return:
        """
        if lower_case:
            self.data['lemmatized'] = self.data['summary'].apply(lambda s: self._lemmatize_sentence(s.lower()))
        else:
            self.data['lemmatized'] = self.data['summary'].apply(lambda s: self._lemmatize_sentence(s))

        return self.data['lemmatized']

    def print_examples(self, n: int = 10) -> None:
        """
        Print the first n rows of the internal dataframe to console output.

        MUST HAVE APPLIED TOKENIZE, STEM, AND LEMMATIZE FUNCTIONS TO INTERNAL DATA BEFORE EXECUTING THIS.

        :param n: The first number of rows to use as examples. Default is 10.
        """
        example_data: pd.DataFrame = self.data[['summary', 'tokenized', 'stemmed', 'lemmatized']].head(n)

        for i, (summary, tokenized, stemmed, lemmatized) in example_data.iterrows():
            print(f'Row #{i}:')
            print(f'\tSummary: {summary}')
            tokenized_str: str = '", "'.join(tokenized)
            print(f'\tTokenized: ["{tokenized_str}"]')
            print(f'\tStemmed: {stemmed}')
            print(f'\tLemmatized: {lemmatized}')


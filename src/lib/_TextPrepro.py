from itertools import tee
from ._Tokenizer import _Tokenizer


class _TextPrepro(_Tokenizer):
    """

    """
    # punctuation = string.punctuation
    # negation_pattern = r'\b(?:not|never|no|can\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t| didn\'t)\b[\w\s]+[^\w\s]'
    # tknz = nltk.tokenize.TweetTokenizer()
    # lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

    def __init__(self, text):
        """

        :param text: text generator
        """
        self.text = text

    # ----------------- helper methods ------------------------------------

    def get_text(self):
        """
        get text file, keep self.text generator
        :return:
        """
        self.text, text = tee(self.text)
        return text


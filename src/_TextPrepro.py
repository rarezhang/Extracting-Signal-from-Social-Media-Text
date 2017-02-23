import nltk, re, string



class _TextPrepro:
    """

    """
    # punctuation = string.punctuation
    # negation_pattern = r'\b(?:not|never|no|can\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t| didn\'t)\b[\w\s]+[^\w\s]'
    tknz = nltk.tokenize.TweetTokenizer()
    # lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

    def __init__(self, text):
        """

        :param text: text generator
        """
        self.text = text

    # ----------------- helper methods ------------------------------------
    @staticmethod
    def _tokenizer(text_string):  # todo put this into parent class
        """
        tokenize each single tweet
        :param text_string:
        :return: list of token
        """
        return _TextPrepro.tknz.tokenize(text_string)
import nltk

class _Tokenizer:
    """
    """
    tknz = nltk.tokenize.TweetTokenizer()

    @staticmethod
    def _tokenizer(text_string):  # todo put this into parent class
        """
        tokenize each single tweet
        :param text_string:
        :return: list of token
        """
        return _Tokenizer.tknz.tokenize(text_string)
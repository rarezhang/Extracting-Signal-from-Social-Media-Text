import nltk, re
from functools import reduce


class CleanAndFeature:
    """

    """
    tknz = nltk.tokenize.TweetTokenizer()
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

    def __init__(self, text):
        """

        :param text: text generator
        """
        self.text = text

    def _nested_fun(self, funs, value):
        """

        :param funs: list of functions (iterable)
        :param value:
        :return: f1(f2(value))
        """
        return reduce(lambda res, f: f(res), funs, value)

    # ------------------- support methods for clean() -------------------
    def _tokenizer(self, text_string):
        """
        tokenize each single tweet
        :param text_string:
        :return: list of token
        """
        return self.tknz.tokenize(text_string)

    def _convert_lower_case(self, token):
        """
        convert all characters to lower case
        :param token:
        :return:
        """
        return token.lower()

    def _convert_user_name(self, token):
        """
        convert @username to single token  _at_user_
        :param token:
        :return:
        """
        return re.sub('@[^\s]+', '_at_user_', token)

    def _convert_url(self, token):
        """
        convert www.* or https?://* to single token _url_
        :param token:
        :return:
        """
        return re.sub('((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))', '_url_', token)

    def _convert_number(self, token):
        """
        convert all numbers to a single token '_number_'
        :param token:
        :return:
        """
        return re.sub('[^\s]*[\d]+[^\s]*', '_number_', token)

    def _convert_duplicate_characters(self, token):
        """
         remove duplicate characters. e.g., sooo gooooood -> soo good
        :param token:
        :return:
        """
        return re.sub(r'([a-z])\2+', r'\2', token)

    def _lemmatization(self, token):
        """
        stem and lemmatizate a token, nltk can only process 'ascii' codec
        :param token:
        :return:
        """
        try:
            return self.lmtzr.stem(token)
        except:
            return token

    #@load_or_make  # todo
    def clean(self):
        """

        :return:
        """
        funs = (self._convert_lower_case, self._convert_user_name, self._convert_url, self._convert_number, self._lemmatization)
        for ts in self.text:  # ts = text_string
            token_list = self._tokenizer(ts)  # return list
            token_list = [self._nested_fun(funs, tk) for tk in token_list]
            yield ' '.join(token_list)

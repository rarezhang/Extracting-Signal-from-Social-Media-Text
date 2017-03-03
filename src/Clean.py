import nltk, re, string
from utils import nested_fun
from _TextPrepro import _TextPrepro


class Clean(_TextPrepro):
    """

    """
    punctuation = string.punctuation
    negation_pattern = r'\b(?:not|never|no|can\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t| didn\'t)\b[\w\s]+[^\w\s]'
    # tknz = nltk.tokenize.TweetTokenizer()
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)

    # def __init__(self, text):
    #     """
    #
    #     :param text: text generator
    #     """
    #     self.text = text

    # ----------------- helper methods ------------------------------------
    # @staticmethod
    # def _tokenizer(text_string):  # todo put this into parent class
    #     """
    #     tokenize each single tweet
    #     :param text_string:
    #     :return: list of token
    #     """
    #     return Clean.tknz.tokenize(text_string)

    # ------------------- support methods for clean() -------------------
    @staticmethod
    def _convert_user_name(token):
        """
        convert @username to single token  _at_user_
        :param token:
        :return:
        """
        return re.sub('@[^\s]+', '_at_user_', token)

    @staticmethod
    def _convert_url(token):
        """
        convert www.* or https?://* to single token _url_
        :param token:
        :return:
        """
        return re.sub('((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))', '_url_', token)

    @staticmethod
    def _convert_number(token):
        """
        convert all numbers to a single token '_number_'
        :param token:
        :return:
        """
        return re.sub('[^\s]*[\d]+[^\s]*', '_number_', token)

    @staticmethod
    def _convert_duplicate_characters(token):
        """
         remove duplicate characters. e.g., sooo gooooood -> soo good
        :param token:
        :return:
        """
        # return re.search('([a-zA-Z])\\1{2,}', token)
        return re.sub('([a-zA-Z])\\1{2,}', '\\1\\1', token)

    @staticmethod
    def _convert_lemmatization(token):
        """
        stem and lemmatizate a token, nltk can only process 'ascii' codec
        :param token:
        :return:
        """
        try:
            return Clean.lmtzr.stem(token)
        except:
            return token

    @staticmethod
    def _convert_lower_case(text_string):
        """
        convert all characters to lower case
        :param token:
        :return:
        """
        return text_string.lower()

    @staticmethod
    def _convert_negation(text_string):
        """
        defined a negated context as a segment of a text that
        starts with a negation word (e.g., no, shouldn't)
        and ends with one of the punctuation marks: .,:;!?
        add `_not_` prefix to each word following the negation word
        :param str: string
        :return:
        """
        text_string += '.'  # need an end sign for a str
        return re.sub(Clean.negation_pattern, lambda match: re.sub(r'(\s+)(\w+)', r'\1_not_\2', match.group(0)), text_string,
                      flags=re.IGNORECASE)

    @staticmethod
    def _convert_punctuation(text_string):
        """
        remove punctuation
        :param text_string:
        :return:
        """
        return text_string.translate(Clean.translator)

    # todo remove all non-ascii; remove all non-utf8 ?  may not
    def _clean(self, text):
        """
        clean text use token_funs and string_funs
        :return: generator
        """
        # self.function_name and cls.function_name both work
        token_funs = (self._convert_user_name, self._convert_url, self._convert_number,
                      self._convert_duplicate_characters, self._convert_lemmatization)
        # _convert_punctuation must be after _convert_negation
        string_funs = (self._convert_lower_case, self._convert_negation, self._convert_punctuation)
        # text = self.text
        # self.text, text = tee(self.text)  # keep generator
        for ts in text:  # ts = text_string
            token_list = self._tokenizer(ts)  # return list
            token_list = [nested_fun(token_funs, tk) for tk in token_list]
            text_string = ' '.join(token_list)  # return string
            yield nested_fun(string_funs, text_string)

    def clean(self):
        """
        keep the ori
        :return:
        """
        text = self.get_text()
        for ts in self._clean(text):
            yield ts


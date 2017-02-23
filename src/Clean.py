import nltk, re, string
from utils import nested_fun


class Clean:
    """

    """
    punctuation = string.punctuation
    negation_pattern = r'\b(?:not|never|no|can\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t| didn\'t)\b[\w\s]+[^\w\s]'
    tknz = nltk.tokenize.TweetTokenizer()
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

    def __init__(self, text):
        """

        :param text: text generator
        """
        self.text = text

    # ----------------- helper methods ------------------------------------
    @staticmethod
    def _tokenizer(text_string):
        """
        tokenize each single tweet
        :param text_string:
        :return: list of token
        """
        return Clean.tknz.tokenize(text_string)

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


    # todo remove all non-ascii; remove all non-utf8 ?  may not
    def clean(self):
        """
        clean text use token_funs and string_funs
        :return: generator
        """
        token_funs = (Clean._convert_user_name, Clean._convert_url, Clean._convert_number,
                      Clean._convert_duplicate_characters, Clean._convert_lemmatization)
        string_funs = (Clean._convert_lower_case, Clean._convert_negation)
        # text = self.text
        # self.text, text = tee(self.text)  # keep generator
        for ts in self.text:  # ts = text_string
            token_list = self._tokenizer(ts)  # return list
            token_list = [nested_fun(token_funs, tk) for tk in token_list]
            text_string = ' '.join(token_list)  # return string
            yield nested_fun(string_funs, text_string)


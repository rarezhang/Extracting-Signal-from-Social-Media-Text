import nltk, re, string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from ._Tokenizer import _Tokenizer


class _FeatureMake(_Tokenizer):
    """

    """
    punctuation = string.punctuation
    # negation_pattern = r'\b(?:not|never|no|can\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t| didn\'t)\b[\w\s]+[^\w\s]'
    # tknz = nltk.tokenize.TweetTokenizer()
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()


    # ------------------ feature before clean -----------------
    @staticmethod
    def _retweet(token):
        """
        if the token is 'RT'
        :param token:
        :return: True or False
        """
        return 1 if re.fullmatch('RT', token) else 0

    @staticmethod
    def _hashtag(token):
        """
        if the token is a hash-tagged word
        :param token:
        :return:
        """
        return 1 if re.fullmatch('(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', token) else 0

    @staticmethod
    def _upper(token):
        """
        if all the character in the token is in uppercase
        :param token:
        :return:
        """
        return 1 if token.isupper() else 0

    @staticmethod
    def _punctuation(token):
        """
        number of punctuation in the token
        :param token
        :return:
        """
        return sum([1 for tk in token if tk in _FeatureMake.punctuation])

    @staticmethod
    def _elongated(token):
        """
        if the token is an elongated word
        similar to _convert_duplicate_characters
        :param token:
        :return:
        """
        return 1 if re.search("([a-zA-Z])\\1{2,}", token) else 0

    def feature_token_based(self, text):
        """
        token based features: check tokens one by one
        :return:
        """
        feature_token_funs = (self._retweet, self._hashtag, self._upper, self._punctuation, self._elongated)
        # ts = text_string; tk = token
        return np.asarray([[sum([f(tk) for tk in self._tokenizer(ts)]) for f in feature_token_funs] for ts in text])
        # for ts in text:
            # token_list = self._tokenizer(ts)
            # results = []
            # for f in funs:
            #     result = sum([f(tk) for tk in token_list])
            #     results.append(result)
            # yield results  # feature values for a single tweet

    # todo: emoticons


    # ------------------ feature after clean -----------------
    @staticmethod
    def _length_num2cat(length, short_threshold=10, long_threshold=25):
        """
        convert length from numerical to categorical
        :param length: int, number of tokens
        :param short_threshold: <= short_threshold, short text
        :param long_threshold: <= long_threshold, medium text, else long text
        :return:
        """
        if length <= short_threshold:
            return 0  # short text
        elif length <= long_threshold:
            return 1  # medium text
        else:
            return 2  # long text

    # @load_or_make(join_file_path())
    def feature_text_length(self, text):
        """
        the length of tweets,
        :return:
        """
        # text = self.clean()  # return iterable
        result = np.asarray([self._length_num2cat(len(self._tokenizer(ts))) for ts in text])
        return result.reshape((result.shape[0], 1))  # result.shape[0] -> how many rows
        # for ts in text:
        #     token_list = self._tokenizer(ts)
        #     yield len(token_list)  # length of single text

    def _pos_tag(self, text):
        """

        :param text:
        :return:
        """
        # text = self.clean()
        for ts in text:
            token_list = self._tokenizer(ts)  # inherited method from _TextPrepro Class
            yield nltk.pos_tag(token_list)  # return list of tuple [(token, tag), (), ...]

    def feature_pos_count(self, text):
        """
        count number of:
        nouns: pos tag [NN*]
        verbs: pos tag [VB*]
        adj / adv: pos tag [JJ*|RB*]
        :return:
        """
        pos_count = []
        for pos_tagged in self._pos_tag(text):
            NN = sum([1 for _, tag in pos_tagged if tag.startswith('NN')])
            VB = sum([1 for _, tag in pos_tagged if tag.startswith('VB')])
            AA = sum([1 for _, tag in pos_tagged if tag.startswith(('VB', 'RB'))])
            pos_count.append([NN, VB, AA])
        return np.asarray(pos_count)

    # todo parameters tuning: mindf and maxdf
    def feature_pos_ngram(self, text, vb=None, anly='word', mindf=2, maxdf=1.0, ngram=(3,3), stp_w=None):
        """

        :param text:
        :param vb:
        :param anly:
        :param mindf: 0.0 -> 0%, 0.01 -> 1%
        :param maxdf: 0.99 -> 99%, 1.0 -> 100%
        :param ngram:
        :return:
        """
        def _pos_string(text):
            for ts in self._pos_tag(text):
                yield " ".join(["_".join((token, tag)) for token, tag in ts])
        return self.feature_word_ngram(_pos_string(text), vb=vb, anly=anly, mindf=mindf, maxdf=maxdf, ngram=ngram, stp_w=stp_w)

    # todo: test 'char_wb' or 'char'
    def feature_char_ngram(self, text, vb=None, anly='char', mindf=0.05, maxdf=0.99, ngram=(3,5), stp_w=None):
        """
        character n-gram [3-5]
        :param anly: Option ‘char_wb’ creates character n-grams only from text inside word boundaries.
        :return:
        """
        return self.feature_word_ngram(text, vb=vb, anly=anly, mindf=mindf, maxdf=maxdf, ngram=ngram, stp_w=stp_w)

    def feature_word_ngram(self, text, vb=None, anly='word', mindf=2, maxdf=0.99, ngram=(1,4), stp_w='english'):
        """
        - document frequency threshold for CountVectorizer() -> word n-gram [1-4], won't include punctuations
        - mindf and maxdf:
            + If float: proportion of documents, integer: absolute counts.
            + max_df can be set to a value in the range [0.7, 1.0) to automatically detect and
              filter stop words based on intra corpus document frequency of terms.

        :param vb: vocabulary for CountVectorizer()
        :param anly: analyzer {‘word’, ‘char’, ‘char_wb’}, Option ‘char_wb’ creates character n-grams only from text inside word boundaries.
        :param mindf: float in range [0.0, 1.0] or int, default=1
        :param maxdf: float in range [0.0, 1.0] or int default=1.0
        :param: ngram: ngram range for CountVectorizer() -> (1,n), min_n <= n <= max_n will be used
        :return: return term-document matrix and learn the vocabulary dictionary
        """
        # stop_words='english' # tokenizer=self._tokenizer Only applies if analyzer == 'word'.
        count_vec = CountVectorizer(vocabulary=vb, analyzer=anly, min_df=mindf, max_df=maxdf, ngram_range=ngram,
                                    decode_error='ignore', stop_words=stp_w, tokenizer=None)#tokenizer=self._tokenizer
        data = count_vec.fit_transform(text).toarray()
        return data, count_vec.vocabulary_


    # ------------- clustering LDA score -------------
    # todo: topic related-score of each token, separate class


    # ---------------- make and combine features --------------
import nltk, re, string
import numpy as np
from itertools import tee
from sklearn.feature_extraction.text import CountVectorizer
from utils import join_file_path, file_remove, check_file_exist, dump_pickle, load_pickle, nested_fun


class CleanAndFeature:
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
    def _tokenizer(self, text_string):
        """
        tokenize each single tweet
        :param text_string:
        :return: list of token
        """
        return self.tknz.tokenize(text_string)

    # ------------------- support methods for clean() -------------------
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
        # return re.search('([a-zA-Z])\\1{2,}', token)
        return re.sub('([a-zA-Z])\\1{2,}', '\\1\\1', token)

    def _convert_lemmatization(self, token):
        """
        stem and lemmatizate a token, nltk can only process 'ascii' codec
        :param token:
        :return:
        """
        try:
            return self.lmtzr.stem(token)
        except:
            return token

    def _convert_lower_case(self, text_string):
        """
        convert all characters to lower case
        :param token:
        :return:
        """
        return text_string.lower()

    def _convert_negation(self, text_string):
        """
        defined a negated context as a segment of a text that
        starts with a negation word (e.g., no, shouldn't)
        and ends with one of the punctuation marks: .,:;!?
        add `_not_` prefix to each word following the negation word
        :param str: string
        :return:
        """
        text_string += '.'  # need an end sign for a str
        return re.sub(self.negation_pattern, lambda match: re.sub(r'(\s+)(\w+)', r'\1_not_\2', match.group(0)), text_string,
                      flags=re.IGNORECASE)


    # todo remove all non-ascii; remove all non-utf8 ?  may not
    def clean(self):
        """
        clean text use token_funs and string_funs
        :return: generator
        """
        token_funs = (self._convert_user_name, self._convert_url, self._convert_number,
                      self._convert_duplicate_characters, self._convert_lemmatization)
        string_funs = (self._convert_lower_case, self._convert_negation)
        # text = self.text
        self.text, text = tee(self.text)  # keep generator
        for ts in text:  # ts = text_string
            token_list = self._tokenizer(ts)  # return list
            token_list = [nested_fun(token_funs, tk) for tk in token_list]
            text_string = ' '.join(token_list)  # return string
            yield nested_fun(string_funs, text_string)

    # ------------------ feature before clean -----------------
    def _retweet(self, token):
        """
        if the token is 'RT'
        :param token:
        :return: True or False
        """
        return 1 if re.fullmatch('RT', token) else 0

    def _hashtag(self, token):
        """
        if the token is a hash-tagged word
        :param token:
        :return:
        """
        return 1 if re.fullmatch('(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', token) else 0

    def _upper(self, token):
        """
        if all the character in the token is in uppercase
        :param token:
        :return:
        """
        return 1 if token.isupper() else 0

    def _punctuation(self, token):
        """
        number of punctuation in the token
        :param token
        :return:
        """
        return sum([1 for tk in token if tk in self.punctuation])

    def _elongated(self, token):
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

    # todo emoticons


    # ------------------ feature after clean -----------------
    def _length_num2cat(self, length, short_threshold=10, long_threshold=25):
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
            token_list = self._tokenizer(ts)
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

    def feature_pos_ngram(self, text, vb=None, anly='word', mindf=0.05, maxdf=0.99, ngram=(3,3)): # todo parameters tuning
        """

        :param text:
        :param vb:
        :param anly:
        :param mindf: 5%
        :param maxdf: 99%
        :param ngram:
        :return:
        """
        def _pos_string(text):
            for ts in self._pos_tag(text):
                yield " ".join(["_".join((token, tag)) for token, tag in ts])
        return self.feature_word_ngram(_pos_string(text), vb=vb, anly=anly, mindf=mindf, maxdf=maxdf, ngram=ngram)

    def feature_char_ngram(self, text, vb=None, anly='char_wb', mindf=1, maxdf=1.0, ngram=(3,5)):
        """
        character n-gram [3-5]
        :param anly: Option ‘char_wb’ creates character n-grams only from text inside word boundaries.
        :return:
        """
        return self.feature_word_ngram(text, vb=vb, anly=anly, mindf=mindf, maxdf=maxdf, ngram=ngram)

    def feature_word_ngram(self, text, vb=None, anly='word', mindf=1, maxdf=1.0, ngram=(1,4)):
        """
        word n-gram [1-4], won't include punctuations
        :param vb: vocabulary for CountVectorizer()
        :param anly: analyzer {‘word’, ‘char’, ‘char_wb’}, Option ‘char_wb’ creates character n-grams only from text inside word boundaries.
        :param mindf: document frequency threshold for CountVectorizer() ->
        :param maxdf: If float: proportion of documents, integer: absolute counts.
        :param: ngram: ngram range for CountVectorizer() -> (1,n), min_n <= n <= max_n will be used
        :return: return term-document matrix and learn the vocabulary dictionary
        """
        # text = self.clean()  # return iterable
        count_vec = CountVectorizer(vocabulary=vb, analyzer=anly, min_df=mindf, max_df=maxdf, ngram_range=ngram, decode_error='ignore', stop_words='english')
        data = count_vec.fit_transform(text).toarray()
        return data, count_vec.vocabulary_


    # ------------- clustering LDA score -------------
    # todo: topic related-score of each token


    # ---------------- make and combine features --------------
    feature_funs = (feature_token_based, feature_text_length,
                    feature_pos_count, feature_pos_ngram,
                    feature_char_ngram, feature_word_ngram)

    # todo: way to tuning feature parameters
    def make_feature(self, path_feature, remake=False, training_set=True):
        """

        :return:
        """
        for f in self.feature_funs:
            feature_name = f.__name__
            feature_path = join_file_path(path_feature, feature_name)

            if remake: file_remove(feature_path)
            if not check_file_exist(feature_path):
                print("dumping feature: {}....".format(feature_name))
                # text: before clean or after clean
                self.text, text = tee(self.text)
                text = text if feature_name == 'feature_token_based' else self.clean()  # generators
                # make feature
                if 'ngram' in feature_name:
                    vocabulary_name = ''.join((feature_name, '_vocabulary'))
                    vocabulary_path = join_file_path(path_feature, vocabulary_name)
                    if training_set:  # dump vocabulary
                        feature, vocabulary = f(self, text)
                        dump_pickle(vocabulary_path, vocabulary)
                    else:
                        vocabulary = load_pickle(vocabulary_path)
                        feature, _ = f(self, text, vb=vocabulary)
                else:
                    feature = f(self, text)
                dump_pickle(feature_path, feature)

    def combine_feature(self, path_feature, feature_type='ALL'):
        """
        load feature and combine features  -> choose feature types
        :param path_feature:
        :param feature_type: 'ALL': all feature type; []: list of string
        :return:
        """
        assert (feature_type == 'ALL') or (isinstance(feature_type, list) and len(feature_type)>0)
        self.make_feature(path_feature)  # make sure there are features can be loaded later

        result = None
        feature_list = [f.__name__ for f in self.feature_funs]

        # filter features based on feature type
        if feature_type is not 'ALL':
            feature_list = set(sum([[fea for fea in feature_list if type in fea] for type in feature_type], []))
        print("will return the concatenate features: {} ".format("\n".join(feature_list)))
        # create feature path
        for feature_name in feature_list:
            print("loading feature: {}".format(feature_name))
            feature_path = join_file_path(path_feature, feature_name)
            # load feature
            feature = load_pickle(feature_path)
            print("{} shape: {}".format(feature_name, feature.shape))
            # combine feature
            if result is None:
                result = feature
            else:  # concatenate features
                result = np.concatenate((result, feature), axis=1)  # column wise
        return result





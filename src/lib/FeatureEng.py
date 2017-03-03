import nltk, re, string
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
# local
from .Clean import Clean
from .utils import pop_var, load_or_make, join_file_path, file_remove, check_file_exist, check_make_dir, dump_pickle, load_pickle, nested_fun


dataset = pop_var()

path = join_file_path('../data/feature_engineering/', dataset)
# general_path_model = join_file_path(path, 'model')
general_path_feature = join_file_path(path, 'feature')
check_make_dir(general_path_feature)
# check_make_dir(general_path_model)
# assert general_path_feature is not None and general_path_model is not None
# print(f'path to model: {general_path_model}')
print(f'path to feature: {general_path_feature}')


class FeatureEng(Clean):
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
        return sum([1 for tk in token if tk in FeatureEng.punctuation])

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
    feature_funs = (feature_token_based, feature_text_length, feature_pos_count,
                    feature_pos_ngram, feature_char_ngram, feature_word_ngram, )

    # todo: way to tuning feature parameters
    def make_feature(self, remake=False, default_vocabulary=True):
        """

        :return:
        """
        # check_make_dir(path_feature)  # check and make directory

        for f in self.feature_funs:
            feature_name = f.__name__
            feature_path = join_file_path(general_path_feature, feature_name)

            if remake: file_remove(feature_path)
            if not check_file_exist(feature_path):
                print("dumping feature: {}....".format(feature_name))
                # text: before clean or after clean
                text = self.get_text()  # inherit from _TextPrepro
                text = text if feature_name == 'feature_token_based' else self.clean()  # generators
                # make feature
                if 'ngram' in feature_name:
                    vocabulary_name = ''.join((feature_name, '_vocabulary'))
                    vocabulary_path = join_file_path(general_path_feature, vocabulary_name)
                    if default_vocabulary:  # dump vocabulary
                        feature, vocabulary = f(self, text)
                        dump_pickle(vocabulary_path, vocabulary)
                    else:
                        vocabulary = load_pickle(vocabulary_path)
                        feature, _ = f(self, text, vb=vocabulary)
                else:
                    feature = f(self, text)
                dump_pickle(feature_path, coo_matrix(feature))

    # @load_or_make(join_file_path(general_path_feature, 'doc.feature'))
    def _feature_engineering(self, feature_type='ALL', default_vocabulary=True):
        """
        load feature and combine features  -> choose feature types
        :param path_feature:
        :param feature_type: 'ALL': all feature type; []: list of string
        :return:
        """
        assert (feature_type == 'ALL') or (isinstance(feature_type, list) and len(feature_type)>0)
        self.make_feature(default_vocabulary=default_vocabulary)  # make sure there are features can be loaded later

        result = None
        feature_list = [f.__name__ for f in self.feature_funs]

        # filter features based on feature type
        if feature_type is not 'ALL':
            feature_list = set(sum([[fea for fea in feature_list if type in fea] for type in feature_type], []))
        print("will return the concatenate features: {} ".format("\n".join(feature_list)))
        # create feature path
        for feature_name in feature_list:
            print("loading feature: {}".format(feature_name))
            feature_path = join_file_path(general_path_feature, feature_name)
            # load feature
            feature = load_pickle(feature_path)
            print("{} shape: {}".format(feature_name, feature.shape))
            # combine feature
            # result = feature if result is None else np.concatenate((result, feature), axis=1)
            result = feature if result is None else hstack((result, feature))
            # if result is None:
            #     result = feature
            # else:  # concatenate features
            #     result = np.concatenate((result, feature), axis=1)  # column wise
        return coo_matrix(result)
        # Advantages of the COO format
            # facilitates fast conversion among sparse formats
            # permits duplicate entries (see example)
            # very fast conversion to and from CSR/CSC formats
        # Disadvantages of the COO format
            # does not directly support:
            # arithmetic operations
            # slicing

    @load_or_make(join_file_path(general_path_feature, 'doc.feature'))
    def feature_engineering(self, feature_type='ALL', default_vocabulary=True):
        return self._feature_engineering(feature_type=feature_type, default_vocabulary=default_vocabulary)
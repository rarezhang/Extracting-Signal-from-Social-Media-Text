"""

"""
import logging
import numpy as np
from scipy.sparse import coo_matrix
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# local
from .Clean import Clean
from .utils import pop_var, load_or_make, check_make_dir, check_file_exist, dump_pickle, load_pickle, join_file_path, files_remove

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataset = pop_var()
path = join_file_path('../data/word_embedding/', dataset)
general_path_model = join_file_path(path, 'model')
general_path_feature = join_file_path(path, 'feature')
check_make_dir(general_path_feature);check_make_dir(general_path_model)
# assert general_path_feature is not None and general_path_model is not None
print(f'path to model: {general_path_model}')
print(f'path to feature: {general_path_feature}')


class FeatureVec(Clean):
    """

    """
    def __init__(self, text, label, rebuild=False):
        """

        :param text: training data
        :param label:
        :param path_model:
        :param path_feature:
        """
        Clean.__init__(self, text)
        self.label = label
        self.w2v_model = None
        self.lr_model = None
        self.idf_model = None
        self.vector_size = 10  # todo add to argument
        if rebuild:
            files_remove(general_path_feature)
            files_remove(general_path_model)
        self.train_word2vector_model()
        self.train_idf_model()

    def _sentences(self):
        """
        return input file for word2vector model
        :return:
        """
        text = self.get_text()
        # gensim wants to iterate over the data multiple times
        # so generator doesn't work
        return [self._tokenizer(ts) for ts in text]

    def train_word2vector_model(self):  # todo: add parameters tuning
        """

        :param path_model: path to model
        :return:
        """
        # path_model = join_file_path(self.path_model, 'w2v.model')
        path_model = join_file_path(general_path_model, 'w2v.model')

        if not check_file_exist(path_model):
            # sg: sg=0 --> CBOW; sg=1 --> skip-gram
            # hs = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and negative is non-zero, negative sampling will be used.
            # negative = if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). Default is 5. If set to 0, no negative sampling is used.
            # min_count:  ignore all words with total frequency lower than this. default=5
            # window: the maximum distance between the current and predicted word within a sentence.
            # size: size of the NN layers, degrees of freedom, default=100
            # workers: how many workers, need Cython, default=1
            # iter = number of iterations (epochs) over the corpus. Default is 5.
            # todo: parameter tuning
            # Initialize the model from an iterable of sentences. Each sentence is a list of words
            # (unicode strings) that will be used for training.
            self.w2v_model = Word2Vec(self._sentences(), sg=1, min_count=1, window=5, size=self.vector_size, workers=4)
            # storing model
            self.w2v_model.save(path_model)
        else:
            self.w2v_model = Word2Vec.load(path_model)

        # todo: another function
        # if resuming:  # load a model and continue training it with more sentences
        #     self.w2v_model.train(text)
        #     self.w2v_model.save(path_model)

    def train_idf_model(self):
        """

        :return:
        """
        path_model = join_file_path(general_path_model, 'tfidf.model')
        if not check_file_exist(path_model):
            text = self.get_text()
            self.idf_model = TfidfVectorizer(tokenizer=self._tokenizer)
            # tfidf_vec = TfidfVectorizer(tokenizer=None)
            self.idf_model.fit(text)
            dump_pickle(path_model, self.idf_model)
        else:
            self.idf_model = load_pickle(path_model)

    # -------------------------------------------------------------
    def _word_vector(self, word):
        """

        :param model:
        :param word:
        :return: vector of word
        """
        try:
            vector = self.w2v_model[word]
        except KeyError:
            vector = np.zeros(shape=(1, self.vector_size), dtype='float32')
        return vector

    def _logistic_regression(self, X, y):
        """

        :return: coefficient
        """
        self.lr_model = LogisticRegression(fit_intercept=True, C=1.0, penalty='l2', tol=0.0001, multi_class='multinomial', solver='newton-cg')
        self.lr_model.fit(X, y)

    @load_or_make(path=join_file_path(general_path_feature, 'token_vector_label_pair'))
    def _token_vector_label_pair(self):
        """

        :return:
        """
        # path_tvlp = join_file_path(self.path_feature, 'token_vector_label_pair')
        # if not check_file_exist(path_tvlp):
        text = self.get_text()  # inherited from _TextPrepro
        tvlp = []
        for y, ts in zip(self.label, text):
            for tk in self._tokenizer(ts):
                vector = np.reshape(self._word_vector(tk), newshape=(1, self.vector_size))
                tvlp.append((tk, vector, y))
        return tvlp

    @load_or_make(path=join_file_path(general_path_feature, 'X_Y'))
    def _make_word_vector(self):
        """
        for logistic regression
        :return:
        """
        # path_X = join_file_path(self.path_feature, 'X')
        # path_Y = join_file_path(self.path_feature, 'Y')
        # if not (check_file_exist(path_X) and check_file_exist(path_Y)):
        X, Y = None, []
        tvlp = self._token_vector_label_pair()
        for _, x, y in tvlp:
            Y.append(y)
            X = x if X is None else np.concatenate((X,x), axis=0)
        return X, Y

    @load_or_make(path=join_file_path(general_path_feature, 'X_adjust'))
    def _make_word_vector_adjust(self):
        """

        :return:
        """
        X, Y = self._make_word_vector()
        self._logistic_regression(X, Y)
        dist = np.diag(self.lr_model.decision_function(X))
        X_adjust = np.dot(dist, X)
        return X_adjust

    @load_or_make(join_file_path(general_path_feature, 'vec.dic'))
    def _word_adjust_vector(self):
        """

        :param path_vector_feature:
        :return:
        """
        X_adjust = self._make_word_vector_adjust()
        tvlp = self._token_vector_label_pair()
        vec_dic = {}
        for tk, _, _ in tvlp:
            for adj_vec in X_adjust:
                if tk in vec_dic:
                    vec_dic[tk] = np.add(vec_dic[tk], adj_vec)/2
                else:
                    vec_dic[tk] = adj_vec
        return vec_dic

    @load_or_make(path=join_file_path(general_path_feature, 'idf.dic'))
    def _word_idf_score(self):
        """

        :return:
        """
        idf_score = np.log(self.idf_model.idf_)
        vb = self.idf_model.vocabulary_
        idf_dic = {tk: idf_score[vb[tk]] for tk in vb}
        return idf_dic

    # -------------------------------------------------------------
    @load_or_make(path=join_file_path(general_path_feature, 'doc.feature'))
    def feature_embedding(self, text):
        """

        :param text: testing data
        :return:
        """
        vec_dic = self._word_adjust_vector()
        idf_dic = self._word_idf_score()
        document_vec = None
        for ts in text:
            # sentence feature
            sentence_vec = np.empty((1, self.vector_size))
            for tk in self._tokenizer(ts):
                vec = vec_dic.get(tk, np.zeros((1, self.vector_size)))
                idf = idf_dic.get(tk, 0)
                vec = vec * idf
                sentence_vec += vec
            # document feature
            document_vec = sentence_vec if document_vec is None else np.concatenate((document_vec, sentence_vec), axis=0)
        return coo_matrix(document_vec)







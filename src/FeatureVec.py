"""

"""
import gensim, logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from Clean import Clean
from utils import check_file_exist, dump_pickle, load_pickle, join_file_path


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class FeatureVec(Clean):
    """

    """
    def __init__(self, text, label, path_model, path_feature):
        """

        """
        Clean.__init__(self, text)
        self.label = label
        self.w2v_model = None
        self.lr_model = None
        self.vector_size = 10
        self.path_model = path_model
        self.path_feature = path_feature


    def word2vector_model(self, rebuild=False, resuming=False):
        """

        :param path_model: path to model
        :return:
        """
        path_model = join_file_path(self.path_model, 'w2v.model')
        text = self.get_text()
        # text = self.clean()  # todo: use clean text
        if rebuild or not check_file_exist(path_model):
            # min_count: cutoff frequency, default=5
            # size: size of the NN layers, degrees of freedom, default=100
            # workers: how many workers, need Cython, default=1
            # text = self.clean()
            self.w2v_model = gensim.models.Word2Vec(text, min_count=1, size=self.vector_size, workers=4)
            # storing model
            self.w2v_model.save(path_model)
        else:
            self.w2v_model = gensim.models.Word2Vec.load(path_model)
        if resuming:  # load a model and continue training it with more sentences
            self.w2v_model.train(text)
            self.w2v_model.save(path_model)
        # return model


    def _word_vector(self, word):  # may do not need this method
        """

        :param model:
        :param word:
        :return: vector of word
        """
        try:
            vector = self.model[word]
        except KeyError:
            vector = np.zeros(shape=(1, self.vector_size), dtype='float32')
        return vector

    def _logistic_regression(self, X, y):
        """

        :return: coefficient
        """
        path_model = join_file_path(self.path_model, 'log_regression.model')
        if not check_file_exist(path_model):
            self.lr_model = LogisticRegression(fit_intercept=True, C=1.0, penalty='l2', tol=0.0001, multi_class='multinomial', solver='newton-cg')
            self.lr_model.fit(X, y)
            dump_pickle(path_model, self.lr_model)
        else:
            self.lr_model = load_pickle(path_model)


    def _label_token_pair(self):
        """

        :return:
        """
        text = self.get_text()  # inherited from _TextPrepro
        for y, ts in zip(self.label, text):
            for tk in self._tokenizer(ts):
                # yield y, tk, self._word_vector(tk)
                vector = np.reshape(self._word_vector(tk), newshape=(1, self.vector_size))
                yield vector, y

    def _make_vec_feature(self):
        """

        :return:
        """
        path_X = join_file_path(self.path_feature, 'X')
        path_Y = join_file_path(self.path_feature, 'Y')
        if not (check_file_exist(path_X) and check_file_exist(path_Y)):
            X, Y = None, []
            ltp = self._label_token_pair()  # ltp: label_token_pair
            for x, y in ltp:
                Y.append(y)
                if X is None:
                    X = x
                else:
                    X = np.concatenate((X,x), axis=0) # row wise
            dump_pickle(path_X, X)
            dump_pickle(path_Y, Y)
        else:
            X = load_pickle(path_X)
            Y = load_pickle(path_Y)
        return X, Y

    def adjust_vec_feature(self):
        """

        :param path_vector_feature:
        :return:
        """
        path_X_adjust = join_file_path(self.path_feature, 'X_adjust')
        if not check_file_exist(path_X_adjust):

            X, Y = self._make_vec_feature()
            self._logistic_regression(X, Y)
            # W, b = self.lr_model.coef_, self.lr_model.intercept_
            c = self.lr_model.classes_  # todo: need this information
            X_adjust = self.lr_model.predict_proba(X)
            dump_pickle(path_X_adjust, X_adjust)
        else:
            X_adjust = load_pickle(path_X_adjust)
        return X_adjust

    def idf_score(self):
        """

        :return:
        """
        path_model = join_file_path(self.path_model, 'tfidf.model')
        if not check_file_exist(path_model):
            text = self.get_text()
            tfidf_vec = TfidfVectorizer()
            tfidf_vec.fit(text)
            dump_pickle(path_model, tfidf_vec)
            # print(type(tfidf_vec))
        else:
            tfidf_vec = load_pickle(path_model)
        idf_score = tfidf_vec.idf_
        vb = tfidf_vec.vocabulary_

        dic = {tk: idf_score[vb[tk]] for tk in vb}
        print(dic)  # todo dump this dictionay
        # return tfidf_vec.vocabulary_, tfidf_vec.idf_


    def test(self):  # todo remove
        text = self.get_text()
        for t in text:
            print(t)

"""

"""
import gensim, logging
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from Clean import Clean
from utils import check_file_exist, dump_pickle, load_pickle, join_file_path, files_remove

# todo: decorator to dump return files 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class FeatureVec(Clean):
    """

    """
    def __init__(self, text, label, path_model, path_feature, rebuild=False):
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
        self.vector_size = 10
        self.path_model = path_model
        self.path_feature = path_feature
        if rebuild:
            files_remove(self.path_feature); files_remove(self.path_model)
        # train word2vector model
        self.word2vector_model()

    def sentences(self):
        """
        return input file for word2vector model
        :return:
        """
        text = self.get_text()
        # gensim wants to iterate over the data multiple times
        # so generator doesn't work
        return [self._tokenizer(ts) for ts in text]

    def word2vector_model(self):
        """

        :param path_model: path to model
        :return:
        """
        path_model = join_file_path(self.path_model, 'w2v.model')

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
            self.w2v_model = Word2Vec(self.sentences(), sg=1, min_count=1, window=5, size=self.vector_size, workers=4)
            # storing model
            self.w2v_model.save(path_model)
        else:
            self.w2v_model = gensim.models.Word2Vec.load(path_model)

        # todo: another function
        # if resuming:  # load a model and continue training it with more sentences
        #     self.w2v_model.train(text)
        #     self.w2v_model.save(path_model)



    def _word_vector(self, word):  # may do not need this method
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

    def _token_vector_label_pair(self):
        """

        :return:
        """
        path_tvlp = join_file_path(self.path_feature, 'token_vector_label_pair')
        if not check_file_exist(path_tvlp):
            text = self.get_text()  # inherited from _TextPrepro
            tvlp = []
            for y, ts in zip(self.label, text):
                for tk in self._tokenizer(ts):
                    # yield y, tk, self._word_vector(tk)
                    vector = np.reshape(self._word_vector(tk), newshape=(1, self.vector_size))
                    tvlp.append((tk, vector, y))
            dump_pickle(path_tvlp, tvlp)
        else:
            tvlp = load_pickle(path_tvlp)
        return tvlp

    def _make_word_vector(self):
        """
        for logistic regression
        :return:
        """
        path_X = join_file_path(self.path_feature, 'X')
        path_Y = join_file_path(self.path_feature, 'Y')
        if not (check_file_exist(path_X) and check_file_exist(path_Y)):
            X, Y = None, []
            tvlp = self._token_vector_label_pair()  # ltp: generator
            for _, x, y in tvlp:
                Y.append(y)
                X = x if X is None else np.concatenate((X,x), axis=0)
                # if X is None:
                #     X = x
                # else:
                #     X = np.concatenate((X,x), axis=0) # row wise
            dump_pickle(path_X, X)
            dump_pickle(path_Y, Y)
        else:
            X = load_pickle(path_X)
            Y = load_pickle(path_Y)
        return X, Y

    def _word_adjust_vector(self):
        """

        :param path_vector_feature:
        :return:
        """
        path_X_adjust = join_file_path(self.path_feature, 'X_adjust')
        path_vec_dic = join_file_path(self.path_feature, 'vec.dic')
        if not check_file_exist(path_X_adjust):

            X, Y = self._make_word_vector()
            self._logistic_regression(X, Y)

            # The confidence score for a sample is the signed distance of that sample to the hyperplane.
            dist = np.diag(self.lr_model.decision_function(X))
            X_adjust = np.dot(dist, X)
            dump_pickle(path_X_adjust, X_adjust)
        else:
            X_adjust = load_pickle(path_X_adjust)

        if not check_file_exist(path_vec_dic):
            tvlp = self._token_vector_label_pair()  # ltp: generator
            # vec_dic = {tk: adj_vec for tk,_,_ in ltp for adj_vec in X_adjust}
            vec_dic = {}
            for tk, _, _ in tvlp:
                for adj_vec in X_adjust:
                    if tk in vec_dic:
                        vec_dic[tk] = np.add(vec_dic[tk], adj_vec)/2
                    else:
                        vec_dic[tk] = adj_vec
            dump_pickle(path_vec_dic, vec_dic)
        else:
            vec_dic = load_pickle(path_vec_dic)
        return vec_dic

    def _word_idf_score(self):
        """

        :return:
        """
        path_model = join_file_path(self.path_model, 'tfidf.model')
        path_idf_dic = join_file_path(self.path_feature, 'idf.dic')
        if not check_file_exist(path_model):
            text = self.get_text()
            tfidf_vec = TfidfVectorizer(tokenizer=self._tokenizer)
            # tfidf_vec = TfidfVectorizer(tokenizer=None)
            tfidf_vec.fit(text)
            dump_pickle(path_model, tfidf_vec)
        else:
            tfidf_vec = load_pickle(path_model)

        if not check_file_exist(path_idf_dic):
            idf_score = np.log(tfidf_vec.idf_)
            vb = tfidf_vec.vocabulary_
            idf_dic = {tk: idf_score[vb[tk]] for tk in vb}
            dump_pickle(path_idf_dic, idf_dic)
        else:
            idf_dic = load_pickle(path_idf_dic)
        return idf_dic


    def feature_sentence_embedding(self, text):
        """

        :param text: testing data
        :return:
        """
        path_doc_feature = join_file_path(self.path_feature, 'doc.feature')
        if not check_file_exist(path_doc_feature):
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
                # if document_vec is None:
                #     document_vec = sentence_vec
                # else:
                #     document_vec = np.concatenate((document_vec, sentence_vec), axis=0)
            dump_pickle(path_doc_feature, document_vec)
        else:
            document_vec = load_pickle(path_doc_feature)
        return document_vec







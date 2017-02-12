"""
Classification
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# todo: cross_validation
# This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.




class Classification:
    """

    """

    def __init__(self):
        """

        """
        pass


    def _normalize_matrix(self, X, ax=0, max_range=1, min_range=0):
        """
        normalize matrix within [min_range, max_range] by columns (ax=0) or by rows (ax=1)
        :param X: numpy array, dtype=int64
        :param ax: axis=0 by columns, axis=1 by rows
        :return: standard matrix
        """
        X_std = (X - X.min(axis=ax)) / (X.max(axis=ax) - X.min(axis=ax))
        return X_std * (max_range - min_range) + min_range

    def k_fold_cross_validation(self, X, Y, model, k=10, normalize=True):
        """

        :param X: feature matrix. numpy array, dtype=int64
        :param Y: label. in any form that can be converted to an array, can be converted by numpy.asarray
        :param model: sklearn classification modle
        :param k: k fold cross validation
        :param normalize: normalize X or not
        :return:
        """
        assert isinstance(X, np.ndarray), 'check the data type of the feature matrix, should be numpy array'
        number_records = len(Y)
        assert number_records == X.shape[0], 'check the number of records in label data Y and feature matrix X'
        label_type = list(set(Y))  # classification_report(labels=list)
        try:
            Y = np.asarray(Y)
        except:
            raise TypeError('label data. in any form that can be converted to an array. '
                            'This includes lists, lists of tuples, tuples, tuples of tuples, '
                            'tuples of lists and ndarrays')
        # normalize the feature matrix or not
        if normalize:
            X = self._normalize_matrix(X)
        # define classifier
        classifier = model()  # todo: parameter tuning
        # define k fold cross validation
        kf = KFold(n_splits=k, shuffle=True, random_state=1)  # todo: parameter tuning

        print('{}_fold classification \n'
              'label types: {} \n'
              'classifier name: {} \n'.format(k, label_type, model.__name__))

        ACC = 0
        PRE, REC, F, SUP = np.zeros(len(label_type)), np.zeros(len(label_type)),np.zeros(len(label_type)),np.zeros(len(label_type))
        for train_index, test_index in kf.split(X):  # kf.split() Generate indices to split data into training and test set.
            # make training and testing data
            X_train, X_test = X[train_index], X[test_index]  # no error
            Y_train, Y_test = Y[train_index], Y[test_index]
            # fit model
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)
            # report results
            # accuracy
            accuracy = accuracy_score(Y_test, Y_pred, normalize=True)


            # precision_recall_fscore_support
            # beta=float: the strength of recall vs precision in F-score
            # labels=list:
            precision, recall, fscore, support = precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, labels=label_type)

            print("accuracy: {} \n".format(accuracy))
            print("precision: {} \n".format(precision))
            print("recall: {} \n".format(recall))
            print("fscore: {} \n".format(fscore))
            print("support: {} \n".format(support))
            ACC += accuracy
            PRE += precision
            REC += recall
            F += fscore
            SUP += support
            # print("precision_recall_fscore_support: \n {} \n".format((precision,recall,fscore,support)))
            # print("-----------------")

            # print("classification_report: \n")
            print(classification_report(Y_test, Y_pred, labels=label_type, digits=3))
            print("-----------------")
        #
        print(ACC/k, PRE/k, REC/k, F/k, SUP/k)




    def predict_label(self):
        pass









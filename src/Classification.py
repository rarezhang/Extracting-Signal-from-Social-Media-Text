"""
Classification
"""
import os
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from utils import join_file_path, check_file_exist, check_make_dir, dump_pickle, load_pickle




# todo: parameter for classification model


class Classification:
    """

    """

    def __init__(self, output_directory, X, Y, normalize=True):
        """
        set output file directory path
        """
        check_make_dir(output_directory)  # check and make output dir

        if normalize:
            X = self._normalize_matrix(X)
        assert isinstance(X, np.ndarray), 'check the data type of the feature matrix, should be numpy array'
        try:
            number_records = len(Y)
            Y = np.asarray(Y)
        except:
            raise TypeError('label data. in any form that can be converted to an array. '
                            'This includes lists, lists of tuples, tuples, tuples of tuples, '
                            'tuples of lists and ndarrays')
        assert number_records == X.shape[0], 'check the number of records in label data Y and feature matrix X'
        self.output_directory = output_directory
        self.normalize = normalize
        self.X = X
        self.Y = Y

    def _write(self, path, content):
        """

        :param path:
        :return:
        """
        with open(path, mode='a') as f:
            f.write(content)

    def _normalize_matrix(self, X, ax=0, max_range=1, min_range=0):
        """
        normalize matrix within [min_range, max_range] by columns (ax=0) or by rows (ax=1)
        :param X: numpy array, dtype=int64
        :param ax: axis=0 by columns, axis=1 by rows
        :return: standard matrix
        """
        X_std = (X - X.min(axis=ax)) / (X.max(axis=ax) - X.min(axis=ax))
        return X_std * (max_range - min_range) + min_range

    classifiers = (
        LinearSVC(), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1),
        MultinomialNB(), GaussianNB(), BernoulliNB(),
        LogisticRegression(), Perceptron(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), ExtraTreesClassifier(), AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        MLPClassifier(alpha=1),
        QuadraticDiscriminantAnalysis())

    def classifier_comparison_cross_validation(self, model_list, k=10):
        """

        :param X: feature matrix. numpy array, dtype=int64
        :param Y: label. in any form that can be converted to an array, can be converted by numpy.asarray
        :param model_list: sklearn classification model, tuple of models
        :param k: k fold cross validation
        :param normalize: normalize X or not
        :return:
        """
        # assert isinstance(model_list, ()), 'list of sklearn classifiers'
        label_type = list(set(self.Y))  # classification_report(labels=list)

        # output file
        output_path = join_file_path(self.output_directory, 'k_fold_cross_validation')

        # define k fold cross validation
        kf = KFold(n_splits=k, shuffle=True, random_state=1)  # todo: parameter tuning

        for model in model_list:
            # define classifier
            classifier = model  # todo: parameter tuning
            # model name
            model_name = str(model)
            to_print = '{}_fold classification \n' \
                  'label types: {} \n' \
                  'classifier name: {} \n'.format(k, label_type, model_name)

            ACC = 0
            PRE, REC, F, SUP = np.zeros(len(label_type)), np.zeros(len(label_type)),np.zeros(len(label_type)),np.zeros(len(label_type))
            for train_index, test_index in kf.split(self.X):  # kf.split() Generate indices to split data into training and test set.
                # make training and testing data
                X_train, X_test = self.X[train_index], self.X[test_index]  # no error
                Y_train, Y_test = self.Y[train_index], self.Y[test_index]
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

                ACC += accuracy
                PRE += precision
                REC += recall
                F += fscore
                SUP += support
                # print("classification_report: \n")
                # print(classification_report(Y_test, Y_pred, labels=label_type, digits=3))
            ACC, PRE, REC, F, SUP = ACC / k, PRE/k, REC/k, F/k, SUP/k
            to_print += 'average_accuracy \t {} \n' \
                       'class \t precision \t recall \t f1-score \t support \n'.format(ACC)
            for i in range(len(label_type)):
                label, pre, rec, f, sup = str(label_type[i]), str(PRE[i]), str(REC[i]), str(F[i]), str(SUP[i])
                to_print += ' '.join((label, '\t', pre, '\t', rec, '\t', f, '\t', sup, '\n'))
            to_print += '-' * 60 + '\n\n\n\n'
            print(to_print)
            self._write(output_path, to_print)

    def predict_label(self, XX, model):
        """

        :param X: training feature matrix, numpy array, dtype=int64
        :param XX: testing feature matrix, numpy array, dtype=int64
        :param model: sklearn classification model
        :param normalize:
        :return: normalize X or not
        """
        # output file
        output_path = join_file_path(self.output_directory, 'predict_label')
        classifier_path = join_file_path(self.output_directory, 'predict_classifier')
        # normalize the feature matrix or not
        if self.normalize:
            XX = self._normalize_matrix(XX)
        # define classifier
        classifier = model
        if not check_file_exist:
            classifier.fit(self.X, self.Y)
            dump_pickle(classifier_path, classifier)
        else:
            classifier = load_pickle(classifier_path)
            # classifier.predict: return array, shape = [n_samples]
            # Predicted class label per sample.
            labels = classifier.predict(XX)
            to_print = str(model) + '\n'
            for lab in labels:
                to_print += str(lab) + '\n'
            self._write(output_path, to_print)














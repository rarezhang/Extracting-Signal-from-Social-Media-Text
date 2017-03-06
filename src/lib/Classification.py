"""
Classification
"""

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, make_scorer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from .utils import time_now, pop_var, join_file_path, check_file_exist, check_make_dir, dump_pickle, load_pickle


dataset = pop_var()

path = join_file_path('../data/result/', dataset)
check_make_dir(path)
print(f'path to output: {path}')



classifiers = (
    # SVM
    SVC(kernel="linear"),  # linear SVM
    SVC(kernel="rbf"),  # RBF kernel
    SVC(kernel='poly'),  # polynomial kernel

    # Naive Bayes
    MultinomialNB(),
    # for multinomially distributed data, one of the two classic naive Bayes variants used in text classification
    # GaussianNB(), # X: array-like, the features is assumed to be Gaussian
    BernoulliNB(),
    # each features is assumed to be a binary-valued(binarize its input (depending on the binarize parameter))

    # Linear model
    LogisticRegression(),
    # “lbfgs”, “sag”(very large dataset) and “newton-cg” (L2 penalization); converge faster for high dimensional data
    # Perceptron(),

    # Tree
    DecisionTreeClassifier(max_depth=5),

    # Neighbors
    # KNeighborsClassifier(3),

    # Gaussian Process
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), # X: array-like, lose efficiency in high dimensional spaces

    # Neural Network
    MLPClassifier(alpha=1),  # Multi-layer Perceptron

    # Discriminant Analysis
    # QuadraticDiscriminantAnalysis(),  # X: array-like

    # Ensemble
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # ExtraTreesClassifier(),
    AdaBoostClassifier(base_estimator=None),
# fit a sequence of weak learners on repeatedly modified versions of the data
)


class Classification:
    """

    """

    def __init__(self, X, Y, normalize=True):
        """
        set output file directory path
        """
        assert isinstance(X, (np.ndarray, coo_matrix)), \
            'check the data type of the feature matrix, should be numpy array or sparse matrix '
        if not isinstance(X, (np.ndarray, np.generic) ):
            X = X.toarray()

        if normalize:
            X = self._normalize_matrix(X)

        try:
            Y = list(Y)
            number_records = len(Y)
            Y = np.asarray(Y)
        except:
            raise TypeError('label data. in any form that can be converted to an array. '
                            'This includes lists, lists of tuples, tuples, tuples of tuples, '
                            'tuples of lists and ndarrays')

        assert number_records == X.shape[0], 'check the number of records in label data Y and feature matrix X'

        # # label data has to be binarized
        # Y = LabelBinarizer().fit(Y)
        # print(Y)

        self.normalize = normalize
        self.X = X
        self.Y = Y

    def _write(self, path, content): # todo print to screen
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


    def classifier_comparison_cross_validation(self, model_list=classifiers, k=10):
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
        output_path = join_file_path(path, 'k_fold_cross_validation')

        # define k fold cross validation
        kf = KFold(n_splits=k, shuffle=True, random_state=1)  # todo: parameter tuning

        to_print = f'{time_now()} \n '
        for model in model_list:
            # define classifier
            classifier = model
            # model name
            model_name = str(model)  # model name with parameters setting info
            to_print += f'{k}_fold classification \n' \
                  f'label types: {label_type} \n' \
                  f'classifier name: {model_name} \n'

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
            to_print += f'average_accuracy \t {ACC} \n' \
                       f'class \t precision \t recall \t f1-score \t support \n'
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
        output_path = join_file_path(path, 'predict_label')
        classifier_path = join_file_path(path, 'predict_classifier')
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

    def _set_scores(self, scores, positive_label):
        """

        :param scores: 'ALL' or ['accuracy', 'f1', 'precision', 'recall']
        :param positive_label:
        :return:
        """
        assert isinstance(scores, list) or scores == 'ALL'

        f1 = make_scorer(f1_score, pos_label=positive_label)
        recall = make_scorer(recall_score, pos_label=positive_label)
        precision = make_scorer(precision_score, pos_label=positive_label)
        accuracy = make_scorer(accuracy_score)
        scores_list = [f1, recall, precision, accuracy]
        if scores is not 'ALL':
            # scores_list = set(sum([[fea for fea in feature_list if type in fea] for type in feature_type], []))
            scores_list = set(sum([[score for score in scores_list if score_string in str(score)] for score_string in scores], []))
        return scores_list


    def parameter_tuning(self, model, parameters, positive_label, scores='ALL', tz=0.1, k=10):
        """
        tuning the hyper-parameter using grid search with cross-validation
        :param model: classifier
        :param parameters: tuned_parameters, dict or list of dictionary
        :param scores: list of score name
        :param tz: test size
        :param k: k cross-validation
        :return:
        """
        assert isinstance(parameters, (list, dict))

        # set scores, if ALL use all score
        scores = self._set_scores(scores, positive_label=positive_label)

        # output file
        model_name = str(model).split('(')[0]
        output_path = join_file_path(path, f'{model_name}_parameter_tuning')

        # Split the data set in two equal parts
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=tz, random_state=5)

        to_print = f'{time_now()} \n '
        for score in scores:
            to_print += f'Tuning hyper-parameters for {score}: \n'
            classifier = GridSearchCV(estimator=model, param_grid=parameters, scoring=score, cv=k, n_jobs=4)
            classifier.fit(X_train, Y_train)

            to_print += f'Best parameters set found on training set: \n' \
                        f'{classifier.best_params_} \n' \
                        f'Grid scores on training set: \n'
            means = classifier.cv_results_['mean_test_score']
            stds = classifier.cv_results_['std_test_score']
            params = classifier.cv_results_['params']
            for mean, std, par in zip(means, stds, params):
                to_print += f'mean: {mean}, std: {std*2} for par: {par} \n'
            to_print += f'The model is trained on the full training set. \n' \
                        f'The scores are computed on the full testing set, test size: {tz} \n'
            Y_pred = classifier.predict(X_test)
            to_print += classification_report(Y_test, Y_pred)
            to_print += '-' * 60 + '\n\n\n\n'
            print(to_print)
            self._write(output_path, to_print)






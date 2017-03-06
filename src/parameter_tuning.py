
"""
A search consists of:
- an estimator
    + SVC: SVM for classification
        * linear
        * RBF kernel (default)
        * polynomial kernel (degree = ?)
    + naive bayes (different naive Bayes classifiers differ mainly by the assumptions regarding the distribution of P(x_i|y).)
        * MultinomialNB
- a parameter space;
- a method for searching or sampling candidates;
    + GridSearchCV: exhaustively considers all parameter combinations
    + RandomizedSearchCV: sample a given number of candidates from a parameter space with a specified distribution
- a cross-validation scheme
- a score function.
"""


from lib.utils import push_var
# dataset = 'asthma'
dataset = 'asthma'
push_var(dataset)


from lib.ReadTextFile import ReadTextFile
from lib.FeatureEng import FeatureEng
from lib.Classification import Classification


# main
general_text = '../data/training/'

# general_output = '../data/result/'


# training data path
if dataset == 'ecig':
    path = f'{general_text}ecig_training_topic.csv'
elif dataset == 'asthma':
    path = f'{general_text}asthma_training_topic.csv'
else:
    path = '../data/training/txte.csv'  # todo test data


rtf = ReadTextFile(path, sep='||')
text = rtf.read_column(1)  # generator, column 1: text

#######################################################################

fea = FeatureEng(text)  # todo add remake to class
X = fea.feature_engineering(feature_type=['feature_word_ngram'])  # only use word ngram

y = rtf.read_column(0)
# y = list(y)
# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# lb.fit(y)
# print(y)


#######################################################################
# Classification

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

if __name__ == '__main__':
    classifiers = (
        # SVM
        SVC(),  # linear SVM: kernel="linear"; RBF kernel: kernel="rbf"; # kernel='poly': polynomial kernel

        # Naive Bayes
        MultinomialNB(),
        # for multinomially distributed data, one of the two classic naive Bayes variants used in text classification
        # GaussianNB(), # X: array-like, the features is assumed to be Gaussian
        BernoulliNB(),
        # each features is assumed to be a binary-valued(binarize its input (depending on the binarize parameter))
        # might perform better on some datasets, especially those with shorter documents

        # Linear model
        LogisticRegression(),
        # “lbfgs”, “sag”(very large dataset) and “newton-cg” (L2 penalization); converge faster for high dimensional data
        Perceptron(),

        # Tree
        DecisionTreeClassifier(),
        # Neural Network
        MLPClassifier(),  # Multi-layer Perceptron
        # Ensemble
        RandomForestClassifier(),
        # ExtraTreesClassifier(),
        AdaBoostClassifier(base_estimator=None),
    )

    parameters = (
        # SVM
        # C: smaller values specify stronger regularization.
        [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]},
         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
         {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4, 5, 6, 7]}],
        # MultinomialNB
        # alpha>=0, alpha=1 Laplace smoothing, alpha<1 Lidstone smoothing, alpha=0 no smoothing
        [{'alpha': [1, 0.5, 0]}],
        # BernoulliNB
        [{'alpha': [1, 0.5, 0]}],
        # LogisticRegression
        # C: Inverse of regularization strength
        [{'solver': ['newton-cg', 'lbfgs', 'sag'], 'C': [1, 10, 100, 1000], 'warm_start': [True]},
         {'solver': ['liblinear'], 'penalty': ['l1', 'l2']}],
        # Perceptron
        [{'penalty': [None, 'l2', 'k1', 'elasticnet'], 'n_iter': [5, 20, 50, 100]}],
        # Tree
        [{'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2', None], 'max_depth': [None, 3, 4, 5, 6], 'min_samples_split': [2, 0.1, 0.2]}],
        # MLPClassifier
        [{'hidden_layer_sizes': [5, 10, 20], 'activation' : ['logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'warm_start': [True]}],
        # RandomForestClassifier
        [{'n_estimators': [10, 20, 30], 'warm_start': [True]}],
        # AdaBoostClassifier
        # todo: wait the best classifier as base_estimator
        # [{''}]
    )
    clf = Classification(X, y, normalize=True)

    for c, p in zip(classifiers, parameters):
        print(c, '\n', p)
        clf.parameter_tuning(c, p, positive_label='R', scores=['accuracy'], k=5)



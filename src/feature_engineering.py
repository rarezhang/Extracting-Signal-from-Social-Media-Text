from lib.utils import push_var
# dataset = 'asthma'
dataset = 'test'
push_var(dataset)


from lib.ReadTextFile import ReadTextFile
from lib.FeatureEng import FeatureEng
from lib.Classification import Classification


# main
general_text = '../data/training/'
# general_feature = '../data/feature/'
general_output = '../data/result/'


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

fea = FeatureEng(text)
# fea.make_feature(path_feature, remake=False)  # todo: remake=F if not necessary
X = fea.feature_engineering(feature_type='ALL')
print('-----------------------', X.shape)
y = rtf.read_column(0)
y = list(y)
print('-----------------------', len(y))
print(type(X))
# X = X.toarray()
# print(type(X))

"""
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

classifiers = (
    Perceptron(),
)

output_directory = general_output + dataset
clf = Classification(output_directory, X, y, normalize=True)
clf.classifier_comparison_cross_validation(classifiers, k=3)
"""


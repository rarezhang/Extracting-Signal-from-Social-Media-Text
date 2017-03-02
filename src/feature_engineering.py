from ReadTextFile import ReadTextFile
from Clean import Clean
from FeatureEng import FeatureEng
from Classification import Classification

# main
general_text = '../data/training/'
general_feature = '../data/feature/'
general_output = '../data/result/'

dataset = 'asthma'

# ReadTextFile
if dataset == 'ecig':
    path = f'{general_text}ecig_training_topic.csv'
elif dataset == 'asthma':
    path = f'{general_text}asthma_training_topic.csv'
else:
    path = '../data/training/txte.csv'  # todo test data


rtf = ReadTextFile(path, sep='||')

text = rtf.read_column(1)  # generator, column 1: text
# for t in text: print(t)

#######################################################################

# Clean
# clean
clean = Clean(text)
clean_text = clean.clean()

path_feature = general_feature + dataset


fea = FeatureEng(text)
fea.make_feature(path_feature, remake=False)  # todo: remake=F if not necessary
X = fea.combine_feature(path_feature)
print('-----------------------', X.shape)
y = rtf.read_column(0)
y = list(y)
print('-----------------------', len(y))



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
clf = Classification(output_directory, X, y)
clf.classifier_comparison_cross_validation(classifiers, k=5)


#######################################################################

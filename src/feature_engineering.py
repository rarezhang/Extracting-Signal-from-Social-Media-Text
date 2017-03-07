from lib.utils import push_var, alarm_when_finish
# dataset = 'asthma'
dataset = 'asthma'
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


#######################################################################
if __name__ == '__main__':
    rtf = ReadTextFile(path, sep='||')
    text = rtf.read_column(1)  # generator, column 1: text

    fea = FeatureEng(text)

    # single feature group
    # feature_group = ('ALL', 'feature_word_ngram', 'feature_pos_ngram', 'feature_pos_count', 'feature_char_ngram', 'feature_text_length', 'feature_token_based')
    feaature_group = ['feature_char_ngram', 'feature_text_length', 'feature_token_based']
    X = fea.feature_engineering(feature_type=feaature_group)
    y = rtf.read_column(0)

    # Classification
    clf = Classification(X, y, normalize=True)

    sep = '\n\n\n' + '+' * 1000 + '\n' + str(feaature_group) + '\n'
    clf._write('D://Projects//signal_extraction//data//result//asthma/k_fold_cross_validation', sep)
    clf.classifier_comparison_cross_validation(work_name='feature_group', k=10)

    alarm_when_finish()





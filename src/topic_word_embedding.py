"""
test the FeatureVec Class
"""
from lib.utils import push_var, alarm_when_finish
dataset = 'asthma'
push_var(dataset)


from lib.ReadTextFile import ReadTextFile
from lib.FeatureVec import FeatureVec
from lib.Clean import Clean



general_text = '../data/training/'
general_feature = '../data/feature/'
general_output = '../data/result/'



# ReadTextFile
if dataset == 'ecig':
    path = f'{general_text}ecig_training_topic.csv'
elif dataset == 'asthma':
    path = f'{general_text}asthma_training_topic.csv'
else:
    path = '../data/training/txte.csv'  # todo test data
print(path)


#######################################################################
if __name__ == '__main__':
    # data for model training
    rtf = ReadTextFile(path, sep='||')
    text = rtf.read_column(1)
    clean = Clean(text)
    text = clean.clean()
    y = rtf.read_column(0)

    # general_path_model = '../data/word_vector/model/'
    # general_path_feature = '../data/word_vector/feature/'
    vec = FeatureVec(text, y, rebuild=True)

    # ====================================================================
    # test file
    rtf = ReadTextFile(path, sep='||')
    text = rtf.read_column(1)
    clean = Clean(text)
    clean_text = clean.clean()
    X = vec.feature_embedding(clean_text)
    print(X.shape)




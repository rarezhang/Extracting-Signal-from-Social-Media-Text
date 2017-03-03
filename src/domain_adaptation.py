from utils import push_var
dataset = 'test'
push_var(dataset)


from FeatureAug import FeatureAug
from ReadTextFile import ReadTextFile
from Clean import Clean
# todo: test data feed in
# todo: test result

# main
path_source = '../data/domain_adaptation/ecig_training_sentiment.csv'
path_target = '../data/domain_adaptation/pima_training_sentiment.csv'
# path_general = '../data/domain_adaptation/general'
# path_target_test = '../data/domain_adaptation/pima_training_sentiment_test.csv'

# text generator
def get_clean_text(path):
    rtf = ReadTextFile(path, sep='||')
    text = rtf.read_column(1)
    clean = Clean(text)
    text = clean.clean()
    return text

text_source = get_clean_text(path_source)
text_target = get_clean_text(path_target)

# for ts in text_source:
#     print(ts)
# for tt in text_target:
#     print(tt)


fa = FeatureAug(text_source, text_target)

# path_feature = '../data/domain_adaptation/feature'
x = fa.augment_feature()
print(x.shape, type(x))



from lib.utils import push_var

# from src import FeatureAug, ReadTextFile, Clean
dataset = 'test'
push_var(dataset)


from lib.FeatureAug import FeatureAug
from lib.ReadTextFile import ReadTextFile
from lib.Clean import Clean
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

fa = FeatureAug(text_source, text_target)

x = fa.augment_feature()
print(x.shape, type(x))


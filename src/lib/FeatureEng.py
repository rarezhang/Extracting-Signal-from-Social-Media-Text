from scipy.sparse import coo_matrix, hstack
# local
from .Clean import Clean
from ._FeatureMake import _FeatureMake
from .utils import pop_var, load_or_make, join_file_path, file_remove, check_file_exist, check_make_dir, dump_pickle, load_pickle, nested_fun


dataset = pop_var()

path = join_file_path('../data/feature_engineering/', dataset)
# general_path_model = join_file_path(path, 'model')
# general_path_feature = join_file_path(path, 'feature')
check_make_dir(path)
# check_make_dir(general_path_model)
# assert general_path_feature is not None and general_path_model is not None
# print(f'path to model: {general_path_model}')
print(f'path to feature: {path}')


class FeatureEng(Clean, _FeatureMake):
    """

    """
    # todo: way to tuning feature parameters
    def make_feature(self, remake=False, default_vocabulary=True):
        """

        :return:
        """
        # check_make_dir(path_feature)  # check and make directory
        feature_funs = (self.feature_token_based, self.feature_text_length, self.feature_pos_count,
                        self.feature_pos_ngram, self.feature_char_ngram, self.feature_word_ngram,)

        for f in feature_funs:
            feature_name = f.__name__
            feature_path = join_file_path(path, feature_name)

            if remake: file_remove(feature_path)
            if not check_file_exist(feature_path):
                print("dumping feature: {}....".format(feature_name))
                # text: before clean or after clean
                text = self.get_text()  # inherit from _TextPrepro
                text = text if feature_name == 'feature_token_based' else self.clean()  # generators
                # make feature
                if 'ngram' in feature_name:
                    vocabulary_name = ''.join((feature_name, '_vocabulary'))
                    vocabulary_path = join_file_path(path, vocabulary_name)
                    if default_vocabulary:  # dump vocabulary
                        feature, vocabulary = f(text)
                        dump_pickle(vocabulary_path, vocabulary)
                    else:
                        vocabulary = load_pickle(vocabulary_path)
                        feature, _ = f(text, vb=vocabulary)
                else:
                    feature = f(text)
                dump_pickle(feature_path, coo_matrix(feature))

    # @load_or_make(join_file_path(general_path_feature, 'doc.feature'))
    def _feature_engineering(self, feature_type='ALL', default_vocabulary=True):
        """
        load feature and combine features  -> choose feature types
        :param path_feature:
        :param feature_type: 'ALL': all feature type; []: list of string
        :return:
        """
        feature_funs = (self.feature_token_based, self.feature_text_length, self.feature_pos_count,
                        self.feature_pos_ngram, self.feature_char_ngram, self.feature_word_ngram,)

        assert (feature_type == 'ALL') or (isinstance(feature_type, list) and len(feature_type)>0)
        self.make_feature(default_vocabulary=default_vocabulary)  # make sure there are features can be loaded later

        result = None
        feature_list = [f.__name__ for f in feature_funs]

        # filter features based on feature type
        if feature_type is not 'ALL':
            feature_list = set(sum([[fea for fea in feature_list if type in fea] for type in feature_type], []))
        print("will return the concatenate features: {} ".format("\n".join(feature_list)))
        # create feature path
        for feature_name in feature_list:
            print("loading feature: {}".format(feature_name))
            feature_path = join_file_path(path, feature_name)
            # load feature
            feature = load_pickle(feature_path)
            print("{} shape: {}".format(feature_name, feature.shape))
            # combine feature
            # result = feature if result is None else np.concatenate((result, feature), axis=1)
            result = feature if result is None else hstack((result, feature))
            # if result is None:
            #     result = feature
            # else:  # concatenate features
            #     result = np.concatenate((result, feature), axis=1)  # column wise
        return coo_matrix(result)
        # Advantages of the COO format
            # facilitates fast conversion among sparse formats
            # permits duplicate entries (see example)
            # very fast conversion to and from CSR/CSC formats
        # Disadvantages of the COO format
            # does not directly support:
            # arithmetic operations
            # slicing

    @load_or_make(join_file_path(path, 'doc.feature'))
    def feature_engineering(self, feature_type='ALL', default_vocabulary=True):
        return self._feature_engineering(feature_type=feature_type, default_vocabulary=default_vocabulary)
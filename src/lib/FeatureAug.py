"""
feature augmentation for domain adaptation
based on:
Daumé III, Hal. "Frustratingly easy domain adaptation." arXiv preprint arXiv:0907.1815 (2009).

Take each feature in the original problem and make three versions of it:
- a general version
- a source-specific version
- a target-specific version
The augmented source data will contain only general and source-specific versions
The augmented target data contains general and target-specific versions.

source_domain = <general, source, 0>
target_domain = <general, 0, target>
"""

import numpy as np
from itertools import tee
from scipy.sparse import coo_matrix
# local
from .FeatureEng import FeatureEng
from .utils import pop_var, load_or_make, join_file_path, check_make_dir, dump_pickle, load_pickle

dataset = pop_var()

path = join_file_path('../data/domain_adaptation/', dataset)
path_source = join_file_path(path, 'source')
path_target = join_file_path(path, 'target')
check_make_dir(path_source)
check_make_dir(path_target)



class FeatureAug:
    """

    """

    def __init__(self, text_source, text_target):
        """

        :param text_source:
        :param text_target:
        """
        self.text_source = text_source
        self.text_target = text_target

    def get_text_source(self):
        text, self.text_source = tee(self.text_source)
        return text

    def get_text_target(self):
        text, self.text_target = tee(self.text_target)
        return text

    def _merge_dictionary(self, dic_a, dic_b):
        """

        :param dic_a: larger dictionary
        :param dic_b: smaller dictionary
        :return:
        """
        position = max(dic_a.values())
        for key_b in dic_b:
            if key_b in dic_a:  # if a given key already exists in a dictionary
                continue
            else:  # add a new key to key_a
                position += 1
                dic_a[key_b] = position
        return dic_a

    def _make_feature_and_vocabulary(self, text, path_domain, default_vocabulary=True):
        """

        :param path_feature_domain: path to feature directory
        :return:
        """
        fea = FeatureEng(text)
        f = fea.feature_word_ngram

        path_vocabulary = join_file_path(path_domain, 'feature_word_ngram_vocabulary')
        if default_vocabulary:
            X, vocabulary = f(text)
            dump_pickle(path_vocabulary, vocabulary)
        else:
            vocabulary = load_pickle(path_vocabulary)
            X, _ = f(text, vb=vocabulary)
        return X

    @load_or_make(path=join_file_path(path_source, 'X'))
    def _make_source(self):
        """

        :return:
        """
        text = self.get_text_source()
        return self._make_feature_and_vocabulary(text, path_source, default_vocabulary=True)

    @load_or_make(path=join_file_path(path_target, 'X'))
    def _make_target(self):
        """

        :return:
        """
        text = self.get_text_target()
        return self._make_feature_and_vocabulary(text, path_target, default_vocabulary=True)

    @load_or_make(path=join_file_path(path, 'feature_word_ngram_vocabulary'))
    def _make_vocabulary_combine(self):
        """
        source + target
        :return:
        """
        path_vocabulary_source = join_file_path(path_source, 'feature_word_ngram_vocabulary')
        path_vocabulary_target = join_file_path(path_target, 'feature_word_ngram_vocabulary')
        vocabulary_source = load_pickle(path_vocabulary_source)
        vocabulary_target = load_pickle(path_vocabulary_target)
        # combine source and target dictionary
        vocabulary_combine = self._merge_dictionary(vocabulary_source, vocabulary_target)
        # override original dic
        dump_pickle(path_vocabulary_source, vocabulary_combine)
        dump_pickle(path_vocabulary_target, vocabulary_combine)
        return vocabulary_combine

    @load_or_make(path=join_file_path(path_source, 'X_combine'))
    def _make_source_combine(self):
        """

        :return:
        """
        text = self.get_text_source()
        return self._make_feature_and_vocabulary(text, path_source, default_vocabulary=False)

    @load_or_make(path=join_file_path(path_target, 'X_combine'))
    def _make_target_combine(self):
        """

        :return:
        """
        text = self.get_text_target()
        return self._make_feature_and_vocabulary(text, path_target, default_vocabulary=False)

    @load_or_make(path=join_file_path(path, 'doc.feature'))
    def augment_feature(self):
            """

            :return: augmented feature matrix
            """
            X_source = self._make_source()
            X_target = self._make_target()
            self._make_vocabulary_combine()
            X_source_combine = self._make_source_combine()
            X_target_combine = self._make_target_combine()

            m, n = X_source.shape[0], X_target.shape[1]
            source = np.concatenate((X_source_combine, X_source, np.zeros((m,n))), axis=1)  # column wise

            m, n = X_target.shape[0], X_source.shape[1]
            target = np.concatenate((X_target_combine, np.zeros((m,n)), X_target), axis=1)  # column wise

            print(X_source.shape, X_target.shape)
            print(X_source_combine.shape, X_target_combine.shape)
            print(source.shape, target.shape)
            return coo_matrix(np.concatenate((source, target), axis=0))










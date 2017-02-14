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
from shutil import copy2
from ReadTextFile import ReadTextFile
from CleanAndFeature import CleanAndFeature
from utils import join_file_path, load_pickle, dump_pickle, concatenate_files, check_file_exist


class FeatureAugmentation:
    """

    """

    def __init__(self, path_text_source, path_text_target):
        """

        :param text_source:
        :param text_target:
        """
        self.path_text_source = path_text_source
        self.path_text_target = path_text_target

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

    def _text_generator(self, path_text, sep='||', col=1):
        """

        :param path_text: path to text file
        :return: text generator
        """
        return CleanAndFeature(ReadTextFile(path_text, sep=sep).read_column(col))  # generator

    def _make_feature_vocabulary(self, text, path_feature_domain, default_vocabulary=True):
        """

        :param path_feature_domain: path to feature directory
        :return:
        """
        # text.make_feature(path_feature)  # directory
        X = text.combine_feature(path_feature_domain, feature_type=['word_ngram'], default_vocabulary=default_vocabulary)
        path_vocabulary = join_file_path(path_feature_domain, 'feature_word_ngram_vocabulary')
        if default_vocabulary:
            vocabulary = load_pickle(path_vocabulary)
            return X, vocabulary
        else:
            return X

    def augment_feature(self, path_feature):
        """

        :return: augmented feature matrix
        """
        text_source = self._text_generator(self.path_text_source)
        path_feature_source = join_file_path(path_feature, 'source')
        X_source, vocabulary_source = self._make_feature_vocabulary(text_source, path_feature_source)

        text_target = self._text_generator(self.path_text_target)
        path_feature_target = join_file_path(path_feature, 'target')
        X_target, vocabulary_target = self._make_feature_vocabulary(text_target, path_feature_target)


        path_feature_general = join_file_path(path_feature, 'general')
        vocabulary_path_general = join_file_path(path_feature_general, 'feature_word_ngram_vocabulary')
        if not check_file_exist(vocabulary_path_general):
            vocabulary_general = self._merge_dictionary(vocabulary_source, vocabulary_target)  # merge vocabulary
            dump_pickle(vocabulary_path_general, vocabulary_general)
        else:
            vocabulary_general = load_pickle(vocabulary_path_general)

        copy2(vocabulary_path_general, join_file_path(path_feature_general, 'source'))
        copy2(vocabulary_path_general, join_file_path(path_feature_general, 'target'))

        X_general_source = self._make_feature_vocabulary(text_source, join_file_path(path_feature_general, 'source'), default_vocabulary=False)
        X_general_target = self._make_feature_vocabulary(text_target, join_file_path(path_feature_general, 'target'), default_vocabulary=False)


        # print(X_source.shape, type(X_source))
        # print(X_general_source.shape, type(X_general_source))
        m, n = X_source.shape[0], X_target.shape[1]
        source = np.concatenate((X_general_source, X_source, np.zeros((m,n))), axis=1)  # column wise

        # print(X_target.shape, type(X_target))
        # print(X_general_target.shape, type(X_general_target))
        m, n = X_target.shape[0], X_source.shape[1]
        target = np.concatenate((X_general_target, np.zeros((m,n)), X_target), axis=1)  # column wise

        # print(source.shape, target.shape)
        return np.concatenate((source, target), axis=0)  # row wise










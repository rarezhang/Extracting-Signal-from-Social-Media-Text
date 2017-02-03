import os, pickle, fileinput, nltk
import datetime, time, codecs
from dateutil import tz


######################################################
# decorator functions
######################################################
def time_it(f):
    """
    decorator function
    :param f: function needs time recording
    :return: higher order function -> f = timeit(f)
    """
    import time

    def timed(*args, **kw):
        begin_time = time.time()
        fun = f(*args, **kw)
        end_time = time.time()
        print(f, 'time used: ', end_time - begin_time)
        return fun

    return timed


def load_or_make(f):
    """
    decorator function
    :param f:
    :return:
    """

    def wrap_fun(*args, **kwargs):
        pickle_path = kwargs['path'] + '.pkl'
        if check_file_exist(pickle_path):
            data = load_pickle(pickle_path)
        else:
            data = f(*args, **kwargs)
            dump_pickle(pickle_path, data)
        return data

    return wrap_fun


######################################################



######################################################
# files
# load and save pickle file
# check if file exist
# concatenate file in a directory
# dump feature: feature matrix -> single feature
######################################################
def dump_pickle(path, data):
    """
    save data as binary file
    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=2)  # protocol 3 is compatible with protocol 2, pickle_load can load protocol 2


def load_pickle(path):
    """
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_file_exist(path):
    """
    check if ``file`` exists
    :param path:
    :return: T/F
    """
    return os.path.isfile(path)


def concatenate_files(input_directory, output_file):
    """
    :param input_directory:
    :param output_file:
    :return:
    """
    assert os.path.isdir(input_directory), 'input path should be a directory'

    if not input_directory.endswith('/'):
        input_directory = ''.join((input_directory, '/'))

    if not check_file_exist(output_file):
        file_names = os.listdir(input_directory)
        file_paths = [''.join((input_directory, f_n)) for f_n in file_names]
        with open(output_file, 'w', encoding='utf-8') as out_file:
            in_file = fileinput.input(files=file_paths, openhook=fileinput.hook_encoded(
                'utf-8'))  # python 2.7.10, fileinput doest not have `__exit__` --> cannot use `with`
            for line in in_file:
                out_file.write(line)
            in_file.close()


def dump_feature(feature_type, feature_path, features, flag_normalize_feature=True):
    """
    if single feature does not exist, dump single feature
    :param feature_type:
    :param feature_path:
    :param features:
    :param flag_normalize_feature
    :return:
    """

    def normalize(l):
        min_x = min(l)
        max_x = max(l)
        return [(float(x) - min_x / max_x - min_x) for x in l]

    for ind, fea in enumerate(feature_type):
        if flag_normalize_feature:
            single_feature_path = ''.join((feature_path, fea.__name__, '_normalized.pkl'))
            single_feature = [r[ind] for r in features]
            single_feature = normalize(single_feature)
        else:  # do not normalize feature
            single_feature_path = ''.join((feature_path, fea.__name__, '.pkl'))
            single_feature = [r[ind] for r in features]

        if not check_file_exist(single_feature_path):
            dump_pickle(single_feature_path, single_feature)


######################################################

def convert_twitter_timedate(twitterdate, fromtimezone='UTC', totimezone='UTC'):
    """
    convert twitter time date to local date time.
    twitterdate: twitter time-date type: e.g., 'Tue Mar 29 08:11:25 +0000 2011'
    fromtimezone: convert from which time zone, twitter using UTC, default UTC
    totimezone: convert to which time zone
    >>> convert_twitter_timedate('Tue Mar 29 08:11:25 +0000 2011', fromtimezone='UTC', totimezone='US/Central')
    2011-03-29 03:11:25-05:00
    """
    date = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(twitterdate, '%a %b %d %H:%M:%S +0000 %Y'))
    from_zone = tz.gettz(fromtimezone)
    to_zone = tz.gettz(totimezone)
    utc = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    # Convert time zone
    return utc.astimezone(to_zone)


def strip_non_ascii(string, starting_ord=32, ending_ord=126):
    """
    Returns the string without non ASCII characters
    :param string:
    :param starting_ord:
    :param ending_ord:
    :return:
    """
    return "".join([c for c in string if starting_ord <= ord(c) <= ending_ord])


@load_or_make
def remove_stop_words(text, path=" "):
    """

    :param tweets:
    :return:
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return [" ".join([word for word in line.split() if word not in stop_words]) for line in text]


def pos_tag(tweet):
    """

    :param words:
    :return:
    """
    pos_t = nltk.pos_tag(strip_non_ascii(tweet).split())  # nltk can only process ascii char
    return pos_t


def python_version():
    """

    :return: 2 or 3
    """
    import sys
    if sys.version_info > (3, 0):
        return 3
    else:
        return 2
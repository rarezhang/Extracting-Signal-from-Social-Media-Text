
from __future__ import division
import io, re, nltk, pylab, numpy, operator
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report


"""
copied from pcci pcci_tweets_cleaning.py
modified for the e-cigarette data set. Use as training data set.
"""


def tokenizer(tweet):
    pattern = r'''(?x)    # set flag to allow verbose regexps
        ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | [<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]   # emoticons
      | [\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?   # emoticons, reverse orientation
      | (?:@[\w_]+)       # Twitter user name
      | (?:\#+[\w_]+[\w\'_\-]*[\w_]+)       # Twitter hashtags
      | ((www\.[^\s]+)|(https?://[^\s]+))   # URLs
      | \w+(-\w+)*        # words with optional internal hyphens
      | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.            # ellipsis
      | [][.,;"'?():-_`]  # these are separate tokens
     '''
    tweet = nltk.regexp_tokenize(tweet, pattern)
    return tweet

def pre_process(tweet): # tweet: string
    tweet = tweet.lower()
    # remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # remove duplicate characters
    tweet = re.sub(r'([a-z])\1+', r'\1', tweet)
    # tokenize: input: string, output: list
    tweet_list = tokenizer(tweet) # return list
    # stem the words, nltk can only process 'ascii' codec
    porter = nltk.PorterStemmer()
    tweet = []
    for t in tweet_list:
        try:
            porter.stem(t)  # only stem 'ascii' codec
            tweet.append(t)
        except:
            tweet.append(t)
    tweet = ' '.join(tweet)
    # convert all numbers to 'NUMBER'
    tweet = re.sub('[^\s]*[\d]+[^\s]*', 'NUMBER', tweet)
    # convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))','URL',tweet)
    # convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    return tweet


def read_file(path):
    """
    path: path to the tweet file: row -> label(0:irrelevant;1relevant), text /n
    return:
    labels -> list [0,1,0,0 ....]
    tweets -> list of string [tweet1, tweet2, ....]
    """
    with io.open(path, mode='r', encoding='utf-8', errors='ignore') as data:
        tweets, labels = [],[]
        for line in data:
            line = line.split(',')
            label = line[0]
            labels.append(label)
            tweet = line[1]
            tweet = pre_process(tweet)
            tweets.append(tweet)
    return labels, tweets

def word_count(tweets):
    words_dic = nltk.FreqDist()
    for tweet in tweets:
        for word in tweet.split():
            words_dic[word]+=1
    return words_dic

def pos_tag(t): # list of words, one tweet
    t = ' '.join(t) # join as a string
    t = ''.join([i for i in t if ord(i)<128]) # remove non ascii (one cha by cha), join back as a string
    pos_t = nltk.pos_tag(t.split()) # nltk can only process ascii char
    return pos_t


def verb_noun_hash_count(tweets):
    nouns, verbs = ['NN','NNP','NNPS','NNS'], ['VB','VBD','VBG','VBN','VBP']
    noun_dic, verb_dic, hash_dic = nltk.FreqDist(), nltk.FreqDist(), nltk.FreqDist()
    pat = r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)'
    for tweet in tweets:
        tweet = tweet.split() #string to list of word
        allhash = [x for x in tweet if re.search(pat, x)] # find all hash tag
        pos_list = pos_tag(tweet)  # pos tag: tup of list: [word, pos]
        for word, pos in pos_list:
            if word in allhash:
                hash_dic[word] += 1
            if pos in nouns:
                noun_dic[word] += 1
            elif pos in verbs:
                verb_dic[word] += 1
    return noun_dic, verb_dic, hash_dic


def conditional_proportions(sorted_relevant,sorted_irrelevant):
    """
    calculate conditional proportions for relevant asthma tweets / irrelevant asthma tweets
    """
    relevant = dict(sorted_relevant)
    irrelevant = dict(sorted_irrelevant)
    relevant_conditional_proportions = {}
    irrelevant_conditional_proportions = {}
    ## for the relevant
    for key in relevant:
            numerator = relevant[key]
            if key in irrelevant:
                # hashtags appear with both #happy and #sad
                denominator = relevant[key] + irrelevant[key]
            else:
                denominator = relevant[key]
            # calculate the conditional proportions
            relevant_conditional_proportions[key] = numerator / denominator
    ## for the #sad hashtags
    for key in irrelevant:
            numerator = irrelevant[key]
            if key in relevant:
                # hashtags appear with both #happy and #sad
                denominator = relevant[key] + irrelevant[key]
            else:
                denominator = irrelevant[key]
            # calculate the conditional proportions
            irrelevant_conditional_proportions[key] = numerator / denominator
    # sort the hashtags associated with #happy and #sad in descending order of their conditional proportions
    sorted_relevant_conditional_proportions = sorted(relevant_conditional_proportions.items(), key=operator.itemgetter(1), reverse=True)
    sorted_irrelevant_conditional_proportions = sorted(irrelevant_conditional_proportions.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_relevant_conditional_proportions,sorted_irrelevant_conditional_proportions

def sort_word_dic(dic):
    # sort the word count based on the # of words
    return sorted(dic.items(), key=lambda tup: tup[1], reverse=True) # list of tuples

def plot_word_bar(dic): # dic: list of two tuples: [(words),(counts)]
    dic = zip(*dic) # return a list of two tuples: [(words),(counts)]
    pylab.figure()
    ax = pylab.subplot(111)
    top_x = 30
    width=0.8
    ax.bar(range(len(dic[0][:top_x])),dic[1][:top_x], width=width)
    ax.set_xticks(numpy.arange(len(dic[0][:top_x])) + width/2)
    ax.set_xticklabels(dic[0][:top_x], rotation=90)
    pylab.show()

# Model negation in features
def feature_negation(tweet_list):
    tweets = []
    for t in tweet_list:
        #print review
        t = re.sub(r'\b(?:no|never|not)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1NOT_\2',match.group(0)), t)
        tweets.append(t)
    return tweets

# Model length of tweet in features
def feature_tweet_length(tweet_list, short_thr = 10, long_thr = 25):
    tweets = []
    for t in tweet_list:
        tweet_len = len(t.split())  # how many words in a single tweet
        if tweet_len < short_thr:
            t += ' tweet_len_short'
        elif tweet_len > long_thr:
            t += ' tweet_len_long'
        else:
            t += ' tweet_len_mid'
        tweets.append(t)
    return tweets # list of string [tweet1, tweet 2]

# Model pos tag of tweet in feature
def feature_pos(tweet_list): #tweet_list: list of string [tweet1, tweet2, ...]
    asthma_key_words = ['asthma','inhaler','wheezing','snezing','runy'] # keywords after cleaning and stem
    tweets = []
    for tweet in tweet_list: # tweet is one tweet
        key_words = [i for i in tweet.split() if i in asthma_key_words]
        pos_feature = ''
        if len(key_words)!=0:
            pos_tweet = pos_tag(tweet.split()) # only do the pos tag when there are key-words in the text
            for k in key_words:
                for ind,wp in enumerate(pos_tweet):  #wp[0]:words, wp[1]:pos tag
                    if wp[0] == k:
                        try: # in case the index out of range
                            pos_feature += ' '+pos_tweet[ind-1][0]+'/'+pos_tweet[ind-1][1]+'_'+wp[0]+'/'+wp[1]+'_'+pos_tweet[ind+1][0]+'/'+pos_tweet[ind+1][1]
                        except:
                            pos_feature += ' '+wp[0]+'/'+wp[1]
        tweet += pos_feature
        tweets.append(tweet)
    return tweets


# define a function to get feature matrix
def get_features(labels, tweets, vb = None, min = 1, max = 1.0, ngram = (1,1)):
    # parameters:
    # labels:
    # tweets:
    # vb: vocabulary for CountVectorizer()
    # max & min: document frequency threshold for CountVectorizer() ->
    # max & min: If float: proportion of documents, integer: absolute counts.
    # ngram: ngram range for CountVectorizer() -> (1,n), min_n <= n <= max_n will be used
    count_vec = CountVectorizer(vocabulary=vb, min_df=min, max_df=max, ngram_range=ngram)
    #features = []
    data = count_vec.fit_transform(tweets).toarray()
    data = lil_matrix(data) # sparse matrices
    target = numpy.array(labels)
    #return feature matrix, target, features
    return data, target, count_vec.vocabulary_


def performance(label,prediction):
    side_by_side = numpy.transpose(numpy.array([label.tolist(), prediction.tolist()]))
    total = 0
    correct = 0
    total_per_class = {}
    predicted_per_class = {}
    correct_per_class = {}
    counts_per_label = {}
    for i in range(side_by_side.shape[0]):
        gold = side_by_side[i,0]
        pred = side_by_side[i,1]
        total += 1
        if(gold == pred):
            correct += 1
            correct_per_class[gold] = correct_per_class.get(gold, 0) + 1
        total_per_class[gold] = total_per_class.get(gold, 0) + 1
        predicted_per_class[pred] = predicted_per_class.get(pred, 0) + 1
    acc = float(correct) / float(total)
    #print "Accuracy:", acc, "correct:", correct, "total:", total
    for l in total_per_class.keys():
        my_correct = correct_per_class.get(l, 0)
        my_pred = predicted_per_class.get(l, 0.001)
        my_total = total_per_class.get(l, 0.001)
        p = float(my_correct) / float(my_pred)
        r = float(my_correct) / float(my_total)
        f1 = 0
        if p != 0 and r != 0:
            f1 = 2*p*r / (p + r)
        #print "Label", l, " => Precision:", p, "Recall:", r, "F1:", f1
        counts_per_label[l]=(p,r,f1) # precision, recall, f1
    return acc, correct, total, counts_per_label

################################################################################################
################################################################################################
################################################################################################
################################################################################################

## tweets cleaning
path = "..//data//ecigarette_training.csv"


labels, tweets = read_file(path) # labels: list, tweets: list of string
#tweets = feature_negation(tweets)
#tweets = feature_tweet_length(tweets)
#tweets = feature_pos(tweets)


data, labels, vb_train = get_features(labels, tweets)
print 'Number of features:', len(vb_train.keys())

#################################################################################
# part 1: simple test
'''
### x-fold validation
clf = LinearSVC()
scores = cross_validation.cross_val_score(clf, data, labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
#################################################################################
# part 2.1: compare different classifiers
'''
### different molds, classification report matrix
models = [LinearSVC, LogisticRegression, MultinomialNB, Perceptron]
target_names = ['irrelevant', 'relevant']
x_fold = 10
for model in models:
    print model.__name__
    accuracy,correct,total =0,0,0
    per_label = {u'1': [0.0, 0.0, 0], u'0': [0.0, 0.0, 0]}
    for i in range(x_fold):
        clf = model()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=x_fold)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        a,c,t,counts_per_label= performance(y_test,y_pred)
        accuracy+=a
        correct+=c
        total+=t
        for lab in per_label:
            per_label[lab][0]+=counts_per_label[lab][0] # Precision
            per_label[lab][1]+=counts_per_label[lab][1] # Recall
            per_label[lab][2]+=counts_per_label[lab][2] # F1
    print '%d_fold validation:' % x_fold
    print "Accuracy:", accuracy/x_fold, "correct:", correct/x_fold, "total:", total/x_fold
    for lab in per_label:
        print "Label", lab, " => Precision:", per_label[lab][0]/x_fold, "Recall:", per_label[lab][1]/x_fold, "F1:", per_label[lab][2]/x_fold
'''
#################################################################################
# part 2.2: compare different classifiers
'''
### decision tree and two tree based ensemble classifier
models = [DecisionTreeClassifier, ExtraTreesClassifier, RandomForestClassifier]
target_names = ['irrelevant', 'relevant']
x_fold = 10
for model in models:
    print model.__name__
    accuracy, correct, total = 0, 0, 0
    per_label = {u'1': [0.0, 0.0, 0], u'0': [0.0, 0.0, 0]}
    for i in range(x_fold):
        clf = model()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=x_fold)
        clf.fit(X_train.toarray(), y_train)
        y_pred = clf.predict(X_test.toarray())
        a,c,t,counts_per_label= performance(y_test,y_pred)
        accuracy+=a
        correct+=c
        total+=t
        for lab in per_label:
            per_label[lab][0] += counts_per_label[lab][0]  # Precision
            per_label[lab][1] += counts_per_label[lab][1]  # Recall
            per_label[lab][2] += counts_per_label[lab][2]  # F1
    print '%d_fold validation:' % x_fold
    print "Accuracy:", accuracy/x_fold, "correct:", correct/x_fold, "total:", total/x_fold
    for lab in per_label:
        print "Label", lab, " => Precision:", per_label[lab][0]/x_fold, "Recall:", per_label[lab][1]/x_fold, "F1:", per_label[lab][2]/x_fold
'''

#################################################################################
# part 3.1: compare different feature frequency, min_range
'''
# feature threshold cut-off, min_df
min_range = numpy.arange(0,0.4,0.03).tolist()
for min_df in min_range:
    print 'Current min_df: ', min_df
    data, labels, vb_train = get_features(labels, tweets, min=min_df )
    print 'Number of features:', len(vb_train.keys())
    #models = [DecisionTreeClassifier,LogisticRegression,LinearSVC] # 3 model with based performance
    models = [MultinomialNB]
    #models = [ExtraTreesClassifier]  # classifier with best performance
    target_names = ['irrelevant', 'relevant']
    x_fold = 10
    for model in models:
        print model.__name__
        accuracy, correct, total =0,0,0
        per_label = {u'1': [0.0, 0.0, 0], u'0': [0.0, 0.0, 0]}
        for i in range(x_fold):
            clf = model()
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=x_fold)
            #clf.fit(X_train.toarray(), y_train)
            #y_pred = clf.predict(X_test.toarray())
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            a,c,t,counts_per_label= performance(y_test,y_pred)
            accuracy+=a
            correct+=c
            total+=t
            for lab in per_label:
                per_label[lab][0]+=counts_per_label[lab][0] # Precision
                per_label[lab][1]+=counts_per_label[lab][1] # Recall
                per_label[lab][2]+=counts_per_label[lab][2] # F1
        print '%d_fold validation:' % x_fold
        print "Accuracy:", accuracy/x_fold, "correct:", correct/x_fold, "total:", total/x_fold
        for lab in per_label:
            print "Label", lab, " => Precision:", per_label[lab][0]/x_fold, "Recall:", per_label[lab][1]/x_fold, "F1:", per_label[lab][2]/x_fold
'''
#################################################################################
# part 3.2: compare different feature frequency, max_df
'''
# feature threshold cut-off, max_df
max_range = numpy.arange(0.5,1.01,0.03).tolist()
for max_df in max_range:
    print 'Current max_df: ', max_df
    data, labels, vb_train = get_features(labels, tweets, max=max_df )
    print 'Number of features:', len(vb_train.keys())
    #models = [DecisionTreeClassifier,LogisticRegression,LinearSVC] # 3 model with based performance
    models = [MultinomialNB]
    #models = [ExtraTreesClassifier]  # classifier with best performance
    target_names = ['irrelevant', 'relevant']
    x_fold = 10
    for model in models:
        print model.__name__
        accuracy,correct,total =0,0,0
        per_label = {u'1': [0.0, 0.0, 0], u'0': [0.0, 0.0, 0]}
        for i in range(x_fold):
            clf = model()
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=x_fold)
            #clf.fit(X_train.toarray(), y_train)
            #y_pred = clf.predict(X_test.toarray())
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            a,c,t,counts_per_label= performance(y_test,y_pred)
            accuracy+=a
            correct+=c
            total+=t
            for lab in per_label:
                per_label[lab][0]+=counts_per_label[lab][0] # Precision
                per_label[lab][1]+=counts_per_label[lab][1] # Recall
                per_label[lab][2]+=counts_per_label[lab][2] # F1
        print '%d_fold validation:' % x_fold
        print "Accuracy:", accuracy/x_fold, "correct:", correct/x_fold, "total:", total/x_fold
        for lab in per_label:
            print "Label", lab, " => Precision:", per_label[lab][0]/x_fold, "Recall:", per_label[lab][1]/x_fold, "F1:", per_label[lab][2]/x_fold
'''
#################################################################################
# part 4: compare different features type
'''
### unigram + bigram + based classifier based on previous experiments
models = [LogisticRegression,LinearSVC]
target_names = ['irrelevant', 'relevant']
x_fold = 10
for model in models:
    print model.__name__
    accuracy,correct,total =0,0,0
    per_label = {u'1': [0.0, 0.0, 0], u'0': [0.0, 0.0, 0]}
    for i in range(x_fold):
        clf = model()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=x_fold)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        a,c,t,counts_per_label= performance(y_test,y_pred)
        accuracy+=a
        correct+=c
        total+=t
        for lab in per_label:
            per_label[lab][0]+=counts_per_label[lab][0] # Precision
            per_label[lab][1]+=counts_per_label[lab][1] # Recall
            per_label[lab][2]+=counts_per_label[lab][2] # F1
    print '%d_fold validation:' % x_fold
    print "Accuracy:", accuracy/x_fold, "correct:", correct/x_fold, "total:", total/x_fold
    for lab in per_label:
        print "Label", lab, " => Precision:", per_label[lab][0]/x_fold, "Recall:", per_label[lab][1]/x_fold, "F1:", per_label[lab][2]/x_fold
'''


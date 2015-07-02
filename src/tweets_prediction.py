
from __future__ import division
import io, re, nltk, numpy, operator
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from utility import *

"""
predicting each tweet is relevant or irrelevant
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


# Part *: get label for the data set
## training part
path = "..//data//training//tweetsAnnotation.csv"
labels, tweets = read_file(path) # labels: list, tweets: list of string
#tweets = feature_negation(tweets)
#tweets = feature_tweet_length(tweets)
#tweets = feature_pos(tweets)

data, labels, vb_train = get_features(labels, tweets)
v = CountVectorizer(vocabulary=vb_train) # get the feature vector

clf = LinearSVC()
clf.fit(data, labels)


## predicting part
#### input file: tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

input_files = ['2013Nov_states','2013Dec_states','2014Jan_states','2014Feb_states', '2014Mar_states', '2014Apr_states', '2014May_states', '2014Jun_states']
input_files = ['2013Dec_states','2014Jan_states','2014Feb_states', '2014Mar_states', '2014Apr_states', '2014May_states', '2014Jun_states']

for file_name in input_files:

    input_path = "..//data//" + file_name
    output_path = "..//data//" + file_name + "_pred"

    with io.open(input_path, 'r', encoding='utf-8', errors='ignore', newline='\n') as infile:
        for line in infile:       
            line = line.split('|')
            #print len(line)
            if len(line) != 5:  ## don't need this if tweets_get_states.py didn't get empty line
                continue
                
            tweet_id,created_at,text,location,time_zone = line[0],line[1],line[2],line[3],line[4].strip() # strip: remove \n
            
            text = pre_process(text)
            # predict label for a single tweet
            tweet_vec = v.fit_transform([text]).toarray() # transform single tweet to array
            prediction = clf.predict(tweet_vec)[0] # get prediction for a single tweet
            
            #text = text.replace('"', '')  #remove " for mysql, use " as ENCLOSED
            #time_zone = time_zone.replace('"', '')
            text = strip_non_ascii(text)
            time_zone = strip_non_ascii(time_zone)            
            
            with io.open(output_path, mode='a',encoding='utf-8') as f:
                f.write('"'+prediction+'"|"'+tweet_id+'"|"'+created_at+'"|"'+text+'"|"'+location+'"|"'+time_zone+'"'+'\n') 



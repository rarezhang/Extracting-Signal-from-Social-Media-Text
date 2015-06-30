__author__ = 'wlz'
import pymongo, re, codecs
from convert_twitter_timedate import *

#######################################################################
# so far can only be used on VM, connect to MongoDB and create csv file
#######################################################################

# connect to mongodb
client = pymongo.MongoClient('127.0.0.1', 27017)  # this is the local host

# connect to database
##db = client.twitterData  # twitter data 2013: 1319150
db = client.twitterData2014 # twitter data 2014: 4197978
print db

# connect to the collection
tweets = db.tweets


########################################################################
# print out tweets by month
##reg = ['Nov', 'Dec']
reg = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

for regexp in reg:
    cur = db.tweets.find({"created_at": {"$regex": regexp}},
                         {'id_str': 1, 'created_at': 1, 'text': 1, 'user.location': 1, 'user.time_zone': 1})

    #path = '../data/2013' + regexp
    path = '../data/2014' + regexp

    print 'path to the result:', path
    with codecs.open(path, 'a', 'utf-8') as f:
        for c in cur:
            # c: a single tweet. type:dictionary
            tweet_id = c['id_str']
            created_at = c['created_at']  # keep all time information
            created_at = str(convert_twitter_timedate(created_at))
            text = c['text'].replace('|', ' ')  # text: remove '|' in text -> use'|'as split
            text = text.replace('\r', ' ').replace('\n', ' ')  # combine multiple lines
            location = c['user']['location'].replace('|', ' ')  # remove '|'-> use'|'as split
            location = location.replace('\r', ' ').replace('\n', ' ')  # combine multiple lines
            try:
                time_zone = c['user']['time_zone'].replace('|', ' ')  # remove '|'-> use'|'as split
                time_zone = time_zone.replace('\r', ' ').replace('\n', ' ')  # combine multiple lines
            except:
                time_zone = 'None'
            # writer.writerow([tweet_id,created_at,text,location])
            # print tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone+'\n'
            f.write(tweet_id + '|' + created_at + '|' + text + '|' + location + '|' + time_zone + '\n')

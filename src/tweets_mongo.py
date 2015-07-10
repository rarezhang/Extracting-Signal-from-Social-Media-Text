__author__ = 'wlz'

import pymongo, re, codecs
from utility import *



# connect to mongodb
client = pymongo.MongoClient('127.0.0.1', 27017) # this is the local host

# connect to database
##db = client.twitterData2013 # twitter data 2013: 1319150
db = client.twitterData2014 # twitter data 2014: 4197978
print db

# connect to the collection
tweets = db.tweets


########################################################################
# print out tweets by month
##reg, year = ['Nov','Dec'], '2013'
reg, year = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun'], '2014'


for regexp in reg:
    # user.location: not null, not tab, not empty string, !!! too slow !!!
    #cur = db.tweets.find({"$and":[{"created_at": {"$regex": regexp}},{"user.location": {'$ne': None}},{"user.location": {'$ne': "\t"}},{"user.location": {'$ne': ""}}]},{'id_str':1,'created_at':1,'text':1,'user.location':1,'user.time_zone':1})
    
    cur = db.tweets.find({"created_at": {"$regex": regexp}},{'id_str':1,'created_at':1,'text':1,'user.location':1,'user.time_zone':1})

    path = '../data/' + year + regexp     
    print 'path to the result:', path
    
    with codecs.open(path, 'a','utf-8',errors='ignore') as f:	
	    for c in cur:
		    # c: a single tweet. type:dictionary
		    location = c['user']['location'] 
		    location = re.sub('[\W_]+', ' ', location) # remove everything except alphanumeric
		    location = re.sub( '\s+', ' ', location).strip() # substitute multiple whitespace with 1
		    if location != "" and location != " ":	
		        tweet_id = c['id_str']
		        created_at = c['created_at']  #keep all time information
		        created_at = str(convert_twitter_timedate(created_at))
		        text = c['text'].replace('|',' ') #text: remove '|' in text -> use'|'as split
		        text = text.replace('\r',' ').replace('\n',' ') #combine multiple lines		    
		        try:
		            time_zone = c['user']['time_zone']
		            time_zone = re.sub('[\W_]+',' ', time_zone) # remove everything except alphanumeric
		        except:
		            time_zone = 'None'
		
		        #print tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone+'\n'	
		        f.write(tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone+'\n')
		

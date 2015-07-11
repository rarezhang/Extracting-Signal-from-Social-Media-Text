# run experiments on training data set
## *. tweets_cleaning.py
###1. 4500 asthma tweets: could run it on ISTA 555 project, already got results; also can use this one, change data path to tweetsAnnotation.csv
###2. e-cigarette: find "# e-cigarette"
###3. 500 fresh asthma tweets: find "# manually re-label 500 fresh tweets"


# Processing 12 months asthma twitter data set 
## 1. tweets_mongo.py
read tweets from mongoDB, month by month.
tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

## 2. tweets_get_states.py
location resolution, only keep tweets whose location can be recognized as US state names.
tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

## 3. tweets_prediction.py
predicting each tweet is relevant (1) or irrelevant (0)
prediction+'|'+tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

## 4. tweets_mysql.py
load prediction results into mysql.
group by 'prediction': relevant / irrelevant
         'location': states name
         'created_at': daily / weekly / monthly

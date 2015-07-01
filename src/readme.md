## *. tweets_cleaning.py
run experiments on training data set (labeled tweets) 

# processing 12 months asthma twitter data set 

## 1. tweets_mongo.py
read tweets from mongoDB, month by month.
tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

## 2. tweets_get_states.py
location resolution, only keep tweets whose location can be recognized as US state names.
tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

## 3. tweets_prediction.py
predicting each tweet is relevant or irrelevant
prediction+'|'+tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone

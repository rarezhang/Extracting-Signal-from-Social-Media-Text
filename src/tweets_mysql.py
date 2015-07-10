#!/user/bin/env python
# coding=utf-8
"""
python connect with mysql
"""

import MySQLdb
import io


# connect python with mysql
db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd='qwer1234',
                     db='TWITTER',   # name of the db
                     local_infile = 1)
# create a Cursor object
## Cursor can execute all the queries
cur = db.cursor()


#############################################################################################
# pre-process the table 
### better run this code in mysql
### mysql --local-infile -uroot -p

# create table 
## prediction+'|'+tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone
'''
DROP TABLE IF EXISTS twitter;
'''

"""
CREATE TABLE twitter 
( 
prediction INT(1) NOT NULL,
tweet_id BIGINT NOT NULL, 
created_at VARCHAR(30) NOT NULL, 
text VARCHAR(255), 
location CHAR(5), 
time_zone VARCHAR(255)
);
"""


# load data infile
"""
LOAD DATA LOCAL INFILE '/home/wenli/Projects/tweetsnlp/data/2014Jun_states_pred' 
INTO TABLE twitter
FIELDS TERMINATED BY '|'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
"""



# change data type of created_at
"""
UPDATE twitter SET created_at = substring(created_at, 1, 19);
ALTER TABLE twitter MODIFY created_at DATETIME;
"""



# build indexes: tweet_id, created_at, location, prediction 
#### DROP INDEX tweet_id ON twitter;
"""
ALTER TABLE twitter ENGINE MyISAM;
ALTER IGNORE TABLE twitter ADD UNIQUE KEY tweet_id(tweet_id);
ALTER TABLE twitter ENGINE InnoDB;

CREATE INDEX created_at on twitter (created_at);
CREATE INDEX location on twitter (location);
CREATE INDEX prediction on twitter (prediction);
"""

#############################################################################################
# show columns
'''
sql = "show columns in twitter;"
'''


#############################################################################################
'''
# twitter data group by "created_at": year / month
# compare with pollution data
sql = ('select count(prediction), location, year(created_at), month(created_at) '
       'from twitter '
       'where prediction = 1 '
       'group by year(created_at), month(created_at), location '
       'order by location;')

# run the qurey
cur.execute(sql)

for row in cur.fetchall():
    ## location , year/month, count(prediction)
    print row[1],str(row[2])+'/'+str(row[3]),row[0]
'''
#############################################################################################

# twitter data group by "location"
# compare with asthma data

output_path = output_path = "..//data//twitter_agg_state" 

sql = ('select count(prediction), location '
       'from twitter '
       'where prediction = 1 '
       'group by location '
       'order by location;')

# run the qurey
cur.execute(sql)

for row in cur.fetchall():
    ## location , count(prediction)
    state,num_tweets = row[1],unicode(row[0])
    with io.open(output_path, mode='a',encoding='utf-8') as f:
        f.write('"'+state+'","'+num_tweets+'"'+'\n') 

__author__ = 'wlz'

import re, json, io

states = {
'alaska': 'AK',
'alabama': 'AL',
'arkansas': 'AR',
'arizona': 'AZ',
'california': 'CA',
'colorado': 'CO',
'connecticut': 'CT',
'delaware': 'DE',
'florida': 'FL',
'georgia': 'GA',
'guam': 'GU',
'hawaii': 'HI',
'iowa': 'IA',
'idaho': 'ID',
'illinois': 'IL',
'indiana': 'IN',
'kansas': 'KS',
'kentucky': 'KY',
'louisiana': 'LA',
'massachusetts': 'MA',
'maryland': 'MD',
'maine': 'ME',
'michigan': 'MI',
'minnesota': 'MN',
'missouri': 'MO',
'mississippi': 'MS',
'montana': 'MT',
'national': 'NA',
'nebraska': 'NE',
'ohio': 'OH',
'oklahoma': 'OK',
'oregon': 'OR',
'pennsylvania': 'PA',
'tennessee': 'TN',
'texas': 'TX',
'utah': 'UT',
'virginia': 'VA',
'vermont': 'VT',
'washington': 'WA',
'wisconsin': 'WI',
'wyoming': 'WY'
}
states_bi = {
'north carolina': 'NC',
'north dakota': 'ND',
'new hampshire': 'NH',
'new jersey': 'NJ',
'new mexico': 'NM',
'nevada': 'NV',
'new york': 'NY',
'west virginia': 'WV',
'puerto rico': 'PR',
'rhode island': 'RI',
'south carolina': 'SC',
'south dakota': 'SD',
'virgin islands': 'VI'
}

city_state_unigram = json.load(open('../data/training/city_state_dic_unigram.txt'))
city_state_bigram = json.load(open('../data/training/city_state_dic_bigram.txt'))

def find_bigrams(input_list):
  bigrams = zip(input_list,input_list[1:])
  result =[]
  for bi in bigrams:
      result.append(' '.join(bi).lower())
  return result

## input: monthly twitter data from MongoDB
## get the data from tweets_mongo.py
## tweet_id | created_at | text | location | time_zone
files = ['2014Jan','2014Feb', '2014Mar', '2014Apr', '2014May', '2014Jun']
files = ['test']
#file_name = '2014Jan'
#file_name = 'test'

for file_name in files:
    input_path = '../data/' + file_name

    ## output: monthly twitter data, only keep tweets whose location can be recognized as US states names.
    output_path = '../data/' + file_name + '_states'

    with io.open(input_path, 'r', encoding='utf-8', errors='ignore', newline='\n' ) as infile:
        for line in infile:
            line = line.split('|')

            # location: split to word token
            loc = line[3].split(' ')  # location splited by space, tokenize        
            location = False
            if len(loc) < 4: # if len of loc>=4, great chance it is not a location            
                for word in loc:
                    if re.match('[A-Z]{2}', word) and word in states.values(): # states abbreviation
                        # already give the states name, abbreviation
                        location = word
                    elif word.lower() in states.keys(): # states full name
                        location = states[word.lower()]
                    elif word.lower() in city_state_unigram.keys(): # city name
                        location = city_state_unigram[word.lower()]
                    else: # check the bigram
                        bi_loc = find_bigrams(loc)
                        for bi_word in bi_loc:
                            if bi_word in states_bi.keys(): # states full name
                                location = states_bi[bi_word]
                            elif bi_word in city_state_bigram.keys(): # city name
                                location = city_state_bigram[bi_word]
            if location:
                with io.open(output_path, mode='a',encoding='utf-8') as f:
                    tweet_id, created_at, text, time_zone = line[0], line[1], line[2], line[4]
                    f.write(tweet_id+'|'+created_at+'|'+text+'|'+location+'|'+time_zone) # don't need /n, since time_zone already have /n 





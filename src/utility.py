__author__ = 'wlz'

def convert_twitter_timedate(twitterdate, fromtimezone='UTC', totimezone='UTC'):
    import datetime, time, codecs
    from dateutil import tz
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
    
#print convert_twitter_timedate('Tue Mar 29 08:11:25 +0000 2011')


def strip_non_ascii(string, starting_ord = 32, ending_ord = 126):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if ord(c)==32 or 65<= ord(c)<=90 or starting_ord <= ord(c) <= ending_ord) # 32:space; 65-90:[A-Z]; 97-121:[a-z]
    #stripped = (c for c in string if starting_ord <= ord(c) <= ending_ord)
    return ''.join(stripped)

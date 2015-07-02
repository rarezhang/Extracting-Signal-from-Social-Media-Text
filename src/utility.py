def strip_non_ascii(string, starting_ord = 97, ending_ord = 122):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if ord(c)==32 or 65<= ord(c)<=90 or starting_ord <= ord(c) <= ending_ord)
    return ''.join(stripped)

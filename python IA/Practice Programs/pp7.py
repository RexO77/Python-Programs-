import re
def validpan(numstr):
    pan_pattern = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')
    if pan_pattern.match(numstr):
        print("It is a Valid PAN.")
    else:
        print("It is not a valid PAN.")
validpan('ABCDE1223E')
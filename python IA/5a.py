import re
def isphnum(numStr):
    if len(numStr) != 12:
        return False
    for i in range(len(numStr)):
        if i==3 or i==7:
            if numStr[i] != '-':
                return False
        else:
            if numStr[i].isdigit()==False:
                return False
    return True
def chkphnum(numStr):
    ph_pattern = re.compile(r'^\d{3}-\d{3}-\d{4}$')
    if ph_pattern.match(numStr):
        return True
    else:
        return False
ph_num= input("Enter a phone Number: ")
if isphnum(ph_num):
    print("Valid Phone Number")
else: 
    print("Invalid Phone Number ")
print("Using Regular Expression ")
if chkphnum(ph_num):
    print("Valid Phone Number")
else: 
    print("Invalid Phone Number ")  
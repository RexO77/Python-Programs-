s = input('Enter a sentence : ')
d=w=u=l=0
length = s.split()
w = len(length)
for c in s:
    if c.isupper():
        u+=1
    if c.islower():
        l+=1
    if c.isdigit():
        d+=1
print("No of words : ",w)
print("No of Uppercase char : ",u)
print("No of lowercase char : ",l)
print("No of digits : ",d)
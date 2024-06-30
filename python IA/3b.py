str1 = input("Enter String 1 : ")
str2 = input("Enter String 2 : ")
if len(str2)<len(str1):
    short = len(str2)
    long = len(str1)
else:
    short = len(str1)
    long = len(str2)
count =0
for i in range(short):
    if str1[i]==str2[i]:
        count+=1
print("Similarity b/w 2 strings are : ",count/long)
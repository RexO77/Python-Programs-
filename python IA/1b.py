print("Palindrome of Integers")
val = int(input("Enter a Integer :"))
str_val = str(val)
if str_val == str_val[::-1]:
    print("Palindrome")
else:
    print("Not Palindrome")

for i in range(10):
    if str_val.count((str(i)) > 0):
        print(str(i)+" is repeated "+str_val.count(str(i))+" times")
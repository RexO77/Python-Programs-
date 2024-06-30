class palstr:
    def __init__(self):
        self.ispali = False
    def chkpali(self,mystr):
        if mystr == mystr[::-1]:
            self.ispali = True
        else:
            self.ispali = False
        return self.ispali
class paliInt(palstr):
    def __init__(self):
        self.ispali = False
    def chkpali(self,val):
        temp = val
        rev = 0
        while(temp!=0):
            rem = temp%10
            rev = (rev*10)+rem
            temp = temp//10
        if val == rev:
            self.ispali = True
        else:
            self.ispali = False
        return self.ispali
st = input("Enter the String : ")
obj = palstr()
if obj.chkpali(st):
    print("It is a Palindrome")
else:
    print("It is not a palindrome")

val = int(input("Enter an Integer :"))
obj1 = paliInt()
if obj1.chkpali(val):
    print("It is a palindrome")
else:
    print("It is not a palindrome")
def romanToDec(romStr):
    romanDict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'K':1000}
    romanBack = list(romStr)[::-1]
    value = 0
    rightVal = romanDict[romanBack[0]]
    for numeral in romanBack:
        leftVal = romanDict[numeral]
        if leftVal<rightVal:
            value-=leftVal
        else:
            value+=leftVal
        rightVal=leftVal
    return value
romanStr = input("Enter a roman number :")
print("Decimal Value : ",romanToDec(romanStr))
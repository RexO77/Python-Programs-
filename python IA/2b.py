def binaryToDecimal(binary):
    print("Binary Value : ",binary)
    decimal,i=0,0
    while(binary!=0):
        dec = binary%10
        decimal = decimal+dec*pow(2,i)
        binary = binary//10
        i+=1
        
    print("Equavalent Decimal Value : ",decimal)
binaryToDecimal(1011)
#octel to Hex
def octalToHex(n):
    print("Octel Value =",n)
    decnum = int(n,8)
    hexadecimal = hex(decnum).replace("ox"," ")
    print("Eqivalent Hex value = ",hexadecimal)
octalToHex('7744')
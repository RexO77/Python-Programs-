m1 = int(input("Enter Marks of Test 1 :"))
m2 = int(input("Enter Marks of Test 2 :"))
m3 = int(input("Enter Marks of Test 3 :"))
if m1<=m2 and m1<=m3:
    avg = (m2+m3)/2
elif m2<=m1 and m2<=m3:                         #Best of 2 marks
    avg = (m1+m3)/2
else:
    avg =(m1+m2)/2
print("Average of best 2 marks is :",avg)
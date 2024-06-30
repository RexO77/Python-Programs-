inputFile = 'file.text'
N = int(input("Enter the N values :"))
with open(inputFile,'r')as filedata:
    read_line= filedata.readlines()
print("First few lines of the file is ",N,"of the text file")
for text in (read_line[:N]):
    print(text,end='')
filedata.close()
word = input("Enter the word to be searched :")
k=0
with open(inputFile,'r')as f:
    for line in f:
        word = line.split()
    for i in word :
        if(i==word):
            k= k+1
print("Occourence is : ",k)
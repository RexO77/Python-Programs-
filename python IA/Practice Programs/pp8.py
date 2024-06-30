inputFile = "C:\Programming\Python Programs\python IA\Practice Programs\\text.txt"
N = int(input("Enter N values : "))
with open(inputFile,'r') as filedata:
    read_line = filedata.readlines()
print("The First ",N,"lines of file are : ")
for text in (read_line[:N]):
    print(text,end='')
filedata.close()
k=0
word = input("Enter the Word to be searched :")
with open(inputFile,'r')as f:
    for line in f:
        word = line.split()
    for i in word:
        if(i == word):
            k=k+1
print(f"Occourence in word {word} is ",k)
with open(r"D:\Exam Stuff\text.txt",'r')as file:
    text = file.read()
import re
phnum = re.findall(r'\+91\d{10}',text)
email= re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',text)
print("phone numbers detected: ",phnum)
print("Email ID detected : ",email)